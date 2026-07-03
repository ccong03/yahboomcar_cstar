#!/usr/bin/env python3
"""
CStar waypoint planner for the single-layer dense RCG framework.

Design:
1. The planner subscribes to one dense RCG graph only:
   /cstar/rcg_nodes
   /cstar/rcg_markers

2. The robot follows this single graph with a simple boustrophedon-style
   waypoint rule.

3. Hole detection is intentionally simple:
   - no dense/backbone layering;
   - no boundary_lap gate;
   - no doorway predictor;
   - only detect during same-lap sweeping;
   - build a thin current-lap scan band instead of a BFS local window;
   - use side-neighbor seeds leaving that scan band;
   - floodfill open connected components from those seeds;
   - classify the component as hole / frontier / normal-lap by size, bbox,
     unknown ratio, and attachment count.
"""

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener


NodeKey = Tuple[int, int]
GridCell = Tuple[int, int]


class CStarWaypointPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_waypoint_planner_node')

        # ========== Single dense RCG input ==========
        self.declare_parameter('rcg_nodes_topic', '/cstar/rcg_nodes')
        self.declare_parameter('rcg_markers_topic', '/cstar/rcg_markers')

        # ========== Map topics ==========
        self.declare_parameter('covered_map_topic', '/cstar/covered_map')
        self.declare_parameter('free_map_topic', '/cstar/free_map')
        self.declare_parameter('obstacle_map_topic', '/cstar/obstacle_map')
        self.declare_parameter('unknown_map_topic', '/cstar/unknown_map')

        # ========== Frames / timing ==========
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('update_period', 0.5)
        self.declare_parameter('position_quantization', 0.05)

        # ========== Goal following ==========
        self.declare_parameter('snap_distance', 0.50)
        self.declare_parameter('goal_center_tolerance', 0.10)
        self.declare_parameter('retreat_center_tolerance', 0.10)
        self.declare_parameter('closed_position_radius', 0.12)
        self.declare_parameter('covered_close_threshold', 50)
        self.declare_parameter('use_covered_map_for_closing', True)

        # Boustrophedon-style neighbor selection.
        self.declare_parameter('initial_sweep_direction', -1.0)
        self.declare_parameter('same_lap_y_tolerance', 0.14)
        self.declare_parameter('same_col_x_tolerance', 0.16)
        self.declare_parameter('allow_diagonal_fallback', True)

        # Dense graph can be locally grid-like. When no direct open neighbor is
        # chosen, optionally walk along graph to the nearest open node.
        self.declare_parameter('enable_graph_transit_to_open', True)
        self.declare_parameter('max_graph_transit_hops', 250)

        # ========== Dead-end retreat / escape ==========
        # Restore the previous retreat logic.  This is only used when the normal
        # C* neighbor policy and graph-transit fallback both cannot find a next
        # open node.  It is independent from hole repair.
        self.declare_parameter('retreat_attach_radius', 0.35)
        self.declare_parameter('allow_open_fallback', True)
        self.declare_parameter('obstacle_buffer', 0.20)
        self.declare_parameter('unknown_buffer', 0.10)
        self.declare_parameter('map_border_buffer', 0.15)
        self.declare_parameter('nearest_safe_search_radius', 0.60)
        self.declare_parameter('escape_resample_step', 0.08)

        # ========== Simple local floodfill hole detection ==========
        self.declare_parameter('enable_hole_detection', True)
        self.declare_parameter('hole_dynamic_update_period', 0.5)
        # 当前 lap 前方扫描带。只在 same-lap 横向覆盖阶段检测侧边 hole。
        self.declare_parameter('hole_local_window_distance', 1.20)
        self.declare_parameter('hole_scan_back_margin', 0.15)
        self.declare_parameter('hole_same_lap_only', True)
        # 旧的 BFS window 已删除，本参数保留读取但不再用于扩展整片局部图。
        self.declare_parameter('hole_local_window_max_hops', 18)

        # A candidate branch is an edge leaving the local current path/window.
        # The edge should have a clear lateral component relative to the local
        # motion direction current_key -> next_goal_key.
        self.declare_parameter('hole_branch_lateral_min_distance', 0.12) #0.12
        self.declare_parameter('hole_branch_lateral_ratio', 1.15)#1.15
        self.declare_parameter('hole_branch_max_along_offset', 0.35) #0.35
        self.declare_parameter('hole_max_branch_seed_distance', 0.55)
        self.declare_parameter('hole_max_seed_candidates', 12)

        # Component filters.
        self.declare_parameter('hole_min_nodes', 4)
        self.declare_parameter('hole_max_nodes', 180)
        self.declare_parameter('hole_min_bbox_area', 0.04)
        self.declare_parameter('hole_max_bbox_area', 5.00)
        self.declare_parameter('hole_max_robot_distance', 2.00)
        self.declare_parameter('hole_max_attachment_count', 3)
        self.declare_parameter('hole_min_closed_attach_count', 1)

        # Unknown / frontier rejection.
        self.declare_parameter('unknown_check_radius', 0.25)
        self.declare_parameter('unknown_reject_ratio', 0.35)

        # Floodfill safety cap.
        self.declare_parameter('floodfill_max_nodes', 600)

        # Repair path visualization over accepted hole component.
        self.declare_parameter('enable_hole_repair_path', True)
        self.declare_parameter('hole_repair_y_tolerance', 0.16)
        self.declare_parameter('hole_repair_max_points', 360)

        # Debug visualization for seed candidates.
        self.declare_parameter('publish_branch_candidates', True)
        self.declare_parameter('publish_rejected_branch_candidates', False)

        # ========== Hole execution state machine ==========
        # 检测到侧边 hole 后，planner 暂停普通 C*，先把 /cstar/goal 切到
        # 当前 lap 上的 entry_base，再把补扫路径发布到 /cstar/escape_path。
        self.declare_parameter('enable_hole_execution', True)
        self.declare_parameter('hole_repair_finish_tolerance', 0.12)
        self.declare_parameter('hole_execution_min_path_points', 2)

        # ========== Resampled hole repair path ==========
        # Hole detection remains graph-based and unchanged.  Once a hole is
        # accepted, the repair route is generated from a local free-space mask
        # by resampling regular boustrophedon waypoints, instead of using the
        # original dense RCG node/edge order inside the hole.
        self.declare_parameter('repair_resample_spacing', 0.22)
        self.declare_parameter('repair_resample_line_spacing', 0.26)
        self.declare_parameter('repair_roi_margin', 0.35)
        self.declare_parameter('repair_obstacle_buffer', 0.10)
        self.declare_parameter('repair_unknown_buffer', 0.05)
        self.declare_parameter('repair_scan_band_block_radius', 0.10)
        self.declare_parameter('repair_min_sample_points', 2)
        self.declare_parameter('repair_replan_on_reached', False)
        self.declare_parameter('repair_skip_covered_samples', True)
        self.declare_parameter('repair_goal_min_spacing', 0.08)
        # Keep hole repair stable: decide the sweep direction before entering
        # the hole and execute goals strictly one by one.
        self.declare_parameter('repair_lock_axis_to_cstar_perpendicular', True)
        self.declare_parameter('repair_max_goal_step', 0.28)
        # Keep the resampled repair mask close to the detected hole component.
        # This prevents the local floodfill from leaking into nearby corridors
        # and prevents repair segments from crossing walls/unknown areas.
        self.declare_parameter('repair_component_inflate_radius', 0.32)
        self.declare_parameter('repair_transition_search_margin', 0.45)

        # ========== Read parameters ==========
        self.rcg_nodes_topic = self.get_parameter('rcg_nodes_topic').value
        self.rcg_markers_topic = self.get_parameter('rcg_markers_topic').value
        self.covered_map_topic = self.get_parameter('covered_map_topic').value
        self.free_map_topic = self.get_parameter('free_map_topic').value
        self.obstacle_map_topic = self.get_parameter('obstacle_map_topic').value
        self.unknown_map_topic = self.get_parameter('unknown_map_topic').value

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.update_period = float(self.get_parameter('update_period').value)
        self.position_quantization = float(self.get_parameter('position_quantization').value)

        self.snap_distance = float(self.get_parameter('snap_distance').value)
        self.goal_center_tolerance = float(self.get_parameter('goal_center_tolerance').value)
        self.retreat_center_tolerance = float(self.get_parameter('retreat_center_tolerance').value)
        self.closed_position_radius = float(self.get_parameter('closed_position_radius').value)
        self.covered_close_threshold = int(self.get_parameter('covered_close_threshold').value)
        self.use_covered_map_for_closing = bool(self.get_parameter('use_covered_map_for_closing').value)

        initial_sweep = float(self.get_parameter('initial_sweep_direction').value)
        self.sweep_dir = -1.0 if initial_sweep < 0.0 else 1.0
        self.same_lap_y_tolerance = float(self.get_parameter('same_lap_y_tolerance').value)
        self.same_col_x_tolerance = float(self.get_parameter('same_col_x_tolerance').value)
        self.allow_diagonal_fallback = bool(self.get_parameter('allow_diagonal_fallback').value)

        self.enable_graph_transit_to_open = bool(self.get_parameter('enable_graph_transit_to_open').value)
        self.max_graph_transit_hops = int(self.get_parameter('max_graph_transit_hops').value)

        self.retreat_attach_radius = float(self.get_parameter('retreat_attach_radius').value)
        self.allow_open_fallback = bool(self.get_parameter('allow_open_fallback').value)
        self.obstacle_buffer = float(self.get_parameter('obstacle_buffer').value)
        self.unknown_buffer = float(self.get_parameter('unknown_buffer').value)
        self.map_border_buffer = float(self.get_parameter('map_border_buffer').value)
        self.nearest_safe_search_radius = float(self.get_parameter('nearest_safe_search_radius').value)
        self.escape_resample_step = float(self.get_parameter('escape_resample_step').value)

        self.enable_hole_detection = bool(self.get_parameter('enable_hole_detection').value)
        self.hole_dynamic_update_period = float(self.get_parameter('hole_dynamic_update_period').value)
        self.hole_local_window_distance = float(self.get_parameter('hole_local_window_distance').value)
        self.hole_scan_back_margin = float(self.get_parameter('hole_scan_back_margin').value)
        self.hole_same_lap_only = bool(self.get_parameter('hole_same_lap_only').value)
        self.hole_local_window_max_hops = int(self.get_parameter('hole_local_window_max_hops').value)
        self.hole_branch_lateral_min_distance = float(self.get_parameter('hole_branch_lateral_min_distance').value)
        self.hole_branch_lateral_ratio = float(self.get_parameter('hole_branch_lateral_ratio').value)
        self.hole_branch_max_along_offset = float(self.get_parameter('hole_branch_max_along_offset').value)
        self.hole_max_branch_seed_distance = float(self.get_parameter('hole_max_branch_seed_distance').value)
        self.hole_max_seed_candidates = int(self.get_parameter('hole_max_seed_candidates').value)

        self.hole_min_nodes = int(self.get_parameter('hole_min_nodes').value)
        self.hole_max_nodes = int(self.get_parameter('hole_max_nodes').value)
        self.hole_min_bbox_area = float(self.get_parameter('hole_min_bbox_area').value)
        self.hole_max_bbox_area = float(self.get_parameter('hole_max_bbox_area').value)
        self.hole_max_robot_distance = float(self.get_parameter('hole_max_robot_distance').value)
        self.hole_max_attachment_count = int(self.get_parameter('hole_max_attachment_count').value)
        self.hole_min_closed_attach_count = int(self.get_parameter('hole_min_closed_attach_count').value)
        self.unknown_check_radius = float(self.get_parameter('unknown_check_radius').value)
        self.unknown_reject_ratio = float(self.get_parameter('unknown_reject_ratio').value)
        self.floodfill_max_nodes = int(self.get_parameter('floodfill_max_nodes').value)
        self.enable_hole_repair_path = bool(self.get_parameter('enable_hole_repair_path').value)
        self.hole_repair_y_tolerance = float(self.get_parameter('hole_repair_y_tolerance').value)
        self.hole_repair_max_points = int(self.get_parameter('hole_repair_max_points').value)
        self.publish_branch_candidates = bool(self.get_parameter('publish_branch_candidates').value)
        self.publish_rejected_branch_candidates = bool(self.get_parameter('publish_rejected_branch_candidates').value)
        self.enable_hole_execution = bool(self.get_parameter('enable_hole_execution').value)
        self.hole_repair_finish_tolerance = float(self.get_parameter('hole_repair_finish_tolerance').value)
        self.hole_execution_min_path_points = int(self.get_parameter('hole_execution_min_path_points').value)

        self.repair_resample_spacing = float(self.get_parameter('repair_resample_spacing').value)
        self.repair_resample_line_spacing = float(self.get_parameter('repair_resample_line_spacing').value)
        self.repair_roi_margin = float(self.get_parameter('repair_roi_margin').value)
        self.repair_obstacle_buffer = float(self.get_parameter('repair_obstacle_buffer').value)
        self.repair_unknown_buffer = float(self.get_parameter('repair_unknown_buffer').value)
        self.repair_scan_band_block_radius = float(self.get_parameter('repair_scan_band_block_radius').value)
        self.repair_min_sample_points = int(self.get_parameter('repair_min_sample_points').value)
        self.repair_replan_on_reached = bool(self.get_parameter('repair_replan_on_reached').value)
        self.repair_skip_covered_samples = bool(self.get_parameter('repair_skip_covered_samples').value)
        self.repair_goal_min_spacing = float(self.get_parameter('repair_goal_min_spacing').value)
        self.repair_lock_axis_to_cstar_perpendicular = bool(self.get_parameter('repair_lock_axis_to_cstar_perpendicular').value)
        self.repair_max_goal_step = float(self.get_parameter('repair_max_goal_step').value)
        self.repair_component_inflate_radius = float(self.get_parameter('repair_component_inflate_radius').value)
        self.repair_transition_search_margin = float(self.get_parameter('repair_transition_search_margin').value)

        # ========== Graph state ==========
        self.nodes: Dict[NodeKey, Tuple[float, float]] = {}
        self.raw_edges: List[Tuple[NodeKey, NodeKey]] = []
        self.adj: Dict[NodeKey, Set[NodeKey]] = {}

        # ========== Coverage / motion state ==========
        self.closed_nodes: Set[NodeKey] = set()
        self.closed_positions: List[Tuple[float, float]] = []
        self.current_goal_key: Optional[NodeKey] = None
        self.selected_path: List[Tuple[float, float]] = []

        # Dead-end retreat state.  This is the old grid-A* retreat mechanism.
        self.escape_active = False
        self.escape_path_xy: List[Tuple[float, float]] = []
        self.last_deadend_key: Optional[NodeKey] = None
        self.latest_retreat_candidates: Set[NodeKey] = set()

        # Hole visualization state.
        self.latest_hole_component: Set[NodeKey] = set()
        self.latest_hole_attachments: Set[NodeKey] = set()
        self.latest_hole_entry_key: Optional[NodeKey] = None
        self.latest_hole_exit_key: Optional[NodeKey] = None
        self.latest_hole_repair_path: List[Tuple[float, float]] = []
        self.latest_branch_edge: Optional[Tuple[NodeKey, NodeKey]] = None
        self.latest_branch_candidates: List[Tuple[NodeKey, NodeKey, str]] = []
        self.latest_scan_band: Set[NodeKey] = set()
        self.last_hole_update_time = None

        # Hole execution state machine.
        # COVERAGE: normal C* sweeping.
        # HOLE_ARMED: a side hole has been accepted; drive to entry_base first.
        # HOLE_REPAIR: track /cstar/escape_path through the hole and return to main lap.
        self.mode = 'COVERAGE'
        self.active_hole_component: Set[NodeKey] = set()
        self.active_hole_attachments: Set[NodeKey] = set()
        self.active_hole_entry_base_key: Optional[NodeKey] = None
        self.active_hole_seed_key: Optional[NodeKey] = None
        self.active_hole_exit_key: Optional[NodeKey] = None
        self.active_hole_repair_path: List[Tuple[float, float]] = []
        # Hole repair is executed as a sequence of normal /cstar/goal targets.
        # This avoids pure-pursuit oscillation on /cstar/escape_path when the
        # repair polyline contains tight turns or graph-layout discontinuities.
        self.active_hole_repair_key_path: List[NodeKey] = []  # legacy field kept for compatibility; not used by resampled repair.
        self.active_hole_repair_goal_index: int = 0
        self.active_hole_repair_points: List[Tuple[float, float]] = []
        self.active_hole_current_goal_xy: Optional[Tuple[float, float]] = None
        self.active_hole_branch_edge: Optional[Tuple[NodeKey, NodeKey]] = None
        self.active_hole_visited_repair_points: List[Tuple[float, float]] = []
        self.active_hole_repair_axis: str = 'resampled'
        self.active_hole_repair_axis_locked: bool = False

        # Maps.
        self.covered_map: Optional[OccupancyGrid] = None
        self.covered_data: Optional[List[int]] = None
        self.free_msg: Optional[OccupancyGrid] = None
        self.obstacle_msg: Optional[OccupancyGrid] = None
        self.unknown_msg: Optional[OccupancyGrid] = None
        self.free_arr: Optional[np.ndarray] = None
        self.obstacle_arr: Optional[np.ndarray] = None
        self.unknown_arr: Optional[np.ndarray] = None

        # TF.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ========== Subscriptions ==========
        self.create_subscription(PoseArray, self.rcg_nodes_topic, self.rcg_nodes_callback, 10)
        self.create_subscription(MarkerArray, self.rcg_markers_topic, self.rcg_markers_callback, 10)
        self.create_subscription(OccupancyGrid, self.covered_map_topic, self.covered_callback, 10)
        self.create_subscription(OccupancyGrid, self.free_map_topic, self.free_callback, 10)
        self.create_subscription(OccupancyGrid, self.obstacle_map_topic, self.obstacle_callback, 10)
        self.create_subscription(OccupancyGrid, self.unknown_map_topic, self.unknown_callback, 10)

        # ========== Publishers ==========
        self.goal_pub = self.create_publisher(PoseStamped, '/cstar/goal', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/cstar/goal_marker', 10)
        self.state_marker_pub = self.create_publisher(MarkerArray, '/cstar/open_closed_markers', 10)
        self.path_pub = self.create_publisher(Path, '/cstar/selected_path', 10)
        self.escape_path_pub = self.create_publisher(Path, '/cstar/escape_path', 10)
        self.retreat_marker_pub = self.create_publisher(Marker, '/cstar/retreat_nodes', 10)

        self.branch_candidate_pub = self.create_publisher(MarkerArray, '/cstar/local_branch_markers', 10)
        self.hole_nodes_pub = self.create_publisher(PoseArray, '/cstar/hole_nodes', 10)
        self.hole_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_markers', 10)
        self.hole_entry_marker_pub = self.create_publisher(MarkerArray, '/cstar/hole_entry_marker', 10)
        self.hole_exit_marker_pub = self.create_publisher(MarkerArray, '/cstar/hole_exit_marker', 10)
        self.hole_repair_path_pub = self.create_publisher(Path, '/cstar/hole_repair_path', 10)
        self.hole_repair_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_repair_markers', 10)

        self.timer = self.create_timer(self.update_period, self.on_timer)

        self.get_logger().info('CStarWaypointPlannerNode single dense RCG mode started.')
        self.get_logger().info(f'rcg_nodes_topic={self.rcg_nodes_topic}')
        self.get_logger().info(f'rcg_markers_topic={self.rcg_markers_topic}')
        self.get_logger().info(
            f'hole_detection={self.enable_hole_detection}, '
            f'window={self.hole_local_window_distance:.2f}, '
            f'min_nodes={self.hole_min_nodes}, max_nodes={self.hole_max_nodes}'
        )
        self.get_logger().info(
            f'hole_execution={self.enable_hole_execution}, '
            f'finish_tol={self.hole_repair_finish_tolerance:.2f}, '
            f'min_path_points={self.hole_execution_min_path_points}'
        )

    # ------------------------------------------------------------------
    # Graph / map callbacks
    # ------------------------------------------------------------------
    def make_key(self, x: float, y: float) -> NodeKey:
        q = self.position_quantization
        return int(round(x / q)), int(round(y / q))

    def rcg_nodes_callback(self, msg: PoseArray) -> None:
        nodes: Dict[NodeKey, Tuple[float, float]] = {}
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            nodes[self.make_key(x, y)] = (x, y)

        self.nodes = nodes
        self.rebuild_adjacency()

        if self.current_goal_key is not None and self.current_goal_key not in self.nodes:
            self.current_goal_key = None
            self.selected_path.clear()

    def rcg_markers_callback(self, msg: MarkerArray) -> None:
        self.raw_edges = self.extract_edges_from_marker_array(msg)
        self.rebuild_adjacency()

    def extract_edges_from_marker_array(self, msg: MarkerArray) -> List[Tuple[NodeKey, NodeKey]]:
        edges: List[Tuple[NodeKey, NodeKey]] = []

        for marker in msg.markers:
            if marker.ns != 'rcg_edges':
                continue

            pts = marker.points
            if len(pts) < 2:
                continue

            for i in range(0, len(pts) - 1, 2):
                p1 = pts[i]
                p2 = pts[i + 1]
                k1 = self.make_key(p1.x, p1.y)
                k2 = self.make_key(p2.x, p2.y)
                if k1 != k2:
                    edges.append((k1, k2))

        return edges

    def rebuild_adjacency(self) -> None:
        adj: Dict[NodeKey, Set[NodeKey]] = {key: set() for key in self.nodes.keys()}

        for k1, k2 in self.raw_edges:
            if k1 not in self.nodes or k2 not in self.nodes:
                continue
            adj[k1].add(k2)
            adj[k2].add(k1)

        self.adj = adj

    def covered_callback(self, msg: OccupancyGrid) -> None:
        self.covered_map = msg
        self.covered_data = list(msg.data)

    def free_callback(self, msg: OccupancyGrid) -> None:
        self.free_msg = msg
        h = msg.info.height
        w = msg.info.width
        self.free_arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w)) > 50

    def obstacle_callback(self, msg: OccupancyGrid) -> None:
        self.obstacle_msg = msg
        h = msg.info.height
        w = msg.info.width
        self.obstacle_arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w)) > 50

    def unknown_callback(self, msg: OccupancyGrid) -> None:
        self.unknown_msg = msg
        h = msg.info.height
        w = msg.info.width
        self.unknown_arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w)) > 50

    # ------------------------------------------------------------------
    # Pose / closed-state helpers
    # ------------------------------------------------------------------
    def get_robot_pose(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            return tf.transform.translation.x, tf.transform.translation.y
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def world_to_covered_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.covered_map is None:
            return None

        info = self.covered_map.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None

        return mx, my

    def world_to_cell(self, x: float, y: float) -> Optional[GridCell]:
        if self.free_msg is None:
            return None

        info = self.free_msg.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)

        if row < 0 or col < 0 or row >= info.height or col >= info.width:
            return None

        return row, col

    def is_inside_covered_map(self, x: float, y: float) -> bool:
        if not self.use_covered_map_for_closing:
            return False

        if self.covered_map is None or self.covered_data is None:
            return False

        cell = self.world_to_covered_cell(x, y)
        if cell is None:
            return False

        mx, my = cell
        idx = my * self.covered_map.info.width + mx

        if idx < 0 or idx >= len(self.covered_data):
            return False

        return int(self.covered_data[idx]) >= self.covered_close_threshold

    def add_closed_position(self, x: float, y: float) -> None:
        if self.closed_positions:
            lx, ly = self.closed_positions[-1]
            if math.hypot(x - lx, y - ly) < 0.05:
                return

        self.closed_positions.append((x, y))
        if len(self.closed_positions) > 5000:
            self.closed_positions = self.closed_positions[-5000:]

    def is_near_closed_position(self, x: float, y: float, radius: Optional[float] = None) -> bool:
        r = self.closed_position_radius if radius is None else radius
        r2 = r * r

        for cx, cy in self.closed_positions:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= r2:
                return True

        return False

    def is_closed_key(self, key: NodeKey) -> bool:
        if key in self.closed_nodes:
            return True

        if key not in self.nodes:
            return False

        x, y = self.nodes[key]
        return self.is_inside_covered_map(x, y) or self.is_near_closed_position(x, y)

    def close_key(self, key: NodeKey) -> None:
        if key not in self.nodes:
            return

        self.closed_nodes.add(key)
        x, y = self.nodes[key]
        self.add_closed_position(x, y)

    def nearest_node_key(self, x: float, y: float) -> Optional[NodeKey]:
        best_key = None
        best_dist = float('inf')

        for key, pos in self.nodes.items():
            d = math.hypot(pos[0] - x, pos[1] - y)
            if d < best_dist:
                best_dist = d
                best_key = key

        if best_dist > self.snap_distance:
            return None

        return best_key

    def is_reached_goal(self, robot_xy: Tuple[float, float]) -> bool:
        if self.current_goal_key is None:
            return True

        if self.current_goal_key not in self.nodes:
            return True

        gx, gy = self.nodes[self.current_goal_key]
        tol = self.retreat_center_tolerance if self.escape_active else self.goal_center_tolerance
        return math.hypot(robot_xy[0] - gx, robot_xy[1] - gy) <= tol

    def is_key_reached(
        self,
        key: NodeKey,
        robot_xy: Tuple[float, float],
        tolerance: Optional[float] = None,
    ) -> bool:
        if key not in self.nodes:
            return True
        x, y = self.nodes[key]
        tol = self.goal_center_tolerance if tolerance is None else tolerance
        return math.hypot(robot_xy[0] - x, robot_xy[1] - y) <= tol

    # ------------------------------------------------------------------
    # Single dense RCG coverage goal selection
    # ------------------------------------------------------------------
    def classify_open_neighbors(self, current_key: NodeKey) -> Dict[str, List[Tuple[float, NodeKey]]]:
        result = {
            'same_forward': [],
            'same_backward': [],
            'left': [],
            'up': [],
            'down': [],
            'right': [],
            'diagonal': [],
        }

        if current_key not in self.nodes:
            return result

        cx, cy = self.nodes[current_key]

        for nb in self.adj.get(current_key, set()):
            if nb not in self.nodes:
                continue

            if self.is_closed_key(nb):
                continue

            x, y = self.nodes[nb]
            dx = x - cx
            dy = y - cy
            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                continue

            if abs(dy) <= self.same_lap_y_tolerance:
                if dx * self.sweep_dir > 0.0:
                    result['same_forward'].append((abs(dx), nb))
                else:
                    result['same_backward'].append((abs(dx), nb))
                continue

            if abs(dx) >= abs(dy):
                if dx < 0.0:
                    result['left'].append((dist, nb))
                else:
                    result['right'].append((dist, nb))
            else:
                if dy > 0.0:
                    result['up'].append((dist, nb))
                else:
                    result['down'].append((dist, nb))

        for key in result:
            result[key].sort(key=lambda item: item[0])

        return result

    def choose_next_goal(self, current_key: NodeKey) -> Optional[NodeKey]:
        candidates = self.classify_open_neighbors(current_key)

        if candidates['same_forward']:
            return candidates['same_forward'][0][1]

        if candidates['same_backward']:
            next_key = candidates['same_backward'][0][1]
            if current_key in self.nodes and next_key in self.nodes:
                cx, _ = self.nodes[current_key]
                nx, _ = self.nodes[next_key]
                self.sweep_dir = -1.0 if (nx - cx) < 0.0 else 1.0
            return next_key

        if candidates['left']:
            self.sweep_dir = -1.0
            return candidates['left'][0][1]

        if candidates['up']:
            self.sweep_dir *= -1.0
            return candidates['up'][0][1]

        if candidates['down']:
            self.sweep_dir *= -1.0
            return candidates['down'][0][1]

        if candidates['right']:
            self.sweep_dir = 1.0
            return candidates['right'][0][1]

        if self.allow_diagonal_fallback and candidates['diagonal']:
            candidates['diagonal'].sort(key=lambda item: item[0])
            return candidates['diagonal'][0][1]

        return None

    def shortest_path_to_nearest_open_node(self, start_key: NodeKey) -> List[NodeKey]:
        if start_key not in self.nodes:
            return []

        q = deque([start_key])
        prev: Dict[NodeKey, Optional[NodeKey]] = {start_key: None}
        depth: Dict[NodeKey, int] = {start_key: 0}
        target: Optional[NodeKey] = None

        while q:
            key = q.popleft()

            if key != start_key and not self.is_closed_key(key):
                target = key
                break

            if depth[key] >= self.max_graph_transit_hops:
                continue

            for nb in sorted(self.adj.get(key, set()), key=lambda k: self.nodes.get(k, (0.0, 0.0))):
                if nb in prev:
                    continue
                if nb not in self.nodes:
                    continue
                prev[nb] = key
                depth[nb] = depth[key] + 1
                q.append(nb)

        if target is None:
            return []

        path: List[NodeKey] = []
        cur: Optional[NodeKey] = target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

    def set_new_goal(self, key: Optional[NodeKey], reason: str) -> None:
        if key is None or key not in self.nodes:
            self.current_goal_key = None
            self.selected_path.clear()
            return

        self.current_goal_key = key
        x, y = self.nodes[key]
        self.selected_path = [(x, y)]
        self.publish_goal(key)
        self.get_logger().info(f'New dense RCG goal: {key}, reason={reason}')

    def publish_goal(self, key: NodeKey) -> None:
        if key not in self.nodes:
            return

        x, y = self.nodes[key]

        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = msg.header.stamp
        marker.ns = 'cstar_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.10
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        if self.escape_active:
            marker.color.r = 1.0
            marker.color.g = 0.35
            marker.color.b = 0.0
            marker.color.a = 0.95
        elif self.mode in ('HOLE_ARMED', 'HOLE_REPAIR'):
            marker.color.r = 0.7
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.95
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 0.95
        self.goal_marker_pub.publish(marker)


    # ------------------------------------------------------------------
    # Dead-end retreat / grid A* escape
    # ------------------------------------------------------------------
    def has_any_open_neighbor(self, key: NodeKey) -> bool:
        if key not in self.nodes:
            return False
        for nb in self.adj.get(key, set()):
            if nb in self.nodes and not self.is_closed_key(nb):
                return True
        return False

    def is_retreat_candidate(self, key: NodeKey, start_key: NodeKey) -> bool:
        if key == start_key:
            return False
        if key not in self.nodes:
            return False
        if self.is_closed_key(key):
            return False

        # A retreat node is an open node attached to already covered space.
        for nb in self.adj.get(key, set()):
            if nb in self.nodes and self.is_closed_key(nb):
                return True

        x, y = self.nodes[key]
        return self.is_near_closed_position(x, y, radius=self.retreat_attach_radius)

    def find_retreat_candidates(self, start_key: NodeKey) -> Set[NodeKey]:
        candidates: Set[NodeKey] = set()
        for key in self.nodes.keys():
            if self.is_retreat_candidate(key, start_key):
                candidates.add(key)
        self.latest_retreat_candidates = candidates
        return candidates

    def dilate_bool(self, mask: np.ndarray, radius_cells: int) -> np.ndarray:
        if radius_cells <= 0:
            return mask.copy()

        h, w = mask.shape
        out = np.zeros_like(mask, dtype=bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return out

        offsets: List[Tuple[int, int]] = []
        r2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dy, dx))

        for dy, dx in offsets:
            y2 = ys + dy
            x2 = xs + dx
            valid = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
            out[y2[valid], x2[valid]] = True
        return out

    def build_safe_free_mask(self) -> Optional[np.ndarray]:
        if self.free_msg is None or self.free_arr is None:
            return None

        info = self.free_msg.info
        res = info.resolution
        h = info.height
        w = info.width
        free = self.free_arr.copy()

        if self.obstacle_arr is not None and self.obstacle_arr.shape == free.shape:
            obstacle = self.obstacle_arr.copy()
        else:
            obstacle = np.logical_not(free)

        if self.unknown_arr is not None and self.unknown_arr.shape == free.shape:
            unknown = self.unknown_arr.copy()
        else:
            unknown = np.zeros_like(free, dtype=bool)

        obstacle_rad = max(0, int(math.ceil(self.obstacle_buffer / res)))
        unknown_rad = max(0, int(math.ceil(self.unknown_buffer / res)))
        border_rad = max(0, int(math.ceil(self.map_border_buffer / res)))

        obstacle_buffer_mask = self.dilate_bool(obstacle, obstacle_rad)
        unknown_buffer_mask = self.dilate_bool(unknown, unknown_rad)

        safe = free.copy()
        safe[obstacle_buffer_mask] = False
        safe[unknown_buffer_mask] = False

        if border_rad > 0:
            safe[:border_rad, :] = False
            safe[h - border_rad:, :] = False
            safe[:, :border_rad] = False
            safe[:, w - border_rad:] = False
        return safe

    def find_nearest_safe_cell(self, cell: GridCell, safe: np.ndarray) -> Optional[GridCell]:
        assert self.free_msg is not None
        row, col = cell
        h, w = safe.shape
        res = self.free_msg.info.resolution
        max_rad = max(1, int(math.ceil(self.nearest_safe_search_radius / res)))

        if 0 <= row < h and 0 <= col < w and safe[row, col]:
            return row, col

        best_cell = None
        best_dist = float('inf')
        for rad in range(1, max_rad + 1):
            r0 = max(0, row - rad)
            r1 = min(h - 1, row + rad)
            c0 = max(0, col - rad)
            c1 = min(w - 1, col + rad)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if not safe[rr, cc]:
                        continue
                    d = math.hypot(rr - row, cc - col)
                    if d < best_dist:
                        best_dist = d
                        best_cell = (rr, cc)
            if best_cell is not None:
                return best_cell
        return None

    def cell_to_world(self, cell: GridCell) -> Tuple[float, float]:
        assert self.free_msg is not None
        row, col = cell
        info = self.free_msg.info
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def grid_line_is_safe(self, a: GridCell, b: GridCell, safe: np.ndarray) -> bool:
        r0, c0 = a
        r1, c1 = b
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        h, w = safe.shape
        for i in range(n + 1):
            t = 0.0 if n == 0 else i / n
            rr = int(round((1.0 - t) * r0 + t * r1))
            cc = int(round((1.0 - t) * c0 + t * c1))
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                return False
            if not safe[rr, cc]:
                return False
        return True

    def simplify_grid_path(self, cells: List[GridCell], safe: np.ndarray) -> List[GridCell]:
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1:
                if self.grid_line_is_safe(cells[i], cells[j], safe):
                    break
                j -= 1
            result.append(cells[j])
            i = j
        return result

    def densify_xy_path(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= 1:
            return points
        step = max(0.02, self.escape_resample_step)
        dense: List[Tuple[float, float]] = [points[0]]
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            dx = x1 - x0
            dy = y1 - y0
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            n = max(1, int(math.ceil(dist / step)))
            for k in range(1, n + 1):
                t = k / n
                dense.append((x0 + t * dx, y0 + t * dy))
        return dense

    def heuristic_to_goal_cells(self, cell: GridCell, goals: Set[GridCell]) -> float:
        r, c = cell
        best = float('inf')
        for gr, gc in goals:
            d = math.hypot(gr - r, gc - c)
            if d < best:
                best = d
        return 0.0 if best == float('inf') else best

    def reconstruct_grid_path(self, prev: Dict[GridCell, Optional[GridCell]], target: GridCell) -> List[GridCell]:
        path: List[GridCell] = []
        cur: Optional[GridCell] = target
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

    def a_star_grid_to_goal_set(
        self,
        start: GridCell,
        goal_to_key: Dict[GridCell, NodeKey],
        safe: np.ndarray,
    ) -> Tuple[List[GridCell], Optional[NodeKey]]:
        goals = set(goal_to_key.keys())
        if not goals:
            return [], None

        h, w = safe.shape
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        ]

        open_heap: List[Tuple[float, float, GridCell]] = []
        g_score: Dict[GridCell, float] = {start: 0.0}
        prev: Dict[GridCell, Optional[GridCell]] = {start: None}
        visited: Set[GridCell] = set()
        heapq.heappush(open_heap, (self.heuristic_to_goal_cells(start, goals), 0.0, start))

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current in goals:
                return self.reconstruct_grid_path(prev, current), goal_to_key[current]

            r, c = current
            for dr, dc, move_cost in neighbors:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if not safe[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    if not safe[r + dr, c] or not safe[r, c + dc]:
                        continue

                nb = (nr, nc)
                tentative_g = current_g + move_cost
                if tentative_g >= g_score.get(nb, float('inf')):
                    continue
                g_score[nb] = tentative_g
                prev[nb] = current
                f = tentative_g + self.heuristic_to_goal_cells(nb, goals)
                heapq.heappush(open_heap, (f, tentative_g, nb))
        return [], None

    def find_grid_escape_path_to_retreat(
        self,
        current_key: NodeKey,
        robot_xy: Tuple[float, float],
    ) -> Tuple[Optional[NodeKey], List[Tuple[float, float]], bool]:
        safe = self.build_safe_free_mask()
        if safe is None:
            self.get_logger().warn('Cannot build safe_free mask for grid A*.')
            return None, [], False

        raw_start = self.world_to_cell(robot_xy[0], robot_xy[1])
        if raw_start is None:
            self.get_logger().warn('Robot is outside free_map. Cannot run grid A*.')
            return None, [], False

        start = self.find_nearest_safe_cell(raw_start, safe)
        if start is None:
            self.get_logger().warn('Cannot find nearest safe start cell for grid A*.')
            return None, [], False

        strict_candidates = self.find_retreat_candidates(current_key)
        goal_to_key: Dict[GridCell, NodeKey] = {}
        for key in strict_candidates:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            raw_goal = self.world_to_cell(x, y)
            if raw_goal is None:
                continue
            safe_goal = self.find_nearest_safe_cell(raw_goal, safe)
            if safe_goal is None:
                continue
            goal_to_key[safe_goal] = key

        grid_path, target_key = self.a_star_grid_to_goal_set(start, goal_to_key, safe)
        if grid_path and target_key is not None:
            simplified = self.simplify_grid_path(grid_path, safe)
            xy_path = [self.cell_to_world(cell) for cell in simplified]
            if target_key in self.nodes:
                tx, ty = self.nodes[target_key]
                if not xy_path or math.hypot(xy_path[-1][0] - tx, xy_path[-1][1] - ty) > 0.03:
                    xy_path.append((tx, ty))
            return target_key, self.densify_xy_path(xy_path), False

        if not self.allow_open_fallback:
            return None, [], False

        fallback_goal_to_key: Dict[GridCell, NodeKey] = {}
        for key in self.nodes.keys():
            if key == current_key:
                continue
            if self.is_closed_key(key):
                continue
            x, y = self.nodes[key]
            raw_goal = self.world_to_cell(x, y)
            if raw_goal is None:
                continue
            safe_goal = self.find_nearest_safe_cell(raw_goal, safe)
            if safe_goal is None:
                continue
            fallback_goal_to_key[safe_goal] = key

        grid_path, target_key = self.a_star_grid_to_goal_set(start, fallback_goal_to_key, safe)
        if grid_path and target_key is not None:
            simplified = self.simplify_grid_path(grid_path, safe)
            xy_path = [self.cell_to_world(cell) for cell in simplified]
            if target_key in self.nodes:
                tx, ty = self.nodes[target_key]
                if not xy_path or math.hypot(xy_path[-1][0] - tx, xy_path[-1][1] - ty) > 0.03:
                    xy_path.append((tx, ty))
            self.get_logger().warn('No strict retreat node found by grid A*. Fallback to reachable open node.')
            return target_key, self.densify_xy_path(xy_path), True

        return None, [], False

    def start_deadend_escape(self, current_key: NodeKey, robot_xy: Tuple[float, float]) -> bool:
        target_key, xy_path, used_fallback = self.find_grid_escape_path_to_retreat(current_key, robot_xy)
        if target_key is None or not xy_path:
            self.escape_active = False
            self.escape_path_xy.clear()
            return False

        self.escape_active = True
        self.escape_path_xy = xy_path
        self.current_goal_key = target_key
        if target_key in self.nodes:
            tx, ty = self.nodes[target_key]
            self.selected_path.append((tx, ty))

        sx, sy = self.nodes.get(current_key, robot_xy)
        gx, gy = self.nodes[target_key]
        self.get_logger().warn(
            f'Dead-end detected at ({sx:.2f}, {sy:.2f}). '
            f'Grid A* retreat path points={len(xy_path)}, '
            f'retreat=({gx:.2f}, {gy:.2f}), '
            f'retreat_candidates={len(self.latest_retreat_candidates)}, '
            f'fallback={used_fallback}'
        )
        return True

    def finish_escape(self, current_key: NodeKey) -> None:
        self.escape_active = False
        self.escape_path_xy.clear()
        if current_key in self.nodes:
            x, y = self.nodes[current_key]
            self.get_logger().info(f'Grid A* retreat finished near ({x:.2f}, {y:.2f}). Resume normal C* coverage.')

    def publish_deadend_escape_path(self) -> None:
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        if self.escape_active:
            for x, y in self.escape_path_xy:
                ps = PoseStamped()
                ps.header = msg.header
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.position.z = 0.08
                ps.pose.orientation.w = 1.0
                msg.poses.append(ps)
        self.escape_path_pub.publish(msg)

    def publish_retreat_nodes(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'retreat_nodes'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.color.r = 1.0
        marker.color.g = 0.45
        marker.color.b = 0.0
        marker.color.a = 0.95
        marker.pose.orientation.w = 1.0
        for key in self.latest_retreat_candidates:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.09
            marker.points.append(p)
        self.retreat_marker_pub.publish(marker)

    # ------------------------------------------------------------------
    # Simple local floodfill hole detection on single dense graph
    # ------------------------------------------------------------------
    def should_update_hole_detection(self) -> bool:
        if self.last_hole_update_time is None:
            self.last_hole_update_time = self.get_clock().now()
            return True

        now = self.get_clock().now()
        elapsed = (now - self.last_hole_update_time).nanoseconds / 1e9
        if elapsed >= self.hole_dynamic_update_period:
            self.last_hole_update_time = now
            return True

        return False

    def current_motion_basis(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Return a stable horizontal motion basis for same-lap coverage.

        Hole detection is only meaningful while the robot is sweeping along the
        current lap. During up/down/diagonal transitions, the area ahead is just
        normal future C* coverage, so we skip hole detection.
        """
        if current_key not in self.nodes:
            return None

        cx, cy = self.nodes[current_key]
        target_key = next_goal_key if next_goal_key in self.nodes else self.current_goal_key

        if target_key in self.nodes and target_key != current_key:
            tx, ty = self.nodes[target_key]
            dx = tx - cx
            dy = ty - cy

            if self.hole_same_lap_only:
                if abs(dy) > self.same_lap_y_tolerance:
                    return None
                if abs(dx) < 1e-4:
                    return None

            direction = -1.0 if dx < 0.0 else 1.0
        else:
            direction = -1.0 if self.sweep_dir < 0.0 else 1.0

        # In the dense grid RCG, the current lap is horizontal.  Keep the hole
        # detector's coordinate frame fixed to the lap instead of using a
        # diagonal/up-down transition vector.
        u = (direction, 0.0)
        v = (0.0, 1.0)
        return u, v

    def collect_current_lap_scan_band(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
    ) -> Set[NodeKey]:
        """
        Collect only the current lap's near-future scan band.

        This replaces the old graph-BFS local window.  The old BFS window could
        swallow the side hole entrance itself, causing late detection.  The new
        band is a thin connected strip on the current lap, so floodfill seeds
        can only come from side edges leaving this strip.
        """
        band: Set[NodeKey] = set()
        if current_key not in self.nodes:
            return band

        basis = self.current_motion_basis(current_key, next_goal_key)
        if basis is None:
            return band

        u, _ = basis
        direction = -1.0 if u[0] < 0.0 else 1.0
        cx, cy = self.nodes[current_key]

        forward_dist = max(0.10, self.hole_local_window_distance)
        back_margin = max(0.0, self.hole_scan_back_margin)
        y_tol = max(0.03, self.same_lap_y_tolerance)

        # Follow only same-lap horizontal graph edges from current_key.  This
        # keeps the scan band on the current coverage lane even when the dense
        # RCG has many vertical cross-lap edges.
        q = deque([current_key])
        visited: Set[NodeKey] = {current_key}

        while q:
            key = q.popleft()
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            along = (x - cx) * direction
            if abs(y - cy) > y_tol:
                continue
            if along < -back_margin or along > forward_dist:
                continue

            band.add(key)

            for nb in self.adj.get(key, set()):
                if nb in visited or nb not in self.nodes:
                    continue

                nx, ny = self.nodes[nb]
                nb_along = (nx - cx) * direction

                # Only expand along the same lap; do not step through vertical
                # grid edges into adjacent laps/hole regions.
                if abs(ny - cy) > y_tol:
                    continue
                if nb_along < -back_margin or nb_along > forward_dist:
                    continue

                visited.add(nb)
                q.append(nb)

        if next_goal_key in self.nodes:
            nx, ny = self.nodes[next_goal_key]
            nb_along = (nx - cx) * direction
            if abs(ny - cy) <= y_tol and -back_margin <= nb_along <= forward_dist:
                band.add(next_goal_key)

        return band

    def collect_local_side_seed_candidates(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
    ) -> Tuple[Set[NodeKey], List[Tuple[float, NodeKey, NodeKey, str]]]:
        scan_band = self.collect_current_lap_scan_band(current_key, next_goal_key, robot_xy)
        self.latest_scan_band = set(scan_band)

        if not scan_band:
            return scan_band, []

        basis = self.current_motion_basis(current_key, next_goal_key)
        if basis is None:
            return scan_band, []

        u, v = basis
        min_lat = max(0.02, self.hole_branch_lateral_min_distance)
        ratio = max(1.0, self.hole_branch_lateral_ratio)
        max_along = max(0.02, self.hole_branch_max_along_offset)
        max_dist = max(min_lat, self.hole_max_branch_seed_distance)

        candidates: List[Tuple[float, NodeKey, NodeKey, str]] = []
        seen_edges: Set[Tuple[NodeKey, NodeKey]] = set()

        for base_key in scan_band:
            if base_key not in self.nodes:
                continue

            bx, by = self.nodes[base_key]
            base_robot_dist = math.hypot(bx - robot_xy[0], by - robot_xy[1])

            for seed_key in self.adj.get(base_key, set()):
                if seed_key not in self.nodes:
                    continue
                if seed_key in scan_band:
                    continue
                if self.is_closed_key(seed_key):
                    continue

                sx, sy = self.nodes[seed_key]

                # Seeds on the same lap are forward/backward normal C* nodes,
                # not side-hole entries.
                if abs(sy - by) <= self.same_lap_y_tolerance:
                    continue

                edge_id = (min(base_key, seed_key), max(base_key, seed_key))
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)

                dx = sx - bx
                dy = sy - by
                edge_dist = math.hypot(dx, dy)

                if edge_dist < 1e-6 or edge_dist > max_dist:
                    continue

                along = dx * u[0] + dy * u[1]
                lateral = abs(dx * v[0] + dy * v[1])

                if lateral < min_lat:
                    continue
                if lateral < ratio * abs(along):
                    continue
                if abs(along) > max_along:
                    continue

                # Earlier bases in front of the robot are preferred, so a side
                # opening can be detected before the robot fully passes it.
                score = base_robot_dist + 0.20 * edge_dist
                candidates.append((score, seed_key, base_key, 'side_seed'))

        candidates.sort(key=lambda item: item[0])
        return scan_band, candidates[:max(1, self.hole_max_seed_candidates)]

    def floodfill_open_component(
        self,
        seed_key: NodeKey,
        barrier_keys: Set[NodeKey],
    ) -> Tuple[Set[NodeKey], Set[NodeKey]]:
        comp: Set[NodeKey] = set()
        attachments: Set[NodeKey] = set()

        if seed_key not in self.nodes:
            return comp, attachments

        q = deque([seed_key])
        visited: Set[NodeKey] = {seed_key}

        while q:
            key = q.popleft()

            if len(comp) >= self.floodfill_max_nodes:
                break

            if key in barrier_keys:
                attachments.add(key)
                continue

            if key not in self.nodes:
                continue

            if self.is_closed_key(key):
                attachments.add(key)
                continue

            comp.add(key)

            for nb in self.adj.get(key, set()):
                if nb in visited:
                    continue

                visited.add(nb)

                if nb in barrier_keys or self.is_closed_key(nb):
                    attachments.add(nb)
                    continue

                q.append(nb)

        # collect all contacts again for stable attachment count
        for key in comp:
            for nb in self.adj.get(key, set()):
                if nb in barrier_keys or self.is_closed_key(nb):
                    attachments.add(nb)

        return comp, attachments

    def classify_component(
        self,
        comp: Set[NodeKey],
        attachments: Set[NodeKey],
        seed_key: NodeKey,
        base_key: NodeKey,
        robot_xy: Tuple[float, float],
    ) -> Tuple[str, str, float]:
        if not comp:
            return 'empty', 'empty_component', 1e9

        if len(comp) < self.hole_min_nodes:
            return 'too_small', f'nodes={len(comp)}', 8e5 + len(comp)

        if len(comp) > self.hole_max_nodes:
            return 'too_large', f'nodes={len(comp)}', 7e5 + len(comp)

        points = [self.nodes[k] for k in comp if k in self.nodes]
        if not points:
            return 'empty', 'no_points', 1e9

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox_area = max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys)))

        if bbox_area < self.hole_min_bbox_area:
            return 'too_small', f'bbox={bbox_area:.2f}', 8e5 + bbox_area

        if bbox_area > self.hole_max_bbox_area:
            return 'too_large', f'bbox={bbox_area:.2f}', 7e5 + bbox_area

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        robot_dist = math.hypot(cx - robot_xy[0], cy - robot_xy[1])
        if robot_dist > self.hole_max_robot_distance:
            return 'too_far', f'robot_dist={robot_dist:.2f}', 6e5 + robot_dist

        unknown_ratio = self.unknown_ratio_near_component(comp)
        if unknown_ratio >= self.unknown_reject_ratio:
            return 'frontier', f'unknown_ratio={unknown_ratio:.2f}', 5e5 + unknown_ratio

        # The component must attach to the current lap scan band, but it does
        # not have to wait until those base nodes become closed.  Waiting for
        # closed attachment was the main reason side holes were detected only
        # after the robot had already passed the whole entrance.
        path_attach_count = len(attachments)
        min_attach = max(1, self.hole_min_closed_attach_count)
        if path_attach_count < min_attach:
            return 'not_attached', f'path_attach={path_attach_count}', 4e5 + path_attach_count

        # A normal neighboring lap usually has many vertical contacts with the
        # scan band.  A hole/pocket usually has only a small entrance.
        if path_attach_count > self.hole_max_attachment_count:
            return 'normal_lap', f'attachments={path_attach_count}', 3e5 + path_attach_count

        score = robot_dist + 0.03 * len(comp) + 0.05 * bbox_area
        return 'hole', (
            f'nodes={len(comp)}, bbox={bbox_area:.2f}, '
            f'unknown={unknown_ratio:.2f}, attach={len(attachments)}'
        ), score

    def unknown_ratio_near_component(self, comp: Set[NodeKey]) -> float:
        if self.unknown_arr is None or self.free_msg is None:
            return 0.0

        total = 0
        unknown_hits = 0
        res = self.free_msg.info.resolution
        rad = max(1, int(math.ceil(self.unknown_check_radius / res)))
        h, w = self.unknown_arr.shape

        visited_cells: Set[GridCell] = set()

        for key in comp:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            cell = self.world_to_cell(x, y)
            if cell is None:
                continue

            row, col = cell
            for rr in range(max(0, row - rad), min(h, row + rad + 1)):
                for cc in range(max(0, col - rad), min(w, col + rad + 1)):
                    if (rr, cc) in visited_cells:
                        continue
                    visited_cells.add((rr, cc))
                    total += 1
                    if bool(self.unknown_arr[rr, cc]):
                        unknown_hits += 1

        if total == 0:
            return 0.0

        return float(unknown_hits) / float(total)

    def choose_hole_exit_key(self, comp: Set[NodeKey], attachments: Set[NodeKey], entry_key: NodeKey) -> Optional[NodeKey]:
        if not comp or not attachments:
            return None

        if entry_key not in self.nodes:
            return None

        ex, ey = self.nodes[entry_key]
        best_key = None
        best_dist = -1.0

        for key in attachments:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            d = math.hypot(x - ex, y - ey)
            if d > best_dist:
                best_dist = d
                best_key = key

        return best_key

    def build_local_repair_path(self, comp: Set[NodeKey], entry_key: NodeKey, exit_key: Optional[NodeKey]) -> List[Tuple[float, float]]:
        if not comp:
            return []

        points = [self.nodes[k] for k in comp if k in self.nodes]
        if not points:
            return []

        y_tol = max(0.05, self.hole_repair_y_tolerance)
        rows: List[List[Tuple[float, float]]] = []

        for p in sorted(points, key=lambda item: (item[1], item[0])):
            placed = False
            for row in rows:
                mean_y = sum(q[1] for q in row) / len(row)
                if abs(p[1] - mean_y) <= y_tol:
                    row.append(p)
                    placed = True
                    break
            if not placed:
                rows.append([p])

        rows.sort(key=lambda row: sum(q[1] for q in row) / len(row))

        if entry_key in self.nodes:
            entry_xy = self.nodes[entry_key]
        else:
            entry_xy = points[0]

        # Choose first row closest to entry, then alternate direction.
        if rows:
            start_index = min(
                range(len(rows)),
                key=lambda idx: min(math.hypot(p[0] - entry_xy[0], p[1] - entry_xy[1]) for p in rows[idx])
            )
            rows = rows[start_index:] + rows[:start_index]

        path: List[Tuple[float, float]] = []
        if entry_key in self.nodes:
            path.append(self.nodes[entry_key])

        forward = True
        for row in rows:
            ordered = sorted(row, key=lambda p: p[0], reverse=not forward)
            for p in ordered:
                if not path or math.hypot(path[-1][0] - p[0], path[-1][1] - p[1]) > 0.03:
                    path.append(p)
            forward = not forward

        if exit_key in self.nodes:
            exit_xy = self.nodes[exit_key]
            if not path or math.hypot(path[-1][0] - exit_xy[0], path[-1][1] - exit_xy[1]) > 0.03:
                path.append(exit_xy)

        return path[:max(1, self.hole_repair_max_points)]

    def clear_hole_state(self) -> None:
        self.latest_hole_component.clear()
        self.latest_hole_attachments.clear()
        self.latest_hole_entry_key = None
        self.latest_hole_exit_key = None
        self.latest_hole_repair_path.clear()
        self.latest_branch_edge = None
        self.latest_branch_candidates.clear()
        self.latest_scan_band.clear()

    def update_hole_detection(self, current_key: NodeKey, next_goal_key: Optional[NodeKey], robot_xy: Tuple[float, float]) -> None:
        if not self.enable_hole_detection:
            self.clear_hole_state()
            return

        if current_key not in self.nodes:
            self.clear_hole_state()
            return

        if not self.should_update_hole_detection():
            return

        barrier_keys, seed_candidates = self.collect_local_side_seed_candidates(
            current_key=current_key,
            next_goal_key=next_goal_key,
            robot_xy=robot_xy,
        )

        self.latest_branch_candidates = []
        best: Optional[Dict[str, object]] = None
        seen_components: Set[frozenset] = set()

        for _, seed_key, base_key, _ in seed_candidates:
            comp, attachments = self.floodfill_open_component(seed_key, barrier_keys)
            if not comp:
                continue

            comp_id = frozenset(comp)
            if comp_id in seen_components:
                continue
            seen_components.add(comp_id)

            label, reason, score = self.classify_component(comp, attachments, seed_key, base_key, robot_xy)
            self.latest_branch_candidates.append((base_key, seed_key, label))

            if label != 'hole':
                continue

            if best is None or score < float(best['score']):
                best = {
                    'score': score,
                    'seed_key': seed_key,
                    'base_key': base_key,
                    'component': comp,
                    'attachments': attachments,
                    'reason': reason,
                }

        if best is None:
            self.latest_hole_component.clear()
            self.latest_hole_attachments.clear()
            self.latest_hole_entry_key = None
            self.latest_hole_exit_key = None
            self.latest_hole_repair_path.clear()
            self.latest_branch_edge = None
            return

        seed_key = best['seed_key']  # type: ignore[assignment]
        base_key = best['base_key']  # type: ignore[assignment]
        comp = best['component']  # type: ignore[assignment]
        attachments = best['attachments']  # type: ignore[assignment]

        assert isinstance(seed_key, tuple)
        assert isinstance(base_key, tuple)
        assert isinstance(comp, set)
        assert isinstance(attachments, set)

        exit_key = self.choose_hole_exit_key(comp, attachments, seed_key)

        self.latest_hole_component = comp
        self.latest_hole_attachments = attachments
        self.latest_hole_entry_key = seed_key
        self.latest_hole_exit_key = exit_key
        self.latest_branch_edge = (base_key, seed_key)

        # Do not publish the old RCG-ordered repair path here.  The stable repair
        # route is generated only after a hole is armed, using locked-axis
        # resampling.
        self.latest_hole_repair_path = []

        self.get_logger().info(
            f'Simple hole detected: seed={seed_key}, base={base_key}, '
            f'nodes={len(comp)}, attachments={len(attachments)}, reason={best.get("reason")}'
        )

        if self.enable_hole_execution and self.mode == 'COVERAGE':
            self.arm_hole_repair(
                entry_base_key=base_key,
                seed_key=seed_key,
                exit_key=exit_key,
                component=comp,
                attachments=attachments,
                repair_path=self.latest_hole_repair_path,
            )


    # ------------------------------------------------------------------
    # Hole execution state machine: resampled repair path
    # ------------------------------------------------------------------
    def is_key_reached(
        self,
        key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
        tolerance: Optional[float] = None,
    ) -> bool:
        if key is None or key not in self.nodes:
            return False

        x, y = self.nodes[key]
        tol = self.goal_center_tolerance if tolerance is None else tolerance
        return math.hypot(robot_xy[0] - x, robot_xy[1] - y) <= tol

    def publish_goal_xy(self, x: float, y: float, reason: str = 'hole repair sample') -> None:
        """Publish a raw XY goal.  This is used only by resampled hole repair."""
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = msg.header.stamp
        marker.ns = 'cstar_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.10
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color.r = 0.7
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        self.goal_marker_pub.publish(marker)

    def distance_xy(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def append_xy_point(
        self,
        path: List[Tuple[float, float]],
        point: Tuple[float, float],
        min_dist: Optional[float] = None,
    ) -> None:
        """Append a repair waypoint without allowing one large goal jump.

        Hole repair is executed through ordinary /cstar/goal targets.  If two
        consecutive resampled points are far apart, insert intermediate goals
        on the same segment so the purple goal_marker advances gradually, just
        like normal C* waypoint following.
        """
        threshold = self.repair_goal_min_spacing if min_dist is None else min_dist
        if not path:
            path.append(point)
            return

        start = path[-1]
        dist = self.distance_xy(start, point)
        if dist < threshold:
            return

        max_step = max(threshold, self.repair_max_goal_step)
        if dist <= max_step:
            path.append(point)
            return

        n = max(1, int(math.ceil(dist / max_step)))
        for i in range(1, n + 1):
            t = float(i) / float(n)
            intermediate = (
                start[0] + t * (point[0] - start[0]),
                start[1] + t * (point[1] - start[1]),
            )
            if self.distance_xy(path[-1], intermediate) >= threshold:
                path.append(intermediate)

    def is_repair_goal_reached(self, robot_xy: Tuple[float, float]) -> bool:
        if self.active_hole_current_goal_xy is None:
            return True
        return self.distance_xy(robot_xy, self.active_hole_current_goal_xy) <= self.hole_repair_finish_tolerance

    def build_repair_safe_mask(self) -> Optional[np.ndarray]:
        """Build a lighter safe mask for local hole repair resampling."""
        if self.free_msg is None or self.free_arr is None:
            return None

        free = self.free_arr.copy()
        if self.obstacle_arr is not None and self.obstacle_arr.shape == free.shape:
            obstacle = self.obstacle_arr.copy()
        else:
            obstacle = np.logical_not(free)

        if self.unknown_arr is not None and self.unknown_arr.shape == free.shape:
            unknown = self.unknown_arr.copy()
        else:
            unknown = np.zeros_like(free, dtype=bool)

        res = self.free_msg.info.resolution
        obstacle_rad = max(0, int(math.ceil(self.repair_obstacle_buffer / res)))
        unknown_rad = max(0, int(math.ceil(self.repair_unknown_buffer / res)))
        border_rad = max(0, int(math.ceil(self.map_border_buffer / res)))

        safe = free.copy()
        if obstacle_rad > 0:
            safe[self.dilate_bool(obstacle, obstacle_rad)] = False
        else:
            safe[obstacle] = False
        if unknown_rad > 0:
            safe[self.dilate_bool(unknown, unknown_rad)] = False
        else:
            safe[unknown] = False

        h, w = safe.shape
        if border_rad > 0:
            safe[:border_rad, :] = False
            safe[h - border_rad:, :] = False
            safe[:, :border_rad] = False
            safe[:, w - border_rad:] = False
        return safe

    def block_scan_band_on_mask(self, mask: np.ndarray, seed_key: NodeKey) -> np.ndarray:
        """Keep the main lap as the hole boundary, but do not erase the seed."""
        if self.free_msg is None:
            return mask
        out = mask.copy()
        res = self.free_msg.info.resolution
        rad = max(0, int(math.ceil(self.repair_scan_band_block_radius / res)))
        h, w = out.shape
        seed_cell = None
        if seed_key in self.nodes:
            seed_cell = self.world_to_cell(*self.nodes[seed_key])

        for key in self.latest_scan_band:
            if key not in self.nodes:
                continue
            cell = self.world_to_cell(*self.nodes[key])
            if cell is None:
                continue
            r, c = cell
            r0 = max(0, r - rad)
            r1 = min(h, r + rad + 1)
            c0 = max(0, c - rad)
            c1 = min(w, c + rad + 1)
            out[r0:r1, c0:c1] = False

        # Make sure the first hole cell remains usable.
        if seed_cell is not None:
            sr, sc = seed_cell
            if 0 <= sr < h and 0 <= sc < w:
                out[sr, sc] = True
        return out

    def local_mask_floodfill(self, start: GridCell, mask: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        rmin, rmax, cmin, cmax = roi
        h, w = mask.shape
        out = np.zeros_like(mask, dtype=bool)
        sr, sc = start
        if sr < 0 or sr >= h or sc < 0 or sc >= w or not mask[sr, sc]:
            return out

        q = deque([(sr, sc)])
        visited = {(sr, sc)}
        out[sr, sc] = True
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while q:
            r, c = q.popleft()
            for dr, dc in neighbors:
                nr = r + dr
                nc = c + dc
                if nr < rmin or nr > rmax or nc < cmin or nc > cmax:
                    continue
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if (nr, nc) in visited:
                    continue
                if not mask[nr, nc]:
                    continue
                # Prevent diagonal corner cutting.
                if dr != 0 and dc != 0:
                    if not mask[r + dr, c] or not mask[r, c + dc]:
                        continue
                visited.add((nr, nc))
                out[nr, nc] = True
                q.append((nr, nc))
        return out

    def component_cells_and_roi(
        self,
        comp: Set[NodeKey],
        seed_key: NodeKey,
    ) -> Tuple[Set[GridCell], Optional[Tuple[int, int, int, int]]]:
        if self.free_msg is None:
            return set(), None

        cells: Set[GridCell] = set()
        for key in comp | {seed_key}:
            if key not in self.nodes:
                continue
            cell = self.world_to_cell(*self.nodes[key])
            if cell is not None:
                cells.add(cell)

        if not cells:
            return set(), None

        h = self.free_msg.info.height
        w = self.free_msg.info.width
        res = self.free_msg.info.resolution
        margin = max(1, int(math.ceil(self.repair_roi_margin / res)))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rmin = max(0, min(rows) - margin)
        rmax = min(h - 1, max(rows) + margin)
        cmin = max(0, min(cols) - margin)
        cmax = min(w - 1, max(cols) + margin)
        return cells, (rmin, rmax, cmin, cmax)

    def build_hole_repair_mask(self, comp: Set[NodeKey], seed_key: NodeKey) -> Optional[np.ndarray]:
        """
        Create a strict local free-space mask for resampled repair.

        Important change from the previous version:
        - Do not floodfill the whole free ROI around the component.
        - First build an inflated envelope from the detected hole component,
          then intersect it with safe free space.

        This keeps the repair route inside the detected hole region and avoids
        accidental leakage into corridors, walls, or unknown space.
        """
        safe = self.build_repair_safe_mask()
        if safe is None or self.free_msg is None:
            return None

        comp_cells, roi = self.component_cells_and_roi(comp, seed_key)
        if roi is None or not comp_cells:
            return None

        h, w = safe.shape
        res = self.free_msg.info.resolution
        rmin, rmax, cmin, cmax = roi

        comp_mask = np.zeros_like(safe, dtype=bool)
        for r, c in comp_cells:
            if 0 <= r < h and 0 <= c < w:
                comp_mask[r, c] = True

        inflate_rad = max(1, int(math.ceil(self.repair_component_inflate_radius / res)))
        envelope = self.dilate_bool(comp_mask, inflate_rad)

        roi_mask = np.zeros_like(safe, dtype=bool)
        roi_mask[rmin:rmax + 1, cmin:cmax + 1] = True

        candidate = safe & envelope & roi_mask
        candidate = self.block_scan_band_on_mask(candidate, seed_key)

        seed_cell = None
        if seed_key in self.nodes:
            seed_cell = self.world_to_cell(*self.nodes[seed_key])
        if seed_cell is None:
            return None

        if not candidate[seed_cell[0], seed_cell[1]]:
            nearest = self.find_nearest_safe_cell(seed_cell, candidate)
            if nearest is not None:
                seed_cell = nearest
            else:
                # Fallback: component cells only.  This is still safer than
                # using the whole ROI, because it cannot create long chords
                # across walls/unknown regions.
                fallback = np.zeros_like(safe, dtype=bool)
                fallback = self.dilate_bool(comp_mask, max(1, inflate_rad // 2)) & safe & roi_mask
                if 0 <= seed_cell[0] < h and 0 <= seed_cell[1] < w:
                    fallback[seed_cell[0], seed_cell[1]] = True
                return fallback

        hole_mask = self.local_mask_floodfill(seed_cell, candidate, roi)
        if np.count_nonzero(hole_mask) == 0:
            fallback = self.dilate_bool(comp_mask, max(1, inflate_rad // 2)) & safe & roi_mask
            if 0 <= seed_cell[0] < h and 0 <= seed_cell[1] < w:
                fallback[seed_cell[0], seed_cell[1]] = True
            return fallback
        return hole_mask

    def sampled_point_is_needed(self, point: Tuple[float, float], always_keep: bool = False) -> bool:
        if always_keep:
            return True
        if not self.repair_skip_covered_samples:
            return True
        if self.is_inside_covered_map(point[0], point[1]):
            return False
        if self.is_near_closed_position(point[0], point[1], radius=max(self.closed_position_radius, 0.10)):
            return False
        return True

    def find_mask_row_point(self, mask: np.ndarray, row_center: int, col: int, band_half: int) -> Optional[GridCell]:
        h, w = mask.shape
        if col < 0 or col >= w:
            return None
        best = None
        best_d = 10 ** 9
        r0 = max(0, row_center - band_half)
        r1 = min(h - 1, row_center + band_half)
        for r in range(r0, r1 + 1):
            if mask[r, col]:
                d = abs(r - row_center)
                if d < best_d:
                    best_d = d
                    best = (r, col)
        return best

    def find_mask_col_point(self, mask: np.ndarray, row: int, col_center: int, band_half: int) -> Optional[GridCell]:
        h, w = mask.shape
        if row < 0 or row >= h:
            return None
        best = None
        best_d = 10 ** 9
        c0 = max(0, col_center - band_half)
        c1 = min(w - 1, col_center + band_half)
        for c in range(c0, c1 + 1):
            if mask[row, c]:
                d = abs(c - col_center)
                if d < best_d:
                    best_d = d
                    best = (row, c)
        return best

    def contiguous_runs(self, values: List[int]) -> List[Tuple[int, int]]:
        if not values:
            return []
        vals = sorted(set(values))
        runs: List[Tuple[int, int]] = []
        start = vals[0]
        prev = vals[0]
        for v in vals[1:]:
            if v == prev + 1:
                prev = v
                continue
            runs.append((start, prev))
            start = v
            prev = v
        runs.append((start, prev))
        return runs

    def mask_line_is_safe(self, a: GridCell, b: GridCell, mask: np.ndarray) -> bool:
        r0, c0 = a
        r1, c1 = b
        n = max(abs(r1 - r0), abs(c1 - c0), 1)
        h, w = mask.shape
        for i in range(n + 1):
            t = i / n
            rr = int(round((1.0 - t) * r0 + t * r1))
            cc = int(round((1.0 - t) * c0 + t * c1))
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                return False
            if not mask[rr, cc]:
                return False
        return True

    def nearest_mask_cell(self, cell: Optional[GridCell], mask: np.ndarray) -> Optional[GridCell]:
        if cell is None:
            return None
        row, col = cell
        h, w = mask.shape
        if 0 <= row < h and 0 <= col < w and mask[row, col]:
            return row, col

        if self.free_msg is None:
            return None
        res = self.free_msg.info.resolution
        max_rad = max(1, int(math.ceil(self.nearest_safe_search_radius / res)))
        best_cell = None
        best_dist = float('inf')
        for rad in range(1, max_rad + 1):
            r0 = max(0, row - rad)
            r1 = min(h - 1, row + rad)
            c0 = max(0, col - rad)
            c1 = min(w - 1, col + rad)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if not mask[rr, cc]:
                        continue
                    d = math.hypot(rr - row, cc - col)
                    if d < best_dist:
                        best_dist = d
                        best_cell = (rr, cc)
            if best_cell is not None:
                return best_cell
        return None

    def a_star_mask_between_cells(
        self,
        start: GridCell,
        goal: GridCell,
        mask: np.ndarray,
    ) -> List[GridCell]:
        if start == goal:
            return [start]
        h, w = mask.shape
        if not (0 <= start[0] < h and 0 <= start[1] < w and mask[start[0], start[1]]):
            return []
        if not (0 <= goal[0] < h and 0 <= goal[1] < w and mask[goal[0], goal[1]]):
            return []

        if self.free_msg is not None:
            res = self.free_msg.info.resolution
            margin = max(2, int(math.ceil(self.repair_transition_search_margin / res)))
        else:
            margin = 10
        rmin = max(0, min(start[0], goal[0]) - margin)
        rmax = min(h - 1, max(start[0], goal[0]) + margin)
        cmin = max(0, min(start[1], goal[1]) - margin)
        cmax = min(w - 1, max(start[1], goal[1]) + margin)

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        ]
        open_heap: List[Tuple[float, float, GridCell]] = []
        g_score: Dict[GridCell, float] = {start: 0.0}
        prev: Dict[GridCell, Optional[GridCell]] = {start: None}
        visited: Set[GridCell] = set()
        h0 = math.hypot(goal[0] - start[0], goal[1] - start[1])
        heapq.heappush(open_heap, (h0, 0.0, start))

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                cells = self.reconstruct_grid_path(prev, current)
                return self.simplify_grid_path(cells, mask)

            r, c = current
            for dr, dc, cost in neighbors:
                nr = r + dr
                nc = c + dc
                if nr < rmin or nr > rmax or nc < cmin or nc > cmax:
                    continue
                if not mask[nr, nc]:
                    continue
                if dr != 0 and dc != 0:
                    if not mask[r + dr, c] or not mask[r, c + dc]:
                        continue
                nb = (nr, nc)
                tentative = current_g + cost
                if tentative >= g_score.get(nb, float('inf')):
                    continue
                g_score[nb] = tentative
                prev[nb] = current
                f = tentative + math.hypot(goal[0] - nr, goal[1] - nc)
                heapq.heappush(open_heap, (f, tentative, nb))
        return []

    def append_safe_xy_connection(
        self,
        path: List[Tuple[float, float]],
        target: Tuple[float, float],
        mask: np.ndarray,
    ) -> bool:
        """Append target without drawing a chord through wall/unknown cells."""
        if not path:
            self.append_xy_point(path, target)
            return True

        start_cell = self.nearest_mask_cell(self.world_to_cell(path[-1][0], path[-1][1]), mask)
        goal_cell = self.nearest_mask_cell(self.world_to_cell(target[0], target[1]), mask)
        if start_cell is None or goal_cell is None:
            return False

        if self.mask_line_is_safe(start_cell, goal_cell, mask):
            # Snap to the requested target if it is close to the safe cell.
            self.append_xy_point(path, target)
            return True

        cells = self.a_star_mask_between_cells(start_cell, goal_cell, mask)
        if not cells:
            return False

        for cell in cells[1:]:
            self.append_xy_point(path, self.cell_to_world(cell))
        # If target is a sampled point inside the mask, it should be safe to
        # append it after the A* goal cell.  The min spacing removes duplicates.
        self.append_xy_point(path, target)
        return True

    def append_exit_connection(
        self,
        path: List[Tuple[float, float]],
        exit_xy: Tuple[float, float],
        mask: np.ndarray,
    ) -> None:
        if not path:
            self.append_xy_point(path, exit_xy)
            return
        goal_cell_raw = self.world_to_cell(exit_xy[0], exit_xy[1])
        goal_cell = self.nearest_mask_cell(goal_cell_raw, mask)
        start_cell = self.nearest_mask_cell(self.world_to_cell(path[-1][0], path[-1][1]), mask)
        if start_cell is not None and goal_cell is not None:
            if self.mask_line_is_safe(start_cell, goal_cell, mask):
                self.append_xy_point(path, self.cell_to_world(goal_cell))
            else:
                cells = self.a_star_mask_between_cells(start_cell, goal_cell, mask)
                for cell in cells[1:]:
                    self.append_xy_point(path, self.cell_to_world(cell))
        # The exit marker is allowed to be just outside the repair mask because
        # it lies on the main-lap contact.  This final segment is short.
        if self.distance_xy(path[-1], exit_xy) <= max(0.45, 2.0 * self.repair_resample_line_spacing):
            self.append_xy_point(path, exit_xy, min_dist=0.02)

    def build_sample_lanes_axis(
        self,
        mask: np.ndarray,
        axis: str,
    ) -> List[Dict[str, object]]:
        if self.free_msg is None or np.count_nonzero(mask) == 0:
            return []

        res = self.free_msg.info.resolution
        sample_step = max(1, int(round(self.repair_resample_spacing / res)))
        line_step = max(1, int(round(self.repair_resample_line_spacing / res)))
        band_half = max(0, line_step // 2)
        ys, xs = np.where(mask)
        rmin, rmax = int(np.min(ys)), int(np.max(ys))
        cmin, cmax = int(np.min(xs)), int(np.max(xs))
        lanes: List[Dict[str, object]] = []

        if axis == 'horizontal':
            centers = list(range(rmin, rmax + 1, line_step))
            if centers and centers[-1] != rmax and rmax - centers[-1] > max(1, line_step // 2):
                centers.append(rmax)
            for row_center in centers:
                r0 = max(0, row_center - band_half)
                r1 = min(mask.shape[0] - 1, row_center + band_half)
                cols = np.where(np.any(mask[r0:r1 + 1, :], axis=0))[0]
                cols = [int(c) for c in cols if cmin <= c <= cmax]
                for run_start, run_end in self.contiguous_runs(cols):
                    lane: List[Tuple[float, float]] = []
                    sampled_cols = list(range(run_start, run_end + 1, sample_step))
                    if sampled_cols and sampled_cols[-1] != run_end:
                        sampled_cols.append(run_end)
                    for c in sampled_cols:
                        cell = self.find_mask_row_point(mask, row_center, c, band_half)
                        if cell is None:
                            continue
                        p = self.cell_to_world(cell)
                        if self.sampled_point_is_needed(p):
                            self.append_xy_point(lane, p)
                    if lane:
                        lanes.append({'center': row_center, 'points': lane})
        else:
            centers = list(range(cmin, cmax + 1, line_step))
            if centers and centers[-1] != cmax and cmax - centers[-1] > max(1, line_step // 2):
                centers.append(cmax)
            for col_center in centers:
                c0 = max(0, col_center - band_half)
                c1 = min(mask.shape[1] - 1, col_center + band_half)
                rows = np.where(np.any(mask[:, c0:c1 + 1], axis=1))[0]
                rows = [int(r) for r in rows if rmin <= r <= rmax]
                for run_start, run_end in self.contiguous_runs(rows):
                    lane = []
                    sampled_rows = list(range(run_start, run_end + 1, sample_step))
                    if sampled_rows and sampled_rows[-1] != run_end:
                        sampled_rows.append(run_end)
                    for r in sampled_rows:
                        cell = self.find_mask_col_point(mask, r, col_center, band_half)
                        if cell is None:
                            continue
                        p = self.cell_to_world(cell)
                        if self.sampled_point_is_needed(p):
                            self.append_xy_point(lane, p)
                    if lane:
                        lanes.append({'center': col_center, 'points': lane})

        lanes.sort(key=lambda item: int(item['center']))
        return lanes

    def assemble_lanes_safely(
        self,
        lanes: List[Dict[str, object]],
        start_ref: Tuple[float, float],
        mask: np.ndarray,
        reverse_order: bool,
    ) -> List[Tuple[float, float]]:
        if reverse_order:
            ordered_lanes = list(reversed(lanes))
        else:
            ordered_lanes = list(lanes)

        path: List[Tuple[float, float]] = []
        prefer_forward = True
        for lane_info in ordered_lanes:
            lane = list(lane_info['points'])  # type: ignore[index]
            if not lane:
                continue

            lane_fwd = lane
            lane_rev = list(reversed(lane))
            ref = path[-1] if path else start_ref

            if path:
                df = self.distance_xy(ref, lane_fwd[0])
                dr = self.distance_xy(ref, lane_rev[0])
                ordered = lane_fwd if df <= dr else lane_rev
            else:
                df = self.distance_xy(ref, lane_fwd[0])
                dr = self.distance_xy(ref, lane_rev[0])
                ordered = lane_fwd if df <= dr else lane_rev
                prefer_forward = ordered is lane_rev

            # Keep the general boustrophedon alternation, but never force an
            # unsafe long chord.  If the alternated end is much worse, use the
            # near end and let the safe connector handle the transition.
            if path and prefer_forward:
                alt = lane_fwd
            elif path:
                alt = lane_rev
            else:
                alt = ordered
            if path and self.distance_xy(path[-1], alt[0]) <= self.distance_xy(path[-1], ordered[0]) + self.repair_resample_line_spacing:
                ordered = alt

            if path:
                if not self.append_safe_xy_connection(path, ordered[0], mask):
                    # Do not draw an unsafe line.  Skip disconnected lanes.
                    continue
            else:
                self.append_xy_point(path, ordered[0])

            for p in ordered[1:]:
                if not self.append_safe_xy_connection(path, p, mask):
                    # Within a lane this should rarely fail, because each lane
                    # comes from a contiguous mask run.  Stop this lane instead
                    # of creating a chord through an obstacle.
                    break
            prefer_forward = not prefer_forward
        return path

    def determine_locked_repair_axis(self, entry_base_key: NodeKey, seed_key: NodeKey) -> str:
        """Choose repair sweep direction before entering the hole.

        The repair stroke direction should be perpendicular to the normal C*
        sweep direction at the moment the side hole is found.  In the current
        dense RCG, normal C* same-lap sweeping is horizontal, so the repair
        path should use vertical lanes.  The generic branch below keeps this
        valid if a future map uses vertical same-lap motion.
        """
        if not self.repair_lock_axis_to_cstar_perpendicular:
            return 'vertical'

        basis = self.current_motion_basis(entry_base_key, self.current_goal_key)
        if basis is None:
            # Current implementation's stable C* sweep is horizontal.
            return 'vertical'
        u, _ = basis
        if abs(u[0]) >= abs(u[1]):
            return 'vertical'
        return 'horizontal'

    def locked_lane_direction_sign(
        self,
        axis: str,
        entry_xy: Tuple[float, float],
        exit_xy: Tuple[float, float],
    ) -> int:
        """Order repair lanes from entry toward exit/main C* progress."""
        if axis == 'vertical':
            delta = exit_xy[0] - entry_xy[0]
            if abs(delta) > 0.05:
                return 1 if delta > 0.0 else -1
            return 1 if self.sweep_dir > 0.0 else -1

        delta = exit_xy[1] - entry_xy[1]
        if abs(delta) > 0.05:
            return 1 if delta > 0.0 else -1
        return 1

    def order_lanes_from_entry(
        self,
        lanes: List[Dict[str, object]],
        axis: str,
        seed_xy: Tuple[float, float],
        exit_xy: Tuple[float, float],
    ) -> List[Dict[str, object]]:
        """Start at the lane nearest the entry seed and move monotonically.

        This avoids the old behavior where the selected route could begin at a
        far corner of the hole, causing the purple goal_marker to jump from one
        side of the room to the other.
        """
        if not lanes or self.free_msg is None:
            return lanes

        seed_cell = self.world_to_cell(seed_xy[0], seed_xy[1])
        exit_cell = self.world_to_cell(exit_xy[0], exit_xy[1])
        if seed_cell is None:
            return lanes

        seed_center = seed_cell[1] if axis == 'vertical' else seed_cell[0]
        exit_center = None
        if exit_cell is not None:
            exit_center = exit_cell[1] if axis == 'vertical' else exit_cell[0]

        sign = self.locked_lane_direction_sign(axis, seed_xy, exit_xy)
        sorted_lanes = sorted(lanes, key=lambda item: int(item['center']))

        start_idx = min(
            range(len(sorted_lanes)),
            key=lambda idx: abs(int(sorted_lanes[idx]['center']) - seed_center),
        )

        if sign >= 0:
            forward = sorted_lanes[start_idx:]
            # Keep a small behind-entry allowance only if the exit lies behind.
            if exit_center is not None and exit_center < seed_center:
                forward = sorted_lanes[:start_idx + 1][::-1]
        else:
            forward = sorted_lanes[:start_idx + 1][::-1]
            if exit_center is not None and exit_center > seed_center:
                forward = sorted_lanes[start_idx:]

        return forward if forward else sorted_lanes

    def assemble_lanes_locked(
        self,
        lanes: List[Dict[str, object]],
        axis: str,
        seed_xy: Tuple[float, float],
        exit_xy: Tuple[float, float],
        mask: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Assemble a fixed-axis boustrophedon route.

        Lane order is locked from entry toward exit; only the direction inside
        each lane alternates.  We never reverse the whole route by scoring, so
        the repair plan cannot flip between left-right and up-down forms.
        """
        ordered_lanes = self.order_lanes_from_entry(lanes, axis, seed_xy, exit_xy)
        path: List[Tuple[float, float]] = []
        reverse_inside = False

        for lane_info in ordered_lanes:
            lane = list(lane_info['points'])  # type: ignore[index]
            if not lane:
                continue

            lane_fwd = lane
            lane_rev = list(reversed(lane))

            if not path:
                # For the first lane, start at the endpoint closest to the seed.
                ordered = lane_fwd if self.distance_xy(seed_xy, lane_fwd[0]) <= self.distance_xy(seed_xy, lane_rev[0]) else lane_rev
                reverse_inside = ordered is lane_rev
            else:
                # After that, strict boustrophedon alternation.  If the intended
                # connector is impossible in the mask, skip the lane rather than
                # drawing a chord across unknown/wall cells.
                ordered = lane_rev if not reverse_inside else lane_fwd

            if path:
                if not self.append_safe_xy_connection(path, ordered[0], mask):
                    continue
            else:
                self.append_xy_point(path, ordered[0])

            ok = True
            for p in ordered[1:]:
                if not self.append_safe_xy_connection(path, p, mask):
                    ok = False
                    break
            if ok:
                reverse_inside = not reverse_inside

        return path

    def resample_mask_axis(
        self,
        mask: np.ndarray,
        axis: str,
        entry_xy: Tuple[float, float],
        seed_xy: Tuple[float, float],
        exit_xy: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        lanes = self.build_sample_lanes_axis(mask, axis)
        if not lanes:
            return []

        path = self.assemble_lanes_locked(
            lanes=lanes,
            axis=axis,
            seed_xy=seed_xy,
            exit_xy=exit_xy,
            mask=mask,
        )
        return path[:max(1, self.hole_repair_max_points)]

    def path_score(self, path: List[Tuple[float, float]], sample_count: int) -> float:
        if len(path) <= 1:
            return 1e9
        length = 0.0
        turn_penalty = 0.0
        for i in range(len(path) - 1):
            length += self.distance_xy(path[i], path[i + 1])
        for i in range(1, len(path) - 1):
            ax = path[i][0] - path[i - 1][0]
            ay = path[i][1] - path[i - 1][1]
            bx = path[i + 1][0] - path[i][0]
            by = path[i + 1][1] - path[i][1]
            na = math.hypot(ax, ay)
            nb = math.hypot(bx, by)
            if na < 1e-6 or nb < 1e-6:
                continue
            cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
            turn_penalty += abs(math.acos(cosv))
        return length + 0.18 * turn_penalty - 0.03 * float(sample_count)

    def choose_execution_exit_key(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
        comp: Set[NodeKey],
        attachments: Set[NodeKey],
        preferred_exit_key: Optional[NodeKey],
    ) -> NodeKey:
        if entry_base_key not in self.nodes:
            return entry_base_key

        ex, ey = self.nodes[entry_base_key]
        direction = -1.0 if self.sweep_dir < 0.0 else 1.0
        possible = set(attachments)
        if preferred_exit_key is not None:
            possible.add(preferred_exit_key)
        possible.add(entry_base_key)

        candidates: List[Tuple[float, NodeKey]] = []
        for key in possible:
            if key not in self.nodes:
                continue
            if key != entry_base_key and not any(nb in comp for nb in self.adj.get(key, set())):
                continue
            x, y = self.nodes[key]
            along = (x - ex) * direction
            same_lap_penalty = abs(y - ey)
            if key == entry_base_key:
                score = 1000.0
            elif key in self.latest_scan_band and along >= -0.05:
                score = same_lap_penalty - 2.0 * along
            elif along >= -0.05:
                score = 20.0 + same_lap_penalty - along
            else:
                score = 100.0 + same_lap_penalty + abs(along)
            candidates.append((score, key))

        if not candidates:
            return entry_base_key
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def build_resampled_repair_points(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
        comp: Set[NodeKey],
        attachments: Set[NodeKey],
        preferred_exit_key: Optional[NodeKey],
        start_xy: Optional[Tuple[float, float]] = None,
        include_entry_prefix: bool = True,
    ) -> Tuple[List[Tuple[float, float]], NodeKey, str]:
        if entry_base_key not in self.nodes or seed_key not in self.nodes:
            return [], entry_base_key, 'none'

        entry_xy = self.nodes[entry_base_key]
        seed_xy = self.nodes[seed_key]
        exit_key = self.choose_execution_exit_key(entry_base_key, seed_key, comp, attachments, preferred_exit_key)
        exit_xy = self.nodes[exit_key] if exit_key in self.nodes else entry_xy

        # The axis is decided before entering the hole and then locked.
        # Do not compare horizontal/vertical plans every replan cycle, because
        # that is exactly what made the purple goal jump between corners.
        if self.active_hole_repair_axis in ('horizontal', 'vertical'):
            axis = self.active_hole_repair_axis
        else:
            axis = self.determine_locked_repair_axis(entry_base_key, seed_key)

        mask = self.build_hole_repair_mask(comp, seed_key)
        if mask is not None and np.count_nonzero(mask) > 0:
            inner = self.resample_mask_axis(mask, axis, entry_xy, seed_xy, exit_xy)
            path: List[Tuple[float, float]] = []
            if include_entry_prefix:
                self.append_xy_point(path, entry_xy, min_dist=0.02)
                self.append_xy_point(path, seed_xy, min_dist=0.02)
            elif start_xy is not None:
                self.append_xy_point(path, start_xy, min_dist=0.02)

            for pnt in inner:
                if not self.append_safe_xy_connection(path, pnt, mask):
                    continue
            self.append_exit_connection(path, exit_xy, mask)

            if len(path) >= max(2, self.hole_execution_min_path_points):
                return path[:max(1, self.hole_repair_max_points)], exit_key, axis

        # Conservative fallback: enter and leave without using unstable RCG
        # ordering.  Keep the same locked axis label for logging.
        path = []
        if include_entry_prefix:
            self.append_xy_point(path, entry_xy, min_dist=0.02)
        elif start_xy is not None:
            self.append_xy_point(path, start_xy, min_dist=0.02)
        self.append_xy_point(path, seed_xy, min_dist=0.02)
        self.append_xy_point(path, exit_xy, min_dist=0.02)
        return path[:max(1, self.hole_repair_max_points)], exit_key, axis

    def first_unreached_index_in_xy_path(
        self,
        path: List[Tuple[float, float]],
        robot_xy: Tuple[float, float],
        start_index: int = 0,
    ) -> int:
        idx = max(0, start_index)
        while idx < len(path):
            if self.distance_xy(path[idx], robot_xy) > max(self.hole_repair_finish_tolerance, 0.08):
                return idx
            idx += 1
        return idx

    def set_repair_sample_goal(self, index: int, reason: str = 'resampled hole repair step') -> None:
        if index < 0 or index >= len(self.active_hole_repair_points):
            self.active_hole_current_goal_xy = None
            return
        self.active_hole_repair_goal_index = index
        self.active_hole_current_goal_xy = self.active_hole_repair_points[index]
        self.current_goal_key = None
        self.selected_path = self.active_hole_repair_points[index:]
        x, y = self.active_hole_current_goal_xy
        self.publish_goal_xy(x, y, reason)

    def rebuild_resampled_repair_plan_from_current(self, robot_xy: Tuple[float, float]) -> bool:
        if self.active_hole_entry_base_key not in self.nodes or self.active_hole_seed_key not in self.nodes:
            return False

        points, exit_key, axis = self.build_resampled_repair_points(
            entry_base_key=self.active_hole_entry_base_key,  # type: ignore[arg-type]
            seed_key=self.active_hole_seed_key,  # type: ignore[arg-type]
            comp=self.active_hole_component,
            attachments=self.active_hole_attachments,
            preferred_exit_key=self.active_hole_exit_key,
            start_xy=robot_xy,
            include_entry_prefix=False,
        )
        if not points:
            return False

        self.active_hole_exit_key = exit_key
        self.active_hole_repair_axis = axis
        self.active_hole_repair_points = points
        self.active_hole_repair_goal_index = self.first_unreached_index_in_xy_path(points, robot_xy, 0)
        self.active_hole_repair_path = list(points)
        self.latest_hole_repair_path = list(points)
        self.latest_hole_exit_key = exit_key
        return self.active_hole_repair_goal_index < len(self.active_hole_repair_points)

    def build_execution_repair_path(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
        comp: Set[NodeKey],
        attachments: Set[NodeKey],
        preferred_exit_key: Optional[NodeKey],
    ) -> Tuple[List[NodeKey], List[Tuple[float, float]], Optional[NodeKey]]:
        # Keep this public method name for compatibility with arm_hole_repair(),
        # but return a resampled XY route instead of an RCG key route.
        self.active_hole_repair_axis = self.determine_locked_repair_axis(entry_base_key, seed_key)
        self.active_hole_repair_axis_locked = True
        points, exit_key, axis = self.build_resampled_repair_points(
            entry_base_key=entry_base_key,
            seed_key=seed_key,
            comp=comp,
            attachments=attachments,
            preferred_exit_key=preferred_exit_key,
            include_entry_prefix=True,
        )
        self.active_hole_repair_axis = axis
        return [], points, exit_key

    def arm_hole_repair(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
        exit_key: Optional[NodeKey],
        component: Set[NodeKey],
        attachments: Set[NodeKey],
        repair_path: List[Tuple[float, float]],
    ) -> None:
        if entry_base_key not in self.nodes or seed_key not in self.nodes:
            return
        if not component:
            return

        _, xy_path, execution_exit_key = self.build_execution_repair_path(
            entry_base_key=entry_base_key,
            seed_key=seed_key,
            comp=component,
            attachments=attachments,
            preferred_exit_key=exit_key,
        )

        if len(xy_path) < max(2, self.hole_execution_min_path_points):
            self.get_logger().warn('Detected hole but resampled repair path is too short; keep detection only.')
            return

        self.mode = 'HOLE_ARMED'
        self.escape_active = False
        self.escape_path_xy.clear()
        self.publish_empty_escape_path()

        self.active_hole_component = set(component)
        self.active_hole_attachments = set(attachments)
        self.active_hole_entry_base_key = entry_base_key
        self.active_hole_seed_key = seed_key
        self.active_hole_exit_key = execution_exit_key
        self.active_hole_repair_path = list(xy_path)
        self.active_hole_repair_points = list(xy_path)
        self.active_hole_repair_key_path = []
        self.active_hole_repair_goal_index = 0
        self.active_hole_current_goal_xy = None
        self.active_hole_visited_repair_points = []
        self.active_hole_branch_edge = (entry_base_key, seed_key)

        self.latest_hole_component = set(component)
        self.latest_hole_attachments = set(attachments)
        self.latest_hole_entry_key = seed_key
        self.latest_hole_exit_key = execution_exit_key
        self.latest_hole_repair_path = list(xy_path)
        self.latest_branch_edge = (entry_base_key, seed_key)

        self.set_new_goal(entry_base_key, 'hole entry_base')
        self.get_logger().info(
            f'Hole repair armed with resampled path: entry_base={entry_base_key}, '
            f'seed={seed_key}, exit={execution_exit_key}, points={len(xy_path)}, '
            f'axis={self.active_hole_repair_axis}'
        )

    def handle_hole_armed(self, robot_xy: Tuple[float, float]) -> None:
        self.publish_empty_escape_path()
        if self.active_hole_entry_base_key is None or self.active_hole_entry_base_key not in self.nodes:
            self.abort_hole_execution('entry_base disappeared')
            return

        if self.is_key_reached(
            self.active_hole_entry_base_key,
            robot_xy,
            tolerance=max(self.goal_center_tolerance, 0.08),
        ):
            self.close_key(self.active_hole_entry_base_key)
            self.mode = 'HOLE_REPAIR'
            self.active_hole_repair_goal_index = self.first_unreached_index_in_xy_path(
                self.active_hole_repair_points,
                robot_xy,
                start_index=0,
            )
            if self.active_hole_repair_goal_index >= len(self.active_hole_repair_points):
                self.finish_hole_repair()
                return
            self.set_repair_sample_goal(self.active_hole_repair_goal_index, 'start resampled hole repair')
            return

        self.set_new_goal(self.active_hole_entry_base_key, 'hole entry_base')

    def handle_hole_repair(self, robot_xy: Tuple[float, float]) -> None:
        # Keep controller in ordinary /cstar/goal mode.  /cstar/escape_path is
        # reserved for dead-end retreat only.
        self.publish_empty_escape_path()

        if not self.active_hole_repair_points:
            if not self.rebuild_resampled_repair_plan_from_current(robot_xy):
                self.abort_hole_execution('empty resampled repair plan')
                return

        if self.active_hole_current_goal_xy is None:
            self.active_hole_repair_goal_index = self.first_unreached_index_in_xy_path(
                self.active_hole_repair_points,
                robot_xy,
                start_index=self.active_hole_repair_goal_index,
            )
            if self.active_hole_repair_goal_index >= len(self.active_hole_repair_points):
                self.finish_hole_repair()
                return
            self.set_repair_sample_goal(self.active_hole_repair_goal_index, 'resampled hole repair step')
            return

        if self.is_repair_goal_reached(robot_xy):
            reached = self.active_hole_current_goal_xy
            self.active_hole_visited_repair_points.append(reached)
            self.add_closed_position(reached[0], reached[1])

            if self.repair_replan_on_reached:
                # Optional: update only after reaching the current sample.
                # The repair axis remains locked, so the plan cannot flip
                # between horizontal and vertical boustrophedon forms.  Default
                # is false for maximum stability.
                if self.rebuild_resampled_repair_plan_from_current(robot_xy):
                    self.set_repair_sample_goal(self.active_hole_repair_goal_index, 'updated locked-axis repair step')
                    return

            self.active_hole_repair_goal_index += 1
            self.active_hole_repair_goal_index = self.first_unreached_index_in_xy_path(
                self.active_hole_repair_points,
                robot_xy,
                start_index=self.active_hole_repair_goal_index,
            )
            if self.active_hole_repair_goal_index >= len(self.active_hole_repair_points):
                self.finish_hole_repair()
                return
            self.set_repair_sample_goal(self.active_hole_repair_goal_index, 'resampled hole repair step')
            return

        # Re-publish the current raw XY goal every timer cycle so controller
        # restarts do not miss it.
        if self.active_hole_current_goal_xy is not None:
            x, y = self.active_hole_current_goal_xy
            self.publish_goal_xy(x, y, 'resampled hole repair step')

    def finish_hole_repair(self) -> None:
        for key in self.active_hole_component:
            if key in self.nodes:
                self.closed_nodes.add(key)
                x, y = self.nodes[key]
                self.add_closed_position(x, y)

        for key in (
            self.active_hole_entry_base_key,
            self.active_hole_seed_key,
            self.active_hole_exit_key,
        ):
            if key in self.nodes:
                self.closed_nodes.add(key)
                x, y = self.nodes[key]
                self.add_closed_position(x, y)

        for x, y in self.active_hole_repair_points:
            self.add_closed_position(x, y)

        self.publish_empty_escape_path()
        self.get_logger().info(
            f'Resampled hole repair finished: samples={len(self.active_hole_repair_points)}, '
            f'axis={self.active_hole_repair_axis}, exit={self.active_hole_exit_key}'
        )

        self.mode = 'COVERAGE'
        self.active_hole_component.clear()
        self.active_hole_attachments.clear()
        self.active_hole_entry_base_key = None
        self.active_hole_seed_key = None
        self.active_hole_exit_key = None
        self.active_hole_repair_path.clear()
        self.active_hole_repair_key_path.clear()
        self.active_hole_repair_points.clear()
        self.active_hole_repair_goal_index = 0
        self.active_hole_current_goal_xy = None
        self.active_hole_branch_edge = None
        self.active_hole_visited_repair_points.clear()
        self.active_hole_repair_axis = 'resampled'
        self.active_hole_repair_axis_locked = False
        self.current_goal_key = None
        self.selected_path.clear()
        self.last_hole_update_time = None

    def abort_hole_execution(self, reason: str) -> None:
        self.get_logger().warn(f'Hole execution aborted: {reason}')
        self.publish_empty_escape_path()
        self.mode = 'COVERAGE'
        self.active_hole_component.clear()
        self.active_hole_attachments.clear()
        self.active_hole_entry_base_key = None
        self.active_hole_seed_key = None
        self.active_hole_exit_key = None
        self.active_hole_repair_path.clear()
        self.active_hole_repair_key_path.clear()
        self.active_hole_repair_points.clear()
        self.active_hole_repair_goal_index = 0
        self.active_hole_current_goal_xy = None
        self.active_hole_branch_edge = None
        self.active_hole_visited_repair_points.clear()
        self.active_hole_repair_axis = 'resampled'
        self.active_hole_repair_axis_locked = False
        self.current_goal_key = None
        self.selected_path.clear()

    def publish_escape_path(self, path: List[Tuple[float, float]]) -> None:
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.05
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.escape_path_pub.publish(msg)

    def publish_empty_escape_path(self) -> None:
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        self.escape_path_pub.publish(msg)

    # ------------------------------------------------------------------
    # Timer / publishing
    # ------------------------------------------------------------------
    def on_timer(self) -> None:
        robot_xy = self.get_robot_pose()
        if robot_xy is None:
            return

        if not self.nodes or not self.adj:
            return

        nearest_key = self.nearest_node_key(robot_xy[0], robot_xy[1])
        if nearest_key is None:
            return

        # Hole execution states have priority over normal C* goal selection.
        # During these states we do not run new hole detection and do not choose
        # a new normal coverage goal.
        if self.mode == 'HOLE_ARMED':
            self.handle_hole_armed(robot_xy)
        elif self.mode == 'HOLE_REPAIR':
            self.handle_hole_repair(robot_xy)
        else:
            self.mode = 'COVERAGE'
            reached_goal = self.is_reached_goal(robot_xy)
            current_key = nearest_key

            if reached_goal:
                if self.current_goal_key is not None:
                    self.close_key(self.current_goal_key)
                    current_key = self.current_goal_key
                else:
                    self.close_key(current_key)

                if self.escape_active:
                    self.finish_escape(current_key)

                next_goal = self.choose_next_goal(current_key)

                if next_goal is None and self.enable_graph_transit_to_open:
                    path = self.shortest_path_to_nearest_open_node(current_key)
                    if len(path) >= 2:
                        next_goal = path[1]

                if next_goal is not None:
                    self.escape_active = False
                    self.escape_path_xy.clear()
                    self.last_deadend_key = None
                    self.set_new_goal(next_goal, 'single dense C*')
                    self.update_hole_detection(current_key, next_goal, robot_xy)
                else:
                    if self.has_any_open_neighbor(current_key):
                        x, y = self.nodes[current_key]
                        self.get_logger().warn(
                            f'Node ({x:.2f}, {y:.2f}) still has open graph neighbors, '
                            f'but the boustrophedon policy rejected them. '
                            f'Consider increasing same_lap_y_tolerance / same_col_x_tolerance '
                            f'or enabling allow_diagonal_fallback.'
                        )

                    if not self.start_deadend_escape(current_key, robot_xy):
                        self.current_goal_key = None
                        if self.last_deadend_key != current_key:
                            self.last_deadend_key = current_key
                            x, y = self.nodes[current_key]
                            self.get_logger().warn(
                                f'Dead-end: no grid A* retreat path from ({x:.2f}, {y:.2f}). '
                                f'Robot will stop until map/RCG updates.'
                            )
            else:
                if self.current_goal_key is not None and not self.escape_active:
                    self.update_hole_detection(nearest_key, self.current_goal_key, robot_xy)

        # Important for launch-order robustness: /cstar/goal is not latched.
        # Re-publish the current goal every timer cycle, so a controller that is
        # started/restarted after the planner still receives the active target.
        if self.current_goal_key is not None and self.current_goal_key in self.nodes:
            self.publish_goal(self.current_goal_key)

        if self.escape_active:
            self.publish_deadend_escape_path()
        elif self.mode == 'COVERAGE':
            # Clear stale escape paths while normal C* is running.  Hole repair
            # states already publish an empty path inside their handlers.
            self.publish_empty_escape_path()

        self.publish_state_markers()
        self.publish_selected_path()
        self.publish_branch_candidate_markers()
        self.publish_retreat_nodes()
        self.publish_hole_outputs()

    def publish_state_markers(self) -> None:
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        open_marker = Marker()
        open_marker.header.frame_id = self.map_frame
        open_marker.header.stamp = stamp
        open_marker.ns = 'open_nodes'
        open_marker.id = 0
        open_marker.type = Marker.SPHERE_LIST
        open_marker.action = Marker.ADD
        open_marker.scale.x = 0.06
        open_marker.scale.y = 0.06
        open_marker.scale.z = 0.06
        open_marker.color.r = 0.1
        open_marker.color.g = 0.7
        open_marker.color.b = 1.0
        open_marker.color.a = 0.55
        open_marker.pose.orientation.w = 1.0

        closed_marker = Marker()
        closed_marker.header.frame_id = self.map_frame
        closed_marker.header.stamp = stamp
        closed_marker.ns = 'closed_nodes'
        closed_marker.id = 1
        closed_marker.type = Marker.SPHERE_LIST
        closed_marker.action = Marker.ADD
        closed_marker.scale.x = 0.075
        closed_marker.scale.y = 0.075
        closed_marker.scale.z = 0.075
        closed_marker.color.r = 1.0
        closed_marker.color.g = 0.05
        closed_marker.color.b = 0.05
        closed_marker.color.a = 0.80
        closed_marker.pose.orientation.w = 1.0

        for key, (x, y) in self.nodes.items():
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.09
            if self.is_closed_key(key):
                closed_marker.points.append(p)
            else:
                open_marker.points.append(p)

        ma.markers.append(open_marker)
        ma.markers.append(closed_marker)
        self.state_marker_pub.publish(ma)

    def publish_selected_path(self) -> None:
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in self.selected_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.05
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.path_pub.publish(msg)

    def publish_branch_candidate_markers(self) -> None:
        if not self.publish_branch_candidates:
            return

        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        accepted = Marker()
        accepted.header.frame_id = self.map_frame
        accepted.header.stamp = stamp
        accepted.ns = 'accepted_local_branch'
        accepted.id = 0
        accepted.type = Marker.LINE_LIST
        accepted.action = Marker.ADD
        accepted.scale.x = 0.045
        accepted.color.r = 0.65
        accepted.color.g = 0.0
        accepted.color.b = 1.0
        accepted.color.a = 0.95
        accepted.pose.orientation.w = 1.0

        rejected = Marker()
        rejected.header.frame_id = self.map_frame
        rejected.header.stamp = stamp
        rejected.ns = 'rejected_local_branch'
        rejected.id = 1
        rejected.type = Marker.LINE_LIST
        rejected.action = Marker.ADD
        rejected.scale.x = 0.025
        rejected.color.r = 1.0
        rejected.color.g = 0.8
        rejected.color.b = 0.0
        rejected.color.a = 0.65
        rejected.pose.orientation.w = 1.0

        scan_band = Marker()
        scan_band.header.frame_id = self.map_frame
        scan_band.header.stamp = stamp
        scan_band.ns = 'current_lap_scan_band'
        scan_band.id = 2
        scan_band.type = Marker.SPHERE_LIST
        scan_band.action = Marker.ADD
        scan_band.scale.x = 0.075
        scan_band.scale.y = 0.075
        scan_band.scale.z = 0.075
        scan_band.color.r = 0.0
        scan_band.color.g = 0.95
        scan_band.color.b = 1.0
        scan_band.color.a = 0.80
        scan_band.pose.orientation.w = 1.0

        for key in self.latest_scan_band:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            p = Point(); p.x = x; p.y = y; p.z = 0.18
            scan_band.points.append(p)

        for base, seed, label in self.latest_branch_candidates:
            if base not in self.nodes or seed not in self.nodes:
                continue
            bx, by = self.nodes[base]
            sx, sy = self.nodes[seed]
            p1 = Point(); p1.x = bx; p1.y = by; p1.z = 0.17
            p2 = Point(); p2.x = sx; p2.y = sy; p2.z = 0.17
            if label == 'hole':
                accepted.points.append(p1)
                accepted.points.append(p2)
            elif self.publish_rejected_branch_candidates:
                rejected.points.append(p1)
                rejected.points.append(p2)

        ma.markers.append(scan_band)
        ma.markers.append(accepted)
        if self.publish_rejected_branch_candidates:
            ma.markers.append(rejected)
        self.branch_candidate_pub.publish(ma)

    def publish_hole_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.publish_hole_nodes(stamp)
        self.publish_hole_markers(stamp)
        self.publish_entry_exit_markers(stamp)
        self.publish_hole_repair_path(stamp)

    def publish_hole_nodes(self, stamp) -> None:
        msg = PoseArray()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = stamp

        for key in self.latest_hole_component:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.05
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.hole_nodes_pub.publish(msg)

    def publish_hole_markers(self, stamp) -> None:
        ma = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        nodes_marker = Marker()
        nodes_marker.header.frame_id = self.map_frame
        nodes_marker.header.stamp = stamp
        nodes_marker.ns = 'simple_hole_nodes'
        nodes_marker.id = 0
        nodes_marker.type = Marker.SPHERE_LIST
        nodes_marker.action = Marker.ADD
        nodes_marker.scale.x = 0.10
        nodes_marker.scale.y = 0.10
        nodes_marker.scale.z = 0.10
        nodes_marker.color.r = 0.7
        nodes_marker.color.g = 0.0
        nodes_marker.color.b = 1.0
        nodes_marker.color.a = 0.95
        nodes_marker.pose.orientation.w = 1.0

        for key in self.latest_hole_component:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.14
            nodes_marker.points.append(p)

        branch_marker = Marker()
        branch_marker.header.frame_id = self.map_frame
        branch_marker.header.stamp = stamp
        branch_marker.ns = 'simple_hole_branch_edge'
        branch_marker.id = 1
        branch_marker.type = Marker.LINE_LIST
        branch_marker.action = Marker.ADD
        branch_marker.scale.x = 0.06
        branch_marker.color.r = 1.0
        branch_marker.color.g = 0.0
        branch_marker.color.b = 0.0
        branch_marker.color.a = 0.95
        branch_marker.pose.orientation.w = 1.0

        if self.latest_branch_edge is not None:
            base, seed = self.latest_branch_edge
            if base in self.nodes and seed in self.nodes:
                bx, by = self.nodes[base]
                sx, sy = self.nodes[seed]
                p1 = Point(); p1.x = bx; p1.y = by; p1.z = 0.18
                p2 = Point(); p2.x = sx; p2.y = sy; p2.z = 0.18
                branch_marker.points.append(p1)
                branch_marker.points.append(p2)

        ma.markers.append(nodes_marker)
        ma.markers.append(branch_marker)
        self.hole_markers_pub.publish(ma)

    def publish_entry_exit_markers(self, stamp) -> None:
        entry_ma = MarkerArray()
        exit_ma = MarkerArray()

        for arr in (entry_ma, exit_ma):
            delete_all = Marker()
            delete_all.header.frame_id = self.map_frame
            delete_all.header.stamp = stamp
            delete_all.action = Marker.DELETEALL
            arr.markers.append(delete_all)

        if self.latest_hole_entry_key in self.nodes:
            x, y = self.nodes[self.latest_hole_entry_key]
            entry_ma.markers.append(
                self.make_sphere_marker('hole_entry', 0, x, y, 0.18, (0.0, 1.0, 1.0, 0.95), stamp)
            )

        if self.latest_hole_exit_key in self.nodes:
            x, y = self.nodes[self.latest_hole_exit_key]
            exit_ma.markers.append(
                self.make_sphere_marker('hole_exit', 0, x, y, 0.18, (1.0, 0.55, 0.0, 0.95), stamp)
            )

        self.hole_entry_marker_pub.publish(entry_ma)
        self.hole_exit_marker_pub.publish(exit_ma)

    def make_sphere_marker(
        self,
        ns: str,
        marker_id: int,
        x: float,
        y: float,
        size: float,
        color: Tuple[float, float, float, float],
        stamp,
    ) -> Marker:
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.18
        m.pose.orientation.w = 1.0
        m.scale.x = size
        m.scale.y = size
        m.scale.z = size
        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]
        m.color.a = color[3]
        return m

    def publish_hole_repair_path(self, stamp) -> None:
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = stamp

        for x, y in self.latest_hole_repair_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.05
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.hole_repair_path_pub.publish(msg)

        ma = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        line = Marker()
        line.header.frame_id = self.map_frame
        line.header.stamp = stamp
        line.ns = 'hole_repair_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.035
        line.color.r = 1.0
        line.color.g = 0.2
        line.color.b = 0.0
        line.color.a = 0.90
        line.pose.orientation.w = 1.0

        for x, y in self.latest_hole_repair_path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.13
            line.points.append(p)

        ma.markers.append(line)
        self.hole_repair_markers_pub.publish(ma)



def main(args=None) -> None:
    rclpy.init(args=args)
    node = CStarWaypointPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
