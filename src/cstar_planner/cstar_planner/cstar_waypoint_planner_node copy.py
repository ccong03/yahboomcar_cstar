#!/usr/bin/env python3
#repair路线会穿墙并且会被固定版本。
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


        # Debug visualization for seed candidates.
        self.declare_parameter('publish_branch_candidates', True)
        self.declare_parameter('publish_rejected_branch_candidates', False)

        # ========== Hole execution state machine ==========
        # 检测到侧边 hole 后，planner 暂停普通 C*，先把 /cstar/goal 切到
        # 当前 lap 上的 entry_base，再把补扫路径发布到 /cstar/escape_path。
        self.declare_parameter('enable_hole_execution', True)

        # ========== Hole-local LocalRepairRCG validation ==========
        # This stage deliberately does NOT generate or execute a repair route.
        # It only stabilizes and visualizes:
        #   1) the virtual gate line;
        #   2) the active gate-clipped hole mask;
        #   3) a hole-local RCG-like resampling grid.
        self.declare_parameter('repair_roi_margin', 0.80)
        self.declare_parameter('repair_obstacle_buffer', 0.15)
        self.declare_parameter('repair_unknown_buffer', 0.08)
        self.declare_parameter('repair_scan_band_block_radius', 0.10)
        self.declare_parameter('repair_gate_tolerance', 0.06)

        # LocalRepairRCG sampling parameters, adapted from cstar_rcg_node.py.
        self.declare_parameter('local_repair_lap_spacing', 0.30)
        self.declare_parameter('local_repair_sample_spacing', 0.34)
        self.declare_parameter('local_repair_min_run_length', 0.30)
        self.declare_parameter('local_repair_min_node_keep_distance', 0.15)
        self.declare_parameter('local_repair_enable_interlap_edges', True)
        self.declare_parameter('local_repair_interlap_max_dist', 0.55)
        self.declare_parameter('local_repair_interlap_col_tolerance', 0.18)
        self.declare_parameter('local_repair_mask_viz_stride', 2)
        self.declare_parameter('local_repair_gate_viz_length', 2.0)

        # Repair route preview built from LocalRepairRCG lanes.  This stage
        # only visualizes a one-stroke candidate route; it still does not
        # command the robot to execute the hole route.
        self.declare_parameter('local_repair_build_route_preview', True)
        self.declare_parameter('local_repair_route_include_entry_segment', True)
        self.declare_parameter('local_repair_route_include_exit_segment', True)
        self.declare_parameter('local_repair_route_min_lane_nodes', 2)

        # Executable LocalRepair route.  The route is still built from the
        # stable LocalRepairRCG lanes, but once the robot reaches entry_base we
        # lock the current preview route and publish each waypoint on /cstar/goal
        # sequentially.  Normal C* resumes after the locked route is finished.
        self.declare_parameter('local_repair_execute_route', True)
        self.declare_parameter('local_repair_goal_tolerance', 0.12)
        self.declare_parameter('local_repair_goal_passed_tolerance', 0.18)
        self.declare_parameter('local_repair_min_execute_points', 3)

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
        self.publish_branch_candidates = bool(self.get_parameter('publish_branch_candidates').value)
        self.publish_rejected_branch_candidates = bool(self.get_parameter('publish_rejected_branch_candidates').value)
        self.enable_hole_execution = bool(self.get_parameter('enable_hole_execution').value)

        self.repair_roi_margin = float(self.get_parameter('repair_roi_margin').value)
        self.repair_obstacle_buffer = float(self.get_parameter('repair_obstacle_buffer').value)
        self.repair_unknown_buffer = float(self.get_parameter('repair_unknown_buffer').value)
        self.repair_scan_band_block_radius = float(self.get_parameter('repair_scan_band_block_radius').value)
        self.repair_gate_tolerance = float(self.get_parameter('repair_gate_tolerance').value)
        self.local_repair_lap_spacing = float(self.get_parameter('local_repair_lap_spacing').value)
        self.local_repair_sample_spacing = float(self.get_parameter('local_repair_sample_spacing').value)
        self.local_repair_min_run_length = float(self.get_parameter('local_repair_min_run_length').value)
        self.local_repair_min_node_keep_distance = float(self.get_parameter('local_repair_min_node_keep_distance').value)
        self.local_repair_enable_interlap_edges = bool(self.get_parameter('local_repair_enable_interlap_edges').value)
        self.local_repair_interlap_max_dist = float(self.get_parameter('local_repair_interlap_max_dist').value)
        self.local_repair_interlap_col_tolerance = float(self.get_parameter('local_repair_interlap_col_tolerance').value)
        self.local_repair_mask_viz_stride = max(1, int(self.get_parameter('local_repair_mask_viz_stride').value))
        self.local_repair_gate_viz_length = float(self.get_parameter('local_repair_gate_viz_length').value)
        self.local_repair_build_route_preview = bool(self.get_parameter('local_repair_build_route_preview').value)
        self.local_repair_route_include_entry_segment = bool(self.get_parameter('local_repair_route_include_entry_segment').value)
        self.local_repair_route_include_exit_segment = bool(self.get_parameter('local_repair_route_include_exit_segment').value)
        self.local_repair_route_min_lane_nodes = max(1, int(self.get_parameter('local_repair_route_min_lane_nodes').value))
        self.local_repair_execute_route = bool(self.get_parameter('local_repair_execute_route').value)
        self.local_repair_goal_tolerance = float(self.get_parameter('local_repair_goal_tolerance').value)
        self.local_repair_goal_passed_tolerance = float(self.get_parameter('local_repair_goal_passed_tolerance').value)
        self.local_repair_min_execute_points = max(2, int(self.get_parameter('local_repair_min_execute_points').value))

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
        # Hole-local LocalRepairRCG validation state.
        # No one-stroke repair route is generated in this validation version.
        self.active_hole_gate_origin: Optional[Tuple[float, float]] = None
        self.active_hole_gate_normal: Optional[Tuple[float, float]] = None
        self.active_hole_mask: Optional[np.ndarray] = None
        self.active_hole_roi: Optional[Tuple[int, int, int, int]] = None
        self.local_repair_axis: str = 'vertical'
        self.local_repair_nodes: List[Dict[str, object]] = []
        self.local_repair_edges: Set[Tuple[int, int]] = set()
        self.local_repair_route_points: List[Tuple[float, float]] = []
        self.active_hole_execute_route_points: List[Tuple[float, float]] = []
        self.active_hole_route_goal_index: int = 0
        self.active_hole_current_goal_xy: Optional[Tuple[float, float]] = None
        self.active_hole_executed_route_points: List[Tuple[float, float]] = []

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
        self.hole_gate_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_gate_markers', 10)
        self.active_hole_mask_markers_pub = self.create_publisher(Marker, '/cstar/active_hole_mask_marker', 10)
        self.local_repair_rcg_nodes_pub = self.create_publisher(PoseArray, '/cstar/local_repair_rcg_nodes', 10)
        self.local_repair_rcg_markers_pub = self.create_publisher(MarkerArray, '/cstar/local_repair_rcg_markers', 10)

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
            f'hole_execution={self.enable_hole_execution}, mode=LocalRepairRCG validation only'
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

    def publish_xy_goal(self, xy: Tuple[float, float]) -> None:
        """Publish a non-RCG repair waypoint on /cstar/goal and /cstar/goal_marker."""
        x, y = xy

        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
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
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
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
            self.arm_hole_local_repair_rcg(
                entry_base_key=base_key,
                seed_key=seed_key,
                exit_key=exit_key,
                component=comp,
                attachments=attachments,
            )


    # ------------------------------------------------------------------
    # Hole-local LocalRepairRCG validation utilities
    # ------------------------------------------------------------------
    def distance_xy(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def build_repair_safe_mask(self) -> Optional[np.ndarray]:
        """Build a safe free-space mask for the hole-local RCG sampler."""
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
        """Treat the current C* scan band as the outside boundary of the hole."""
        if self.free_msg is None:
            return mask
        out = mask.copy()
        res = self.free_msg.info.resolution
        rad = max(0, int(math.ceil(self.repair_scan_band_block_radius / res)))
        h, w = out.shape
        seed_cell = self.world_to_cell(*self.nodes[seed_key]) if seed_key in self.nodes else None

        for key in self.latest_scan_band:
            if key not in self.nodes:
                continue
            cell = self.world_to_cell(*self.nodes[key])
            if cell is None:
                continue
            r, c = cell
            out[max(0, r - rad):min(h, r + rad + 1), max(0, c - rad):min(w, c + rad + 1)] = False

        if seed_cell is not None:
            sr, sc = seed_cell
            if 0 <= sr < h and 0 <= sc < w:
                out[sr, sc] = True
        return out

    def component_cells_and_roi(
        self,
        comp: Set[NodeKey],
        seed_key: NodeKey,
    ) -> Tuple[Set[GridCell], Optional[Tuple[int, int, int, int]]]:
        """Return component cells and a soft ROI that can grow with active_hole_mask."""
        if self.free_msg is None:
            return set(), None

        cells: Set[GridCell] = set()
        for key in comp | {seed_key}:
            if key not in self.nodes:
                continue
            cell = self.world_to_cell(*self.nodes[key])
            if cell is not None:
                cells.add(cell)

        # Important change from the old repair route builder: the ROI is no
        # longer locked only to the initial component.  Once active_hole_mask
        # exists, include its current bbox so the mask can expand gradually as
        # lidar reveals more free cells behind the same fixed gate.
        if self.active_hole_mask is not None:
            rows, cols = np.where(self.active_hole_mask)
            if len(rows) > 0:
                cells.add((int(rows.min()), int(cols.min())))
                cells.add((int(rows.max()), int(cols.max())))

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

    def build_hole_gate(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Build a virtual gate through seed; normal points from C* lap into hole."""
        if entry_base_key not in self.nodes or seed_key not in self.nodes:
            return None
        entry_xy = self.nodes[entry_base_key]
        seed_xy = self.nodes[seed_key]
        dx = seed_xy[0] - entry_xy[0]
        dy = seed_xy[1] - entry_xy[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        normal = (dx / norm, dy / norm)
        return seed_xy, normal

    def set_active_hole_gate(self, entry_base_key: NodeKey, seed_key: NodeKey) -> bool:
        gate = self.build_hole_gate(entry_base_key, seed_key)
        if gate is None:
            self.active_hole_gate_origin = None
            self.active_hole_gate_normal = None
            return False
        self.active_hole_gate_origin, self.active_hole_gate_normal = gate
        return True

    def point_inside_active_gate(self, xy: Tuple[float, float], tolerance: Optional[float] = None) -> bool:
        if self.active_hole_gate_origin is None or self.active_hole_gate_normal is None:
            return True
        tol = self.repair_gate_tolerance if tolerance is None else tolerance
        ox, oy = self.active_hole_gate_origin
        nx, ny = self.active_hole_gate_normal
        signed = (xy[0] - ox) * nx + (xy[1] - oy) * ny
        return signed >= -tol

    def build_gate_half_plane_mask(self, gate_origin=None, gate_normal=None) -> Optional[np.ndarray]:
        if self.free_msg is None:
            return None
        origin = gate_origin if gate_origin is not None else self.active_hole_gate_origin
        normal = gate_normal if gate_normal is not None else self.active_hole_gate_normal
        if origin is None or normal is None:
            return None
        h = self.free_msg.info.height
        w = self.free_msg.info.width
        info = self.free_msg.info
        ox, oy = origin
        nx, ny = normal
        mask = np.zeros((h, w), dtype=bool)
        for r in range(h):
            y = info.origin.position.y + (r + 0.5) * info.resolution
            for c in range(w):
                x = info.origin.position.x + (c + 0.5) * info.resolution
                signed = (x - ox) * nx + (y - oy) * ny
                if signed >= -self.repair_gate_tolerance:
                    mask[r, c] = True
        return mask

    def find_nearest_true_cell_in_roi(
        self,
        start: GridCell,
        mask: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Optional[GridCell]:
        sr, sc = start
        rmin, rmax, cmin, cmax = roi
        if rmin <= sr <= rmax and cmin <= sc <= cmax and mask[sr, sc]:
            return sr, sc
        best = None
        best_d = float('inf')
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                if not mask[r, c]:
                    continue
                d = (r - sr) * (r - sr) + (c - sc) * (c - sc)
                if d < best_d:
                    best_d = d
                    best = (r, c)
        return best

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
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            r, c = q.popleft()
            for dr, dc in neighbors:
                nr = r + dr
                nc = c + dc
                if nr < rmin or nr > rmax or nc < cmin or nc > cmax:
                    continue
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if (nr, nc) in visited or not mask[nr, nc]:
                    continue
                visited.add((nr, nc))
                out[nr, nc] = True
                q.append((nr, nc))
        return out

    def build_active_hole_mask(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
        comp: Set[NodeKey],
    ) -> bool:
        """Build/update the gate-clipped active_hole_mask from current maps."""
        safe = self.build_repair_safe_mask()
        if safe is None or self.free_msg is None:
            return False
        if self.active_hole_gate_origin is None or self.active_hole_gate_normal is None:
            if not self.set_active_hole_gate(entry_base_key, seed_key):
                return False
        gate_mask = self.build_gate_half_plane_mask()
        if gate_mask is None:
            return False

        _, roi = self.component_cells_and_roi(comp, seed_key)
        if roi is None:
            return False
        rmin, rmax, cmin, cmax = roi

        candidate = safe & gate_mask
        candidate = self.block_scan_band_on_mask(candidate, seed_key)
        roi_mask = np.zeros_like(candidate, dtype=bool)
        roi_mask[rmin:rmax + 1, cmin:cmax + 1] = True
        candidate &= roi_mask

        seed_cell = self.world_to_cell(*self.nodes[seed_key]) if seed_key in self.nodes else None
        if seed_cell is None:
            return False
        start = self.find_nearest_true_cell_in_roi(seed_cell, candidate, roi)
        if start is None:
            return False

        new_mask = self.local_mask_floodfill(start, candidate, roi)
        if not np.any(new_mask):
            return False

        if self.active_hole_mask is not None and self.active_hole_mask.shape == new_mask.shape:
            combined = (self.active_hole_mask | new_mask) & candidate
            # If current safe/candidate temporarily shrinks due to map noise,
            # avoid erasing the whole visualization in one cycle.
            if np.any(combined):
                self.active_hole_mask = combined
            else:
                self.active_hole_mask = new_mask
        else:
            self.active_hole_mask = new_mask
        self.active_hole_roi = roi
        return True

    def determine_local_repair_axis(self) -> str:
        """Choose LocalRepairRCG lane direction from the fixed gate normal."""
        if self.active_hole_gate_normal is None:
            return 'vertical'
        nx, ny = self.active_hole_gate_normal
        # If the robot enters mainly along Y, use vertical lanes; if it enters
        # mainly along X, use horizontal lanes.  This keeps the local sampler
        # aligned with the hole depth direction instead of arbitrary map rows.
        return 'vertical' if abs(ny) >= abs(nx) else 'horizontal'

    def collect_local_laps_from_mask(self, mask: np.ndarray, axis: str, lap_step: int) -> List[int]:
        if axis == 'horizontal':
            has_safe = np.any(mask, axis=1)
        else:
            has_safe = np.any(mask, axis=0)
        n = len(has_safe)
        bands: List[Tuple[int, int]] = []
        inside = False
        start = 0
        for i in range(n):
            if has_safe[i] and not inside:
                inside = True
                start = i
            elif not has_safe[i] and inside:
                inside = False
                end = i - 1
                if end >= start:
                    bands.append((start, end))
        if inside:
            bands.append((start, n - 1))

        laps: Set[int] = set()
        for start, end in bands:
            width = end - start + 1
            if width <= max(2, lap_step // 2):
                laps.add((start + end) // 2)
                continue
            i = start
            local: List[int] = []
            while i <= end:
                local.append(i)
                i += lap_step
            if local:
                laps.update(local)
                if end - local[-1] >= max(2, lap_step // 2):
                    laps.add(end)
        return sorted(laps)

    def find_local_safe_runs(self, mask: np.ndarray, axis: str, lap_index: int) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        if self.free_msg is None:
            return runs
        res = self.free_msg.info.resolution
        min_cells = max(1, int(round(self.local_repair_min_run_length / res)))
        if axis == 'horizontal':
            arr = mask[lap_index, :]
        else:
            arr = mask[:, lap_index]

        inside = False
        start = 0
        for i, val in enumerate(arr):
            if bool(val) and not inside:
                inside = True
                start = i
            elif not bool(val) and inside:
                inside = False
                end = i - 1
                if end >= start and (end - start + 1) >= min_cells:
                    runs.append((start, end))
        if inside:
            end = len(arr) - 1
            if end >= start and (end - start + 1) >= min_cells:
                runs.append((start, end))
        return runs

    def local_sample_segment(self, start: int, end: int, step: int) -> List[int]:
        if end <= start:
            return [start]
        length = end - start + 1
        if length <= max(2, int(0.75 * step)):
            return [(start + end) // 2]
        values = list(range(start, end + 1, step))
        if values[-1] != end:
            values.append(end)
        return sorted(set(values))

    def filter_local_close_candidates(self, candidates: List[Tuple[int, bool, int]]) -> List[Tuple[int, bool, int]]:
        if not candidates or self.free_msg is None:
            return candidates
        candidates = sorted(candidates, key=lambda item: item[0])
        res = self.free_msg.info.resolution
        min_dist_cells = max(1, int(round(self.local_repair_min_node_keep_distance / res)))
        kept: List[Tuple[int, bool, int]] = []
        for cand in candidates:
            value, endpoint, order = cand
            if not kept:
                kept.append(cand)
                continue
            prev_value, prev_endpoint, _ = kept[-1]
            if abs(value - prev_value) >= min_dist_cells:
                kept.append(cand)
                continue
            if endpoint and not prev_endpoint:
                kept[-1] = cand
        last = candidates[-1]
        if last not in kept:
            if not kept or abs(last[0] - kept[-1][0]) >= min_dist_cells:
                kept.append(last)
        return kept

    def local_cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        assert self.free_msg is not None
        info = self.free_msg.info
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def local_line_is_safe(self, mask: np.ndarray, a: Dict[str, object], b: Dict[str, object]) -> bool:
        r0 = int(a['row']); c0 = int(a['col'])
        r1 = int(b['row']); c1 = int(b['col'])
        n = max(abs(r1 - r0), abs(c1 - c0)) + 1
        h, w = mask.shape
        for i in range(n + 1):
            t = 0.0 if n == 0 else i / n
            rr = int(round((1.0 - t) * r0 + t * r1))
            cc = int(round((1.0 - t) * c0 + t * c1))
            if rr < 0 or rr >= h or cc < 0 or cc >= w or not mask[rr, cc]:
                return False
        return True

    def local_segments_overlap_or_near(self, a0: int, a1: int, b0: int, b1: int, tol: int) -> bool:
        return min(a1, b1) + tol >= max(a0, b0) - tol

    def try_add_local_edge(self, edges: Set[Tuple[int, int]], nodes: List[Dict[str, object]], i: int, j: int, mask: np.ndarray) -> bool:
        if i == j:
            return False
        a, b = min(i, j), max(i, j)
        if (a, b) in edges:
            return False
        if not self.local_line_is_safe(mask, nodes[a], nodes[b]):
            return False
        edges.add((a, b))
        return True

    def build_local_repair_rcg(self) -> None:
        """Build an internal RCG-like grid inside active_hole_mask only."""
        self.local_repair_nodes = []
        self.local_repair_edges = set()
        self.local_repair_route_points = []
        if self.active_hole_mask is None or self.free_msg is None:
            return
        mask = self.active_hole_mask
        if not np.any(mask):
            return

        res = self.free_msg.info.resolution
        lap_step = max(1, int(round(self.local_repair_lap_spacing / res)))
        sample_step = max(1, int(round(self.local_repair_sample_spacing / res)))
        interlap_max_cells = max(1.0, self.local_repair_interlap_max_dist / res)
        interlap_tol_cells = max(1, int(round(self.local_repair_interlap_col_tolerance / res)))

        axis = self.determine_local_repair_axis()
        self.local_repair_axis = axis
        laps = self.collect_local_laps_from_mask(mask, axis, lap_step)
        lap_seg_to_nodes: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        lap_seg_bounds: Dict[Tuple[int, int], Tuple[int, int]] = {}

        for lap_id, lap_index in enumerate(laps):
            runs = self.find_local_safe_runs(mask, axis, lap_index)
            for seg_id, (start, end) in enumerate(runs):
                samples = self.local_sample_segment(start, end, sample_step)
                candidates: List[Tuple[int, bool, int]] = []
                for order, along in enumerate(samples):
                    endpoint = (order == 0 or order == len(samples) - 1)
                    candidates.append((along, endpoint, order))
                candidates = self.filter_local_close_candidates(candidates)
                kept: List[Dict[str, object]] = []
                for along, endpoint, order in candidates:
                    if axis == 'horizontal':
                        row, col = lap_index, along
                    else:
                        row, col = along, lap_index
                    if row < 0 or row >= mask.shape[0] or col < 0 or col >= mask.shape[1] or not mask[row, col]:
                        continue
                    x, y = self.local_cell_to_world(row, col)
                    node: Dict[str, object] = {
                        'idx': len(self.local_repair_nodes),
                        'row': row,
                        'col': col,
                        'x': x,
                        'y': y,
                        'lap_id': lap_id,
                        'seg_id': seg_id,
                        'endpoint': endpoint,
                    }
                    self.local_repair_nodes.append(node)
                    kept.append(node)
                for a, b in zip(kept[:-1], kept[1:]):
                    self.try_add_local_edge(self.local_repair_edges, self.local_repair_nodes, int(a['idx']), int(b['idx']), mask)
                lap_seg_to_nodes[(lap_id, seg_id)] = kept
                lap_seg_bounds[(lap_id, seg_id)] = (start, end)

        if not self.local_repair_enable_interlap_edges:
            return

        for lap_id in range(len(laps) - 1):
            if abs(laps[lap_id + 1] - laps[lap_id]) > interlap_max_cells:
                continue
            curr_seg_ids = sorted(key[1] for key in lap_seg_to_nodes.keys() if key[0] == lap_id)
            next_seg_ids = sorted(key[1] for key in lap_seg_to_nodes.keys() if key[0] == lap_id + 1)
            for curr_seg_id in curr_seg_ids:
                seg_a = lap_seg_to_nodes.get((lap_id, curr_seg_id), [])
                if not seg_a:
                    continue
                a0, a1 = lap_seg_bounds[(lap_id, curr_seg_id)]
                for next_seg_id in next_seg_ids:
                    seg_b = lap_seg_to_nodes.get((lap_id + 1, next_seg_id), [])
                    if not seg_b:
                        continue
                    b0, b1 = lap_seg_bounds[(lap_id + 1, next_seg_id)]
                    if not self.local_segments_overlap_or_near(a0, a1, b0, b1, interlap_tol_cells):
                        continue
                    pair_candidates: List[Tuple[float, float, Dict[str, object], Dict[str, object]]] = []
                    for a in seg_a:
                        for b in seg_b:
                            along_a = int(a['col']) if axis == 'horizontal' else int(a['row'])
                            along_b = int(b['col']) if axis == 'horizontal' else int(b['row'])
                            lap_a = int(a['row']) if axis == 'horizontal' else int(a['col'])
                            lap_b = int(b['row']) if axis == 'horizontal' else int(b['col'])
                            dalong = float(along_b - along_a)
                            dlap = float(lap_b - lap_a)
                            dist = math.hypot(dalong, dlap)
                            if dist > interlap_max_cells or abs(dalong) > interlap_tol_cells:
                                continue
                            if not self.local_line_is_safe(mask, a, b):
                                continue
                            pair_candidates.append((abs(dalong), dist, a, b))
                    pair_candidates.sort(key=lambda item: (item[0], item[1]))
                    used_a: Set[int] = set()
                    used_b: Set[int] = set()
                    for _, _, a, b in pair_candidates:
                        ai = int(a['idx']); bi = int(b['idx'])
                        if ai in used_a or bi in used_b:
                            continue
                        if self.try_add_local_edge(self.local_repair_edges, self.local_repair_nodes, ai, bi, mask):
                            used_a.add(ai)
                            used_b.add(bi)

    def append_route_point(self, route: List[Tuple[float, float]], xy: Tuple[float, float]) -> None:
        if route:
            lx, ly = route[-1]
            if math.hypot(xy[0] - lx, xy[1] - ly) < 0.03:
                return
        route.append((float(xy[0]), float(xy[1])))

    def local_node_xy(self, node: Dict[str, object]) -> Tuple[float, float]:
        return float(node['x']), float(node['y'])

    def local_node_lap_coord(self, node: Dict[str, object]) -> int:
        # The lap coordinate is the coordinate perpendicular to a cleaning lane.
        # For horizontal lanes it is the row; for vertical lanes it is the col.
        return int(node['row']) if self.local_repair_axis == 'horizontal' else int(node['col'])

    def local_node_along_coord(self, node: Dict[str, object]) -> int:
        # The along coordinate is the coordinate along a cleaning lane.
        return int(node['col']) if self.local_repair_axis == 'horizontal' else int(node['row'])

    def local_repair_lane_groups(self) -> List[Dict[str, object]]:
        groups: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for node in self.local_repair_nodes:
            key = (int(node['lap_id']), int(node['seg_id']))
            groups.setdefault(key, []).append(node)

        lanes: List[Dict[str, object]] = []
        min_nodes = max(1, self.local_repair_route_min_lane_nodes)
        for (lap_id, seg_id), nodes in groups.items():
            ordered = sorted(nodes, key=self.local_node_along_coord)
            if len(ordered) < min_nodes:
                continue
            along_values = [self.local_node_along_coord(n) for n in ordered]
            lap_values = [self.local_node_lap_coord(n) for n in ordered]
            length_cells = max(along_values) - min(along_values) if along_values else 0
            lanes.append({
                'lap_id': lap_id,
                'seg_id': seg_id,
                'nodes': ordered,
                'lap_coord': int(round(sum(lap_values) / max(1, len(lap_values)))),
                'length_cells': length_cells,
            })
        lanes.sort(key=lambda lane: (int(lane['lap_coord']), int(lane['seg_id'])))
        return lanes

    def route_exit_lap_coordinate(self) -> Optional[int]:
        if self.active_hole_exit_key is None or self.active_hole_exit_key not in self.nodes:
            return None
        ex, ey = self.nodes[self.active_hole_exit_key]
        cell = self.world_to_cell(ex, ey)
        if cell is None:
            return None
        row, col = cell
        return row if self.local_repair_axis == 'horizontal' else col

    def order_local_repair_lanes_for_route(self, lanes: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if len(lanes) <= 1:
            return lanes
        exit_coord = self.route_exit_lap_coordinate()
        if exit_coord is None:
            return lanes
        min_coord = int(lanes[0]['lap_coord'])
        max_coord = int(lanes[-1]['lap_coord'])
        # Make the last lane the one nearer to the chosen exit side.  The route
        # therefore sweeps from one side of the hole toward the other side.
        if abs(max_coord - exit_coord) <= abs(min_coord - exit_coord):
            return lanes
        return list(reversed(lanes))

    def local_graph_node_path(self, start_idx: int, goal_idx: int) -> List[int]:
        if start_idx == goal_idx:
            return [start_idx]
        adj: Dict[int, List[int]] = {}
        for i, j in self.local_repair_edges:
            adj.setdefault(i, []).append(j)
            adj.setdefault(j, []).append(i)
        q = deque([start_idx])
        prev: Dict[int, Optional[int]] = {start_idx: None}
        while q:
            cur = q.popleft()
            if cur == goal_idx:
                break
            for nb in adj.get(cur, []):
                if nb in prev:
                    continue
                prev[nb] = cur
                q.append(nb)
        if goal_idx not in prev:
            return []
        path: List[int] = []
        cur: Optional[int] = goal_idx
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return path

    def append_connector_to_route(
        self,
        route: List[Tuple[float, float]],
        from_node: Optional[Dict[str, object]],
        to_node: Dict[str, object],
    ) -> None:
        if from_node is None:
            self.append_route_point(route, self.local_node_xy(to_node))
            return
        mask = self.active_hole_mask
        if mask is not None and self.local_line_is_safe(mask, from_node, to_node):
            self.append_route_point(route, self.local_node_xy(to_node))
            return
        graph_path = self.local_graph_node_path(int(from_node['idx']), int(to_node['idx']))
        if len(graph_path) >= 2:
            for idx in graph_path[1:]:
                if 0 <= idx < len(self.local_repair_nodes):
                    self.append_route_point(route, self.local_node_xy(self.local_repair_nodes[idx]))
            return
        # Last-resort visualization fallback.  If this happens in RViz, the gap
        # means LocalRepairRCG has no safe connector between the chosen lanes.
        self.append_route_point(route, self.local_node_xy(to_node))

    def build_local_repair_route_preview(self) -> None:
        """
        Build a non-executable one-stroke repair-route preview from LocalRepairRCG lanes.

        The LocalRepairRCG edges are not used to decide the sweeping order.  The
        order is fixed by lane positions.  Edges are used only when a connector
        between two already selected lane endpoints needs a safe local fallback.
        """
        self.local_repair_route_points = []
        self.latest_hole_repair_path = []
        if not self.local_repair_nodes:
            return

        lanes = self.local_repair_lane_groups()
        if not lanes:
            return
        lanes = self.order_local_repair_lanes_for_route(lanes)

        seed_xy: Optional[Tuple[float, float]] = None
        if self.active_hole_seed_key is not None and self.active_hole_seed_key in self.nodes:
            seed_xy = self.nodes[self.active_hole_seed_key]
        entry_xy: Optional[Tuple[float, float]] = None
        if self.active_hole_entry_base_key is not None and self.active_hole_entry_base_key in self.nodes:
            entry_xy = self.nodes[self.active_hole_entry_base_key]

        route: List[Tuple[float, float]] = []
        if self.local_repair_route_include_entry_segment and entry_xy is not None:
            self.append_route_point(route, entry_xy)
        if seed_xy is not None:
            self.append_route_point(route, seed_xy)

        prev_node: Optional[Dict[str, object]] = None
        first_lane = True
        forward = True
        for lane in lanes:
            nodes = list(lane['nodes'])
            if not nodes:
                continue

            if first_lane:
                # Start the first lane from the endpoint closer to the seed, so
                # the visible route first enters toward one side of the hole.
                if seed_xy is not None:
                    first = nodes[0]
                    last = nodes[-1]
                    d_first = math.hypot(float(first['x']) - seed_xy[0], float(first['y']) - seed_xy[1])
                    d_last = math.hypot(float(last['x']) - seed_xy[0], float(last['y']) - seed_xy[1])
                    forward = d_first <= d_last
                ordered_nodes = nodes if forward else list(reversed(nodes))
                first_lane = False
            else:
                ordered_nodes = nodes if forward else list(reversed(nodes))

            self.append_connector_to_route(route, prev_node, ordered_nodes[0])
            for node in ordered_nodes[1:]:
                self.append_route_point(route, self.local_node_xy(node))
            prev_node = ordered_nodes[-1]
            forward = not forward

        if self.local_repair_route_include_exit_segment:
            if self.active_hole_exit_key is not None and self.active_hole_exit_key in self.nodes:
                self.append_route_point(route, self.nodes[self.active_hole_exit_key])

        self.local_repair_route_points = route
        self.latest_hole_repair_path = list(route)

    def update_hole_local_repair_debug(self) -> bool:
        if self.active_hole_entry_base_key is None or self.active_hole_seed_key is None:
            return False
        if self.active_hole_entry_base_key not in self.nodes or self.active_hole_seed_key not in self.nodes:
            return False
        ok = self.build_active_hole_mask(
            self.active_hole_entry_base_key,
            self.active_hole_seed_key,
            self.active_hole_component,
        )
        if ok:
            self.build_local_repair_rcg()
            if self.local_repair_build_route_preview:
                self.build_local_repair_route_preview()
            else:
                self.local_repair_route_points = []
                self.latest_hole_repair_path = []
        return ok

    def arm_hole_local_repair_rcg(
        self,
        entry_base_key: NodeKey,
        seed_key: NodeKey,
        exit_key: Optional[NodeKey],
        component: Set[NodeKey],
        attachments: Set[NodeKey],
    ) -> None:
        """Arm LocalRepairRCG and prepare for entry-first repair execution."""
        if entry_base_key not in self.nodes or seed_key not in self.nodes or not component:
            return

        self.mode = 'HOLE_ARMED'
        self.escape_active = False
        self.escape_path_xy.clear()
        self.publish_empty_escape_path()

        self.active_hole_component = set(component)
        self.active_hole_attachments = set(attachments)
        self.active_hole_entry_base_key = entry_base_key
        self.active_hole_seed_key = seed_key
        self.active_hole_exit_key = exit_key
        self.active_hole_mask = None
        self.active_hole_roi = None
        self.local_repair_nodes = []
        self.local_repair_edges = set()
        self.local_repair_route_points = []
        self.active_hole_execute_route_points = []
        self.active_hole_route_goal_index = 0
        self.active_hole_current_goal_xy = None
        self.active_hole_executed_route_points = []
        self.set_active_hole_gate(entry_base_key, seed_key)
        self.update_hole_local_repair_debug()

        self.latest_hole_component = set(component)
        self.latest_hole_attachments = set(attachments)
        self.latest_hole_entry_key = seed_key
        self.latest_hole_exit_key = exit_key
        self.latest_hole_repair_path = []
        self.latest_branch_edge = (entry_base_key, seed_key)

        self.set_new_goal(entry_base_key, 'hole entry_base before LocalRepair execution')
        self.get_logger().info(
            f'Hole LocalRepair execution armed: entry_base={entry_base_key}, '
            f'seed={seed_key}, exit={exit_key}, local_nodes={len(self.local_repair_nodes)}, '
            f'local_edges={len(self.local_repair_edges)}, axis={self.local_repair_axis}'
        )

    def first_unreached_repair_route_index(
        self,
        route: List[Tuple[float, float]],
        robot_xy: Tuple[float, float],
        start_index: int = 0,
    ) -> int:
        tol = max(0.03, self.local_repair_goal_tolerance)
        for idx in range(max(0, start_index), len(route)):
            if self.distance_xy(route[idx], robot_xy) > tol:
                return idx
        return len(route)

    def set_repair_route_goal(self, index: int) -> None:
        if index < 0 or index >= len(self.active_hole_execute_route_points):
            self.active_hole_current_goal_xy = None
            return
        self.active_hole_route_goal_index = index
        xy = self.active_hole_execute_route_points[index]
        self.active_hole_current_goal_xy = xy
        self.current_goal_key = None
        self.selected_path = [xy]
        self.publish_xy_goal(xy)

    def route_goal_reached_or_passed(
        self,
        robot_xy: Tuple[float, float],
        index: int,
    ) -> bool:
        if index < 0 or index >= len(self.active_hole_execute_route_points):
            return True

        goal = self.active_hole_execute_route_points[index]
        if self.distance_xy(robot_xy, goal) <= max(0.03, self.local_repair_goal_tolerance):
            return True

        if index <= 0:
            return False

        prev = self.active_hole_execute_route_points[index - 1]
        vx = goal[0] - prev[0]
        vy = goal[1] - prev[1]
        seg_len2 = vx * vx + vy * vy
        if seg_len2 < 1e-8:
            return False

        wx = robot_xy[0] - prev[0]
        wy = robot_xy[1] - prev[1]
        proj = (wx * vx + wy * vy) / seg_len2
        if proj < 1.0:
            return False

        closest_x = prev[0] + proj * vx
        closest_y = prev[1] + proj * vy
        lateral = math.hypot(robot_xy[0] - closest_x, robot_xy[1] - closest_y)
        return lateral <= max(self.local_repair_goal_tolerance, self.local_repair_goal_passed_tolerance)

    def start_hole_repair_execution(self, robot_xy: Tuple[float, float]) -> bool:
        if not self.local_repair_execute_route:
            return False

        # Lock the current route.  After this point the robot follows this fixed
        # route monotonically; LocalRepairRCG can still be visualized, but it no
        # longer moves the current purple goal backward.
        route = list(self.local_repair_route_points)
        route = self.compact_xy_route(route)
        if len(route) < self.local_repair_min_execute_points:
            self.get_logger().warn(
                f'Cannot start hole repair: route points={len(route)} < '
                f'{self.local_repair_min_execute_points}'
            )
            return False

        self.mode = 'HOLE_REPAIR'
        self.current_goal_key = None
        self.escape_active = False
        self.escape_path_xy.clear()
        self.active_hole_execute_route_points = route
        self.active_hole_executed_route_points = []
        self.latest_hole_repair_path = list(route)

        idx = self.first_unreached_repair_route_index(route, robot_xy, start_index=0)
        if idx >= len(route):
            self.finish_hole_repair(robot_xy)
            return True

        self.set_repair_route_goal(idx)
        self.get_logger().info(
            f'Start LocalRepair route execution: points={len(route)}, start_index={idx}, '
            f'entry={self.active_hole_entry_base_key}, exit={self.active_hole_exit_key}'
        )
        return True

    def compact_xy_route(self, route: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        compact: List[Tuple[float, float]] = []
        min_dist = max(0.02, 0.5 * self.local_repair_goal_tolerance)
        for xy in route:
            if not compact or self.distance_xy(compact[-1], xy) >= min_dist:
                compact.append((float(xy[0]), float(xy[1])))
        return compact

    def handle_hole_armed(self, robot_xy: Tuple[float, float]) -> None:
        """Drive to entry_base, then lock and execute the current repair route."""
        self.publish_empty_escape_path()
        self.update_hole_local_repair_debug()
        if self.active_hole_entry_base_key is None or self.active_hole_entry_base_key not in self.nodes:
            self.abort_hole_execution('entry_base disappeared')
            return

        if self.is_key_reached(self.active_hole_entry_base_key, robot_xy, tolerance=max(self.goal_center_tolerance, self.local_repair_goal_tolerance)):
            if self.start_hole_repair_execution(robot_xy):
                return
            self.abort_hole_execution('no executable LocalRepair route')
            return

        self.set_new_goal(self.active_hole_entry_base_key, 'hole entry_base before LocalRepair execution')

    def handle_hole_repair(self, robot_xy: Tuple[float, float]) -> None:
        if not self.active_hole_execute_route_points:
            self.abort_hole_execution('empty executable repair route')
            return

        idx = self.active_hole_route_goal_index
        advanced = False
        while idx < len(self.active_hole_execute_route_points) and self.route_goal_reached_or_passed(robot_xy, idx):
            reached_xy = self.active_hole_execute_route_points[idx]
            self.active_hole_executed_route_points.append(reached_xy)
            self.add_closed_position(reached_xy[0], reached_xy[1])
            idx += 1
            advanced = True

        if idx >= len(self.active_hole_execute_route_points):
            self.finish_hole_repair(robot_xy)
            return

        if advanced or self.active_hole_current_goal_xy is None:
            self.set_repair_route_goal(idx)
        else:
            # Keep the current purple target alive for controllers that start or
            # reconnect after the planner.  Do not recompute it from the latest
            # map during execution.
            self.publish_xy_goal(self.active_hole_execute_route_points[idx])

    def finish_hole_repair(self, robot_xy: Tuple[float, float]) -> None:
        route = list(self.active_hole_execute_route_points)
        for x, y in route:
            self.add_closed_position(x, y)

        # Mark the original graph component as closed, so normal C* will not
        # rediscover or re-enter the same repaired hole as an open branch.
        for key in list(self.active_hole_component):
            if key in self.nodes:
                self.close_key(key)
        if self.active_hole_entry_base_key is not None and self.active_hole_entry_base_key in self.nodes:
            self.close_key(self.active_hole_entry_base_key)

        exit_key = self.active_hole_exit_key
        if exit_key is not None and exit_key in self.nodes:
            # End the repair near the exit attachment and let the normal C*
            # policy choose the next open neighbor from that neighborhood.
            self.close_key(exit_key)

        self.get_logger().info(
            f'LocalRepair route finished: route_points={len(route)}, '
            f'closed_component={len(self.active_hole_component)}, resume_exit={exit_key}'
        )

        self.mode = 'COVERAGE'
        self.current_goal_key = None
        self.selected_path.clear()
        self.escape_active = False
        self.escape_path_xy.clear()
        self.active_hole_component.clear()
        self.active_hole_attachments.clear()
        self.active_hole_entry_base_key = None
        self.active_hole_seed_key = None
        self.active_hole_exit_key = None
        self.active_hole_gate_origin = None
        self.active_hole_gate_normal = None
        self.active_hole_mask = None
        self.active_hole_roi = None
        self.local_repair_nodes = []
        self.local_repair_edges = set()
        self.local_repair_route_points = route
        self.active_hole_execute_route_points = []
        self.active_hole_route_goal_index = 0
        self.active_hole_current_goal_xy = None
        self.active_hole_executed_route_points = []
        self.latest_hole_repair_path = list(route)
        self.publish_empty_escape_path()

    def abort_hole_execution(self, reason: str) -> None:
        self.get_logger().warn(f'Hole LocalRepair execution aborted: {reason}')
        self.publish_empty_escape_path()
        self.mode = 'COVERAGE'
        self.active_hole_component.clear()
        self.active_hole_attachments.clear()
        self.active_hole_entry_base_key = None
        self.active_hole_seed_key = None
        self.active_hole_exit_key = None
        self.active_hole_gate_origin = None
        self.active_hole_gate_normal = None
        self.active_hole_mask = None
        self.active_hole_roi = None
        self.local_repair_nodes = []
        self.local_repair_edges = set()
        self.local_repair_route_points = []
        self.active_hole_execute_route_points = []
        self.active_hole_route_goal_index = 0
        self.active_hole_current_goal_xy = None
        self.active_hole_executed_route_points = []
        self.latest_hole_repair_path = []
        self.current_goal_key = None
        self.selected_path.clear()

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

        # Hole execution states have priority over normal C* goal selection.
        # They may temporarily follow LocalRepair samples that are not exactly
        # on a global RCG node, so do not require nearest_key here.
        if self.mode == 'HOLE_ARMED':
            self.handle_hole_armed(robot_xy)
        elif self.mode == 'HOLE_REPAIR':
            self.handle_hole_repair(robot_xy)
        else:
            if nearest_key is None:
                return
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
        self.publish_hole_gate_markers(stamp)
        self.publish_active_hole_mask_marker(stamp)
        self.publish_local_repair_rcg_outputs(stamp)

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
        """Publish the non-executable LocalRepairRCG one-stroke route preview."""
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = stamp
        for x, y in self.latest_hole_repair_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.10
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
        line.ns = 'local_repair_route_preview'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.045
        line.color.r = 1.0
        line.color.g = 0.15
        line.color.b = 0.05
        line.color.a = 0.95
        line.pose.orientation.w = 1.0

        points = Marker()
        points.header.frame_id = self.map_frame
        points.header.stamp = stamp
        points.ns = 'local_repair_route_points'
        points.id = 1
        points.type = Marker.SPHERE_LIST
        points.action = Marker.ADD
        points.scale.x = 0.08
        points.scale.y = 0.08
        points.scale.z = 0.08
        points.color.r = 1.0
        points.color.g = 0.05
        points.color.b = 0.05
        points.color.a = 0.90
        points.pose.orientation.w = 1.0

        for x, y in self.latest_hole_repair_path:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.20
            line.points.append(p)
            pp = Point(); pp.x = float(x); pp.y = float(y); pp.z = 0.22
            points.points.append(pp)

        ma.markers.append(line)
        ma.markers.append(points)
        self.hole_repair_markers_pub.publish(ma)

    def publish_hole_gate_markers(self, stamp) -> None:
        ma = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        if self.active_hole_gate_origin is not None and self.active_hole_gate_normal is not None:
            ox, oy = self.active_hole_gate_origin
            nx, ny = self.active_hole_gate_normal
            tx, ty = -ny, nx

            length = max(0.5, self.local_repair_gate_viz_length)
            if self.active_hole_mask is not None and self.free_msg is not None and np.any(self.active_hole_mask):
                rows, cols = np.where(self.active_hole_mask)
                res = self.free_msg.info.resolution
                extent = max((rows.max() - rows.min() + 1) * res, (cols.max() - cols.min() + 1) * res)
                length = max(length, 0.75 * extent)

            gate_line = Marker()
            gate_line.header.frame_id = self.map_frame
            gate_line.header.stamp = stamp
            gate_line.ns = 'hole_virtual_gate_line'
            gate_line.id = 0
            gate_line.type = Marker.LINE_LIST
            gate_line.action = Marker.ADD
            gate_line.scale.x = 0.045
            gate_line.color.r = 1.0
            gate_line.color.g = 1.0
            gate_line.color.b = 0.0
            gate_line.color.a = 0.95
            gate_line.pose.orientation.w = 1.0
            p1 = Point(); p1.x = ox - tx * length; p1.y = oy - ty * length; p1.z = 0.24
            p2 = Point(); p2.x = ox + tx * length; p2.y = oy + ty * length; p2.z = 0.24
            gate_line.points.append(p1); gate_line.points.append(p2)
            ma.markers.append(gate_line)

            normal_arrow = Marker()
            normal_arrow.header.frame_id = self.map_frame
            normal_arrow.header.stamp = stamp
            normal_arrow.ns = 'hole_gate_normal_arrow'
            normal_arrow.id = 1
            normal_arrow.type = Marker.ARROW
            normal_arrow.action = Marker.ADD
            normal_arrow.scale.x = 0.04
            normal_arrow.scale.y = 0.09
            normal_arrow.scale.z = 0.09
            normal_arrow.color.r = 0.0
            normal_arrow.color.g = 1.0
            normal_arrow.color.b = 0.2
            normal_arrow.color.a = 0.95
            normal_arrow.pose.orientation.w = 1.0
            a0 = Point(); a0.x = ox; a0.y = oy; a0.z = 0.26
            a1 = Point(); a1.x = ox + nx * 0.55; a1.y = oy + ny * 0.55; a1.z = 0.26
            normal_arrow.points.append(a0); normal_arrow.points.append(a1)
            ma.markers.append(normal_arrow)

        self.hole_gate_markers_pub.publish(ma)

    def publish_active_hole_mask_marker(self, stamp) -> None:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = stamp
        marker.ns = 'active_hole_mask_cells'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.8
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.28

        if self.active_hole_mask is not None and self.free_msg is not None:
            res = self.free_msg.info.resolution
            stride = max(1, self.local_repair_mask_viz_stride)
            marker.scale.x = res * stride
            marker.scale.y = res * stride
            marker.scale.z = 0.02
            rows, cols = np.where(self.active_hole_mask)
            for r, c in zip(rows, cols):
                if int(r) % stride != 0 or int(c) % stride != 0:
                    continue
                x, y = self.local_cell_to_world(int(r), int(c))
                p = Point(); p.x = x; p.y = y; p.z = 0.04
                marker.points.append(p)
        else:
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.02

        self.active_hole_mask_markers_pub.publish(marker)

    def publish_local_repair_rcg_outputs(self, stamp) -> None:
        pose_msg = PoseArray()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.header.stamp = stamp
        for node in self.local_repair_nodes:
            pose = Pose()
            pose.position.x = float(node['x'])
            pose.position.y = float(node['y'])
            pose.position.z = 0.06
            pose.orientation.w = 1.0
            pose_msg.poses.append(pose)
        self.local_repair_rcg_nodes_pub.publish(pose_msg)

        ma = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        nodes_marker = Marker()
        nodes_marker.header.frame_id = self.map_frame
        nodes_marker.header.stamp = stamp
        nodes_marker.ns = 'local_repair_rcg_nodes'
        nodes_marker.id = 0
        nodes_marker.type = Marker.SPHERE_LIST
        nodes_marker.action = Marker.ADD
        nodes_marker.scale.x = 0.065
        nodes_marker.scale.y = 0.065
        nodes_marker.scale.z = 0.065
        nodes_marker.color.r = 1.0
        nodes_marker.color.g = 0.45
        nodes_marker.color.b = 0.0
        nodes_marker.color.a = 0.95
        nodes_marker.pose.orientation.w = 1.0

        edge_marker = Marker()
        edge_marker.header.frame_id = self.map_frame
        edge_marker.header.stamp = stamp
        edge_marker.ns = 'local_repair_rcg_edges'
        edge_marker.id = 1
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.025
        edge_marker.color.r = 1.0
        edge_marker.color.g = 0.65
        edge_marker.color.b = 0.0
        edge_marker.color.a = 0.70
        edge_marker.pose.orientation.w = 1.0

        for node in self.local_repair_nodes:
            p = Point(); p.x = float(node['x']); p.y = float(node['y']); p.z = 0.16
            nodes_marker.points.append(p)

        for i, j in sorted(self.local_repair_edges):
            if i < 0 or j < 0 or i >= len(self.local_repair_nodes) or j >= len(self.local_repair_nodes):
                continue
            ni = self.local_repair_nodes[i]
            nj = self.local_repair_nodes[j]
            p1 = Point(); p1.x = float(ni['x']); p1.y = float(ni['y']); p1.z = 0.12
            p2 = Point(); p2.x = float(nj['x']); p2.y = float(nj['y']); p2.z = 0.12
            edge_marker.points.append(p1); edge_marker.points.append(p2)

        ma.markers.append(nodes_marker)
        ma.markers.append(edge_marker)
        self.local_repair_rcg_markers_pub.publish(ma)



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
