#!/usr/bin/env python3
#配合cstar_rcgnodecopy的版本。
import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


NodeKey = Tuple[int, int]
GridCell = Tuple[int, int]  # row, col


class CStarWaypointPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_waypoint_planner_node')

        self.declare_parameter('rcg_nodes_topic', '/cstar/rcg_nodes')
        self.declare_parameter('rcg_markers_topic', '/cstar/rcg_markers')
        self.declare_parameter('covered_map_topic', '/cstar/covered_map')

        self.declare_parameter('free_map_topic', '/cstar/free_map')
        self.declare_parameter('obstacle_map_topic', '/cstar/obstacle_map')
        self.declare_parameter('unknown_map_topic', '/cstar/unknown_map')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        self.declare_parameter('update_period', 0.5)
        self.declare_parameter('position_quantization', 0.05)

        self.declare_parameter('snap_distance', 0.45)

        # 旧参数保留，方便兼容之前的启动命令，但现在 goal 到达判断不用它。
        self.declare_parameter('reached_distance', 0.18)

        # 新逻辑：必须等 base_footprint 底盘中心靠近 goal_marker 中心后才更新 goal。
        # 普通绿色 goal 建议 0.07~0.10；橙色 retreat node 建议 0.08~0.10。
        self.declare_parameter('goal_center_tolerance', 0.10)
        self.declare_parameter('retreat_center_tolerance', 0.10)

        # 这里不要太大。太大会把附近 open 采样点提前染成 closed。
        self.declare_parameter('closed_position_radius', 0.12)
        self.declare_parameter('covered_close_threshold', 50)
        self.declare_parameter('use_covered_map_for_closing', True)

        # Boustrophedon 普通覆盖选点参数
        # -1 表示初始先向左；1 表示初始先向右。
        self.declare_parameter('initial_sweep_direction', -1.0)
        self.declare_parameter('same_lap_y_tolerance', 0.14)
        self.declare_parameter('same_col_x_tolerance', 0.16)

        # 如果严格横向/纵向邻居都没有，是否允许用斜边兜底，避免误判 dead-end。
        # 想让路径更规整，可以启动时改成 false。
        self.declare_parameter('allow_diagonal_fallback', True)

        # retreat node 判定
        self.declare_parameter('retreat_attach_radius', 0.35)
        self.declare_parameter('allow_open_fallback', True)

        # grid A* 安全缓冲区，建议和 cstar_rcg_node.py 一致或略小一点
        self.declare_parameter('obstacle_buffer', 0.20)
        self.declare_parameter('unknown_buffer', 0.10)
        self.declare_parameter('map_border_buffer', 0.15)
        self.declare_parameter('nearest_safe_search_radius', 0.60)

        # A* path 重采样距离。越小，escape_path 越密，controller 越好跟。
        self.declare_parameter('escape_resample_step', 0.08)

        # ========== RCG-based coverage hole detection 可视化 ==========
        # 注意：这里只做检测和可视化，不接管小车运动，不影响当前 C* goal 选择。
        self.declare_parameter('enable_hole_detection', True)

        # 只在当前 current_key 到达并选出 next_goal_key 之后检测。
        # floodfill 从 current_key 的 Open 邻居开始，next_goal_key 作为边界，不进入。
        self.declare_parameter('hole_min_nodes', 4)
        self.declare_parameter('hole_max_nodes', 160)

        # 用 open component 的 bbox 面积过滤太小的边角碎片和太大的未探索区域。
        self.declare_parameter('hole_min_bbox_area', 0.10)
        self.declare_parameter('hole_max_bbox_area', 8.00)

        # hole 必须贴近 closed path / covered 区域，避免把远处普通 open 区域误判为 hole。
        self.declare_parameter('hole_min_closed_attach_nodes', 1)

        # unknown 不再一票否决。
        # 只有 unknown 面积和 unknown 节点比例同时超过阈值，才认为它是真正 frontier，不是 hole。
        self.declare_parameter('hole_unknown_check_radius', 0.25)
        self.declare_parameter('hole_unknown_reject_area', 0.30)
        self.declare_parameter('hole_unknown_reject_ratio', 0.35)

        # 与机器人距离太远的 component 不检测，避免远处房间提前触发。
        self.declare_parameter('hole_max_robot_distance', 1.80)

        # hole seed 搜索半径：
        # 原来只从 current_key 的直接 Open 邻居开始 floodfill；
        # 现在扩展为 current_key 附近一定半径内的 Open 节点，提高漏检 hole 的召回率。
        self.declare_parameter('hole_seed_search_radius', 0.55)

        # 局部 hole 检测门控：
        # 只允许检测当前 C* 正常覆盖路径附近的旁支 hole，
        # 避免把后续正常牛耕会覆盖到的区域或远处房间提前判成 hole。
        self.declare_parameter('hole_local_gate_enable', True)
        self.declare_parameter('hole_local_robot_distance', 1.60)
        self.declare_parameter('hole_local_segment_distance', 0.85)
        self.declare_parameter('hole_local_backward_extension', 0.25)
        self.declare_parameter('hole_local_forward_extension', 1.30)
        self.declare_parameter('hole_keep_nearest_components', 1)

        # escape 过程中不检测 hole，避免撤退路径和 hole 可视化混在一起。
        self.declare_parameter('hole_disable_during_escape', True)

        # 主干道保护过滤：
        # 当 next_goal 仍在当前 lap 上时，说明普通牛耕马上会继续覆盖当前主干道。
        # 这时 current lap 附近、当前运动方向上的 open component 不应被误判为 hole。
        self.declare_parameter('hole_main_lap_protect_enable', True)
        self.declare_parameter('hole_main_lap_width_factor', 1.5)
        self.declare_parameter('hole_main_lap_ratio_threshold', 0.45)
        self.declare_parameter('hole_ahead_ratio_threshold', 0.50)

        # 边界 lap 触发门控：
        # 只有当前小车所在的整条 lap 是贴近 buffer / 墙壁 / 障碍物的第一条 lap 时，
        # 才允许进行 hole detection 和 repair path 生成。
        # 这样可以避免在中间 lap 上过早把后续正常 C* 会覆盖的区域误判为 hole。
        self.declare_parameter('hole_enable_boundary_lap_gate', True)
        self.declare_parameter('hole_boundary_probe_distance', 0.35)
        self.declare_parameter('hole_boundary_min_ratio', 0.35)
        self.declare_parameter('hole_boundary_ignore_endpoint_count', 1)
        self.declare_parameter('hole_boundary_lap_y_tolerance_factor', 1.25)

        # entry / exit 固定在 boundary lap 上，而不是选在 hole 内部。
        # 这样 repair path 会从正常 C* 主扫描线进入 hole，再从同一条主扫描线退出。
        self.declare_parameter('hole_entry_exit_on_boundary_lap', True)
        self.declare_parameter('hole_boundary_entry_s_margin', 0.25)
        self.declare_parameter('hole_boundary_entry_search_distance', 0.55)

        # branch-seed hole predictor:
        # 不再使用 doorway gap。只在 boundary_lap 上寻找“向 hole 内部延伸的侧向 RCG 分支边”。
        # boundary_key 可以是 closed sample；seed_key 必须是分支边另一端的 open sample。
        # 本验证版只使用 branch seed，不使用普通 seed 兜底。
        self.declare_parameter('hole_enable_branch_seed_predictor', True)
        self.declare_parameter('hole_branch_lookahead_distance', 1.20)
        self.declare_parameter('hole_branch_no_backtrack_margin', 0.08)
        self.declare_parameter('hole_branch_lateral_min_distance', 0.18)
        self.declare_parameter('hole_branch_lateral_ratio', 1.40)
        self.declare_parameter('hole_branch_max_along_offset', 0.25)
        self.declare_parameter('hole_branch_max_seed_distance', 0.75)
        self.declare_parameter('hole_branch_max_candidates', 8)


        # ========== dynamic hole entry/exit + orthogonal repair 可视化 ==========
        # 不固定 hole_area；只在当前活跃 hole 上锁定 entry/exit。
        # hole 内部重采样和 repair path 会随着地图更新动态刷新。
        self.declare_parameter('hole_incomplete_unknown_area', 0.06)
        self.declare_parameter('hole_incomplete_unknown_ratio', 0.12)
        self.declare_parameter('hole_entry_exit_lock_enable', True)
        self.declare_parameter('hole_active_match_distance', 0.80)
        self.declare_parameter('hole_entry_reselect_distance', 0.45)
        self.declare_parameter('hole_entry_sample_skip_radius', 0.10)
        self.declare_parameter('hole_dynamic_update_during_motion', True)
        self.declare_parameter('hole_dynamic_update_period', 0.5)

        # repair entry/exit 选择：
        # entry/exit 优先选在靠近当前 C* 主通道的 doorway band 上，
        # 再沿当前运动方向选前后两个端点，避免 entry 和 exit 重合。
        self.declare_parameter('hole_doorway_band_width', 0.35)
        self.declare_parameter('hole_min_entry_exit_distance', 0.35)

        # repair 路径排序：
        # 不再使用普通最近邻造成杂乱跨越，而是使用局部牛耕式排序。

        # ========== hole 内部局部加密采样可视化 ==========
        # 这里只生成 /cstar/hole_samples 和 /cstar/hole_sample_markers，
        # 不接管小车运动，也不修改当前 C* goal。
        self.declare_parameter('enable_hole_sampling', True)
        self.declare_parameter('hole_sample_lap_spacing', 0.16)
        self.declare_parameter('hole_sample_spacing', 0.16)
        self.declare_parameter('hole_sample_margin', 0.15)
        self.declare_parameter('hole_sample_obstacle_buffer', 0.12)
        self.declare_parameter('hole_sample_unknown_buffer', 0.05)
        self.declare_parameter('hole_sample_exclude_covered', True)
        self.declare_parameter('hole_sample_max_points_per_hole', 220)


        # ========== doorway-constrained orthogonal hole repair path ==========
        # 这部分替代普通 repair：检测到 hole 后，在 hole 内部按垂直于当前 C* 主方向的
        # lap 进行重采样，生成类似论文图(c)(d)的局部牛耕补扫路径。
        self.declare_parameter('enable_hole_repair_path', True)
        self.declare_parameter('hole_repair_lap_spacing', 0.25)
        self.declare_parameter('hole_repair_sample_spacing', 0.12)
        self.declare_parameter('hole_repair_min_lap_length', 0.35)
        self.declare_parameter('hole_repair_region_margin', 0.18)
        self.declare_parameter('hole_repair_force_even_laps', True)
        self.declare_parameter('hole_repair_max_path_points', 360)
        self.declare_parameter('hole_repair_max_connector_length', 1.60)
        self.declare_parameter('hole_repair_astar_max_expansions', 12000)

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
        self.reached_distance = float(self.get_parameter('reached_distance').value)

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

        self.retreat_attach_radius = float(self.get_parameter('retreat_attach_radius').value)
        self.allow_open_fallback = bool(self.get_parameter('allow_open_fallback').value)

        self.obstacle_buffer = float(self.get_parameter('obstacle_buffer').value)
        self.unknown_buffer = float(self.get_parameter('unknown_buffer').value)
        self.map_border_buffer = float(self.get_parameter('map_border_buffer').value)
        self.nearest_safe_search_radius = float(self.get_parameter('nearest_safe_search_radius').value)
        self.escape_resample_step = float(self.get_parameter('escape_resample_step').value)

        self.enable_hole_detection = bool(self.get_parameter('enable_hole_detection').value)
        self.hole_min_nodes = int(self.get_parameter('hole_min_nodes').value)
        self.hole_max_nodes = int(self.get_parameter('hole_max_nodes').value)
        self.hole_min_bbox_area = float(self.get_parameter('hole_min_bbox_area').value)
        self.hole_max_bbox_area = float(self.get_parameter('hole_max_bbox_area').value)
        self.hole_min_closed_attach_nodes = int(self.get_parameter('hole_min_closed_attach_nodes').value)
        self.hole_unknown_check_radius = float(self.get_parameter('hole_unknown_check_radius').value)
        self.hole_unknown_reject_area = float(self.get_parameter('hole_unknown_reject_area').value)
        self.hole_unknown_reject_ratio = float(self.get_parameter('hole_unknown_reject_ratio').value)
        self.hole_max_robot_distance = float(self.get_parameter('hole_max_robot_distance').value)
        
        self.hole_seed_search_radius = float(self.get_parameter('hole_seed_search_radius').value)

        self.hole_local_gate_enable = bool(
            self.get_parameter('hole_local_gate_enable').value
        )
        self.hole_local_robot_distance = float(
            self.get_parameter('hole_local_robot_distance').value
        )
        self.hole_local_segment_distance = float(
            self.get_parameter('hole_local_segment_distance').value
        )
        self.hole_local_backward_extension = float(
            self.get_parameter('hole_local_backward_extension').value
        )
        self.hole_local_forward_extension = float(
            self.get_parameter('hole_local_forward_extension').value
        )
        self.hole_keep_nearest_components = int(
            self.get_parameter('hole_keep_nearest_components').value
        )

        self.hole_disable_during_escape = bool(self.get_parameter('hole_disable_during_escape').value)

        self.hole_main_lap_protect_enable = bool(
            self.get_parameter('hole_main_lap_protect_enable').value
        )
        self.hole_main_lap_width_factor = float(
            self.get_parameter('hole_main_lap_width_factor').value
        )
        self.hole_main_lap_ratio_threshold = float(
            self.get_parameter('hole_main_lap_ratio_threshold').value
        )
        self.hole_ahead_ratio_threshold = float(
            self.get_parameter('hole_ahead_ratio_threshold').value
        )

        self.hole_enable_boundary_lap_gate = bool(
            self.get_parameter('hole_enable_boundary_lap_gate').value
        )
        self.hole_boundary_probe_distance = float(
            self.get_parameter('hole_boundary_probe_distance').value
        )
        self.hole_boundary_min_ratio = float(
            self.get_parameter('hole_boundary_min_ratio').value
        )
        self.hole_boundary_ignore_endpoint_count = int(
            self.get_parameter('hole_boundary_ignore_endpoint_count').value
        )
        self.hole_boundary_lap_y_tolerance_factor = float(
            self.get_parameter('hole_boundary_lap_y_tolerance_factor').value
        )
        self.hole_entry_exit_on_boundary_lap = bool(
            self.get_parameter('hole_entry_exit_on_boundary_lap').value
        )
        self.hole_boundary_entry_s_margin = float(
            self.get_parameter('hole_boundary_entry_s_margin').value
        )
        self.hole_boundary_entry_search_distance = float(
            self.get_parameter('hole_boundary_entry_search_distance').value
        )

        self.hole_enable_branch_seed_predictor = bool(
            self.get_parameter('hole_enable_branch_seed_predictor').value
        )
        self.hole_branch_lookahead_distance = float(
            self.get_parameter('hole_branch_lookahead_distance').value
        )
        self.hole_branch_no_backtrack_margin = float(
            self.get_parameter('hole_branch_no_backtrack_margin').value
        )
        self.hole_branch_lateral_min_distance = float(
            self.get_parameter('hole_branch_lateral_min_distance').value
        )
        self.hole_branch_lateral_ratio = float(
            self.get_parameter('hole_branch_lateral_ratio').value
        )
        self.hole_branch_max_along_offset = float(
            self.get_parameter('hole_branch_max_along_offset').value
        )
        self.hole_branch_max_seed_distance = float(
            self.get_parameter('hole_branch_max_seed_distance').value
        )
        self.hole_branch_max_candidates = int(
            self.get_parameter('hole_branch_max_candidates').value
        )


        self.hole_incomplete_unknown_area = float(
            self.get_parameter('hole_incomplete_unknown_area').value
        )
        self.hole_incomplete_unknown_ratio = float(
            self.get_parameter('hole_incomplete_unknown_ratio').value
        )
        self.hole_entry_exit_lock_enable = bool(
            self.get_parameter('hole_entry_exit_lock_enable').value
        )
        self.hole_active_match_distance = float(
            self.get_parameter('hole_active_match_distance').value
        )
        self.hole_entry_reselect_distance = float(
            self.get_parameter('hole_entry_reselect_distance').value
        )
        self.hole_entry_sample_skip_radius = float(
            self.get_parameter('hole_entry_sample_skip_radius').value
        )
        self.hole_dynamic_update_during_motion = bool(
            self.get_parameter('hole_dynamic_update_during_motion').value
        )
        self.hole_dynamic_update_period = float(
            self.get_parameter('hole_dynamic_update_period').value
        )
        self.hole_doorway_band_width = float(
            self.get_parameter('hole_doorway_band_width').value
        )
        self.hole_min_entry_exit_distance = float(
            self.get_parameter('hole_min_entry_exit_distance').value
        )

        self.enable_hole_sampling = bool(self.get_parameter('enable_hole_sampling').value)
        self.hole_sample_lap_spacing = float(self.get_parameter('hole_sample_lap_spacing').value)
        self.hole_sample_spacing = float(self.get_parameter('hole_sample_spacing').value)
        self.hole_sample_margin = float(self.get_parameter('hole_sample_margin').value)
        self.hole_sample_obstacle_buffer = float(self.get_parameter('hole_sample_obstacle_buffer').value)
        self.hole_sample_unknown_buffer = float(self.get_parameter('hole_sample_unknown_buffer').value)
        self.hole_sample_exclude_covered = bool(self.get_parameter('hole_sample_exclude_covered').value)
        self.hole_sample_max_points_per_hole = int(self.get_parameter('hole_sample_max_points_per_hole').value)


        self.enable_hole_repair_path = bool(self.get_parameter('enable_hole_repair_path').value)
        self.hole_repair_lap_spacing = float(self.get_parameter('hole_repair_lap_spacing').value)
        self.hole_repair_sample_spacing = float(self.get_parameter('hole_repair_sample_spacing').value)
        self.hole_repair_min_lap_length = float(self.get_parameter('hole_repair_min_lap_length').value)
        self.hole_repair_region_margin = float(self.get_parameter('hole_repair_region_margin').value)
        self.hole_repair_force_even_laps = bool(self.get_parameter('hole_repair_force_even_laps').value)
        self.hole_repair_max_path_points = int(self.get_parameter('hole_repair_max_path_points').value)
        self.hole_repair_max_connector_length = float(self.get_parameter('hole_repair_max_connector_length').value)
        self.hole_repair_astar_max_expansions = int(self.get_parameter('hole_repair_astar_max_expansions').value)

        self.nodes: Dict[NodeKey, Tuple[float, float]] = {}
        self.raw_edges: List[Tuple[NodeKey, NodeKey]] = []
        self.adjacency: Dict[NodeKey, Set[NodeKey]] = {}

        self.closed_nodes: Set[NodeKey] = set()
        self.closed_positions: List[Tuple[float, float]] = []

        self.current_goal_key: Optional[NodeKey] = None
        self.selected_path: List[Tuple[float, float]] = []

        self.escape_active = False
        self.escape_path_xy: List[Tuple[float, float]] = []
        self.last_deadend_key: Optional[NodeKey] = None
        self.latest_retreat_candidates: Set[NodeKey] = set()

        # hole 可视化状态。这里只记录，不影响 C* 控制。
        self.latest_hole_components: List[Set[NodeKey]] = []
        self.latest_hole_infos: List[Tuple[float, float, int, float, float, float, str]] = []
        # info: center_x, center_y, node_count, bbox_area, unknown_area, unknown_ratio, label

        # hole 内部局部加密采样结果，只用于可视化和后续 repair 输入。
        self.latest_hole_samples: List[Tuple[float, float]] = []

        # dynamic hole repair 可视化状态。只固定当前活跃 hole 的 entry/exit，
        # hole_nodes / hole_samples / repair_path 会动态更新。
        self.active_hole_center: Optional[Tuple[float, float]] = None
        self.active_hole_entry: Optional[Tuple[float, float]] = None
        self.active_hole_exit: Optional[Tuple[float, float]] = None
        self.active_hole_entry_key: Optional[NodeKey] = None
        self.active_hole_exit_key: Optional[NodeKey] = None
        # branch seed 当前帧给出的 entry/exit。
        # entry 优先使用侧向 RCG 分支的 seed；exit 固定在 boundary_lap 上。
        # 同一个 active hole 后续动态更新时仍保持锁定，不让 entry/exit 漂移。
        self.pending_hole_entry_key: Optional[NodeKey] = None
        self.pending_hole_exit_key: Optional[NodeKey] = None

        # branch trigger 预判状态：
        # 在当前 sample 选择 next_goal 时，如果 next_goal 本身就是 boundary_lap 上
        # 带侧向 RCG 分支边的 boundary_key，就先把 seed 存起来；
        # 等小车真正到达这个 next_goal 后，再从该 seed 立即 floodfill/repair。
        self.pending_branch_trigger_goal_key: Optional[NodeKey] = None
        self.pending_branch_trigger_seed_key: Optional[NodeKey] = None
        self.pending_branch_trigger_boundary_key: Optional[NodeKey] = None
        self.pending_branch_trigger_score: float = 0.0

        self.latest_hole_repair_path: List[Tuple[float, float]] = []
        self.hole_detection_armed = False
        self.last_hole_dynamic_update_time = None

        self.covered_map: Optional[OccupancyGrid] = None
        self.covered_data: Optional[List[int]] = None

        self.free_msg: Optional[OccupancyGrid] = None
        self.obstacle_msg: Optional[OccupancyGrid] = None
        self.unknown_msg: Optional[OccupancyGrid] = None

        self.free_arr: Optional[np.ndarray] = None
        self.obstacle_arr: Optional[np.ndarray] = None
        self.unknown_arr: Optional[np.ndarray] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseArray, self.rcg_nodes_topic, self.rcg_callback, 10)
        self.create_subscription(MarkerArray, self.rcg_markers_topic, self.rcg_markers_callback, 10)
        self.create_subscription(OccupancyGrid, self.covered_map_topic, self.covered_callback, 10)

        self.create_subscription(OccupancyGrid, self.free_map_topic, self.free_callback, 10)
        self.create_subscription(OccupancyGrid, self.obstacle_map_topic, self.obstacle_callback, 10)
        self.create_subscription(OccupancyGrid, self.unknown_map_topic, self.unknown_callback, 10)

        self.goal_pub = self.create_publisher(PoseStamped, '/cstar/goal', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/cstar/goal_marker', 10)
        self.state_marker_pub = self.create_publisher(MarkerArray, '/cstar/open_closed_markers', 10)
        self.path_pub = self.create_publisher(Path, '/cstar/selected_path', 10)
        self.escape_path_pub = self.create_publisher(Path, '/cstar/escape_path', 10)
        self.retreat_marker_pub = self.create_publisher(Marker, '/cstar/retreat_nodes', 10)

        # 只保留这两个 hole topic，旧的 /cstar/graph_hole_* 全部删除。
        self.hole_nodes_pub = self.create_publisher(PoseArray, '/cstar/hole_nodes', 10)
        self.hole_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_markers', 10)

        # hole 内部加密采样可视化。下一步 repair 会直接使用 /cstar/hole_samples。
        self.hole_samples_pub = self.create_publisher(PoseArray, '/cstar/hole_samples', 10)
        self.hole_sample_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_sample_markers', 10)

        # dynamic hole entry/exit + repair 可视化，不接管控制。
        self.hole_entry_marker_pub = self.create_publisher(MarkerArray, '/cstar/hole_entry_marker', 10)
        self.hole_exit_marker_pub = self.create_publisher(MarkerArray, '/cstar/hole_exit_marker', 10)
        self.hole_repair_path_pub = self.create_publisher(Path, '/cstar/hole_repair_path', 10)
        self.hole_repair_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_repair_markers', 10)

        self.timer = self.create_timer(self.update_period, self.on_timer)

        self.get_logger().info('CStarWaypointPlannerNode started.')
        self.get_logger().info(f'rcg_nodes_topic={self.rcg_nodes_topic}')
        self.get_logger().info(f'rcg_markers_topic={self.rcg_markers_topic}')
        self.get_logger().info(f'covered_map_topic={self.covered_map_topic}')
        self.get_logger().info(f'free_map_topic={self.free_map_topic}')
        self.get_logger().info(
            f'sweep_dir={self.sweep_dir}, '
            f'same_lap_y_tolerance={self.same_lap_y_tolerance:.2f}, '
            f'same_col_x_tolerance={self.same_col_x_tolerance:.2f}, '
            f'allow_diagonal_fallback={self.allow_diagonal_fallback}'
        )
        self.get_logger().info(
            f'goal_center_tolerance={self.goal_center_tolerance:.2f}, '
            f'retreat_center_tolerance={self.retreat_center_tolerance:.2f}'
        )
        self.get_logger().info(
            f'hole_detection={self.enable_hole_detection}, '
            f'unknown_reject_area={self.hole_unknown_reject_area:.2f}, '
            f'unknown_reject_ratio={self.hole_unknown_reject_ratio:.2f}, '
            f'unknown_check_radius={self.hole_unknown_check_radius:.2f}'
        )
        self.get_logger().info(
            f'hole_main_lap_protect={self.hole_main_lap_protect_enable}, '
            f'width_factor={self.hole_main_lap_width_factor:.2f}, '
            f'lap_ratio={self.hole_main_lap_ratio_threshold:.2f}, '
            f'ahead_ratio={self.hole_ahead_ratio_threshold:.2f}'
        )
        self.get_logger().info(
            f'hole_boundary_lap_gate={self.hole_enable_boundary_lap_gate}, '
            f'probe_dist={self.hole_boundary_probe_distance:.2f}, '
            f'min_ratio={self.hole_boundary_min_ratio:.2f}, '
            f'ignore_endpoints={self.hole_boundary_ignore_endpoint_count}'
        )
        self.get_logger().info(
            f'hole_entry_exit_on_boundary_lap={self.hole_entry_exit_on_boundary_lap}, '
            f's_margin={self.hole_boundary_entry_s_margin:.2f}, '
            f'search_distance={self.hole_boundary_entry_search_distance:.2f}'
        )
        self.get_logger().info(
            f'hole_branch_seed_predictor={self.hole_enable_branch_seed_predictor}, '
            f'lookahead={self.hole_branch_lookahead_distance:.2f}, '
            f'lateral_min={self.hole_branch_lateral_min_distance:.2f}, '
            f'lateral_ratio={self.hole_branch_lateral_ratio:.2f}'
        )
        self.get_logger().info(
            f'hole_local_gate={self.hole_local_gate_enable}, '
            f'robot_dist={self.hole_local_robot_distance:.2f}, '
            f'seg_dist={self.hole_local_segment_distance:.2f}, '
            f'keep_nearest={self.hole_keep_nearest_components}'
        )
        self.get_logger().info(
            f'hole_sampling={self.enable_hole_sampling}, '
            f'lap_spacing={self.hole_sample_lap_spacing:.2f}, '
            f'sample_spacing={self.hole_sample_spacing:.2f}, '
            f'margin={self.hole_sample_margin:.2f}'
        )
        self.get_logger().info(
            f'hole_repair_path={self.enable_hole_repair_path}, '
            f'repair_lap_spacing={self.hole_repair_lap_spacing:.2f}, '
            f'repair_sample_spacing={self.hole_repair_sample_spacing:.2f}, '
            f'force_even_laps={self.hole_repair_force_even_laps}'
        )

    def make_key(self, x: float, y: float) -> NodeKey:
        q = self.position_quantization
        return int(round(x / q)), int(round(y / q))

    def rcg_callback(self, msg: PoseArray) -> None:
        new_nodes: Dict[NodeKey, Tuple[float, float]] = {}

        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            key = self.make_key(x, y)
            new_nodes[key] = (x, y)

        self.nodes = new_nodes
        self.rebuild_adjacency()

        if self.current_goal_key is not None and self.current_goal_key not in self.nodes:
            self.current_goal_key = None
            self.escape_active = False
            self.escape_path_xy.clear()

    def rcg_markers_callback(self, msg: MarkerArray) -> None:
        raw_edges: List[Tuple[NodeKey, NodeKey]] = []

        for marker in msg.markers:
            if marker.ns != 'rcg_edges':
                continue

            points = marker.points
            if len(points) < 2:
                continue

            for i in range(0, len(points) - 1, 2):
                p1 = points[i]
                p2 = points[i + 1]

                k1 = self.make_key(p1.x, p1.y)
                k2 = self.make_key(p2.x, p2.y)

                if k1 != k2:
                    raw_edges.append((k1, k2))

        self.raw_edges = raw_edges
        self.rebuild_adjacency()

    def rebuild_adjacency(self) -> None:
        adjacency: Dict[NodeKey, Set[NodeKey]] = {key: set() for key in self.nodes.keys()}

        for k1, k2 in self.raw_edges:
            if k1 not in self.nodes or k2 not in self.nodes:
                continue

            adjacency[k1].add(k2)
            adjacency[k2].add(k1)

        self.adjacency = adjacency

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

    def get_robot_pose(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.1)
            )
            return tf.transform.translation.x, tf.transform.translation.y
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def world_to_cell(self, x: float, y: float) -> Optional[GridCell]:
        if self.free_msg is None:
            return None

        info = self.free_msg.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)

        if row < 0 or col < 0 or row >= info.height or col >= info.width:
            return None

        return row, col

    def cell_to_world(self, cell: GridCell) -> Tuple[float, float]:
        assert self.free_msg is not None

        row, col = cell
        info = self.free_msg.info

        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def world_to_covered_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if self.covered_map is None:
            return None

        info = self.covered_map.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None

        return mx, my

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

        if self.is_inside_covered_map(x, y):
            return True

        if self.is_near_closed_position(x, y):
            return True

        return False

    def close_key(self, key: NodeKey) -> None:
        if key not in self.nodes:
            return

        self.closed_nodes.add(key)

        x, y = self.nodes[key]
        self.add_closed_position(x, y)

    def nearest_node_key(self, x: float, y: float) -> Optional[NodeKey]:
        if not self.nodes:
            return None

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
        """
        新到达判断：
        只看 base_footprint 底盘中心到当前 goal 节点中心的距离。
        """
        if self.current_goal_key is None:
            return True

        if self.current_goal_key not in self.nodes:
            return True

        gx, gy = self.nodes[self.current_goal_key]

        if self.escape_active:
            tolerance = self.retreat_center_tolerance
        else:
            tolerance = self.goal_center_tolerance

        return math.hypot(robot_xy[0] - gx, robot_xy[1] - gy) <= tolerance

    def has_any_open_neighbor(self, current_key: NodeKey) -> bool:
        for nb in self.adjacency.get(current_key, set()):
            if nb in self.nodes and not self.is_closed_key(nb):
                return True
        return False

    def choose_diagonal_fallback(self, current_key: NodeKey) -> Optional[NodeKey]:
        if current_key not in self.nodes:
            return None

        candidates: List[Tuple[float, float, NodeKey]] = []
        cx, cy = self.nodes[current_key]

        for nb in self.adjacency.get(current_key, set()):
            if nb not in self.nodes:
                continue

            if self.is_closed_key(nb):
                continue

            x, y = self.nodes[nb]
            dx = abs(x - cx)
            dy = abs(y - cy)
            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                continue

            # 越接近横/竖，ratio 越小；越接近 45 度，ratio 越大。
            ratio = min(dx, dy) / max(dx, dy, 1e-6)
            candidates.append((ratio, dist, nb))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]
    
    def classify_open_neighbors(self, current_key: NodeKey) -> Dict[str, List[Tuple[float, NodeKey]]]:
        """
        简洁稳定版候选分类：

        1. same_forward / same_backward：
           当前 lap 上的 Open 邻居，优先用于把当前 lap 跑完。

        2. left / up / down / right：
           当前 lap 没有可走 Open 节点后，再按“左 -> 上 -> 下 -> 右”
           选择其他方向的 Open 邻居。

        这样既保留牛耕的连续性，又避免刚换 lap 后左右都有点时直接乱跳。
        """
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

        for nb in self.adjacency.get(current_key, set()):
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

            # 当前 lap 上的横向邻居：第一优先级，保证先跑完当前 lap。
            if abs(dy) <= self.same_lap_y_tolerance:
                if dx * self.sweep_dir > 0.0:
                    result['same_forward'].append((abs(dx), nb))
                else:
                    result['same_backward'].append((abs(dx), nb))
                continue

            # 当前 lap 跑完后，再按地图坐标方向分类。
            # left/right 看 x 方向，up/down 看 y 方向。
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

        for direction in result:
            result[direction].sort(key=lambda item: item[0])

        return result

    def choose_next_normal_goal(self, current_key: NodeKey) -> Optional[NodeKey]:
        """
        普通 C* goal 选择逻辑：

        1. 优先跑完当前 lap：same_forward -> same_backward；
        2. 当前 lap 没有 Open 节点后，再按 left -> up -> down -> right；
        3. 最后才允许 diagonal fallback。
        """
        candidates = self.classify_open_neighbors(current_key)

        # 1. 优先沿当前 sweep_dir 跑完当前 lap。
        if candidates['same_forward']:
            return candidates['same_forward'][0][1]

        # 2. 如果当前方向没有，但当前 lap 反方向还有 Open 点，也先补完当前 lap。
        if candidates['same_backward']:
            next_key = candidates['same_backward'][0][1]

            if current_key in self.nodes and next_key in self.nodes:
                cx, _ = self.nodes[current_key]
                nx, _ = self.nodes[next_key]
                self.sweep_dir = -1.0 if (nx - cx) < 0.0 else 1.0

            return next_key

        # 3. 当前 lap 跑完后，再按论文式顺序选择其他方向。
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

        # 4. 最后才允许斜边兜底，避免误判 dead-end。
        if self.allow_diagonal_fallback and candidates['diagonal']:
            return self.choose_diagonal_fallback(current_key)

        return None

    def is_retreat_candidate(self, key: NodeKey, start_key: NodeKey) -> bool:
        if key == start_key:
            return False

        if key not in self.nodes:
            return False

        if self.is_closed_key(key):
            return False

        # retreat node 是从已覆盖轨迹重新进入 open 区域的入口。
        for nb in self.adjacency.get(key, set()):
            if nb in self.nodes and self.is_closed_key(nb):
                return True

        x, y = self.nodes[key]
        if self.is_near_closed_position(x, y, radius=self.retreat_attach_radius):
            return True

        return False

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

        if best == float('inf'):
            return 0.0

        return best

    def reconstruct_grid_path(
        self,
        prev: Dict[GridCell, Optional[GridCell]],
        target: GridCell
    ) -> List[GridCell]:
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
        safe: np.ndarray
    ) -> Tuple[List[GridCell], Optional[NodeKey]]:
        goals = set(goal_to_key.keys())

        if not goals:
            return [], None

        h, w = safe.shape

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        open_heap: List[Tuple[float, float, GridCell]] = []
        g_score: Dict[GridCell, float] = {start: 0.0}
        prev: Dict[GridCell, Optional[GridCell]] = {start: None}
        visited: Set[GridCell] = set()

        h0 = self.heuristic_to_goal_cells(start, goals)
        heapq.heappush(open_heap, (h0, 0.0, start))

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

                # 禁止斜向切角
                if dr != 0 and dc != 0:
                    if not safe[r + dr, c]:
                        continue
                    if not safe[r, c + dc]:
                        continue

                nb = (nr, nc)
                tentative_g = current_g + move_cost

                if tentative_g >= g_score.get(nb, float('inf')):
                    continue

                g_score[nb] = tentative_g
                prev[nb] = current

                h_cost = self.heuristic_to_goal_cells(nb, goals)
                f = tentative_g + h_cost

                heapq.heappush(open_heap, (f, tentative_g, nb))

        return [], None

    def find_grid_escape_path_to_retreat(
        self,
        current_key: NodeKey,
        robot_xy: Tuple[float, float]
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

            # 把真正的 retreat_node 坐标追加到 escape path 末尾，
            # 避免 A* 最后一个 safe cell 和黄色 retreat_node 不重合。
            if target_key in self.nodes:
                tx, ty = self.nodes[target_key]
                if not xy_path or math.hypot(xy_path[-1][0] - tx, xy_path[-1][1] - ty) > 0.03:
                    xy_path.append((tx, ty))

            xy_path = self.densify_xy_path(xy_path)
            return target_key, xy_path, False

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

            xy_path = self.densify_xy_path(xy_path)

            self.get_logger().warn(
                'No strict retreat node found by grid A*. Fallback to reachable Open node.'
            )
            return target_key, xy_path, True

        return None, [], False

    # ==========================
    # RCG-based hole detection
    # ==========================



    def collect_branch_hole_seed_pairs(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> List[Tuple[NodeKey, NodeKey, float]]:
        """
        在当前 boundary_lap 前方寻找“侧向 RCG 分支边”。

        返回：
            [(seed_key, boundary_key, score), ...]

        boundary_key:
            位于当前 boundary_lap 上的采样点，可以是 closed sample。
            因为小车出生点不确定，boundary_lap 上已经被覆盖/染红的点仍可能是有效门口点。

        seed_key:
            boundary_key 的邻接点，必须是 open sample；
            它是向 hole 内部延伸的分支边另一端，用作 floodfill seed。

        这版特意不加入普通 seed 兜底，方便单独验证“侧向 RCG 分支边”
        是否足以提前发现 hole。
        """
        if not self.hole_enable_branch_seed_predictor:
            return []

        if current_key not in self.nodes:
            return []

        if next_goal_key is None or next_goal_key not in self.nodes:
            return []

        if not self.is_next_goal_on_same_lap(current_key, next_goal_key):
            return []

        basis = self.get_current_lap_motion_basis(current_key, next_goal_key)
        if basis is None:
            return []

        u, n1, _, origin = basis
        vx, vy = n1

        lap_nodes = self.collect_same_lap_segment_nodes(current_key)
        if not lap_nodes:
            return []

        robot_s = (robot_xy[0] - origin[0]) * u[0] + (robot_xy[1] - origin[1]) * u[1]

        no_back = max(0.0, self.hole_branch_no_backtrack_margin)
        lookahead = max(0.05, self.hole_branch_lookahead_distance)
        min_lat = max(0.02, self.hole_branch_lateral_min_distance)
        ratio = max(1.0, self.hole_branch_lateral_ratio)
        max_along = max(0.02, self.hole_branch_max_along_offset)
        max_dist = max(min_lat, self.hole_branch_max_seed_distance)

        candidates: List[Tuple[float, NodeKey, NodeKey]] = []

        for boundary_key in lap_nodes:
            if boundary_key not in self.nodes:
                continue

            bx, by = self.nodes[boundary_key]
            boundary_s = (bx - origin[0]) * u[0] + (by - origin[1]) * u[1]

            # 只看机器人当前前方一段 boundary_lap；允许极小 backtrack 容差。
            if boundary_s < robot_s - no_back:
                continue

            if boundary_s > robot_s + lookahead:
                continue

            # boundary_key 可以 closed，但 seed_key 必须 open。
            for seed_key in self.adjacency.get(boundary_key, set()):
                if seed_key not in self.nodes:
                    continue

                if seed_key == current_key or seed_key == next_goal_key:
                    continue

                if self.is_closed_key(seed_key):
                    continue

                sx, sy = self.nodes[seed_key]
                dx = sx - bx
                dy = sy - by
                edge_dist = math.hypot(dx, dy)

                if edge_dist < 1e-6 or edge_dist > max_dist:
                    continue

                edge_along = dx * u[0] + dy * u[1]
                edge_lateral_signed = dx * vx + dy * vy
                edge_lateral = abs(edge_lateral_signed)

                # 不是同一条 lap 上的普通左右边，而是明显向侧方伸出的分支边。
                if edge_lateral < min_lat:
                    continue

                if edge_lateral < ratio * abs(edge_along):
                    continue

                if abs(edge_along) > max_along:
                    continue

                # seed 应该确实离开当前 boundary_lap，而不是 y 方向抖动造成的假侧边。
                _, seed_t = self.project_in_sweep_frame(sx, sy, origin, u, (vx, vy))
                if abs(seed_t) < min_lat:
                    continue

                score = max(0.0, boundary_s - robot_s) + 0.15 * edge_dist
                candidates.append((score, seed_key, boundary_key))

        candidates.sort(key=lambda item: item[0])

        out: List[Tuple[NodeKey, NodeKey, float]] = []
        used_seeds: Set[NodeKey] = set()
        used_edges: Set[Tuple[NodeKey, NodeKey]] = set()

        for score, seed_key, boundary_key in candidates:
            if seed_key in used_seeds:
                continue

            edge_id = (seed_key, boundary_key)
            if edge_id in used_edges:
                continue

            used_seeds.add(seed_key)
            used_edges.add(edge_id)
            out.append((seed_key, boundary_key, score))

            if len(out) >= max(1, self.hole_branch_max_candidates):
                break

        return out

    def select_branch_exit_key_for_component(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
        boundary_key: Optional[NodeKey]
    ) -> Optional[NodeKey]:
        """
        为 branch-seed hole 选择 exit。

        约束：
        - exit 必须在当前 boundary_lap 上；
        - exit 尽量位于当前 C* 前进方向上、hole component 投影范围的后端；
        - boundary_key 可以是 closed sample，因此 exit 也允许是 closed sample；
        - 如果找不到足够好的后端点，则退回到 boundary_key，保证先能可视化验证。
        """
        if not comp or current_key not in self.nodes:
            return None

        if next_goal_key is None or next_goal_key not in self.nodes:
            return None

        basis = self.get_current_lap_motion_basis(current_key, next_goal_key)
        if basis is None:
            return None

        u, n1, _, origin = basis
        v = n1

        lap_nodes = self.collect_same_lap_segment_nodes(current_key)
        if not lap_nodes:
            return boundary_key if boundary_key in self.nodes else None

        comp_s: List[float] = []
        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            s_coord, _ = self.project_in_sweep_frame(x, y, origin, u, v)
            comp_s.append(s_coord)

        if not comp_s:
            return boundary_key if boundary_key in self.nodes else None

        min_s = min(comp_s)
        max_s = max(comp_s)

        boundary_s = None
        if boundary_key in self.nodes:
            boundary_s, _ = self.project_in_sweep_frame(
                self.nodes[boundary_key][0],
                self.nodes[boundary_key][1],
                origin,
                u,
                v
            )

        robot_s = (robot_xy[0] - origin[0]) * u[0] + (robot_xy[1] - origin[1]) * u[1]
        lower_s = max(robot_s - self.hole_branch_no_backtrack_margin, min_s - self.hole_boundary_entry_s_margin)
        if boundary_s is not None:
            lower_s = max(lower_s, boundary_s - 0.05)

        target_s = max_s + self.hole_boundary_entry_s_margin
        max_dist_to_hole = max(0.10, self.hole_boundary_entry_search_distance)
        min_sep = max(0.05, self.hole_min_entry_exit_distance)

        candidates: List[Tuple[float, float, float, NodeKey]] = []

        bx = by = None
        if boundary_key in self.nodes:
            bx, by = self.nodes[boundary_key]

        for key in lap_nodes:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            s_coord, t_coord = self.project_in_sweep_frame(x, y, origin, u, v)

            if s_coord < lower_s:
                continue

            if abs(t_coord) > max(self.same_lap_y_tolerance * 2.0, self.hole_doorway_band_width):
                continue

            dist_to_hole = self.min_distance_from_node_to_component(key, comp)
            in_projection_window = (s_coord >= min_s - self.hole_boundary_entry_s_margin and
                                    s_coord <= max_s + self.hole_boundary_entry_s_margin)
            near_hole = dist_to_hole <= max_dist_to_hole

            if not in_projection_window and not near_hole:
                continue

            if bx is not None and by is not None:
                sep = math.hypot(x - bx, y - by)
                if sep < min_sep:
                    # 不直接丢掉，给较大惩罚，避免小 hole 完全找不到 exit。
                    sep_penalty = min_sep - sep
                else:
                    sep_penalty = 0.0
            else:
                sep_penalty = 0.0

            # 越靠近 hole 投影后端、越靠近 component，越适合作为 exit。
            score = abs(s_coord - target_s) + 0.35 * dist_to_hole + 2.0 * sep_penalty
            # 负的 s_coord 轻微惩罚，避免选到机器人后方。
            if s_coord < robot_s:
                score += 1.0 + (robot_s - s_coord)

            candidates.append((score, -s_coord, dist_to_hole, key))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[2], item[1]))
            return candidates[0][3]

        return boundary_key if boundary_key in self.nodes else None

    def nearest_component_key_to_xy(
        self,
        comp: Set[NodeKey],
        xy: Tuple[float, float]
    ) -> Optional[NodeKey]:
        best_key = None
        best_dist = float('inf')

        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            d = math.hypot(x - xy[0], y - xy[1])
            if d < best_dist:
                best_dist = d
                best_key = key

        return best_key

    def collect_hole_seed_nodes(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey]
    ) -> List[NodeKey]:
        """
        本验证版不再使用普通 seed 兜底。

        hole detection 的 seed 只来自 collect_branch_hole_seed_pairs()：
        boundary_lap 上的 sample 如果存在明显向 hole 内部延伸的侧向 RCG 边，
        则把这条边另一端的 open sample 作为 floodfill seed。
        """
        return []

    def is_next_goal_on_same_lap(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey]
    ) -> bool:
        if current_key not in self.nodes:
            return False

        if next_goal_key is None or next_goal_key not in self.nodes:
            return False

        _, cy = self.nodes[current_key]
        _, gy = self.nodes[next_goal_key]

        return abs(gy - cy) <= self.same_lap_y_tolerance


    def is_lap_switch_goal(
        self,
        current_key: Optional[NodeKey],
        next_goal_key: Optional[NodeKey]
    ) -> bool:
        """
        判断 current_key -> next_goal_key 是否属于“切换不同 lap”。

        hole detection / repair 只在沿同一条 lap 正常前进时触发；
        当普通 C* 正在从一条 lap 切换到另一条 lap（上/下/斜向过渡）时，
        暂停 hole 检测，避免把换行通道、门口附近区域误判成 hole。
        """
        if current_key is None or next_goal_key is None:
            return False

        if current_key not in self.nodes or next_goal_key not in self.nodes:
            return False

        cx, cy = self.nodes[current_key]
        gx, gy = self.nodes[next_goal_key]
        dx = gx - cx
        dy = gy - cy

        # 主要横向且 y 差很小，认为仍在当前 lap。
        if abs(dy) <= self.same_lap_y_tolerance and abs(dx) >= 1e-6:
            return False

        return True

    def collect_same_lap_segment_nodes(self, current_key: NodeKey) -> List[NodeKey]:
        """
        收集 current_key 所在的同一条 lap 段。

        这里不用“全图 y 坐标相近”的所有节点，避免不同房间/不同走廊里
        刚好同 y 的节点被错误合并。只沿 RCG 邻接边，在 y 差较小的边上做 BFS，
        得到当前连通 lap 段。
        """
        if current_key not in self.nodes:
            return []

        y_tol = max(
            self.same_lap_y_tolerance,
            self.same_lap_y_tolerance * max(1.0, self.hole_boundary_lap_y_tolerance_factor)
        )

        visited: Set[NodeKey] = set()
        q = deque()
        q.append(current_key)
        visited.add(current_key)

        while q:
            key = q.popleft()
            if key not in self.nodes:
                continue

            _, y0 = self.nodes[key]

            for nb in self.adjacency.get(key, set()):
                if nb not in self.nodes or nb in visited:
                    continue

                _, y1 = self.nodes[nb]
                if abs(y1 - y0) <= y_tol:
                    visited.add(nb)
                    q.append(nb)

        lap_nodes = [key for key in visited if key in self.nodes]
        lap_nodes.sort(key=lambda k: self.nodes[k][0])
        return lap_nodes

    def boundary_probe_hits_buffer(
        self,
        x: float,
        y: float,
        nx: float,
        ny: float,
        safe: np.ndarray
    ) -> bool:
        """
        从一个 lap 采样点沿法向探测，看 probe_distance 内是否遇到 buffer / 障碍 / 边界。

        safe 来自 build_safe_free_mask()，已经扣除了 obstacle_buffer、unknown_buffer、map_border_buffer。
        因此探测到 not safe，就等价于靠近墙壁、障碍物、未知区或地图边界。
        """
        if self.free_msg is None:
            return False

        res = self.free_msg.info.resolution
        max_dist = max(res, self.hole_boundary_probe_distance)
        step = max(res, 0.5 * res)
        n = max(1, int(math.ceil(max_dist / step)))

        for i in range(1, n + 1):
            d = min(max_dist, i * step)
            cell = self.world_to_cell(x + nx * d, y + ny * d)

            if cell is None:
                return True

            row, col = cell
            if row < 0 or col < 0 or row >= safe.shape[0] or col >= safe.shape[1]:
                return True

            if not bool(safe[row, col]):
                return True

        return False

    def is_current_lap_near_buffer(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey]
    ) -> bool:
        """
        判断当前小车所在的整条 lap 是否为靠近 buffer 的边界侧 lap。

        触发原则：
        - 必须是同一条 lap 上的正常前进，不在换 lap 阶段；
        - 不能只看当前点或端点，因为一条 lap 的端点天然靠近墙；
        - 收集当前连通 lap 段，去掉两端 endpoint 后，沿 lap 法向两侧探测；
        - 如果某一侧有足够比例的中间采样点在短距离内探测到 buffer，
          则认为这是最靠近墙/障碍物的一条 boundary lap，允许 hole detection。
        """
        if not self.hole_enable_boundary_lap_gate:
            return True

        if current_key not in self.nodes:
            return False

        if next_goal_key is None or next_goal_key not in self.nodes:
            return False

        # 不在同一条 lap 上，说明正在换行/过渡，此时不检测 hole。
        if not self.is_next_goal_on_same_lap(current_key, next_goal_key):
            return False

        safe = self.build_safe_free_mask()
        if safe is None:
            return False

        lap_nodes = self.collect_same_lap_segment_nodes(current_key)
        if len(lap_nodes) < 3:
            return False

        ignore_n = max(0, self.hole_boundary_ignore_endpoint_count)
        if len(lap_nodes) > 2 * ignore_n + 1:
            test_nodes = lap_nodes[ignore_n: len(lap_nodes) - ignore_n]
        else:
            # lap 很短时仍至少保留中间节点，避免全部被忽略。
            mid = len(lap_nodes) // 2
            test_nodes = [lap_nodes[mid]]

        if not test_nodes:
            return False

        cx, cy = self.nodes[current_key]
        gx, gy = self.nodes[next_goal_key]
        ux = gx - cx
        uy = gy - cy
        u_norm = math.hypot(ux, uy)

        if u_norm < 1e-6:
            # 当前工程里 lap 基本是水平的，兜底用 x 方向。
            ux, uy = 1.0, 0.0
        else:
            ux /= u_norm
            uy /= u_norm

        # lap 法向两侧。
        nx1, ny1 = -uy, ux
        nx2, ny2 = uy, -ux

        side1_hits = 0
        side2_hits = 0
        valid_count = 0

        for key in test_nodes:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            valid_count += 1

            if self.boundary_probe_hits_buffer(x, y, nx1, ny1, safe):
                side1_hits += 1

            if self.boundary_probe_hits_buffer(x, y, nx2, ny2, safe):
                side2_hits += 1

        if valid_count <= 0:
            return False

        side1_ratio = float(side1_hits) / float(valid_count)
        side2_ratio = float(side2_hits) / float(valid_count)
        best_ratio = max(side1_ratio, side2_ratio)

        return best_ratio >= self.hole_boundary_min_ratio


    def get_current_lap_motion_basis(
        self,
        current_key: NodeKey,
        next_goal_key: NodeKey
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        返回当前 boundary_lap 的运动坐标系：
        u：当前 C* 前进方向；n1/n2：lap 两侧法向；origin：current_key 坐标。
        """
        if current_key not in self.nodes or next_goal_key not in self.nodes:
            return None

        cx, cy = self.nodes[current_key]
        gx, gy = self.nodes[next_goal_key]
        ux = gx - cx
        uy = gy - cy
        norm = math.hypot(ux, uy)

        if norm < 1e-6:
            ux = 1.0 if self.sweep_dir >= 0.0 else -1.0
            uy = 0.0
        else:
            ux /= norm
            uy /= norm

        n1 = (-uy, ux)
        n2 = (uy, -ux)
        return (ux, uy), n1, n2, (cx, cy)

    def node_s_on_lap(
        self,
        key: NodeKey,
        origin: Tuple[float, float],
        u: Tuple[float, float]
    ) -> float:
        x, y = self.nodes[key]
        return (x - origin[0]) * u[0] + (y - origin[1]) * u[1]

    def component_main_lap_ratio(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey
    ) -> float:
        if current_key not in self.nodes or not comp:
            return 0.0

        _, cy = self.nodes[current_key]
        width = max(
            self.same_lap_y_tolerance,
            self.same_lap_y_tolerance * max(1.0, self.hole_main_lap_width_factor)
        )

        same_lap_count = 0
        valid_count = 0

        for key in comp:
            if key not in self.nodes:
                continue

            _, y = self.nodes[key]
            valid_count += 1

            if abs(y - cy) <= width:
                same_lap_count += 1

        if valid_count <= 0:
            return 0.0

        return float(same_lap_count) / float(valid_count)

    def component_ahead_ratio_on_current_sweep(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        next_goal_key: NodeKey
    ) -> float:
        if current_key not in self.nodes or next_goal_key not in self.nodes or not comp:
            return 0.0

        cx, cy = self.nodes[current_key]
        gx, _ = self.nodes[next_goal_key]

        dx_goal = gx - cx
        if abs(dx_goal) < 1e-6:
            return 0.0

        sweep_sign = 1.0 if dx_goal > 0.0 else -1.0
        width = max(
            self.same_lap_y_tolerance,
            self.same_lap_y_tolerance * max(1.0, self.hole_main_lap_width_factor)
        )

        same_lap_count = 0
        ahead_count = 0

        for key in comp:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]

            if abs(y - cy) > width:
                continue

            same_lap_count += 1

            # 允许 -5cm 的误差，避免机器人刚过节点时把紧邻节点误排除。
            if (x - cx) * sweep_sign >= -0.05:
                ahead_count += 1

        if same_lap_count <= 0:
            return 0.0

        return float(ahead_count) / float(same_lap_count)

    def should_protect_component_as_main_sweep(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey]
    ) -> bool:
        """
        主干道保护过滤：

        如果 next_goal 仍在当前 lap 上，说明当前 C* 牛耕还没有离开这条主扫描线。
        此时位于当前 lap 附近、并且处在当前运动方向上的 open component，
        不是 coverage hole，而是马上会被普通 C* 覆盖的区域。

        这一步只过滤误检，不会改变当前 C* goal，也不会接管运动控制。
        """
        if not self.hole_main_lap_protect_enable:
            return False

        if next_goal_key is None or next_goal_key not in self.nodes:
            return False

        if not self.is_next_goal_on_same_lap(current_key, next_goal_key):
            return False

        main_lap_ratio = self.component_main_lap_ratio(comp, current_key)
        ahead_ratio = self.component_ahead_ratio_on_current_sweep(
            comp,
            current_key,
            next_goal_key
        )

        if main_lap_ratio >= self.hole_main_lap_ratio_threshold:
            return True

        if ahead_ratio >= self.hole_ahead_ratio_threshold:
            return True

        return False

    def point_to_segment_metrics(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float
    ) -> Tuple[float, float, float]:
        """
        返回点到线段的：
        dist：垂直/端点距离；
        along：投影点距离线段起点的有符号长度；
        seg_len：线段长度。
        """
        vx = bx - ax
        vy = by - ay
        seg_len2 = vx * vx + vy * vy

        if seg_len2 < 1e-9:
            return math.hypot(px - ax, py - ay), 0.0, 0.0

        seg_len = math.sqrt(seg_len2)
        t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
        t_clamped = max(0.0, min(1.0, t))

        qx = ax + t_clamped * vx
        qy = ay + t_clamped * vy
        dist = math.hypot(px - qx, py - qy)
        along = t * seg_len

        return dist, along, seg_len


    def component_local_gate_metrics(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        next_goal_key: NodeKey,
        robot_xy: Tuple[float, float]
    ) -> Tuple[float, float, bool]:
        """
        计算 component 与当前 C* 前视局部走廊的关系。

        这里不再只看 current->next 这一条很短的小线段，而是把它沿前进方向
        向前扩展 hole_local_forward_extension。这样能更早识别洞口旁边的 hole，
        但仍不会把远处房间全部拉进来。
        """
        if current_key not in self.nodes or next_goal_key not in self.nodes:
            return float('inf'), float('inf'), False

        ax, ay = self.nodes[current_key]
        bx, by = self.nodes[next_goal_key]

        vx = bx - ax
        vy = by - ay
        seg_len = math.hypot(vx, vy)

        if seg_len < 1e-6:
            ux, uy = 1.0, 0.0
            seg_len = 0.0
        else:
            ux, uy = vx / seg_len, vy / seg_len

        min_robot_dist = float('inf')
        min_corridor_dist = float('inf')
        has_node_in_gate = False

        for key in comp:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]

            robot_dist = math.hypot(x - robot_xy[0], y - robot_xy[1])
            if robot_dist < min_robot_dist:
                min_robot_dist = robot_dist

            dx = x - ax
            dy = y - ay
            along = dx * ux + dy * uy
            lateral = abs(-uy * dx + ux * dy)

            if lateral < min_corridor_dist:
                min_corridor_dist = lateral

            in_along_window = (
                along >= -self.hole_local_backward_extension and
                along <= seg_len + self.hole_local_forward_extension
            )
            in_lateral_window = lateral <= self.hole_local_segment_distance

            if in_along_window and in_lateral_window:
                has_node_in_gate = True

        return min_robot_dist, min_corridor_dist, has_node_in_gate

    def component_local_hole_score(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> float:
        if next_goal_key is None or next_goal_key not in self.nodes:
            cx, cy = self.component_center(comp)
            return math.hypot(cx - robot_xy[0], cy - robot_xy[1])

        min_robot_dist, min_segment_dist, _ = self.component_local_gate_metrics(
            comp,
            current_key,
            next_goal_key,
            robot_xy
        )

        cx, cy = self.component_center(comp)
        center_robot_dist = math.hypot(cx - robot_xy[0], cy - robot_xy[1])

        # 越靠近当前机器人、越靠近 current->next 局部运动线段，越优先。
        return 0.55 * min_robot_dist + 0.35 * min_segment_dist + 0.10 * center_robot_dist

    def component_passes_local_hole_gate(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> bool:
        """
        局部 hole 门控：
        只保留当前 C* 正常覆盖路径附近的旁支区域。

        目的：
        1. 不提前检测远处房间；
        2. 不把后续正常牛耕会覆盖的区域判为 hole；
        3. 只检测当前机器人附近、当前 current->next 局部运动线段旁边的 hole。
        """
        if not self.hole_local_gate_enable:
            return True

        if next_goal_key is None or next_goal_key not in self.nodes:
            return True

        min_robot_dist, _, has_node_in_gate = self.component_local_gate_metrics(
            comp,
            current_key,
            next_goal_key,
            robot_xy
        )

        if min_robot_dist > self.hole_local_robot_distance:
            return False

        if not has_node_in_gate:
            return False

        return True

    def clear_pending_branch_trigger(self) -> None:
        self.pending_branch_trigger_goal_key = None
        self.pending_branch_trigger_seed_key = None
        self.pending_branch_trigger_boundary_key = None
        self.pending_branch_trigger_score = 0.0

    def find_branch_trigger_for_goal(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> Optional[Tuple[NodeKey, NodeKey, float]]:
        """
        判断刚选出来的 goal_marker 是否就是 boundary_key。

        新逻辑：
        - 不再“先缓存，等到达后执行”；
        - 只要 next_goal_key 本身就是带侧向 RCG 分支边的 boundary_key，
          就立即返回 (seed_key, boundary_key, score)，由 on_timer 直接进入 hole 过程。
        """
        if not self.enable_hole_detection:
            return None

        if current_key not in self.nodes:
            return None

        if next_goal_key is None or next_goal_key not in self.nodes:
            return None

        if self.hole_disable_during_escape and self.escape_active:
            return None

        if self.is_lap_switch_goal(current_key, next_goal_key):
            return None

        if self.hole_enable_boundary_lap_gate:
            if not self.is_current_lap_near_buffer(current_key, next_goal_key):
                return None

        branch_pairs = self.collect_branch_hole_seed_pairs(
            current_key,
            next_goal_key,
            robot_xy
        )

        matched: List[Tuple[float, NodeKey, NodeKey]] = []
        for seed_key, boundary_key, score in branch_pairs:
            if boundary_key == next_goal_key:
                matched.append((score, seed_key, boundary_key))

        if not matched:
            return None

        matched.sort(key=lambda item: item[0])
        score, seed_key, boundary_key = matched[0]
        return seed_key, boundary_key, score

    def arm_branch_trigger_for_next_goal(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> None:
        """
        兼容旧接口。当前版本不再使用“预判后等待到达”的触发方式。
        这里只做清空，真正触发由 find_branch_trigger_for_goal() 在选出 goal_marker 后立即完成。
        """
        self.clear_pending_branch_trigger()

    def consume_branch_trigger_if_arrived(
        self,
        current_key: NodeKey
    ) -> Optional[Tuple[NodeKey, NodeKey, float]]:
        """
        如果小车当前真正到达了之前预判的 boundary_key，则取出对应 seed。
        返回 (seed_key, boundary_key, score)。
        """
        if self.pending_branch_trigger_goal_key is None:
            return None

        if current_key != self.pending_branch_trigger_goal_key:
            return None

        seed_key = self.pending_branch_trigger_seed_key
        boundary_key = self.pending_branch_trigger_boundary_key
        score = self.pending_branch_trigger_score

        self.clear_pending_branch_trigger()

        if seed_key is None or boundary_key is None:
            return None

        if seed_key not in self.nodes or boundary_key not in self.nodes:
            return None

        return seed_key, boundary_key, score

    def detect_hole_from_goal_branch_trigger(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
        seed_key: NodeKey,
        boundary_key: NodeKey,
        branch_score: float
    ) -> None:
        """
        当刚选出的 goal_marker 本身就是 boundary_key 时，立即从对应 branch seed 做 floodfill。

        这一版不等待小车到达 boundary_key；goal_marker 一旦被判定为 boundary_key，
        就直接进入 hole detection / repair 可视化过程。
        """
        self.latest_hole_components = []
        self.latest_hole_infos = []
        self.latest_hole_samples = []
        self.latest_hole_repair_path = []
        self.pending_hole_entry_key = None
        self.pending_hole_exit_key = None

        if not self.enable_hole_detection:
            return

        if current_key not in self.nodes:
            return

        if next_goal_key is None or next_goal_key not in self.nodes:
            return

        if self.hole_disable_during_escape and self.escape_active:
            return

        if self.hole_enable_boundary_lap_gate:
            if not self.is_current_lap_near_buffer(current_key, next_goal_key):
                return

        visited: Set[NodeKey] = set()
        comp = self.floodfill_open_component(seed_key, next_goal_key, visited)
        if not comp:
            return

        if self.should_protect_component_as_main_sweep(comp, current_key, next_goal_key):
            return

        if not self.component_passes_local_hole_gate(comp, current_key, next_goal_key, robot_xy):
            return

        ok, info = self.validate_hole_component(comp, robot_xy)
        if not ok or info is None:
            return

        entry_key = seed_key if seed_key in comp else self.nearest_component_key_to_xy(comp, robot_xy)
        exit_key = self.select_branch_exit_key_for_component(
            comp,
            current_key,
            next_goal_key,
            robot_xy,
            boundary_key
        )

        self.latest_hole_components = [comp]
        self.latest_hole_infos = [info]
        self.pending_hole_entry_key = entry_key
        self.pending_hole_exit_key = exit_key

        if self.pending_hole_entry_key in self.nodes and self.pending_hole_exit_key in self.nodes:
            self.build_hole_repair_path(current_key, robot_xy, next_goal_key)

        entry_msg = str(self.pending_hole_entry_key) if self.pending_hole_entry_key is not None else 'None'
        exit_msg = str(self.pending_hole_exit_key) if self.pending_hole_exit_key is not None else 'None'
        self.get_logger().warn(
            f'Direct goal-branch hole: entry={entry_msg}, exit={exit_msg}, '
            f'score={branch_score:.3f}, hole_samples={len(self.latest_hole_samples)}, '
            f'repair_points={len(self.latest_hole_repair_path)}.'
        )

    def detect_holes_after_next_goal(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> None:
        """
        Branch-seed 验证版 hole detection。

        本版故意删去 doorway gap predictor 和普通 seed 兜底：
        1. 先判断当前是否处于 boundary_lap 正常前进；
        2. 在当前 boundary_lap 前方寻找“向 hole 内部延伸的侧向 RCG 分支边”；
        3. 只用分支边另一端的 open sample 作为 floodfill seed；
        4. boundary_key 可以是 closed sample；
        5. entry 优先使用 branch seed，exit 固定在 boundary_lap 上。
        """
        self.latest_hole_components = []
        self.latest_hole_infos = []
        self.latest_hole_samples = []
        self.latest_hole_repair_path = []
        self.pending_hole_entry_key = None
        self.pending_hole_exit_key = None
        self.pending_branch_trigger_goal_key = None
        self.pending_branch_trigger_seed_key = None
        self.pending_branch_trigger_boundary_key = None
        self.pending_branch_trigger_score = 0.0

        if not self.enable_hole_detection:
            return

        if current_key not in self.nodes:
            return

        if next_goal_key is None or next_goal_key not in self.nodes:
            return

        if self.hole_disable_during_escape and self.escape_active:
            return

        if self.hole_enable_boundary_lap_gate:
            if not self.is_current_lap_near_buffer(current_key, next_goal_key):
                return

        branch_pairs = self.collect_branch_hole_seed_pairs(
            current_key,
            next_goal_key,
            robot_xy
        )

        if not branch_pairs:
            return

        candidate_holes: List[Tuple[
            float,
            Set[NodeKey],
            Tuple[float, float, int, float, float, float, str],
            Optional[NodeKey],
            Optional[NodeKey]
        ]] = []

        visited: Set[NodeKey] = set()
        seen_components: Set[frozenset] = set()

        for seed_key, boundary_key, branch_score in branch_pairs:
            if seed_key not in self.nodes:
                continue

            if seed_key in visited:
                continue

            comp = self.floodfill_open_component(seed_key, next_goal_key, visited)
            if not comp:
                continue

            comp_id = frozenset(comp)
            if comp_id in seen_components:
                continue
            seen_components.add(comp_id)

            if self.should_protect_component_as_main_sweep(comp, current_key, next_goal_key):
                continue

            if not self.component_passes_local_hole_gate(comp, current_key, next_goal_key, robot_xy):
                continue

            ok, info = self.validate_hole_component(comp, robot_xy)
            if not ok or info is None:
                continue

            if seed_key in comp and seed_key in self.nodes:
                entry_key = seed_key
            else:
                # 理论上 seed 会在 comp 内；这里只做防御性兜底，仍不使用普通 seed。
                entry_key = self.nearest_component_key_to_xy(comp, robot_xy)

            exit_key = self.select_branch_exit_key_for_component(
                comp,
                current_key,
                next_goal_key,
                robot_xy,
                boundary_key
            )

            score = (
                branch_score +
                0.35 * self.component_local_hole_score(comp, current_key, next_goal_key, robot_xy)
            )

            candidate_holes.append((score, comp, info, entry_key, exit_key))

        candidate_holes.sort(key=lambda item: item[0])
        keep_n = max(1, self.hole_keep_nearest_components)
        candidate_holes = candidate_holes[:keep_n]

        if not candidate_holes:
            return

        self.latest_hole_components = [item[1] for item in candidate_holes]
        self.latest_hole_infos = [item[2] for item in candidate_holes]

        self.pending_hole_entry_key = candidate_holes[0][3]
        self.pending_hole_exit_key = candidate_holes[0][4]

        if self.pending_hole_entry_key in self.nodes and self.pending_hole_exit_key in self.nodes:
            self.build_hole_repair_path(current_key, robot_xy, next_goal_key)

        if self.latest_hole_infos:
            entry_msg = str(self.pending_hole_entry_key) if self.pending_hole_entry_key is not None else 'None'
            exit_msg = str(self.pending_hole_exit_key) if self.pending_hole_exit_key is not None else 'None'
            self.get_logger().warn(
                f'Branch-seed hole detection: detected {len(self.latest_hole_infos)} component(s), '
                f'branch_pairs={len(branch_pairs)}, '
                f'entry={entry_msg}, exit={exit_msg}, '
                f'hole_samples={len(self.latest_hole_samples)}, '
                f'repair_points={len(self.latest_hole_repair_path)}.'
            )

    def floodfill_open_component(
        self,
        seed: NodeKey,
        protected_goal_key: NodeKey,
        global_visited: Set[NodeKey]
    ) -> Set[NodeKey]:
        comp: Set[NodeKey] = set()
        q = deque()

        q.append(seed)
        global_visited.add(seed)

        while q:
            key = q.popleft()

            if key not in self.nodes:
                continue

            if key == protected_goal_key:
                continue

            if self.is_closed_key(key):
                continue

            comp.add(key)

            for nb in self.adjacency.get(key, set()):
                if nb not in self.nodes:
                    continue

                if nb in global_visited:
                    continue

                if nb == protected_goal_key:
                    continue

                if self.is_closed_key(nb):
                    continue

                global_visited.add(nb)
                q.append(nb)

        return comp

    def validate_hole_component(
        self,
        comp: Set[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> Tuple[bool, Optional[Tuple[float, float, int, float, float, float, str]]]:
        node_count = len(comp)

        if node_count < self.hole_min_nodes:
            return False, None

        if node_count > self.hole_max_nodes:
            return False, None

        center_x, center_y = self.component_center(comp)
        robot_dist = math.hypot(center_x - robot_xy[0], center_y - robot_xy[1])

        if robot_dist > self.hole_max_robot_distance:
            return False, None

        bbox_area = self.component_bbox_area(comp)

        if bbox_area < self.hole_min_bbox_area:
            return False, None

        if bbox_area > self.hole_max_bbox_area:
            return False, None

        closed_attach_nodes = self.component_closed_attach_count(comp)

        if closed_attach_nodes < self.hole_min_closed_attach_nodes:
            return False, None

        unknown_area, unknown_ratio = self.component_unknown_stats(comp)

        # 关键改动：unknown 不再一票否决。
        # 只有 unknown 面积和比例同时较大，才认为这个 component 接触真正 frontier，不是 hole。
        significant_unknown = (
            unknown_area >= self.hole_unknown_reject_area and
            unknown_ratio >= self.hole_unknown_reject_ratio
        )

        if significant_unknown:
            return False, None

        incomplete_unknown = (
            unknown_area >= self.hole_incomplete_unknown_area or
            unknown_ratio >= self.hole_incomplete_unknown_ratio
        )

        if incomplete_unknown:
            label = 'incomplete'
        elif unknown_area > 1e-6:
            label = 'potential'
        else:
            label = 'confirmed'

        return True, (
            center_x,
            center_y,
            node_count,
            bbox_area,
            unknown_area,
            unknown_ratio,
            label
        )

    def component_center(self, comp: Set[NodeKey]) -> Tuple[float, float]:
        if not comp:
            return 0.0, 0.0

        sx = 0.0
        sy = 0.0
        count = 0

        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            sx += x
            sy += y
            count += 1

        if count == 0:
            return 0.0, 0.0

        return sx / float(count), sy / float(count)

    def component_bbox_area(self, comp: Set[NodeKey]) -> float:
        if not comp:
            return 0.0

        xs: List[float] = []
        ys: List[float] = []

        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            xs.append(x)
            ys.append(y)

        if not xs or not ys:
            return 0.0

        res = 0.05
        if self.free_msg is not None:
            res = self.free_msg.info.resolution

        width = max(max(xs) - min(xs), res)
        height = max(max(ys) - min(ys), res)

        return width * height

    def component_closed_attach_count(self, comp: Set[NodeKey]) -> int:
        count = 0

        for key in comp:
            if key not in self.nodes:
                continue

            attached = False

            # RCG 图上邻接 closed node
            for nb in self.adjacency.get(key, set()):
                if nb in self.nodes and self.is_closed_key(nb):
                    attached = True
                    break

            if attached:
                count += 1

        return count

    def component_unknown_stats(self, comp: Set[NodeKey]) -> Tuple[float, float]:
        """
        返回：
        unknown_area:
            component 周围检查半径内的 unknown 面积，单位 m^2。
        unknown_ratio:
            component 中有多少比例的节点靠近 unknown。

        只有 unknown_area 和 unknown_ratio 同时较大时，才认为它是真正 frontier。
        """
        if self.free_msg is None or self.unknown_arr is None:
            return 0.0, 0.0

        if not comp:
            return 0.0, 0.0

        info = self.free_msg.info
        res = info.resolution
        rad = max(1, int(math.ceil(self.hole_unknown_check_radius / res)))

        h, w = self.unknown_arr.shape
        unknown_cells: Set[GridCell] = set()
        nodes_near_unknown = 0

        for key in comp:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            cell = self.world_to_cell(x, y)

            if cell is None:
                continue

            row, col = cell
            near_unknown = False

            r0 = max(0, row - rad)
            r1 = min(h - 1, row + rad)
            c0 = max(0, col - rad)
            c1 = min(w - 1, col + rad)

            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    dr = rr - row
                    dc = cc - col

                    if dr * dr + dc * dc > rad * rad:
                        continue

                    if self.unknown_arr[rr, cc]:
                        unknown_cells.add((rr, cc))
                        near_unknown = True

            if near_unknown:
                nodes_near_unknown += 1

        unknown_area = len(unknown_cells) * res * res
        unknown_ratio = float(nodes_near_unknown) / float(max(1, len(comp)))

        return unknown_area, unknown_ratio

    def node_has_unknown_near(self, key: NodeKey) -> bool:
        if key not in self.nodes:
            return False

        if self.free_msg is None or self.unknown_arr is None:
            return False

        x, y = self.nodes[key]
        cell = self.world_to_cell(x, y)
        if cell is None:
            return False

        row, col = cell
        res = self.free_msg.info.resolution
        rad = max(1, int(math.ceil(self.hole_unknown_check_radius / res)))
        h, w = self.unknown_arr.shape

        r0 = max(0, row - rad)
        r1 = min(h - 1, row + rad)
        c0 = max(0, col - rad)
        c1 = min(w - 1, col + rad)

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                dr = rr - row
                dc = cc - col
                if dr * dr + dc * dc > rad * rad:
                    continue
                if self.unknown_arr[rr, cc]:
                    return True

        return False

    def choose_active_hole_index(
        self,
        robot_xy: Tuple[float, float]
    ) -> Optional[int]:
        """
        选择当前最应该可视化/补扫的一个 hole component。

        这里仍然只做可视化，不接管 C* 运动。
        优先保持和 active_hole_center 匹配的 component；没有匹配时选离机器人最近的 component。
        """
        if not self.latest_hole_components or not self.latest_hole_infos:
            return None

        if self.active_hole_center is not None:
            ax, ay = self.active_hole_center
            best_i = None
            best_d = float('inf')

            for i, info in enumerate(self.latest_hole_infos):
                cx, cy = info[0], info[1]
                d = math.hypot(cx - ax, cy - ay)
                if d < best_d:
                    best_d = d
                    best_i = i

            if best_i is not None and best_d <= self.hole_active_match_distance:
                return best_i

        best_i = None
        best_d = float('inf')

        for i, info in enumerate(self.latest_hole_infos):
            cx, cy = info[0], info[1]
            d = math.hypot(cx - robot_xy[0], cy - robot_xy[1])
            if d < best_d:
                best_d = d
                best_i = i

        return best_i

    def get_main_sweep_basis(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
        comp: Optional[Set[NodeKey]] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
        """
        返回当前 C* 主运动方向 u、其垂直方向 v、投影原点 origin、当前段长度 seg_len。

        u：沿 current_key -> next_goal_key，即正常 C* 当前运动方向；
        v：垂直于 u，hole 内部补扫优先沿 v 做往返运动。
        """
        if current_key in self.nodes:
            ox, oy = self.nodes[current_key]
        else:
            ox, oy = robot_xy

        if next_goal_key is not None and next_goal_key in self.nodes and current_key in self.nodes:
            gx, gy = self.nodes[next_goal_key]
            dx = gx - ox
            dy = gy - oy
            seg_len = math.hypot(dx, dy)
        else:
            seg_len = 0.0
            dx = 0.0
            dy = 0.0
            if comp:
                cx, cy = self.component_center(comp)
                dx = cx - ox
                dy = cy - oy
                seg_len = math.hypot(dx, dy)

        if seg_len < 1e-6:
            ux, uy = 1.0, 0.0
            seg_len = 0.0
        else:
            ux, uy = dx / seg_len, dy / seg_len

        # 左手垂直方向。符号本身不重要，后续会尝试正反两种内部排序。
        vx, vy = -uy, ux
        return (ux, uy), (vx, vy), (ox, oy), seg_len

    def project_in_sweep_frame(
        self,
        x: float,
        y: float,
        origin: Tuple[float, float],
        u: Tuple[float, float],
        v: Tuple[float, float]
    ) -> Tuple[float, float]:
        dx = x - origin[0]
        dy = y - origin[1]
        s = dx * u[0] + dy * u[1]
        t = dx * v[0] + dy * v[1]
        return s, t

    def min_distance_from_node_to_component(
        self,
        key: NodeKey,
        comp: Set[NodeKey]
    ) -> float:
        if key not in self.nodes or not comp:
            return float('inf')

        x, y = self.nodes[key]
        best = float('inf')

        for ck in comp:
            if ck not in self.nodes:
                continue
            cx, cy = self.nodes[ck]
            d = math.hypot(cx - x, cy - y)
            if d < best:
                best = d

        return best

    def select_boundary_lap_entry_exit_keys_for_hole(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        robot_xy: Tuple[float, float],
        next_goal_key: Optional[NodeKey]
    ) -> Tuple[Optional[NodeKey], Optional[NodeKey]]:
        """
        从当前 boundary lap 上选择 hole repair 的 start/end。

        关键区别：
        - entry / exit 不再从 hole component 内部选；
        - 只从 current_key 所在的同一条 boundary lap 段上选；
        - 通过 hole component 在当前 C* 主方向上的投影范围确定 doorway 区间；
        - entry 是沿当前 C* 前进方向最先遇到的 boundary-lap 节点；
          exit 是最后遇到的 boundary-lap 节点。

        这样即使 hole 内部 unknown area 后续动态补全，start/end 也稳定停留在
        正常 C* 的主扫描线上，不会跑进 hole 内部。
        """
        if not comp or current_key not in self.nodes:
            return None, None

        if next_goal_key is None or next_goal_key not in self.nodes:
            return None, None

        lap_nodes = self.collect_same_lap_segment_nodes(current_key)
        if len(lap_nodes) < 2:
            return None, None

        u, v, origin, _ = self.get_main_sweep_basis(
            current_key,
            next_goal_key,
            robot_xy,
            comp
        )

        comp_proj: List[Tuple[float, float]] = []
        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            comp_proj.append(self.project_in_sweep_frame(x, y, origin, u, v))

        if not comp_proj:
            return None, None

        min_s = min(s for s, _ in comp_proj)
        max_s = max(s for s, _ in comp_proj)
        mean_t = sum(t for _, t in comp_proj) / float(len(comp_proj))
        side_sign = 1.0 if mean_t >= 0.0 else -1.0

        s_margin = max(0.05, self.hole_boundary_entry_s_margin)
        max_dist_to_hole = max(0.10, self.hole_boundary_entry_search_distance)
        band_width = max(0.08, self.hole_doorway_band_width)

        candidates: List[Tuple[float, float, float, NodeKey]] = []
        fallback: List[Tuple[float, float, float, NodeKey]] = []

        for key in lap_nodes:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            s_coord, t_coord = self.project_in_sweep_frame(x, y, origin, u, v)

            # boundary lap 节点应在当前主扫描线附近；偏离太多的节点不作为入口/出口。
            if abs(t_coord) > max(self.same_lap_y_tolerance * 2.0, band_width):
                continue

            dist_to_hole = self.min_distance_from_node_to_component(key, comp)
            in_s_window = (s_coord >= min_s - s_margin and s_coord <= max_s + s_margin)
            near_hole = dist_to_hole <= max_dist_to_hole

            # side_score 只作为排序兜底，不改变 start/end 的沿 lap 顺序。
            side_score = abs(t_coord) + 0.10 * max(0.0, -side_sign * t_coord)

            if in_s_window and near_hole:
                candidates.append((s_coord, dist_to_hole, side_score, key))

            range_gap = 0.0
            if s_coord < min_s:
                range_gap = min_s - s_coord
            elif s_coord > max_s:
                range_gap = s_coord - max_s
            fallback.append((range_gap + 0.3 * dist_to_hole, dist_to_hole, side_score, key))

        if len(candidates) < 2:
            fallback.sort(key=lambda item: (item[0], item[1], item[2]))
            keep = min(max(2, min(8, len(fallback))), len(fallback))
            candidates = []
            for item in fallback[:keep]:
                key = item[3]
                if key not in self.nodes:
                    continue
                x, y = self.nodes[key]
                s_coord, _ = self.project_in_sweep_frame(x, y, origin, u, v)
                candidates.append((s_coord, item[1], item[2], key))

        if len(candidates) < 2:
            return None, None

        unique: Dict[NodeKey, Tuple[float, float, float, NodeKey]] = {}
        for item in candidates:
            unique[item[3]] = item
        candidates = list(unique.values())
        candidates.sort(key=lambda item: item[0])

        entry_key = candidates[0][3]
        exit_key = candidates[-1][3]

        if entry_key in self.nodes and exit_key in self.nodes:
            ex, ey = self.nodes[entry_key]
            xx, xy = self.nodes[exit_key]
            if math.hypot(xx - ex, xy - ey) < self.hole_min_entry_exit_distance:
                best_pair = None
                best_sep = -1.0
                for a in candidates:
                    for b in candidates:
                        if a[3] == b[3]:
                            continue
                        ax, ay = self.nodes[a[3]]
                        bx, by = self.nodes[b[3]]
                        dxy = math.hypot(bx - ax, by - ay)
                        sep = abs(b[0] - a[0])
                        if dxy >= self.hole_min_entry_exit_distance and sep > best_sep:
                            best_sep = sep
                            best_pair = (a, b)

                if best_pair is not None:
                    a, b = best_pair
                    if a[0] <= b[0]:
                        entry_key, exit_key = a[3], b[3]
                    else:
                        entry_key, exit_key = b[3], a[3]

        if entry_key not in self.nodes or exit_key not in self.nodes:
            return None, None

        return entry_key, exit_key

    def select_entry_exit_keys_for_hole(
        self,
        comp: Set[NodeKey],
        current_key: NodeKey,
        robot_xy: Tuple[float, float],
        next_goal_key: Optional[NodeKey]
    ) -> Tuple[Optional[NodeKey], Optional[NodeKey]]:
        """
        选择 repair start/end。

        工程稳定版：
        - 默认只允许 entry / exit 落在当前 boundary_lap 上；
        - 不再从 hole component 内部兜底选择 entry/exit，避免 start/end 跑进 hole；
        - 如果 boundary_lap 上无法选出合理 start/end，则本轮只显示 hole，
          不生成 repair path。
        """
        if not comp:
            return None, None

        if self.hole_entry_exit_on_boundary_lap:
            return self.select_boundary_lap_entry_exit_keys_for_hole(
                comp,
                current_key,
                robot_xy,
                next_goal_key
            )

        # 只有显式关闭 boundary-lap entry/exit 时，才使用旧的 component 内部兜底逻辑。
        u, v, origin, seg_len = self.get_main_sweep_basis(
            current_key,
            next_goal_key,
            robot_xy,
            comp
        )

        metrics: List[Tuple[float, float, float, NodeKey]] = []
        min_lateral = float('inf')

        for key in comp:
            if key not in self.nodes:
                continue

            x, y = self.nodes[key]
            s_coord, t_coord = self.project_in_sweep_frame(x, y, origin, u, v)
            lateral = abs(t_coord)
            metrics.append((s_coord, lateral, math.hypot(x - robot_xy[0], y - robot_xy[1]), key))
            min_lateral = min(min_lateral, lateral)

        if len(metrics) < 2:
            return None, None

        band_width = max(0.05, self.hole_doorway_band_width)
        doorway: List[Tuple[float, float, float, NodeKey]] = []

        for item in metrics:
            s_coord, lateral, _, _ = item
            in_lateral_band = lateral <= min_lateral + band_width
            in_forward_window = (
                s_coord >= -self.hole_local_backward_extension and
                s_coord <= seg_len + self.hole_local_forward_extension
            )
            if in_lateral_band and in_forward_window:
                doorway.append(item)

        if len(doorway) < 2:
            sorted_by_lateral = sorted(metrics, key=lambda item: (item[1], item[2]))
            keep_n = min(max(2, min(8, len(sorted_by_lateral))), len(sorted_by_lateral))
            doorway = sorted_by_lateral[:keep_n]

        doorway.sort(key=lambda item: item[0])
        return doorway[0][3], doorway[-1][3]

    def nearest_safe_cell_to_xy(
        self,
        xy: Tuple[float, float],
        safe: np.ndarray,
        max_radius_m: Optional[float] = None
    ) -> Optional[GridCell]:
        raw = self.world_to_cell(xy[0], xy[1])
        if raw is None:
            return None

        if max_radius_m is None:
            return self.find_nearest_safe_cell(raw, safe)

        assert self.free_msg is not None
        row, col = raw
        h, w = safe.shape
        res = self.free_msg.info.resolution
        max_rad = max(1, int(math.ceil(max_radius_m / res)))

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

    def build_hole_repair_region_mask(
        self,
        comp: Set[NodeKey]
    ) -> Optional[np.ndarray]:
        """
        构造 hole 内部补扫区域 mask。

        不是简单使用 bbox 内所有 free，而是：
        1. 用 hole_sample_safe_mask 作为候选 free/uncovered/safe 区域；
        2. 从当前 hole component 的 RCG 节点附近作为 seed；
        3. 在 bbox 范围内 floodfill，得到与 hole component 相连的局部区域。

        这样既能在 hole 内做加密采样，又不容易把远处区域一起扫进去。
        """
        sample_safe = self.build_hole_sample_safe_mask()
        if sample_safe is None or self.free_msg is None:
            return None

        bbox = self.component_bbox_cells_with_margin(
            comp,
            max(self.hole_sample_margin, self.hole_repair_region_margin)
        )
        if bbox is None:
            return None

        r_min, r_max, c_min, c_max = bbox
        h, w = sample_safe.shape

        bbox_gate = np.zeros_like(sample_safe, dtype=bool)
        bbox_gate[r_min:r_max + 1, c_min:c_max + 1] = True

        seeds: List[GridCell] = []
        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            raw = self.world_to_cell(x, y)
            if raw is None:
                continue

            rr, cc = raw
            if 0 <= rr < h and 0 <= cc < w and sample_safe[rr, cc] and bbox_gate[rr, cc]:
                seeds.append((rr, cc))
                continue

            near = self.find_nearest_safe_cell(raw, sample_safe & bbox_gate)
            if near is not None:
                seeds.append(near)

        if not seeds:
            return sample_safe & bbox_gate

        region = np.zeros_like(sample_safe, dtype=bool)
        visited = np.zeros_like(sample_safe, dtype=bool)
        q = deque()

        for cell in seeds:
            r, c = cell
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if not sample_safe[r, c] or not bbox_gate[r, c]:
                continue
            if visited[r, c]:
                continue
            visited[r, c] = True
            q.append((r, c))

        while q:
            r, c = q.popleft()
            region[r, c] = True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if visited[nr, nc]:
                    continue
                if not bbox_gate[nr, nc]:
                    continue
                if not sample_safe[nr, nc]:
                    continue

                # 斜向不切角
                if dr != 0 and dc != 0:
                    if not sample_safe[r + dr, c] or not sample_safe[r, c + dc]:
                        continue

                visited[nr, nc] = True
                q.append((nr, nc))

        if int(np.count_nonzero(region)) == 0:
            return sample_safe & bbox_gate

        return region

    def a_star_grid_path_between_cells(
        self,
        start: GridCell,
        goal: GridCell,
        safe: np.ndarray
    ) -> List[GridCell]:
        if start == goal:
            return [start]

        h, w = safe.shape
        sr, sc = start
        gr, gc = goal

        if sr < 0 or sr >= h or sc < 0 or sc >= w:
            return []
        if gr < 0 or gr >= h or gc < 0 or gc >= w:
            return []
        if not safe[sr, sc] or not safe[gr, gc]:
            return []

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        ]

        open_heap: List[Tuple[float, float, GridCell]] = []
        heapq.heappush(open_heap, (math.hypot(gr - sr, gc - sc), 0.0, start))

        g_score: Dict[GridCell, float] = {start: 0.0}
        prev: Dict[GridCell, Optional[GridCell]] = {start: None}
        visited: Set[GridCell] = set()
        expansions = 0
        max_expansions = max(1000, self.hole_repair_astar_max_expansions)

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            expansions += 1

            if expansions > max_expansions:
                return []

            if current == goal:
                path: List[GridCell] = []
                cur: Optional[GridCell] = current
                while cur is not None:
                    path.append(cur)
                    cur = prev.get(cur)
                path.reverse()
                return path

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
                h_cost = math.hypot(gr - nr, gc - nc)
                heapq.heappush(open_heap, (tentative_g + h_cost, tentative_g, nb))

        return []

    def a_star_grid_path_between_xy(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        safe: np.ndarray
    ) -> List[Tuple[float, float]]:
        start = self.nearest_safe_cell_to_xy(start_xy, safe, max_radius_m=self.nearest_safe_search_radius)
        goal = self.nearest_safe_cell_to_xy(goal_xy, safe, max_radius_m=self.nearest_safe_search_radius)

        if start is None or goal is None:
            return []

        cells = self.a_star_grid_path_between_cells(start, goal, safe)
        if not cells:
            return []

        simplified = self.simplify_grid_path(cells, safe)
        return [self.cell_to_world(cell) for cell in simplified]

    def xy_to_sweep_xy(
        self,
        s: float,
        t: float,
        origin: Tuple[float, float],
        u: Tuple[float, float],
        v: Tuple[float, float]
    ) -> Tuple[float, float]:
        x = origin[0] + s * u[0] + t * v[0]
        y = origin[1] + s * u[1] + t * v[1]
        return x, y

    def generate_repair_laps_from_region(
        self,
        region_mask: np.ndarray,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float],
        comp: Set[NodeKey],
        entry_xy: Tuple[float, float],
        exit_xy: Tuple[float, float]
    ) -> List[List[Tuple[float, float]]]:
        """
        在 hole 内部生成“垂直于正常 C* 主方向”的局部牛耕 laps。

        返回值是若干条 lap，每条 lap 是一串 xy 点；后续会用 A* 连接
        entry -> lap_0 -> lap_1 -> ... -> exit。
        """
        if self.free_msg is None:
            return []

        cells = np.argwhere(region_mask)
        if cells.size == 0:
            return []

        u, v, origin, _ = self.get_main_sweep_basis(
            current_key,
            next_goal_key,
            robot_xy,
            comp
        )

        projected: List[Tuple[float, float]] = []
        for row, col in cells:
            x, y = self.cell_to_world((int(row), int(col)))
            s, t = self.project_in_sweep_frame(x, y, origin, u, v)
            projected.append((s, t))

        if not projected:
            return []

        min_s = min(p[0] for p in projected)
        max_s = max(p[0] for p in projected)
        min_t = min(p[1] for p in projected)
        max_t = max(p[1] for p in projected)

        lap_spacing = max(0.08, self.hole_repair_lap_spacing)
        sample_spacing = max(0.05, self.hole_repair_sample_spacing)
        min_lap_len = max(0.10, self.hole_repair_min_lap_length)

        offsets = [0.0, 0.25 * lap_spacing, -0.25 * lap_spacing,
                   0.50 * lap_spacing, -0.50 * lap_spacing]

        entry_s, entry_t = self.project_in_sweep_frame(entry_xy[0], entry_xy[1], origin, u, v)
        exit_s, exit_t = self.project_in_sweep_frame(exit_xy[0], exit_xy[1], origin, u, v)

        best_laps: List[List[Tuple[float, float]]] = []
        best_score = float('inf')

        for offset in offsets:
            s0 = min_s + offset
            while s0 > min_s:
                s0 -= lap_spacing

            s_values: List[float] = []
            s = s0
            while s <= max_s + 1e-6:
                if s >= min_s - 1e-6:
                    s_values.append(s)
                s += lap_spacing

            raw_laps: List[List[Tuple[float, float]]] = []

            for s_line in s_values:
                t_values: List[float] = []
                t = min_t
                while t <= max_t + 1e-6:
                    t_values.append(t)
                    t += sample_spacing
                if t_values and abs(t_values[-1] - max_t) > 1e-6:
                    t_values.append(max_t)

                runs: List[List[Tuple[float, float]]] = []
                current_run: List[Tuple[float, float]] = []
                last_cell: Optional[GridCell] = None

                for t_line in t_values:
                    x, y = self.xy_to_sweep_xy(s_line, t_line, origin, u, v)
                    cell = self.world_to_cell(x, y)
                    valid = False

                    if cell is not None:
                        rr, cc = cell
                        if 0 <= rr < region_mask.shape[0] and 0 <= cc < region_mask.shape[1]:
                            valid = bool(region_mask[rr, cc])

                    if valid:
                        if cell != last_cell:
                            wx, wy = self.cell_to_world(cell)
                            current_run.append((wx, wy))
                            last_cell = cell
                    else:
                        if current_run:
                            runs.append(current_run)
                            current_run = []
                            last_cell = None

                if current_run:
                    runs.append(current_run)

                for run in runs:
                    if len(run) < 2:
                        continue
                    length = 0.0
                    for a, b in zip(run[:-1], run[1:]):
                        length += math.hypot(b[0] - a[0], b[1] - a[1])
                    if length >= min_lap_len:
                        raw_laps.append(run)

            if not raw_laps:
                continue

            raw_laps.sort(
                key=lambda lap: sum(self.project_in_sweep_frame(x, y, origin, u, v)[0] for x, y in lap) / float(len(lap))
            )

            variants: List[List[List[Tuple[float, float]]]] = [raw_laps]
            if len(raw_laps) > 2:
                variants.append(raw_laps[1:])
                variants.append(raw_laps[:-1])
                # 删除最短的一条，作为奇偶/边界兜底。
                lengths = []
                for i, lap in enumerate(raw_laps):
                    lap_len = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(lap[:-1], lap[1:]))
                    lengths.append((lap_len, i))
                _, shortest_i = min(lengths, key=lambda item: item[0])
                variants.append([lap for i, lap in enumerate(raw_laps) if i != shortest_i])

            for laps0 in variants:
                laps = list(laps0)
                if len(laps) < 1:
                    continue

                # 若 entry/exit 都在同一侧，偶数条 lap 更容易从另一端正常出 hole。
                parity_penalty = 0.0
                if self.hole_repair_force_even_laps and len(laps) % 2 != 0:
                    parity_penalty = 5.0

                # 根据 entry_s -> exit_s 决定横向扫描顺序。
                if exit_s < entry_s:
                    laps = list(reversed(laps))

                for first_forward in [True, False]:
                    ordered_laps: List[List[Tuple[float, float]]] = []
                    forward = first_forward

                    for lap in laps:
                        if forward:
                            ordered_laps.append(lap)
                        else:
                            ordered_laps.append(list(reversed(lap)))
                        forward = not forward

                    first = ordered_laps[0][0]
                    last = ordered_laps[-1][-1]
                    score = parity_penalty
                    score += math.hypot(first[0] - entry_xy[0], first[1] - entry_xy[1])
                    score += math.hypot(last[0] - exit_xy[0], last[1] - exit_xy[1])

                    for a_lap, b_lap in zip(ordered_laps[:-1], ordered_laps[1:]):
                        a = a_lap[-1]
                        b = b_lap[0]
                        score += 0.7 * math.hypot(b[0] - a[0], b[1] - a[1])

                    # 少量偏好更多 lap 覆盖，但不要压过入口/出口合理性。
                    score -= 0.02 * len(ordered_laps)

                    if score < best_score:
                        best_score = score
                        best_laps = ordered_laps

        return best_laps

    def append_connector_and_points(
        self,
        path: List[Tuple[float, float]],
        target_points: List[Tuple[float, float]],
        travel_safe: np.ndarray
    ) -> None:
        if not target_points:
            return

        if not path:
            path.extend(target_points)
            return

        start_xy = path[-1]
        first_xy = target_points[0]
        direct_dist = math.hypot(first_xy[0] - start_xy[0], first_xy[1] - start_xy[1])

        connector = self.a_star_grid_path_between_xy(start_xy, first_xy, travel_safe)

        if connector:
            # 如果 A* 连接过长，通常说明不是同一个局部 hole，避免画出夸张绕路。
            conn_len = 0.0
            for a, b in zip(connector[:-1], connector[1:]):
                conn_len += math.hypot(b[0] - a[0], b[1] - a[1])
            if conn_len <= max(self.hole_repair_max_connector_length, direct_dist * 4.0):
                if math.hypot(connector[0][0] - path[-1][0], connector[0][1] - path[-1][1]) < 1e-6:
                    path.extend(connector[1:])
                else:
                    path.extend(connector)
            elif direct_dist <= 0.25:
                path.append(first_xy)
        elif direct_dist <= 0.25:
            path.append(first_xy)

        if path and target_points:
            if math.hypot(path[-1][0] - target_points[0][0], path[-1][1] - target_points[0][1]) < 1e-6:
                path.extend(target_points[1:])
            else:
                path.extend(target_points)

    def build_hole_repair_path(
        self,
        current_key: NodeKey,
        robot_xy: Tuple[float, float],
        next_goal_key: Optional[NodeKey]
    ) -> None:
        """
        Branch-seed constrained orthogonal hole repair path。

        新逻辑：
        1. hole detection 仍使用当前稳定的 RCG-based floodfill；
        2. entry 优先使用 branch seed，exit 固定在 boundary_lap 上；
        3. hole 内部不再用普通 repair/最近邻，也不只用稀疏 RCG nodes；
        4. 在 hole 内部按“垂直于当前 C* 主方向”的平行 lap 加密重采样；
        5. 自动尝试不同 offset 和奇偶 lap 数，优先生成从 entry 进、从 exit 出的牛耕路径；
        6. lap 之间、entry/exit 连接使用 grid A*，避免跨越墙壁或出现长距离直线。
        """
        self.latest_hole_repair_path = []
        self.latest_hole_samples = []

        previous_center = self.active_hole_center
        previous_entry_key = self.active_hole_entry_key
        previous_exit_key = self.active_hole_exit_key

        if not self.enable_hole_repair_path:
            self.active_hole_center = None
            self.active_hole_entry = None
            self.active_hole_exit = None
            return

        active_i = self.choose_active_hole_index(robot_xy)
        if active_i is None:
            self.active_hole_center = None
            self.active_hole_entry = None
            self.active_hole_exit = None
            return

        if active_i >= len(self.latest_hole_components) or active_i >= len(self.latest_hole_infos):
            return

        comp = set(self.latest_hole_components[active_i])
        comp = {key for key in comp if key in self.nodes and not self.is_closed_key(key)}

        if len(comp) < 2:
            self.active_hole_center = None
            self.active_hole_entry = None
            self.active_hole_exit = None
            return

        cx, cy = self.latest_hole_infos[active_i][0], self.latest_hole_infos[active_i][1]
        new_center = (cx, cy)

        # 如果仍然是同一个 active hole，则保持最初的 boundary-lap entry/exit 不变。
        # 后续 hole 内部 unknown area 被补全时，只动态更新 repair laps / repair path，
        # 不再让 start/end 跑进 hole 内部。
        use_locked_entry_exit = False
        if (
            self.hole_entry_exit_lock_enable and
            previous_center is not None and
            previous_entry_key in self.nodes and
            previous_exit_key in self.nodes
        ):
            center_shift = math.hypot(new_center[0] - previous_center[0], new_center[1] - previous_center[1])
            if center_shift <= self.hole_active_match_distance:
                use_locked_entry_exit = True

        pending_entry_key = self.pending_hole_entry_key
        pending_exit_key = self.pending_hole_exit_key

        if use_locked_entry_exit:
            entry_key = previous_entry_key
            exit_key = previous_exit_key
        elif pending_entry_key in self.nodes and pending_exit_key in self.nodes:
            # branch seed 当前帧给出的 entry/exit 优先级最高。
            # 初次检测时用它们锁定 start/end；同一 active hole 后续动态更新时不再漂移。
            entry_key = pending_entry_key
            exit_key = pending_exit_key
        else:
            entry_key, exit_key = self.select_entry_exit_keys_for_hole(
                comp,
                current_key,
                robot_xy,
                next_goal_key
            )

        self.active_hole_center = new_center

        if entry_key is None or exit_key is None:
            self.active_hole_entry_key = None
            self.active_hole_exit_key = None
            self.active_hole_entry = None
            self.active_hole_exit = None
            return

        self.active_hole_entry_key = entry_key
        self.active_hole_exit_key = exit_key
        self.active_hole_entry = self.nodes[entry_key]
        self.active_hole_exit = self.nodes[exit_key]

        region_mask = self.build_hole_repair_region_mask(comp)
        travel_safe = self.build_safe_free_mask()

        if region_mask is None or travel_safe is None:
            return

        entry_xy = self.nodes[entry_key]
        exit_xy = self.nodes[exit_key]

        laps = self.generate_repair_laps_from_region(
            region_mask,
            current_key,
            next_goal_key,
            robot_xy,
            comp,
            entry_xy,
            exit_xy
        )

        # 可视化用：显示加密后的 hole repair samples。
        samples: List[Tuple[float, float]] = []
        for lap in laps:
            samples.extend(lap)
        self.latest_hole_samples = samples[:max(1, self.hole_sample_max_points_per_hole)]

        if not laps:
            return

        repair_path: List[Tuple[float, float]] = [entry_xy]

        for lap in laps:
            self.append_connector_and_points(repair_path, lap, travel_safe)
            if len(repair_path) >= self.hole_repair_max_path_points:
                repair_path = repair_path[:self.hole_repair_max_path_points]
                break

        if len(repair_path) < self.hole_repair_max_path_points:
            self.append_connector_and_points(repair_path, [exit_xy], travel_safe)

        # 去掉连续重复点，避免 Path/Marker 过密。
        compact: List[Tuple[float, float]] = []
        for p in repair_path:
            if not compact:
                compact.append(p)
                continue
            if math.hypot(p[0] - compact[-1][0], p[1] - compact[-1][1]) >= 0.015:
                compact.append(p)

        self.latest_hole_repair_path = compact[:max(2, self.hole_repair_max_path_points)]


    def publish_hole_entry_exit_markers(self) -> None:
        entry_ma = MarkerArray()
        exit_ma = MarkerArray()

        delete_entry = Marker()
        delete_entry.header.stamp = self.get_clock().now().to_msg()
        delete_entry.header.frame_id = self.map_frame
        delete_entry.action = Marker.DELETEALL
        entry_ma.markers.append(delete_entry)

        delete_exit = Marker()
        delete_exit.header.stamp = self.get_clock().now().to_msg()
        delete_exit.header.frame_id = self.map_frame
        delete_exit.action = Marker.DELETEALL
        exit_ma.markers.append(delete_exit)

        if self.active_hole_entry is not None:
            x, y = self.active_hole_entry

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.map_frame
            marker.ns = 'hole_entry'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.23
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.10
            marker.scale.y = 0.10
            marker.scale.z = 0.10
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.95
            entry_ma.markers.append(marker)

            text = Marker()
            text.header = marker.header
            text.ns = 'hole_entry_label'
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.40
            text.pose.orientation.w = 1.0
            text.scale.z = 0.13
            text.color.r = 0.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = 'entry'
            entry_ma.markers.append(text)

        if self.active_hole_exit is not None:
            x, y = self.active_hole_exit

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = self.map_frame
            marker.ns = 'hole_exit'
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.23
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.10
            marker.scale.y = 0.10
            marker.scale.z = 0.10
            marker.color.r = 0.2
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 0.95
            exit_ma.markers.append(marker)

            text = Marker()
            text.header = marker.header
            text.ns = 'hole_exit_label'
            text.id = 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.40
            text.pose.orientation.w = 1.0
            text.scale.z = 0.13
            text.color.r = 0.2
            text.color.g = 1.0
            text.color.b = 0.2
            text.color.a = 1.0
            text.text = 'exit'
            exit_ma.markers.append(text)

        self.hole_entry_marker_pub.publish(entry_ma)
        self.hole_exit_marker_pub.publish(exit_ma)

    def publish_hole_repair_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame

        for x, y in self.latest_hole_repair_path:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.18
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.hole_repair_path_pub.publish(path)


    def publish_hole_repair_markers(self) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        line = Marker()
        line.header.stamp = self.get_clock().now().to_msg()
        line.header.frame_id = self.map_frame
        line.ns = 'hole_repair_path'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.035
        line.color.r = 1.0
        line.color.g = 0.35
        line.color.b = 0.0
        line.color.a = 0.95
        line.pose.orientation.w = 1.0

        for x, y in self.latest_hole_repair_path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.20
            line.points.append(p)

        ma.markers.append(line)
        self.hole_repair_markers_pub.publish(ma)


    def publish_hole_nodes(self) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for comp in self.latest_hole_components:
            for key in comp:
                if key not in self.nodes:
                    continue

                x, y = self.nodes[key]
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = 0.09
                pose.orientation.w = 1.0
                msg.poses.append(pose)

        self.hole_nodes_pub.publish(msg)

    def publish_hole_markers(self) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        hole_points = Marker()
        hole_points.header.stamp = self.get_clock().now().to_msg()
        hole_points.header.frame_id = self.map_frame
        hole_points.ns = 'hole_nodes'
        hole_points.id = 0
        hole_points.type = Marker.POINTS
        hole_points.action = Marker.ADD
        hole_points.scale.x = 0.10
        hole_points.scale.y = 0.10
        hole_points.color.r = 1.0
        hole_points.color.g = 0.0
        hole_points.color.b = 1.0
        hole_points.color.a = 0.90
        hole_points.pose.orientation.w = 1.0

        centers = Marker()
        centers.header.stamp = self.get_clock().now().to_msg()
        centers.header.frame_id = self.map_frame
        centers.ns = 'hole_centers'
        centers.id = 1
        centers.type = Marker.SPHERE_LIST
        centers.action = Marker.ADD
        centers.scale.x = 0.18
        centers.scale.y = 0.18
        centers.scale.z = 0.18
        centers.color.r = 1.0
        centers.color.g = 0.0
        centers.color.b = 1.0
        centers.color.a = 0.95
        centers.pose.orientation.w = 1.0

        for comp in self.latest_hole_components:
            for key in comp:
                if key not in self.nodes:
                    continue

                x, y = self.nodes[key]
                p = Point()
                p.x = x
                p.y = y
                p.z = 0.12
                hole_points.points.append(p)

        for i, (cx, cy, node_count, bbox_area, unknown_area, unknown_ratio, label) in enumerate(
            self.latest_hole_infos
        ):
            p = Point()
            p.x = cx
            p.y = cy
            p.z = 0.16
            centers.points.append(p)

            text = Marker()
            text.header.stamp = self.get_clock().now().to_msg()
            text.header.frame_id = self.map_frame
            text.ns = 'hole_labels'
            text.id = 100 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cx
            text.pose.position.y = cy
            text.pose.position.z = 0.35
            text.pose.orientation.w = 1.0
            text.scale.z = 0.16
            text.color.r = 1.0
            text.color.g = 0.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                f'{label}\n'
                f'N={node_count}\n'
                f'A={bbox_area:.2f}\n'
                f'U={unknown_area:.2f}/{unknown_ratio:.2f}'
            )
            ma.markers.append(text)

        ma.markers.append(hole_points)
        ma.markers.append(centers)

        self.hole_markers_pub.publish(ma)


    # ==========================
    # Hole internal dense sampling
    # ==========================

    def get_covered_bool_array(self) -> Optional[np.ndarray]:
        if self.covered_map is None or self.covered_data is None:
            return None

        h = self.covered_map.info.height
        w = self.covered_map.info.width

        if len(self.covered_data) != h * w:
            return None

        arr = np.asarray(self.covered_data, dtype=np.int16).reshape((h, w))
        return arr >= self.covered_close_threshold

    def build_hole_sample_safe_mask(self) -> Optional[np.ndarray]:
        """
        生成 hole 内部采样用的安全栅格：
        free - obstacle_buffer - unknown_buffer - covered(optional)

        这一步比 RCG 主图采样更偏向“补扫”：
        已经 covered 的地方默认不再采样，减少重复清扫。
        """
        if self.free_msg is None or self.free_arr is None:
            return None

        info = self.free_msg.info
        res = info.resolution

        free = self.free_arr.copy()

        if self.obstacle_arr is not None and self.obstacle_arr.shape == free.shape:
            obstacle = self.obstacle_arr.copy()
        else:
            obstacle = np.logical_not(free)

        if self.unknown_arr is not None and self.unknown_arr.shape == free.shape:
            unknown = self.unknown_arr.copy()
        else:
            unknown = np.zeros_like(free, dtype=bool)

        obstacle_rad = max(0, int(math.ceil(self.hole_sample_obstacle_buffer / res)))
        unknown_rad = max(0, int(math.ceil(self.hole_sample_unknown_buffer / res)))

        obstacle_block = self.dilate_bool(obstacle, obstacle_rad)
        unknown_block = self.dilate_bool(unknown, unknown_rad)

        safe = free.copy()
        safe[obstacle_block] = False
        safe[unknown_block] = False

        if self.hole_sample_exclude_covered:
            covered = self.get_covered_bool_array()
            if covered is not None and covered.shape == safe.shape:
                safe[covered] = False

        return safe

    def component_bbox_cells_with_margin(
        self,
        comp: Set[NodeKey],
        margin: float
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        返回带 margin 的 bbox cell 范围：r_min, r_max, c_min, c_max。
        """
        if self.free_msg is None or not comp:
            return None

        xs: List[float] = []
        ys: List[float] = []

        for key in comp:
            if key not in self.nodes:
                continue
            x, y = self.nodes[key]
            xs.append(x)
            ys.append(y)

        if not xs or not ys:
            return None

        min_x = min(xs) - margin
        max_x = max(xs) + margin
        min_y = min(ys) - margin
        max_y = max(ys) + margin

        cell_min = self.world_to_cell(min_x, min_y)
        cell_max = self.world_to_cell(max_x, max_y)

        if cell_min is None or cell_max is None:
            info = self.free_msg.info
            res = info.resolution
            h = info.height
            w = info.width

            c_min = int((min_x - info.origin.position.x) / res)
            c_max = int((max_x - info.origin.position.x) / res)
            r_min = int((min_y - info.origin.position.y) / res)
            r_max = int((max_y - info.origin.position.y) / res)

            r_min = max(0, min(h - 1, r_min))
            r_max = max(0, min(h - 1, r_max))
            c_min = max(0, min(w - 1, c_min))
            c_max = max(0, min(w - 1, c_max))
        else:
            r_min, c_min = cell_min
            r_max, c_max = cell_max

        if r_min > r_max:
            r_min, r_max = r_max, r_min
        if c_min > c_max:
            c_min, c_max = c_max, c_min

        return r_min, r_max, c_min, c_max

    def generate_hole_samples_for_component(
        self,
        comp: Set[NodeKey]
    ) -> List[Tuple[float, float]]:
        if not self.enable_hole_sampling:
            return []

        safe = self.build_hole_sample_safe_mask()
        if safe is None or self.free_msg is None:
            return []

        res = self.free_msg.info.resolution
        row_step = max(1, int(round(self.hole_sample_lap_spacing / res)))
        col_step = max(1, int(round(self.hole_sample_spacing / res)))
        max_per_hole = max(1, self.hole_sample_max_points_per_hole)

        bbox = self.component_bbox_cells_with_margin(comp, self.hole_sample_margin)
        if bbox is None:
            return []

        r_min, r_max, c_min, c_max = bbox
        samples: List[Tuple[float, float]] = []
        used_cells: Set[GridCell] = set()

        rows = list(range(r_min, r_max + 1, row_step))
        if rows and rows[-1] != r_max:
            rows.append(r_max)

        reverse = False
        for row in rows:
            cols = list(range(c_min, c_max + 1, col_step))
            if cols and cols[-1] != c_max:
                cols.append(c_max)

            if reverse:
                cols = list(reversed(cols))
            reverse = not reverse

            for col in cols:
                if row < 0 or col < 0 or row >= safe.shape[0] or col >= safe.shape[1]:
                    continue

                if not safe[row, col]:
                    continue

                cell = (row, col)
                if cell in used_cells:
                    continue

                x, y = self.cell_to_world(cell)
                used_cells.add(cell)
                samples.append((x, y))

                if len(samples) >= max_per_hole:
                    break

            if len(samples) >= max_per_hole:
                break

        return samples

    def generate_hole_samples(self) -> List[Tuple[float, float]]:
        """
        对当前检测到的 hole component 生成局部加密采样点。
        该结果不固定，每次 hole detection 都会基于最新地图重算。
        """
        if not self.enable_hole_sampling:
            return []

        if not self.latest_hole_components:
            return []

        all_samples: List[Tuple[float, float]] = []
        used_keys: Set[NodeKey] = set()

        for comp in self.latest_hole_components:
            comp_samples = self.generate_hole_samples_for_component(comp)

            for x, y in comp_samples:
                key = self.make_key(x, y)
                if key in used_keys:
                    continue
                used_keys.add(key)
                all_samples.append((x, y))

        return all_samples

    def publish_hole_samples(self) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for x, y in self.latest_hole_samples:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.10
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.hole_samples_pub.publish(msg)

    def publish_hole_sample_markers(self) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        samples = Marker()
        samples.header.stamp = self.get_clock().now().to_msg()
        samples.header.frame_id = self.map_frame
        samples.ns = 'hole_samples'
        samples.id = 0
        samples.type = Marker.POINTS
        samples.action = Marker.ADD
        samples.scale.x = 0.075
        samples.scale.y = 0.075
        samples.color.r = 1.0
        samples.color.g = 0.55
        samples.color.b = 0.0
        samples.color.a = 0.92
        samples.pose.orientation.w = 1.0

        for x, y in self.latest_hole_samples:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.15
            samples.points.append(p)

        ma.markers.append(samples)
        self.hole_sample_markers_pub.publish(ma)

    # ==========================
    # Publish common visualizations
    # ==========================

    def publish_goal(self, goal_key: NodeKey) -> None:
        if goal_key not in self.nodes:
            return

        gx, gy = self.nodes[goal_key]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        marker = Marker()
        marker.header = msg.header
        marker.ns = 'cstar_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = msg.pose
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18

        if self.escape_active:
            marker.color.r = 1.0
            marker.color.g = 0.3
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        self.goal_marker_pub.publish(marker)

    def publish_selected_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame

        for x, y in self.selected_path:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.03
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)

    def publish_escape_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame

        if self.escape_active:
            for x, y in self.escape_path_xy:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.08
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)

        self.escape_path_pub.publish(path)

    def publish_retreat_nodes(self) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
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

    def publish_open_closed_markers(self) -> None:
        marker_array = MarkerArray()

        open_marker = Marker()
        open_marker.header.stamp = self.get_clock().now().to_msg()
        open_marker.header.frame_id = self.map_frame
        open_marker.ns = 'open_nodes'
        open_marker.id = 0
        open_marker.type = Marker.POINTS
        open_marker.action = Marker.ADD
        open_marker.scale.x = 0.06
        open_marker.scale.y = 0.06
        open_marker.color.r = 0.0
        open_marker.color.g = 0.4
        open_marker.color.b = 1.0
        open_marker.color.a = 0.7

        closed_marker = Marker()
        closed_marker.header = open_marker.header
        closed_marker.ns = 'closed_nodes'
        closed_marker.id = 1
        closed_marker.type = Marker.POINTS
        closed_marker.action = Marker.ADD
        closed_marker.scale.x = 0.07
        closed_marker.scale.y = 0.07
        closed_marker.color.r = 1.0
        closed_marker.color.g = 0.0
        closed_marker.color.b = 0.0
        closed_marker.color.a = 0.85

        for key, (x, y) in self.nodes.items():
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.05

            if self.is_closed_key(key):
                closed_marker.points.append(p)
            else:
                open_marker.points.append(p)

        marker_array.markers.append(open_marker)
        marker_array.markers.append(closed_marker)

        self.state_marker_pub.publish(marker_array)

    def set_new_goal(self, goal_key: NodeKey, reason: str) -> None:
        self.current_goal_key = goal_key

        if goal_key in self.nodes:
            gx, gy = self.nodes[goal_key]
            self.selected_path.append((gx, gy))

            self.get_logger().info(
                f'{reason}: new goal=({gx:.2f}, {gy:.2f}), '
                f'sweep_dir={self.sweep_dir}, '
                f'escape_active={self.escape_active}, '
                f'closed_positions={len(self.closed_positions)}, '
                f'total_nodes={len(self.nodes)}'
            )

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

        start_x, start_y = self.nodes[current_key]
        goal_x, goal_y = self.nodes[target_key]

        self.get_logger().warn(
            f'Dead-end detected at ({start_x:.2f}, {start_y:.2f}). '
            f'Grid A* escape path points={len(xy_path)}, '
            f'retreat=({goal_x:.2f}, {goal_y:.2f}), '
            f'retreat_candidates={len(self.latest_retreat_candidates)}, '
            f'fallback={used_fallback}'
        )

        return True

    def finish_escape(self, current_key: NodeKey) -> None:
        self.escape_active = False
        self.escape_path_xy.clear()

        if current_key in self.nodes:
            x, y = self.nodes[current_key]
            self.get_logger().info(
                f'Grid A* escape finished near ({x:.2f}, {y:.2f}). Resume normal C* coverage.'
            )

    def choose_anchor_key(
        self,
        robot_xy: Tuple[float, float],
        nearest_key: Optional[NodeKey],
        reached_goal: bool
    ) -> Optional[NodeKey]:
        # 关键：如果已经到达当前 goal，则以 current_goal_key 作为 anchor，
        # 不再单纯使用机器人当前位置最近点。这样可以减少误判 dead-end。
        if reached_goal and self.current_goal_key is not None and self.current_goal_key in self.nodes:
            return self.current_goal_key

        return nearest_key

    def clear_hole_visual_state(self) -> None:
        self.latest_hole_components = []
        self.latest_hole_infos = []
        self.latest_hole_samples = []
        self.latest_hole_repair_path = []
        self.active_hole_center = None
        self.active_hole_entry = None
        self.active_hole_exit = None
        self.active_hole_entry_key = None
        self.active_hole_exit_key = None
        self.pending_hole_entry_key = None
        self.pending_hole_exit_key = None

    def publish_hole_visuals(self) -> None:
        self.publish_hole_nodes()
        self.publish_hole_markers()
        self.publish_hole_samples()
        self.publish_hole_sample_markers()
        self.publish_hole_entry_exit_markers()
        self.publish_hole_repair_path()
        self.publish_hole_repair_markers()


    def should_update_hole_during_motion(self) -> bool:
        if not self.hole_dynamic_update_during_motion:
            return False

        if not self.hole_detection_armed:
            return False

        if self.escape_active:
            return False

        if self.current_goal_key is None:
            return False

        now = self.get_clock().now()

        if self.last_hole_dynamic_update_time is None:
            self.last_hole_dynamic_update_time = now
            return True

        dt = now - self.last_hole_dynamic_update_time
        if dt.nanoseconds / 1e9 >= max(0.1, self.hole_dynamic_update_period):
            self.last_hole_dynamic_update_time = now
            return True

        return False

    def on_timer(self) -> None:
        if not self.nodes:
            self.clear_hole_visual_state()
            self.publish_hole_visuals()
            return

        robot_xy = self.get_robot_pose()

        if robot_xy is None:
            self.get_logger().warn('Cannot get robot pose in map frame.')
            return

        rx, ry = robot_xy
        self.add_closed_position(rx, ry)

        nearest_key = self.nearest_node_key(rx, ry)

        if nearest_key is not None:
            self.close_key(nearest_key)

        reached_goal = self.is_reached_goal(robot_xy)

        # 刚启动时 current_goal_key 为 None，这一轮只生成第一个 goal，不能做 hole detection。
        reached_from_existing_goal = self.current_goal_key is not None

        # escape 结束这一轮不做 hole detection，避免撤退和 hole 可视化混在一起。
        was_escape_active = self.escape_active

        current_key = self.choose_anchor_key(robot_xy, nearest_key, reached_goal)

        if current_key is None:
            self.publish_open_closed_markers()
            self.publish_selected_path()
            self.publish_escape_path()
            self.publish_retreat_nodes()
            self.publish_hole_visuals()
            return

        self.close_key(current_key)

        # 切换不同 lap 的过渡阶段不做 hole detection / repair。
        # 这个阶段常常位于门口或走廊转接处，容易把正常换行区域误判为 hole。
        current_goal_is_lap_switch = self.is_lap_switch_goal(
            current_key,
            self.current_goal_key
        )

        if current_goal_is_lap_switch:
            self.clear_hole_visual_state()

        # 本版采用“预判 -> 到达后执行”的 branch-trigger 逻辑。
        # 行进途中不再重新触发新的 hole floodfill，避免还没到 boundary_key 就提前生成 repair。
        # 后续如果需要动态更新 repair path，可以只在 latest_hole_components 已存在时单独刷新。

        if reached_goal:
            if self.current_goal_key is not None:
                self.close_key(self.current_goal_key)

            if self.escape_active:
                self.finish_escape(current_key)
                self.clear_hole_visual_state()

            normal_goal = self.choose_next_normal_goal(current_key)

            if normal_goal is not None:
                self.last_deadend_key = None

                # 先锁定正常 C* goal，保证 goal_marker 的选择不被 hole detection 干扰。
                self.set_new_goal(normal_goal, 'C* normal')

                next_goal_is_lap_switch = self.is_lap_switch_goal(current_key, normal_goal)

                # 新触发条件：只要刚选出来的 goal_marker 本身就是 boundary_key，
                # 就立即从对应 branch seed 做 floodfill / repair。
                # 不再缓存到“到达该 goal_marker 后”才执行。
                self.clear_pending_branch_trigger()
                direct_trigger = None
                if not was_escape_active and not next_goal_is_lap_switch:
                    direct_trigger = self.find_branch_trigger_for_goal(
                        current_key,
                        normal_goal,
                        robot_xy
                    )

                if direct_trigger is not None:
                    seed_key, boundary_key, branch_score = direct_trigger
                    self.hole_detection_armed = True
                    self.last_hole_dynamic_update_time = self.get_clock().now()
                    self.detect_hole_from_goal_branch_trigger(
                        current_key,
                        normal_goal,
                        robot_xy,
                        seed_key,
                        boundary_key,
                        branch_score
                    )
                else:
                    self.clear_hole_visual_state()

            else:
                if self.has_any_open_neighbor(current_key):
                    x, y = self.nodes[current_key]
                    self.get_logger().warn(
                        f'Node ({x:.2f}, {y:.2f}) still has open graph neighbors, '
                        f'but Boustrophedon policy rejected them. '
                        f'Consider increasing same_lap_y_tolerance / same_col_x_tolerance '
                        f'or enabling allow_diagonal_fallback.'
                    )

                self.clear_hole_visual_state()

                if not self.start_deadend_escape(current_key, robot_xy):
                    self.current_goal_key = None

                    if self.last_deadend_key != current_key:
                        self.last_deadend_key = current_key
                        x, y = self.nodes[current_key]
                        self.get_logger().warn(
                            f'Dead-end: no grid A* retreat path from ({x:.2f}, {y:.2f}). '
                            f'Robot will stop until map/RCG updates.'
                        )

        if self.current_goal_key is not None:
            self.publish_goal(self.current_goal_key)

        self.publish_open_closed_markers()
        self.publish_selected_path()
        self.publish_escape_path()
        self.publish_retreat_nodes()
        self.publish_hole_visuals()



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


# ------------------------------------------------------------------------------------------

#!/usr/bin/env python3
"""
CStar waypoint planner for the Dense-RCG + Sparse-Backbone framework.

Design:
1. The robot follows only the sparse backbone graph:
   /cstar/rcg_nodes_backbone
   /cstar/rcg_markers_backbone

2. The dense graph is used only for topology analysis and hole detection:
   /cstar/rcg_nodes_dense
   /cstar/rcg_markers_dense

3. Old boundary-lap / doorway-gap / static-branch logic has been removed.
   Branch edges are detected dynamically in the planner from the robot's
   current local backbone window:
   local backbone node -> off-backbone dense seed -> floodfill dense component
   -> classify as hole / frontier / normal-lap.
"""

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

        # ========== Input graph topics ==========
        self.declare_parameter('backbone_nodes_topic', '/cstar/rcg_nodes_backbone')
        self.declare_parameter('backbone_markers_topic', '/cstar/rcg_markers_backbone')
        self.declare_parameter('dense_nodes_topic', '/cstar/rcg_nodes_dense')
        self.declare_parameter('dense_markers_topic', '/cstar/rcg_markers_dense')

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
        self.declare_parameter('closed_position_radius', 0.12)
        self.declare_parameter('covered_close_threshold', 50)
        self.declare_parameter('use_covered_map_for_closing', True)

        # Backbone boustrophedon selection.
        self.declare_parameter('initial_sweep_direction', -1.0)
        self.declare_parameter('same_lap_y_tolerance', 0.14)
        self.declare_parameter('same_col_x_tolerance', 0.16)
        self.declare_parameter('allow_diagonal_fallback', True)

        # When no open neighbor exists, the planner moves along the backbone graph
        # toward the nearest still-open backbone node.
        self.declare_parameter('enable_graph_transit_to_open', True)
        self.declare_parameter('max_graph_transit_hops', 200)

        # ========== Dense off-backbone hole detection ==========
        self.declare_parameter('enable_hole_detection', True)
        self.declare_parameter('hole_scan_lookahead_distance', 1.30)
        self.declare_parameter('hole_scan_backtrack_margin', 0.10)
        self.declare_parameter('hole_dynamic_update_period', 0.5)
        # Dynamic branch detection is local to the robot's current backbone corridor.
        self.declare_parameter('local_backbone_window_lateral_radius', 0.38)
        self.declare_parameter('branch_edge_max_distance', 0.75)
        self.declare_parameter('branch_lateral_min_distance', 0.14)
        self.declare_parameter('branch_lateral_ratio', 1.20)
        self.declare_parameter('branch_seed_min_distance_to_window', 0.12)
        self.declare_parameter('publish_nonhole_branch_candidates', True)

        # Candidate component filters.
        self.declare_parameter('hole_min_nodes', 4)
        self.declare_parameter('hole_max_nodes', 220)
        self.declare_parameter('hole_min_bbox_area', 0.04)
        self.declare_parameter('hole_max_bbox_area', 8.00)
        self.declare_parameter('hole_max_robot_distance', 2.00)

        # If a dense component is very close to the backbone line, it is usually
        # just skipped dense samples between sparse backbone nodes, not a hole.
        self.declare_parameter('normal_lap_distance_to_backbone', 0.13)
        self.declare_parameter('hole_min_max_distance_to_backbone', 0.18)

        # If an off-backbone component has many independent contacts with the
        # backbone, it is more likely a normal dense region than a local hole.
        self.declare_parameter('normal_lap_attachment_threshold', 4)

        # Unknown/frontier rejection.
        self.declare_parameter('unknown_check_radius', 0.25)
        self.declare_parameter('unknown_reject_ratio', 0.35)

        # Floodfill safety cap.
        self.declare_parameter('dense_floodfill_max_nodes', 600)

        # Local repair path visualization over the accepted hole component.
        self.declare_parameter('enable_hole_repair_path', True)
        self.declare_parameter('hole_repair_lap_tolerance', 0.18)
        self.declare_parameter('hole_repair_max_points', 360)

        # ========== Read parameters ==========
        self.backbone_nodes_topic = self.get_parameter('backbone_nodes_topic').value
        self.backbone_markers_topic = self.get_parameter('backbone_markers_topic').value
        self.dense_nodes_topic = self.get_parameter('dense_nodes_topic').value
        self.dense_markers_topic = self.get_parameter('dense_markers_topic').value

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

        self.enable_hole_detection = bool(self.get_parameter('enable_hole_detection').value)
        self.hole_scan_lookahead_distance = float(self.get_parameter('hole_scan_lookahead_distance').value)
        self.hole_scan_backtrack_margin = float(self.get_parameter('hole_scan_backtrack_margin').value)
        self.hole_dynamic_update_period = float(self.get_parameter('hole_dynamic_update_period').value)
        self.local_backbone_window_lateral_radius = float(
            self.get_parameter('local_backbone_window_lateral_radius').value
        )
        self.branch_edge_max_distance = float(self.get_parameter('branch_edge_max_distance').value)
        self.branch_lateral_min_distance = float(
            self.get_parameter('branch_lateral_min_distance').value
        )
        self.branch_lateral_ratio = float(self.get_parameter('branch_lateral_ratio').value)
        self.branch_seed_min_distance_to_window = float(
            self.get_parameter('branch_seed_min_distance_to_window').value
        )
        self.publish_nonhole_branch_candidates = bool(
            self.get_parameter('publish_nonhole_branch_candidates').value
        )

        self.hole_min_nodes = int(self.get_parameter('hole_min_nodes').value)
        self.hole_max_nodes = int(self.get_parameter('hole_max_nodes').value)
        self.hole_min_bbox_area = float(self.get_parameter('hole_min_bbox_area').value)
        self.hole_max_bbox_area = float(self.get_parameter('hole_max_bbox_area').value)
        self.hole_max_robot_distance = float(self.get_parameter('hole_max_robot_distance').value)

        self.normal_lap_distance_to_backbone = float(
            self.get_parameter('normal_lap_distance_to_backbone').value
        )
        self.hole_min_max_distance_to_backbone = float(
            self.get_parameter('hole_min_max_distance_to_backbone').value
        )
        self.normal_lap_attachment_threshold = int(
            self.get_parameter('normal_lap_attachment_threshold').value
        )

        self.unknown_check_radius = float(self.get_parameter('unknown_check_radius').value)
        self.unknown_reject_ratio = float(self.get_parameter('unknown_reject_ratio').value)
        self.dense_floodfill_max_nodes = int(self.get_parameter('dense_floodfill_max_nodes').value)

        self.enable_hole_repair_path = bool(self.get_parameter('enable_hole_repair_path').value)
        self.hole_repair_lap_tolerance = float(self.get_parameter('hole_repair_lap_tolerance').value)
        self.hole_repair_max_points = int(self.get_parameter('hole_repair_max_points').value)

        # ========== Graph state ==========
        self.backbone_nodes: Dict[NodeKey, Tuple[float, float]] = {}
        self.backbone_raw_edges: List[Tuple[NodeKey, NodeKey]] = []
        self.backbone_adj: Dict[NodeKey, Set[NodeKey]] = {}

        self.dense_nodes: Dict[NodeKey, Tuple[float, float]] = {}
        self.dense_raw_edges: List[Tuple[NodeKey, NodeKey]] = []
        self.dense_adj: Dict[NodeKey, Set[NodeKey]] = {}

        # Backbone keys that also exist in dense graph. These are barriers for
        # dense off-backbone floodfill.
        self.backbone_keys_in_dense: Set[NodeKey] = set()

        # ========== Coverage / motion state ==========
        self.closed_backbone_nodes: Set[NodeKey] = set()
        self.closed_positions: List[Tuple[float, float]] = []
        self.current_goal_key: Optional[NodeKey] = None
        self.selected_path: List[Tuple[float, float]] = []

        # Hole visualization state.
        self.latest_hole_component: Set[NodeKey] = set()
        self.latest_hole_attachments: Set[NodeKey] = set()
        self.latest_hole_entry_key: Optional[NodeKey] = None
        self.latest_hole_exit_key: Optional[NodeKey] = None
        self.latest_hole_repair_path: List[Tuple[float, float]] = []
        self.latest_branch_edge: Optional[Tuple[NodeKey, NodeKey]] = None
        self.latest_dynamic_branch_edges: List[Tuple[NodeKey, NodeKey, str]] = []
        self.latest_backbone_window: Set[NodeKey] = set()
        self.last_hole_update_time = None

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
        self.create_subscription(PoseArray, self.backbone_nodes_topic, self.backbone_nodes_callback, 10)
        self.create_subscription(MarkerArray, self.backbone_markers_topic, self.backbone_markers_callback, 10)
        self.create_subscription(PoseArray, self.dense_nodes_topic, self.dense_nodes_callback, 10)
        self.create_subscription(MarkerArray, self.dense_markers_topic, self.dense_markers_callback, 10)

        self.create_subscription(OccupancyGrid, self.covered_map_topic, self.covered_callback, 10)
        self.create_subscription(OccupancyGrid, self.free_map_topic, self.free_callback, 10)
        self.create_subscription(OccupancyGrid, self.obstacle_map_topic, self.obstacle_callback, 10)
        self.create_subscription(OccupancyGrid, self.unknown_map_topic, self.unknown_callback, 10)

        # ========== Publishers ==========
        self.goal_pub = self.create_publisher(PoseStamped, '/cstar/goal', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/cstar/goal_marker', 10)
        self.state_marker_pub = self.create_publisher(MarkerArray, '/cstar/open_closed_markers', 10)
        self.path_pub = self.create_publisher(Path, '/cstar/selected_path', 10)

        self.hole_nodes_pub = self.create_publisher(PoseArray, '/cstar/hole_nodes', 10)
        self.hole_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_markers', 10)
        self.hole_entry_marker_pub = self.create_publisher(MarkerArray, '/cstar/hole_entry_marker', 10)
        self.hole_exit_marker_pub = self.create_publisher(MarkerArray, '/cstar/hole_exit_marker', 10)
        self.hole_repair_path_pub = self.create_publisher(Path, '/cstar/hole_repair_path', 10)
        self.hole_repair_markers_pub = self.create_publisher(MarkerArray, '/cstar/hole_repair_markers', 10)
        self.dynamic_branch_marker_pub = self.create_publisher(
            MarkerArray, '/cstar/dynamic_branch_markers', 10
        )

        self.timer = self.create_timer(self.update_period, self.on_timer)

        self.get_logger().info('CStarWaypointPlannerNode dense/backbone mode started.')
        self.get_logger().info(f'backbone_nodes_topic={self.backbone_nodes_topic}')
        self.get_logger().info(f'backbone_markers_topic={self.backbone_markers_topic}')
        self.get_logger().info(f'dense_nodes_topic={self.dense_nodes_topic}')
        self.get_logger().info(f'dense_markers_topic={self.dense_markers_topic}')
        self.get_logger().info(
            f'hole_detection={self.enable_hole_detection}, '
            f'lookahead={self.hole_scan_lookahead_distance:.2f}, '
            f'lateral_radius={self.local_backbone_window_lateral_radius:.2f}, '
            f'branch_lateral_min={self.branch_lateral_min_distance:.2f}, '
            f'min_nodes={self.hole_min_nodes}, max_nodes={self.hole_max_nodes}'
        )

    # ------------------------------------------------------------------
    # Basic graph / map callbacks
    # ------------------------------------------------------------------
    def make_key(self, x: float, y: float) -> NodeKey:
        q = self.position_quantization
        return int(round(x / q)), int(round(y / q))

    def backbone_nodes_callback(self, msg: PoseArray) -> None:
        nodes: Dict[NodeKey, Tuple[float, float]] = {}
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            nodes[self.make_key(x, y)] = (x, y)

        self.backbone_nodes = nodes
        self.rebuild_backbone_adjacency()
        self.refresh_backbone_keys_in_dense()

        if self.current_goal_key is not None and self.current_goal_key not in self.backbone_nodes:
            self.current_goal_key = None
            self.selected_path.clear()

    def backbone_markers_callback(self, msg: MarkerArray) -> None:
        self.backbone_raw_edges = self.extract_edges_from_marker_array(msg)
        self.rebuild_backbone_adjacency()

    def dense_nodes_callback(self, msg: PoseArray) -> None:
        nodes: Dict[NodeKey, Tuple[float, float]] = {}
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            nodes[self.make_key(x, y)] = (x, y)

        self.dense_nodes = nodes
        self.rebuild_dense_adjacency()
        self.refresh_backbone_keys_in_dense()

    def dense_markers_callback(self, msg: MarkerArray) -> None:
        self.dense_raw_edges = self.extract_edges_from_marker_array(msg)
        self.rebuild_dense_adjacency()

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

    def rebuild_backbone_adjacency(self) -> None:
        adj: Dict[NodeKey, Set[NodeKey]] = {key: set() for key in self.backbone_nodes.keys()}

        for k1, k2 in self.backbone_raw_edges:
            if k1 not in self.backbone_nodes or k2 not in self.backbone_nodes:
                continue
            adj[k1].add(k2)
            adj[k2].add(k1)

        self.backbone_adj = adj

    def rebuild_dense_adjacency(self) -> None:
        adj: Dict[NodeKey, Set[NodeKey]] = {key: set() for key in self.dense_nodes.keys()}

        for k1, k2 in self.dense_raw_edges:
            if k1 not in self.dense_nodes or k2 not in self.dense_nodes:
                continue
            adj[k1].add(k2)
            adj[k2].add(k1)

        self.dense_adj = adj

    def refresh_backbone_keys_in_dense(self) -> None:
        self.backbone_keys_in_dense = {
            key for key in self.backbone_nodes.keys()
            if key in self.dense_nodes
        }

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

    def is_near_closed_position(
        self,
        x: float,
        y: float,
        radius: Optional[float] = None
    ) -> bool:
        r = self.closed_position_radius if radius is None else radius
        r2 = r * r

        for cx, cy in self.closed_positions:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= r2:
                return True

        return False

    def is_closed_backbone_key(self, key: NodeKey) -> bool:
        if key in self.closed_backbone_nodes:
            return True

        if key not in self.backbone_nodes:
            return False

        x, y = self.backbone_nodes[key]
        return self.is_inside_covered_map(x, y) or self.is_near_closed_position(x, y)

    def is_closed_dense_key(self, key: NodeKey) -> bool:
        if key not in self.dense_nodes:
            return False

        x, y = self.dense_nodes[key]
        return self.is_inside_covered_map(x, y) or self.is_near_closed_position(x, y)

    def close_backbone_key(self, key: NodeKey) -> None:
        if key not in self.backbone_nodes:
            return

        self.closed_backbone_nodes.add(key)
        x, y = self.backbone_nodes[key]
        self.add_closed_position(x, y)

    def nearest_backbone_key(self, x: float, y: float) -> Optional[NodeKey]:
        best_key = None
        best_dist = float('inf')

        for key, pos in self.backbone_nodes.items():
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

        if self.current_goal_key not in self.backbone_nodes:
            return True

        gx, gy = self.backbone_nodes[self.current_goal_key]
        return math.hypot(robot_xy[0] - gx, robot_xy[1] - gy) <= self.goal_center_tolerance

    # ------------------------------------------------------------------
    # Backbone coverage goal selection
    # ------------------------------------------------------------------
    def classify_open_backbone_neighbors(
        self,
        current_key: NodeKey
    ) -> Dict[str, List[Tuple[float, NodeKey]]]:
        result = {
            'same_forward': [],
            'same_backward': [],
            'left': [],
            'up': [],
            'down': [],
            'right': [],
            'diagonal': [],
        }

        if current_key not in self.backbone_nodes:
            return result

        cx, cy = self.backbone_nodes[current_key]

        for nb in self.backbone_adj.get(current_key, set()):
            if nb not in self.backbone_nodes:
                continue

            if self.is_closed_backbone_key(nb):
                continue

            x, y = self.backbone_nodes[nb]
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

    def choose_next_backbone_goal(self, current_key: NodeKey) -> Optional[NodeKey]:
        candidates = self.classify_open_backbone_neighbors(current_key)

        if candidates['same_forward']:
            return candidates['same_forward'][0][1]

        if candidates['same_backward']:
            next_key = candidates['same_backward'][0][1]
            if current_key in self.backbone_nodes and next_key in self.backbone_nodes:
                cx, _ = self.backbone_nodes[current_key]
                nx, _ = self.backbone_nodes[next_key]
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

    def shortest_path_to_nearest_open_backbone(
        self,
        start_key: NodeKey
    ) -> List[NodeKey]:
        if start_key not in self.backbone_nodes:
            return []

        q = deque([start_key])
        prev: Dict[NodeKey, Optional[NodeKey]] = {start_key: None}
        depth: Dict[NodeKey, int] = {start_key: 0}
        target: Optional[NodeKey] = None

        while q:
            key = q.popleft()

            if key != start_key and not self.is_closed_backbone_key(key):
                target = key
                break

            if depth[key] >= self.max_graph_transit_hops:
                continue

            for nb in sorted(self.backbone_adj.get(key, set()), key=lambda k: self.backbone_nodes.get(k, (0.0, 0.0))):
                if nb in prev:
                    continue
                if nb not in self.backbone_nodes:
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
        if key is None or key not in self.backbone_nodes:
            self.current_goal_key = None
            self.selected_path.clear()
            return

        self.current_goal_key = key
        x, y = self.backbone_nodes[key]
        self.selected_path = [(x, y)]
        self.publish_goal(key)
        self.get_logger().info(f'New backbone goal: {key}, reason={reason}')

    def publish_goal(self, key: NodeKey) -> None:
        if key not in self.backbone_nodes:
            return

        x, y = self.backbone_nodes[key]

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
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 0.95
        self.goal_marker_pub.publish(marker)

    # ------------------------------------------------------------------
    # Dense off-backbone hole detection
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

    def get_backbone_scan_basis(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Return local scan basis (u, v, origin). u is the current backbone
        forward direction; v is the lateral direction. This makes branch
        detection dynamic to the current robot/goal state instead of static
        in the RCG node.
        """
        if current_key in self.backbone_nodes:
            origin = self.backbone_nodes[current_key]
        elif next_goal_key in self.backbone_nodes:
            origin = self.backbone_nodes[next_goal_key]  # type: ignore[index]
        else:
            origin = (0.0, 0.0)

        if (
            current_key in self.backbone_nodes and
            next_goal_key in self.backbone_nodes and
            next_goal_key != current_key
        ):
            cx, cy = self.backbone_nodes[current_key]
            gx, gy = self.backbone_nodes[next_goal_key]  # type: ignore[index]
            dx = gx - cx
            dy = gy - cy
        else:
            dx = self.sweep_dir
            dy = 0.0

        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            u = (1.0 if self.sweep_dir >= 0.0 else -1.0, 0.0)
        else:
            u = (dx / norm, dy / norm)

        v = (-u[1], u[0])
        return u, v, origin

    def project_to_basis(
        self,
        xy: Tuple[float, float],
        origin: Tuple[float, float],
        u: Tuple[float, float],
        v: Tuple[float, float],
    ) -> Tuple[float, float]:
        dx = xy[0] - origin[0]
        dy = xy[1] - origin[1]
        return dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]

    def collect_backbone_window(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> Set[NodeKey]:
        """
        Dynamic local backbone window.

        This replaces the old global/static branch concept. Only backbone
        nodes near the robot's current main corridor and within a forward
        lookahead window are allowed to start branch search.
        """
        window: Set[NodeKey] = set()

        if current_key not in self.backbone_nodes:
            return window

        u, v, origin = self.get_backbone_scan_basis(current_key, next_goal_key)
        lookahead = max(0.10, self.hole_scan_lookahead_distance)
        backtrack = max(0.0, self.hole_scan_backtrack_margin)
        lateral_radius = max(0.05, self.local_backbone_window_lateral_radius)

        # Graph search gives candidates; projection filtering turns it into a
        # local forward corridor instead of a full global neighborhood.
        q = deque([current_key])
        dist_map: Dict[NodeKey, float] = {current_key: 0.0}
        graph_limit = lookahead + backtrack + 1.0

        while q:
            key = q.popleft()
            if key not in self.backbone_nodes:
                continue

            s, t = self.project_to_basis(self.backbone_nodes[key], origin, u, v)
            if -backtrack <= s <= lookahead and abs(t) <= lateral_radius:
                window.add(key)

            kx, ky = self.backbone_nodes[key]
            for nb in self.backbone_adj.get(key, set()):
                if nb not in self.backbone_nodes:
                    continue

                nx, ny = self.backbone_nodes[nb]
                step = math.hypot(nx - kx, ny - ky)
                nd = dist_map[key] + step

                if nd > graph_limit:
                    continue

                if nb in dist_map and nd >= dist_map[nb]:
                    continue

                # Do not expand very far away from the local corridor.
                ns, nt = self.project_to_basis(self.backbone_nodes[nb], origin, u, v)
                if ns < -backtrack - 0.60 or ns > lookahead + 0.60:
                    continue
                if abs(nt) > lateral_radius + 0.80:
                    continue

                dist_map[nb] = nd
                q.append(nb)

        if current_key in self.backbone_nodes:
            window.add(current_key)
        if next_goal_key in self.backbone_nodes:
            window.add(next_goal_key)  # type: ignore[arg-type]

        self.latest_backbone_window = set(window)
        return window

    def local_backbone_segments(
        self,
        window: Set[NodeKey]
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for a in window:
            if a not in self.backbone_nodes:
                continue
            for b in self.backbone_adj.get(a, set()):
                if b not in window or b not in self.backbone_nodes:
                    continue
                if a > b:
                    continue
                segs.append((self.backbone_nodes[a], self.backbone_nodes[b]))
        return segs

    def min_distance_to_local_backbone(
        self,
        xy: Tuple[float, float],
        window: Set[NodeKey]
    ) -> float:
        segs = self.local_backbone_segments(window)
        if segs:
            return min(self.point_to_segment_distance(xy, a, b) for a, b in segs)

        best = float('inf')
        for key in window:
            if key not in self.backbone_nodes:
                continue
            bx, by = self.backbone_nodes[key]
            best = min(best, math.hypot(xy[0] - bx, xy[1] - by))
        return best if best != float('inf') else 0.0

    def is_dynamic_branch_edge(
        self,
        base_key: NodeKey,
        seed_key: NodeKey,
        window: Set[NodeKey],
        u: Tuple[float, float],
        v: Tuple[float, float],
    ) -> bool:
        """
        True branch edges are not static map attributes. They are dynamic
        relations from the current local backbone corridor to an off-backbone
        dense seed.
        """
        if base_key not in self.backbone_nodes or base_key not in self.dense_nodes:
            return False
        if seed_key not in self.dense_nodes:
            return False
        if seed_key in self.backbone_keys_in_dense:
            return False
        if self.is_closed_dense_key(seed_key):
            return False

        bx, by = self.dense_nodes[base_key]
        sx, sy = self.dense_nodes[seed_key]
        dx = sx - bx
        dy = sy - by
        edge_dist = math.hypot(dx, dy)

        if edge_dist < 1e-6 or edge_dist > self.branch_edge_max_distance:
            return False

        edge_along = dx * u[0] + dy * u[1]
        edge_lateral = abs(dx * v[0] + dy * v[1])

        # Exclude dense samples that simply fill sparse gaps along the current
        # backbone line. A dynamic branch should leave the local corridor.
        if edge_lateral < self.branch_lateral_min_distance:
            return False

        if edge_lateral < self.branch_lateral_ratio * abs(edge_along):
            return False

        seed_dist_to_window = self.min_distance_to_local_backbone((sx, sy), window)
        if seed_dist_to_window < self.branch_seed_min_distance_to_window:
            return False

        return True

    def find_dense_branch_components(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> List[Dict[str, object]]:
        if not self.enable_hole_detection:
            self.latest_dynamic_branch_edges = []
            self.latest_backbone_window = set()
            return []

        if not self.dense_nodes or not self.dense_adj:
            self.latest_dynamic_branch_edges = []
            return []

        if not self.backbone_nodes or not self.backbone_adj:
            self.latest_dynamic_branch_edges = []
            return []

        window = self.collect_backbone_window(current_key, next_goal_key, robot_xy)
        if not window:
            self.latest_dynamic_branch_edges = []
            return []

        u, v, _ = self.get_backbone_scan_basis(current_key, next_goal_key)

        candidates: List[Dict[str, object]] = []
        seen_components: Set[frozenset] = set()
        dynamic_edges: List[Tuple[NodeKey, NodeKey, str]] = []

        for base_key in sorted(window):
            if base_key not in self.dense_nodes:
                continue

            for seed_key in self.dense_adj.get(base_key, set()):
                if not self.is_dynamic_branch_edge(base_key, seed_key, window, u, v):
                    continue

                comp, attachments = self.floodfill_dense_off_backbone(seed_key)
                if not comp:
                    continue

                comp_id = frozenset(comp)
                if comp_id in seen_components:
                    continue
                seen_components.add(comp_id)

                label, reason, score = self.classify_off_backbone_component(
                    comp=comp,
                    attachments=attachments,
                    seed_key=seed_key,
                    base_key=base_key,
                    robot_xy=robot_xy,
                )

                dynamic_edges.append((base_key, seed_key, label))
                candidates.append({
                    'label': label,
                    'reason': reason,
                    'score': score,
                    'seed_key': seed_key,
                    'base_key': base_key,
                    'component': comp,
                    'attachments': attachments,
                })

        self.latest_dynamic_branch_edges = dynamic_edges
        candidates.sort(key=lambda item: float(item['score']))
        return candidates

    def floodfill_dense_off_backbone(
        self,
        seed_key: NodeKey
    ) -> Tuple[Set[NodeKey], Set[NodeKey]]:
        comp: Set[NodeKey] = set()
        attachments: Set[NodeKey] = set()

        if seed_key not in self.dense_nodes:
            return comp, attachments

        q = deque([seed_key])
        visited: Set[NodeKey] = {seed_key}

        while q:
            key = q.popleft()

            if len(comp) >= self.dense_floodfill_max_nodes:
                break

            if key in self.backbone_keys_in_dense:
                attachments.add(key)
                continue

            if key not in self.dense_nodes:
                continue

            if self.is_closed_dense_key(key):
                continue

            comp.add(key)

            for nb in self.dense_adj.get(key, set()):
                if nb in visited:
                    continue

                visited.add(nb)

                if nb in self.backbone_keys_in_dense:
                    attachments.add(nb)
                    continue

                q.append(nb)

        # A seed usually comes from a backbone base_key. If the raw graph edge
        # is quantized and not discovered during floodfill, collect contacts again.
        for key in comp:
            for nb in self.dense_adj.get(key, set()):
                if nb in self.backbone_keys_in_dense:
                    attachments.add(nb)

        return comp, attachments

    def classify_off_backbone_component(
        self,
        comp: Set[NodeKey],
        attachments: Set[NodeKey],
        seed_key: NodeKey,
        base_key: NodeKey,
        robot_xy: Tuple[float, float],
    ) -> Tuple[str, str, float]:
        n = len(comp)
        if n < self.hole_min_nodes:
            return 'too_small', f'nodes={n}', 1e6 + n

        if n > self.hole_max_nodes:
            return 'too_large', f'nodes={n}', 1e6 + n

        points = [self.dense_nodes[k] for k in comp if k in self.dense_nodes]
        if not points:
            return 'empty', 'no valid points', 1e6

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        bbox_area = max(dx, 0.05) * max(dy, 0.05)

        if bbox_area < self.hole_min_bbox_area:
            return 'too_small_area', f'bbox={bbox_area:.3f}', 1e6 + bbox_area

        if bbox_area > self.hole_max_bbox_area:
            return 'too_large_area', f'bbox={bbox_area:.3f}', 1e6 + bbox_area

        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        robot_dist = math.hypot(cx - robot_xy[0], cy - robot_xy[1])

        if robot_dist > self.hole_max_robot_distance:
            return 'too_far', f'dist={robot_dist:.2f}', 1e6 + robot_dist

        unknown_ratio = self.unknown_ratio_near_component(comp)
        if unknown_ratio >= self.unknown_reject_ratio:
            return 'frontier', f'unknown_ratio={unknown_ratio:.2f}', 5e5 + unknown_ratio

        mean_d, max_d = self.distance_stats_to_backbone_edges(comp)

        if max_d < self.hole_min_max_distance_to_backbone and mean_d < self.normal_lap_distance_to_backbone:
            return 'normal_lap', f'mean_backbone_dist={mean_d:.2f}, max={max_d:.2f}', 2e5 + mean_d

        if len(attachments) >= self.normal_lap_attachment_threshold:
            return 'normal_lap', f'attachments={len(attachments)}', 2e5 + len(attachments)

        # Prefer small and nearby components, and prefer components with fewer
        # backbone contacts.
        score = robot_dist + 0.35 * bbox_area + 0.15 * len(attachments)
        return 'hole', (
            f'nodes={n}, bbox={bbox_area:.2f}, attach={len(attachments)}, '
            f'unknown={unknown_ratio:.2f}, d_backbone={mean_d:.2f}/{max_d:.2f}'
        ), score

    def unknown_ratio_near_component(self, comp: Set[NodeKey]) -> float:
        if self.unknown_arr is None or self.free_msg is None:
            return 0.0

        info = self.free_msg.info
        rad = max(1, int(math.ceil(self.unknown_check_radius / info.resolution)))

        total = 0
        unknown_hits = 0

        for key in comp:
            if key not in self.dense_nodes:
                continue

            x, y = self.dense_nodes[key]
            cell = self.world_to_cell(x, y)
            if cell is None:
                continue

            row, col = cell
            total += 1

            r0 = max(0, row - rad)
            r1 = min(self.unknown_arr.shape[0], row + rad + 1)
            c0 = max(0, col - rad)
            c1 = min(self.unknown_arr.shape[1], col + rad + 1)

            if bool(np.any(self.unknown_arr[r0:r1, c0:c1])):
                unknown_hits += 1

        if total <= 0:
            return 0.0

        return unknown_hits / total

    def distance_stats_to_backbone_edges(self, comp: Set[NodeKey]) -> Tuple[float, float]:
        if not comp or not self.backbone_raw_edges:
            return 0.0, 0.0

        edge_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for a, b in self.backbone_raw_edges:
            if a in self.backbone_nodes and b in self.backbone_nodes:
                edge_segments.append((self.backbone_nodes[a], self.backbone_nodes[b]))

        if not edge_segments:
            return 0.0, 0.0

        distances: List[float] = []

        for key in comp:
            if key not in self.dense_nodes:
                continue

            p = self.dense_nodes[key]
            best = min(self.point_to_segment_distance(p, a, b) for a, b in edge_segments)
            distances.append(best)

        if not distances:
            return 0.0, 0.0

        return sum(distances) / len(distances), max(distances)

    def point_to_segment_distance(
        self,
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float]
    ) -> float:
        px, py = p
        ax, ay = a
        bx, by = b
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        vv = vx * vx + vy * vy

        if vv < 1e-9:
            return math.hypot(px - ax, py - ay)

        t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        qx = ax + t * vx
        qy = ay + t * vy
        return math.hypot(px - qx, py - qy)

    def update_hole_detection(
        self,
        current_key: NodeKey,
        next_goal_key: Optional[NodeKey],
        robot_xy: Tuple[float, float]
    ) -> None:
        candidates = self.find_dense_branch_components(current_key, next_goal_key, robot_xy)

        hole_candidate = None
        for cand in candidates:
            if cand['label'] == 'hole':
                hole_candidate = cand
                break

        if hole_candidate is None:
            self.clear_hole_state()
            if candidates:
                first = candidates[0]
                self.get_logger().info(
                    f'Dense branch found but not hole: label={first["label"]}, reason={first["reason"]}'
                )
            return

        comp = set(hole_candidate['component'])
        attachments = set(hole_candidate['attachments'])
        seed_key = hole_candidate['seed_key']
        base_key = hole_candidate['base_key']

        self.latest_hole_component = comp
        self.latest_hole_attachments = attachments
        self.latest_hole_entry_key = seed_key
        self.latest_hole_exit_key = self.choose_hole_exit_key(comp, attachments, seed_key)
        self.latest_branch_edge = (base_key, seed_key)

        if self.enable_hole_repair_path:
            self.latest_hole_repair_path = self.build_local_repair_path(
                comp,
                entry_key=self.latest_hole_entry_key,
                exit_key=self.latest_hole_exit_key,
            )
        else:
            self.latest_hole_repair_path = []

        self.get_logger().info(
            f'Hole detected from dense branch: seed={seed_key}, base={base_key}, '
            f'exit={self.latest_hole_exit_key}, reason={hole_candidate["reason"]}'
        )

    def choose_hole_exit_key(
        self,
        comp: Set[NodeKey],
        attachments: Set[NodeKey],
        entry_key: Optional[NodeKey],
    ) -> Optional[NodeKey]:
        if not attachments:
            return None

        if entry_key not in self.dense_nodes:
            return next(iter(attachments))

        ex, ey = self.dense_nodes[entry_key]

        # Prefer a backbone attachment away from the entry seed so the visual
        # repair path has an enter/return direction.
        best_key = None
        best_score = -1.0

        for key in attachments:
            if key not in self.backbone_nodes:
                continue

            x, y = self.backbone_nodes[key]
            d = math.hypot(x - ex, y - ey)

            if d > best_score:
                best_score = d
                best_key = key

        return best_key if best_key is not None else next(iter(attachments))

    def build_local_repair_path(
        self,
        comp: Set[NodeKey],
        entry_key: Optional[NodeKey],
        exit_key: Optional[NodeKey],
    ) -> List[Tuple[float, float]]:
        points = [self.dense_nodes[k] for k in comp if k in self.dense_nodes]
        if not points:
            return []

        # Cluster by y to create a local boustrophedon-like order.
        pts = sorted(points, key=lambda p: (p[1], p[0]))
        clusters: List[List[Tuple[float, float]]] = []

        for p in pts:
            if not clusters:
                clusters.append([p])
                continue

            cy = sum(v[1] for v in clusters[-1]) / len(clusters[-1])
            if abs(p[1] - cy) <= self.hole_repair_lap_tolerance:
                clusters[-1].append(p)
            else:
                clusters.append([p])

        ordered: List[Tuple[float, float]] = []
        left_to_right = True
        for cluster in clusters:
            cluster.sort(key=lambda p: p[0], reverse=not left_to_right)
            ordered.extend(cluster)
            left_to_right = not left_to_right

        if not ordered:
            return []

        # Choose orientation that starts closer to entry.
        if entry_key in self.dense_nodes:
            entry_xy = self.dense_nodes[entry_key]
            d_start = math.hypot(ordered[0][0] - entry_xy[0], ordered[0][1] - entry_xy[1])
            d_end = math.hypot(ordered[-1][0] - entry_xy[0], ordered[-1][1] - entry_xy[1])
            if d_end < d_start:
                ordered.reverse()

        path: List[Tuple[float, float]] = []

        if entry_key in self.dense_nodes:
            path.append(self.dense_nodes[entry_key])

        for p in ordered:
            if not path or math.hypot(path[-1][0] - p[0], path[-1][1] - p[1]) > 0.03:
                path.append(p)

        if exit_key in self.backbone_nodes:
            ex = self.backbone_nodes[exit_key]
            if not path or math.hypot(path[-1][0] - ex[0], path[-1][1] - ex[1]) > 0.03:
                path.append(ex)

        return path[:max(1, self.hole_repair_max_points)]

    def clear_hole_state(self) -> None:
        self.latest_hole_component.clear()
        self.latest_hole_attachments.clear()
        self.latest_hole_entry_key = None
        self.latest_hole_exit_key = None
        self.latest_hole_repair_path.clear()
        self.latest_branch_edge = None

    # ------------------------------------------------------------------
    # Timer and publishing
    # ------------------------------------------------------------------
    def on_timer(self) -> None:
        if not self.backbone_nodes or not self.backbone_adj:
            return

        robot_xy = self.get_robot_pose()
        if robot_xy is None:
            return

        nearest_key = self.nearest_backbone_key(robot_xy[0], robot_xy[1])
        if nearest_key is None:
            return

        reached_goal = self.is_reached_goal(robot_xy)

        if reached_goal:
            if self.current_goal_key is not None:
                self.close_backbone_key(self.current_goal_key)

            current_key = self.current_goal_key if self.current_goal_key in self.backbone_nodes else nearest_key
            self.close_backbone_key(current_key)

            next_goal = self.choose_next_backbone_goal(current_key)

            if next_goal is None and self.enable_graph_transit_to_open:
                path = self.shortest_path_to_nearest_open_backbone(current_key)
                if len(path) >= 2:
                    # Move one graph step at a time along the transit path.
                    next_goal = path[1]
                    self.selected_path = [self.backbone_nodes[k] for k in path if k in self.backbone_nodes]
                    self.get_logger().info(
                        f'No direct open neighbor. Transit along backbone to {path[-1]} via {next_goal}.'
                    )

            if next_goal is not None:
                self.set_new_goal(next_goal, 'backbone coverage')
                if self.enable_hole_detection:
                    self.update_hole_detection(current_key, next_goal, robot_xy)
            else:
                self.current_goal_key = None
                self.selected_path.clear()
                self.clear_hole_state()
                self.get_logger().info('Backbone coverage appears complete: no open goal found.')

        else:
            # Dynamic branch check while moving toward current goal.
            if (
                self.enable_hole_detection and
                self.current_goal_key is not None and
                self.should_update_hole_detection()
            ):
                self.update_hole_detection(nearest_key, self.current_goal_key, robot_xy)

        self.publish_state_markers()
        self.publish_selected_path()
        self.publish_hole_outputs()

        if self.current_goal_key is not None:
            # Republish current goal to keep downstream simple controllers alive.
            self.publish_goal(self.current_goal_key)

    def publish_state_markers(self) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        open_marker = Marker()
        open_marker.header.frame_id = self.map_frame
        open_marker.header.stamp = delete_all.header.stamp
        open_marker.ns = 'backbone_open'
        open_marker.id = 0
        open_marker.type = Marker.SPHERE_LIST
        open_marker.action = Marker.ADD
        open_marker.scale.x = 0.065
        open_marker.scale.y = 0.065
        open_marker.scale.z = 0.065
        open_marker.color.r = 0.0
        open_marker.color.g = 1.0
        open_marker.color.b = 0.25
        open_marker.color.a = 0.75
        open_marker.pose.orientation.w = 1.0

        closed_marker = Marker()
        closed_marker.header.frame_id = self.map_frame
        closed_marker.header.stamp = delete_all.header.stamp
        closed_marker.ns = 'backbone_closed'
        closed_marker.id = 1
        closed_marker.type = Marker.SPHERE_LIST
        closed_marker.action = Marker.ADD
        closed_marker.scale.x = 0.075
        closed_marker.scale.y = 0.075
        closed_marker.scale.z = 0.075
        closed_marker.color.r = 1.0
        closed_marker.color.g = 0.10
        closed_marker.color.b = 0.10
        closed_marker.color.a = 0.85
        closed_marker.pose.orientation.w = 1.0

        for key, (x, y) in self.backbone_nodes.items():
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.06

            if self.is_closed_backbone_key(key):
                closed_marker.points.append(pt)
            else:
                open_marker.points.append(pt)

        ma.markers.append(open_marker)
        ma.markers.append(closed_marker)
        self.state_marker_pub.publish(ma)

    def publish_selected_path(self) -> None:
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        pts = self.selected_path
        if self.current_goal_key in self.backbone_nodes and not pts:
            pts = [self.backbone_nodes[self.current_goal_key]]

        for x, y in pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.03
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.path_pub.publish(msg)


    def publish_dynamic_branch_markers(self, stamp) -> None:
        """
        Planner-side dynamic branch visualization.
        Purple/yellow/red edges here are computed from the current local
        backbone window, not from static RCG generation.
        """
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        window_nodes = Marker()
        window_nodes.header.frame_id = self.map_frame
        window_nodes.header.stamp = stamp
        window_nodes.ns = 'dynamic_backbone_window'
        window_nodes.id = 0
        window_nodes.type = Marker.SPHERE_LIST
        window_nodes.action = Marker.ADD
        window_nodes.scale.x = 0.095
        window_nodes.scale.y = 0.095
        window_nodes.scale.z = 0.095
        window_nodes.color.r = 0.55
        window_nodes.color.g = 0.85
        window_nodes.color.b = 1.0
        window_nodes.color.a = 0.75
        window_nodes.pose.orientation.w = 1.0

        for key in sorted(self.latest_backbone_window):
            if key not in self.backbone_nodes:
                continue
            x, y = self.backbone_nodes[key]
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.20
            window_nodes.points.append(p)

        branch_edges = Marker()
        branch_edges.header.frame_id = self.map_frame
        branch_edges.header.stamp = stamp
        branch_edges.ns = 'dynamic_branch_edges'
        branch_edges.id = 1
        branch_edges.type = Marker.LINE_LIST
        branch_edges.action = Marker.ADD
        branch_edges.scale.x = 0.045
        branch_edges.color.r = 1.0
        branch_edges.color.g = 0.0
        branch_edges.color.b = 1.0
        branch_edges.color.a = 0.95
        branch_edges.pose.orientation.w = 1.0

        rejected_edges = Marker()
        rejected_edges.header.frame_id = self.map_frame
        rejected_edges.header.stamp = stamp
        rejected_edges.ns = 'dynamic_nonhole_branch_edges'
        rejected_edges.id = 2
        rejected_edges.type = Marker.LINE_LIST
        rejected_edges.action = Marker.ADD
        rejected_edges.scale.x = 0.030
        rejected_edges.color.r = 1.0
        rejected_edges.color.g = 0.85
        rejected_edges.color.b = 0.0
        rejected_edges.color.a = 0.70
        rejected_edges.pose.orientation.w = 1.0

        for base, seed, label in self.latest_dynamic_branch_edges:
            if base not in self.backbone_nodes or seed not in self.dense_nodes:
                continue

            bx, by = self.backbone_nodes[base]
            sx, sy = self.dense_nodes[seed]

            p1 = Point()
            p1.x = bx
            p1.y = by
            p1.z = 0.22
            p2 = Point()
            p2.x = sx
            p2.y = sy
            p2.z = 0.22

            # Purple means: dynamically detected branch candidate from the
            # current local backbone window. Hole acceptance is shown separately
            # by /cstar/hole_markers and /cstar/hole_repair_path.
            branch_edges.points.append(p1)
            branch_edges.points.append(p2)

            # Optional yellow overlay for candidates that were explicitly
            # classified as non-hole, useful while tuning filters.
            if label != 'hole' and self.publish_nonhole_branch_candidates:
                rejected_edges.points.append(p1)
                rejected_edges.points.append(p2)

        ma.markers.append(window_nodes)
        ma.markers.append(branch_edges)
        ma.markers.append(rejected_edges)
        self.dynamic_branch_marker_pub.publish(ma)

    def publish_hole_outputs(self) -> None:
        stamp = self.get_clock().now().to_msg()

        # Hole nodes PoseArray.
        pose_array = PoseArray()
        pose_array.header.frame_id = self.map_frame
        pose_array.header.stamp = stamp

        for key in sorted(self.latest_hole_component):
            if key not in self.dense_nodes:
                continue

            x, y = self.dense_nodes[key]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.08
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.hole_nodes_pub.publish(pose_array)

        self.publish_hole_markers(stamp)
        self.publish_entry_exit_markers(stamp)
        self.publish_hole_repair_path(stamp)
        self.publish_dynamic_branch_markers(stamp)

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
        nodes_marker.ns = 'dense_hole_nodes'
        nodes_marker.id = 0
        nodes_marker.type = Marker.SPHERE_LIST
        nodes_marker.action = Marker.ADD
        nodes_marker.scale.x = 0.08
        nodes_marker.scale.y = 0.08
        nodes_marker.scale.z = 0.08
        nodes_marker.color.r = 1.0
        nodes_marker.color.g = 0.0
        nodes_marker.color.b = 1.0
        nodes_marker.color.a = 0.92
        nodes_marker.pose.orientation.w = 1.0

        for key in sorted(self.latest_hole_component):
            if key not in self.dense_nodes:
                continue
            x, y = self.dense_nodes[key]
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = 0.10
            nodes_marker.points.append(pt)

        branch_marker = Marker()
        branch_marker.header.frame_id = self.map_frame
        branch_marker.header.stamp = stamp
        branch_marker.ns = 'dense_hole_branch_edge'
        branch_marker.id = 1
        branch_marker.type = Marker.LINE_LIST
        branch_marker.action = Marker.ADD
        branch_marker.scale.x = 0.04
        branch_marker.color.r = 1.0
        branch_marker.color.g = 0.0
        branch_marker.color.b = 0.0
        branch_marker.color.a = 0.95
        branch_marker.pose.orientation.w = 1.0

        if self.latest_branch_edge is not None:
            base, seed = self.latest_branch_edge
            if base in self.backbone_nodes and seed in self.dense_nodes:
                bx, by = self.backbone_nodes[base]
                sx, sy = self.dense_nodes[seed]

                p1 = Point()
                p1.x = bx
                p1.y = by
                p1.z = 0.16

                p2 = Point()
                p2.x = sx
                p2.y = sy
                p2.z = 0.16

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

        if self.latest_hole_entry_key in self.dense_nodes:
            x, y = self.dense_nodes[self.latest_hole_entry_key]
            entry_ma.markers.append(
                self.make_sphere_marker('hole_entry', 0, x, y, 0.18, (0.0, 1.0, 1.0, 0.95), stamp)
            )

        if self.latest_hole_exit_key in self.backbone_nodes:
            x, y = self.backbone_nodes[self.latest_hole_exit_key]
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