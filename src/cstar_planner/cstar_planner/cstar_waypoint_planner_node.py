#!/usr/bin/env python3
import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseArray, PoseStamped, Point
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
        # 普通绿色 goal 建议 0.07~0.09；橙色 retreat node 建议 0.08~0.10。
        self.declare_parameter('goal_center_tolerance', 0.10) #原始0.08
        self.declare_parameter('retreat_center_tolerance', 0.10) #原始0.09

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

        covered_map 可以继续让节点变红；
        但是 goal_marker / retreat_node 的切换必须等底盘中心真正到达目标点附近。
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

    def classify_open_neighbors(self, current_key: NodeKey) -> Dict[str, List[Tuple[float, NodeKey]]]:
        result = {
            'forward': [],
            'backward': [],
            'up': [],
            'down': [],
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

            # 同一 lap 上的横向运动，是 Boustrophedon 的主运动。
            if abs(dy) <= self.same_lap_y_tolerance:
                if dx * self.sweep_dir > 0.0:
                    result['forward'].append((abs(dx), nb))
                else:
                    result['backward'].append((abs(dx), nb))
                continue

            # 换行运动。只有横向走不动时才优先考虑。
            if abs(dx) <= self.same_col_x_tolerance:
                if dy > 0:
                    result['up'].append((abs(dy), nb))
                else:
                    result['down'].append((abs(dy), nb))
                continue

            result['diagonal'].append((dist, nb))

        for k in result:
            result[k].sort(key=lambda item: item[0])

        return result

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

    def choose_next_normal_goal(self, current_key: NodeKey) -> Optional[NodeKey]:
        candidates = self.classify_open_neighbors(current_key)

        # 1. 优先沿当前 lap 继续扫
        if candidates['forward']:
            return candidates['forward'][0][1]

        # 2. 当前 lap 走到头，尝试上下换行，并反转 sweep 方向
        if candidates['up']:
            self.sweep_dir *= -1.0
            return candidates['up'][0][1]

        if candidates['down']:
            self.sweep_dir *= -1.0
            return candidates['down'][0][1]

        # 3. 上下都不行时，再尝试回头补扫
        if candidates['backward']:
            self.sweep_dir *= -1.0
            return candidates['backward'][0][1]

        # 4. 最后才允许斜边兜底，避免明明有 open 邻居却误判 dead-end
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

    def on_timer(self) -> None:
        if not self.nodes:
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

        current_key = self.choose_anchor_key(robot_xy, nearest_key, reached_goal)

        if current_key is None:
            self.publish_open_closed_markers()
            self.publish_selected_path()
            self.publish_escape_path()
            self.publish_retreat_nodes()
            return

        self.close_key(current_key)

        if reached_goal:
            if self.current_goal_key is not None:
                self.close_key(self.current_goal_key)

            if self.escape_active:
                self.finish_escape(current_key)

            normal_goal = self.choose_next_normal_goal(current_key)

            if normal_goal is not None:
                self.last_deadend_key = None
                self.set_new_goal(normal_goal, 'C* normal')

            else:
                # 如果图上明明有 open 邻居，但策略没选出来，说明方向分类太严格。
                if self.has_any_open_neighbor(current_key):
                    x, y = self.nodes[current_key]
                    self.get_logger().warn(
                        f'Node ({x:.2f}, {y:.2f}) still has open graph neighbors, '
                        f'but Boustrophedon policy rejected them. '
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

        if self.current_goal_key is not None:
            self.publish_goal(self.current_goal_key)

        self.publish_open_closed_markers()
        self.publish_selected_path()
        self.publish_escape_path()
        self.publish_retreat_nodes()


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