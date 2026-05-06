#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class GraphNode:
    idx: int
    row: int
    col: int
    x: float
    y: float
    lap_id: int
    seg_id: int
    essential: bool


class CStarRCGNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_rcg_node')

        self.declare_parameter('free_topic', '/cstar/free_map')
        self.declare_parameter('frontier_topic', '/cstar/frontier_map')
        self.declare_parameter('obstacle_topic', '/cstar/obstacle_map')
        self.declare_parameter('unknown_topic', '/cstar/unknown_map')
        self.declare_parameter('map_frame', 'map')

        self.declare_parameter('lap_spacing', 0.30) #lap的疏密程度 原始0.35 稳定0.30
        self.declare_parameter('sample_spacing', 0.25) #sample的间距 原始0.22 稳定0.25

        self.declare_parameter('frontier_keep_radius', 0.25)
        self.declare_parameter('interlap_max_dist', 0.55)
        self.declare_parameter('prune_stride', 2)
        self.declare_parameter('publish_period', 1.0)

        # 安全缓冲区，单位：米0.2 /0.10 /0.15
        self.declare_parameter('obstacle_buffer', 0.15) #0.20
        self.declare_parameter('unknown_buffer', 0.05) #0.10
        self.declare_parameter('map_border_buffer', 0.10) #0.15

        # 过滤太短的安全自由段，避免在墙角生成零碎点
        self.declare_parameter('min_run_length', 0.30)

        self.declare_parameter('node_size', 0.06)
        self.declare_parameter('edge_width', 0.02)

        self.free_topic = self.get_parameter('free_topic').value
        self.frontier_topic = self.get_parameter('frontier_topic').value
        self.obstacle_topic = self.get_parameter('obstacle_topic').value
        self.unknown_topic = self.get_parameter('unknown_topic').value
        self.map_frame = self.get_parameter('map_frame').value

        self.lap_spacing = float(self.get_parameter('lap_spacing').value)
        self.sample_spacing = float(self.get_parameter('sample_spacing').value)
        self.frontier_keep_radius = float(self.get_parameter('frontier_keep_radius').value)
        self.interlap_max_dist = float(self.get_parameter('interlap_max_dist').value)
        self.prune_stride = max(1, int(self.get_parameter('prune_stride').value))
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.obstacle_buffer = float(self.get_parameter('obstacle_buffer').value)
        self.unknown_buffer = float(self.get_parameter('unknown_buffer').value)
        self.map_border_buffer = float(self.get_parameter('map_border_buffer').value)
        self.min_run_length = float(self.get_parameter('min_run_length').value)

        self.node_size = float(self.get_parameter('node_size').value)
        self.edge_width = float(self.get_parameter('edge_width').value)

        self.free_msg: Optional[OccupancyGrid] = None
        self.frontier_msg: Optional[OccupancyGrid] = None
        self.obstacle_msg: Optional[OccupancyGrid] = None
        self.unknown_msg: Optional[OccupancyGrid] = None

        self.free_arr: Optional[np.ndarray] = None
        self.frontier_arr: Optional[np.ndarray] = None
        self.obstacle_arr: Optional[np.ndarray] = None
        self.unknown_arr: Optional[np.ndarray] = None
        self.safe_free_arr: Optional[np.ndarray] = None

        self.free_sub = self.create_subscription(
            OccupancyGrid, self.free_topic, self.free_callback, 10
        )
        self.frontier_sub = self.create_subscription(
            OccupancyGrid, self.frontier_topic, self.frontier_callback, 10
        )
        self.obstacle_sub = self.create_subscription(
            OccupancyGrid, self.obstacle_topic, self.obstacle_callback, 10
        )
        self.unknown_sub = self.create_subscription(
            OccupancyGrid, self.unknown_topic, self.unknown_callback, 10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/cstar/rcg_markers', 10)
        self.nodes_pub = self.create_publisher(PoseArray, '/cstar/rcg_nodes', 10)

        self.timer = self.create_timer(self.publish_period, self.on_timer)

        self.get_logger().info('CStarRCGNode started.')
        self.get_logger().info(f'free_topic={self.free_topic}')
        self.get_logger().info(f'frontier_topic={self.frontier_topic}')
        self.get_logger().info(f'obstacle_topic={self.obstacle_topic}')
        self.get_logger().info(f'unknown_topic={self.unknown_topic}')
        self.get_logger().info(
            f'lap_spacing={self.lap_spacing:.2f}, sample_spacing={self.sample_spacing:.2f}'
        )
        self.get_logger().info(
            f'obstacle_buffer={self.obstacle_buffer:.2f}, '
            f'unknown_buffer={self.unknown_buffer:.2f}, '
            f'map_border_buffer={self.map_border_buffer:.2f}'
        )

    def free_callback(self, msg: OccupancyGrid) -> None:
        self.free_msg = msg
        h = msg.info.height
        w = msg.info.width
        arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        self.free_arr = arr > 50

    def frontier_callback(self, msg: OccupancyGrid) -> None:
        self.frontier_msg = msg
        h = msg.info.height
        w = msg.info.width
        arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        self.frontier_arr = arr > 50

    def obstacle_callback(self, msg: OccupancyGrid) -> None:
        self.obstacle_msg = msg
        h = msg.info.height
        w = msg.info.width
        arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        self.obstacle_arr = arr > 50

    def unknown_callback(self, msg: OccupancyGrid) -> None:
        self.unknown_msg = msg
        h = msg.info.height
        w = msg.info.width
        arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        self.unknown_arr = arr > 50

    def on_timer(self) -> None:
        if self.free_msg is None or self.frontier_msg is None:
            return
        if self.free_arr is None or self.frontier_arr is None:
            return

        if not self.metadata_ok():
            self.get_logger().warn('RCG input map metadata mismatch.')
            return

        self.safe_free_arr = self.build_safe_free_mask()

        nodes, edges = self.build_graph()
        self.publish_pose_array(nodes)
        self.publish_markers(nodes, edges)

    def metadata_ok(self) -> bool:
        if self.free_msg is None or self.frontier_msg is None:
            return False

        base = self.free_msg.info

        msgs = [self.frontier_msg]
        if self.obstacle_msg is not None:
            msgs.append(self.obstacle_msg)
        if self.unknown_msg is not None:
            msgs.append(self.unknown_msg)

        for msg in msgs:
            info = msg.info
            if info.width != base.width:
                return False
            if info.height != base.height:
                return False
            if abs(info.resolution - base.resolution) > 1e-9:
                return False
            if abs(info.origin.position.x - base.origin.position.x) > 1e-6:
                return False
            if abs(info.origin.position.y - base.origin.position.y) > 1e-6:
                return False

        return True

    def build_safe_free_mask(self) -> np.ndarray:
        """
        safe_free = free - obstacle_buffer - unknown_buffer - map_border_buffer

        这样生成的 RCG 节点不会贴墙、贴未知区、贴地图边界。
        """
        assert self.free_msg is not None
        assert self.free_arr is not None

        info = self.free_msg.info
        res = info.resolution
        h = info.height
        w = info.width

        free = self.free_arr.copy()

        # 如果 obstacle_map / unknown_map 暂时没来，就退化为“非 free 都是不安全区”
        if self.obstacle_arr is not None:
            obstacle = self.obstacle_arr.copy()
        else:
            obstacle = np.logical_not(free)

        if self.unknown_arr is not None:
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

    def build_graph(self) -> Tuple[List[GraphNode], Set[Tuple[int, int]]]:
        info = self.free_msg.info
        res = info.resolution

        lap_step = max(1, int(round(self.lap_spacing / res)))
        sample_step = max(1, int(round(self.sample_spacing / res)))
        frontier_rad = max(1, int(round(self.frontier_keep_radius / res)))
        interlap_max_cells = max(1.0, self.interlap_max_dist / res)

        nodes: List[GraphNode] = []
        edges: Set[Tuple[int, int]] = set()

        lap_to_nodes: Dict[int, List[GraphNode]] = {}

        # 核心修改：不再用全局 row_start，而是根据 safe_free_arr 的安全行带生成 lap
        lap_rows = self.collect_lap_rows_from_safe_mask(lap_step)

        for lap_id, row in enumerate(lap_rows):
            runs = self.find_safe_runs_on_row(row)
            lap_nodes: List[GraphNode] = []

            for seg_id, (start_col, end_col) in enumerate(runs):
                sample_cols = self.sample_segment(start_col, end_col, sample_step)
                kept_nodes: List[GraphNode] = []

                for i, col in enumerate(sample_cols):
                    if not self.is_safe_cell(row, col):
                        continue

                    near_frontier = self.has_frontier_near(row, col, frontier_rad)
                    endpoint = (i == 0 or i == len(sample_cols) - 1)

                    # 第一版弱 pruning：
                    # 1. 保留安全段端点；
                    # 2. 保留靠近 frontier 的点；
                    # 3. 普通区域按 prune_stride 稀疏保留。
                    keep = endpoint or near_frontier or (i % self.prune_stride == 0)

                    if not keep:
                        continue

                    x, y = self.cell_to_world(col, row)
                    node = GraphNode(
                        idx=len(nodes),
                        row=row,
                        col=col,
                        x=x,
                        y=y,
                        lap_id=lap_id,
                        seg_id=seg_id,
                        essential=(endpoint or near_frontier)
                    )
                    nodes.append(node)
                    kept_nodes.append(node)
                    lap_nodes.append(node)

                # 同一条 lap 内相邻节点连边
                for a, b in zip(kept_nodes[:-1], kept_nodes[1:]):
                    if self.line_is_safe(a.row, a.col, b.row, b.col):
                        edges.add((min(a.idx, b.idx), max(a.idx, b.idx)))

            lap_to_nodes[lap_id] = lap_nodes

        # 邻接 lap 之间连边
        for lap_id in range(len(lap_rows) - 1):
            curr_nodes = lap_to_nodes.get(lap_id, [])
            next_nodes = lap_to_nodes.get(lap_id + 1, [])

            if not curr_nodes or not next_nodes:
                continue

            row_gap = abs(lap_rows[lap_id + 1] - lap_rows[lap_id])

            # 如果两条 lap 行间距明显大于 interlap 阈值，直接跳过，避免跨区域尝试连边
            if row_gap > interlap_max_cells:
                continue

            for n1 in curr_nodes:
                best_node: Optional[GraphNode] = None
                best_dist = 1e9

                for n2 in next_nodes:
                    drow = float(n2.row - n1.row)
                    dcol = float(n2.col - n1.col)
                    dist = math.hypot(drow, dcol)

                    if dist > interlap_max_cells:
                        continue

                    if dist < best_dist and self.line_is_safe(n1.row, n1.col, n2.row, n2.col):
                        best_dist = dist
                        best_node = n2

                if best_node is not None:
                    edges.add((min(n1.idx, best_node.idx), max(n1.idx, best_node.idx)))

        self.get_logger().info(
            f'RCG rebuilt: nodes={len(nodes)}, edges={len(edges)}, laps={len(lap_rows)}'
        )
        return nodes, edges

    def find_safe_runs_on_row(self, row: int) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        inside = False
        start = 0

        arr = self.safe_free_arr
        if arr is None:
            return runs

        min_cells = max(1, int(round(self.min_run_length / self.free_msg.info.resolution)))

        for c in range(arr.shape[1]):
            if arr[row, c] and not inside:
                inside = True
                start = c
            elif not arr[row, c] and inside:
                inside = False
                end = c - 1
                if end >= start and (end - start + 1) >= min_cells:
                    runs.append((start, end))

        if inside:
            end = arr.shape[1] - 1
            if end >= start and (end - start + 1) >= min_cells:
                runs.append((start, end))

        return runs

    def sample_segment(self, start_col: int, end_col: int, step: int) -> List[int]:
        if end_col <= start_col:
            return [start_col]

        cols = list(range(start_col, end_col + 1, step))
        if cols[-1] != end_col:
            cols.append(end_col)

        return sorted(set(cols))

    def has_frontier_near(self, row: int, col: int, rad: int) -> bool:
        r0 = max(0, row - rad)
        r1 = min(self.frontier_arr.shape[0], row + rad + 1)
        c0 = max(0, col - rad)
        c1 = min(self.frontier_arr.shape[1], col + rad + 1)
        return bool(np.any(self.frontier_arr[r0:r1, c0:c1]))

    def is_safe_cell(self, row: int, col: int) -> bool:
        if self.safe_free_arr is None:
            return False
        if row < 0 or row >= self.safe_free_arr.shape[0]:
            return False
        if col < 0 or col >= self.safe_free_arr.shape[1]:
            return False
        return bool(self.safe_free_arr[row, col])

    def line_is_safe(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        if self.safe_free_arr is None:
            return False

        n = max(abs(r1 - r0), abs(c1 - c0)) + 1

        for i in range(n + 1):
            t = 0.0 if n == 0 else i / n
            rr = int(round((1.0 - t) * r0 + t * r1))
            cc = int(round((1.0 - t) * c0 + t * c1))

            if not self.is_safe_cell(rr, cc):
                return False

        return True

    def cell_to_world(self, col: int, row: int) -> Tuple[float, float]:
        info = self.free_msg.info
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def publish_pose_array(self, nodes: List[GraphNode]) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for n in nodes:
            p = Pose()
            p.position.x = n.x
            p.position.y = n.y
            p.position.z = 0.02
            p.orientation.w = 1.0
            msg.poses.append(p)

        self.nodes_pub.publish(msg)

    def publish_markers(self, nodes: List[GraphNode], edges: Set[Tuple[int, int]]) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.map_frame
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        normal_nodes = Marker()
        normal_nodes.header.frame_id = self.map_frame
        normal_nodes.header.stamp = self.get_clock().now().to_msg()
        normal_nodes.ns = 'rcg_nodes'
        normal_nodes.id = 0
        normal_nodes.type = Marker.SPHERE_LIST
        normal_nodes.action = Marker.ADD
        normal_nodes.scale.x = self.node_size
        normal_nodes.scale.y = self.node_size
        normal_nodes.scale.z = self.node_size
        normal_nodes.color.r = 0.1
        normal_nodes.color.g = 0.5
        normal_nodes.color.b = 1.0
        normal_nodes.color.a = 0.85
        normal_nodes.pose.orientation.w = 1.0

        essential_nodes = Marker()
        essential_nodes.header.frame_id = self.map_frame
        essential_nodes.header.stamp = self.get_clock().now().to_msg()
        essential_nodes.ns = 'rcg_essential_nodes'
        essential_nodes.id = 1
        essential_nodes.type = Marker.SPHERE_LIST
        essential_nodes.action = Marker.ADD
        essential_nodes.scale.x = self.node_size * 1.2
        essential_nodes.scale.y = self.node_size * 1.2
        essential_nodes.scale.z = self.node_size * 1.2
        essential_nodes.color.r = 1.0
        essential_nodes.color.g = 0.85
        essential_nodes.color.b = 0.1
        essential_nodes.color.a = 0.95
        essential_nodes.pose.orientation.w = 1.0

        edge_marker = Marker()
        edge_marker.header.frame_id = self.map_frame
        edge_marker.header.stamp = self.get_clock().now().to_msg()
        edge_marker.ns = 'rcg_edges'
        edge_marker.id = 2
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = self.edge_width
        edge_marker.color.r = 0.0
        edge_marker.color.g = 1.0
        edge_marker.color.b = 1.0
        edge_marker.color.a = 0.55
        edge_marker.pose.orientation.w = 1.0

        for n in nodes:
            pt = Point()
            pt.x = n.x
            pt.y = n.y
            pt.z = 0.03

            if n.essential:
                essential_nodes.points.append(pt)
            else:
                normal_nodes.points.append(pt)

        for i, j in sorted(edges):
            ni = nodes[i]
            nj = nodes[j]

            p1 = Point()
            p1.x = ni.x
            p1.y = ni.y
            p1.z = 0.015

            p2 = Point()
            p2.x = nj.x
            p2.y = nj.y
            p2.z = 0.015

            edge_marker.points.append(p1)
            edge_marker.points.append(p2)

        ma.markers.append(normal_nodes)
        ma.markers.append(essential_nodes)
        ma.markers.append(edge_marker)

        self.marker_pub.publish(ma)

    def collect_lap_rows_from_safe_mask(self, lap_step: int) -> List[int]:
        """
        从 safe_free_arr 中提取每个连续安全行带，并在每个行带内部独立生成 lap 行。

        这样可以避免原来的全局 row_start 导致：
        1. 上下边界采样距离不一致；
        2. 某些窄区域刚好被全局 lap 跳过；
        3. 房间内部出现明显漏采样带。
        """
        if self.safe_free_arr is None:
            return []

        arr = self.safe_free_arr
        h, _ = arr.shape

        row_has_safe = np.any(arr, axis=1)

        bands: List[Tuple[int, int]] = []
        inside = False
        start = 0

        for r in range(h):
            if row_has_safe[r] and not inside:
                inside = True
                start = r
            elif not row_has_safe[r] and inside:
                inside = False
                end = r - 1
                if end >= start:
                    bands.append((start, end))

        if inside:
            end = h - 1
            if end >= start:
                bands.append((start, end))

        rows: Set[int] = set()

        for start, end in bands:
            band_height = end - start + 1

            # 很窄的安全带，至少取中间一行，避免上下边界两行太挤
            if band_height <= max(2, lap_step // 2):
                rows.add((start + end) // 2)
                continue

            # 从该安全行带自己的上边界开始采样
            r = start
            while r <= end:
                rows.add(r)
                r += lap_step

            # 强制保留下边界附近一行，保证上下边界覆盖一致
            rows.add(end)

        return sorted(rows)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = CStarRCGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
