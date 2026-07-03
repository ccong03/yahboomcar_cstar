#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
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
    endpoint: bool
    near_frontier: bool


Candidate = Tuple[int, bool, bool, int]  # col, endpoint, near_frontier, original_order


class CStarRCGNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_rcg_node')

        self.declare_parameter('free_topic', '/cstar/free_map')
        self.declare_parameter('frontier_topic', '/cstar/frontier_map')
        self.declare_parameter('obstacle_topic', '/cstar/obstacle_map')
        self.declare_parameter('unknown_topic', '/cstar/unknown_map')
        self.declare_parameter('map_frame', 'map')

        # 保留你当前偏连通的采样设置，再通过 node spacing 和端点桥接减少乱边
        self.declare_parameter('lap_spacing', 0.30)
        self.declare_parameter('sample_spacing', 0.15)
        self.declare_parameter('frontier_keep_radius', 0.25)
        self.declare_parameter('prune_stride', 1)
        self.declare_parameter('publish_period', 0.5)

        # 端点主动跨 lap 连接：
        # 不是“端点只能连端点”，而是“端点连到相邻 lap 上最近的 sample”
        self.declare_parameter('interlap_max_dist', 0.55) #0.75
        self.declare_parameter('endpoint_to_sample_col_tolerance', 0.25) #0.55 0.22
        self.declare_parameter('max_interlap_bridges_per_segment_pair', 2) #4
        self.declare_parameter('enable_endpoint_fallback_bridge', True)

        # 安全缓冲区，单位：米
        self.declare_parameter('obstacle_buffer', 0.12) #0.15
        self.declare_parameter('unknown_buffer', 0.06) #0.06
        self.declare_parameter('map_border_buffer', 0.12) #0.15

        # 过滤太短的安全自由段，避免墙角碎点
        self.declare_parameter('min_run_length', 0.30)

        # 节点压缩：减少过近点。端点/frontier 点优先保留
        self.declare_parameter('enable_node_spacing_filter', True)
        self.declare_parameter('min_node_keep_distance', 0.25)

        # 边约束：禁止交叉边
        self.declare_parameter('enable_edge_intersection_check', True)

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
        self.prune_stride = max(1, int(self.get_parameter('prune_stride').value))
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.interlap_max_dist = float(self.get_parameter('interlap_max_dist').value)
        self.endpoint_to_sample_col_tolerance = float(
            self.get_parameter('endpoint_to_sample_col_tolerance').value
        )
        self.max_interlap_bridges_per_segment_pair = max(
            0, int(self.get_parameter('max_interlap_bridges_per_segment_pair').value)
        )
        self.enable_endpoint_fallback_bridge = bool(
            self.get_parameter('enable_endpoint_fallback_bridge').value
        )

        self.obstacle_buffer = float(self.get_parameter('obstacle_buffer').value)
        self.unknown_buffer = float(self.get_parameter('unknown_buffer').value)
        self.map_border_buffer = float(self.get_parameter('map_border_buffer').value)
        self.min_run_length = float(self.get_parameter('min_run_length').value)

        self.enable_node_spacing_filter = bool(
            self.get_parameter('enable_node_spacing_filter').value
        )
        self.min_node_keep_distance = float(
            self.get_parameter('min_node_keep_distance').value
        )

        self.enable_edge_intersection_check = bool(
            self.get_parameter('enable_edge_intersection_check').value
        )

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
        # relaxed_edge_arr 用于边界/障碍物附近端点的特殊连边判断：
        # 不检查 map_border_buffer 和 obstacle_buffer，但仍然禁止穿过真实障碍和 unknown_buffer。
        self.relaxed_edge_arr: Optional[np.ndarray] = None

        self.create_subscription(OccupancyGrid, self.free_topic, self.free_callback, 10)
        self.create_subscription(OccupancyGrid, self.frontier_topic, self.frontier_callback, 10)
        self.create_subscription(OccupancyGrid, self.obstacle_topic, self.obstacle_callback, 10)
        self.create_subscription(OccupancyGrid, self.unknown_topic, self.unknown_callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/cstar/rcg_markers', 10)
        self.nodes_pub = self.create_publisher(PoseArray, '/cstar/rcg_nodes', 10)

        self.timer = self.create_timer(self.publish_period, self.on_timer)

        self.get_logger().info('CStarRCGNode endpoint-to-nearest-sample bridge mode started.')
        self.get_logger().info(
            f'lap_spacing={self.lap_spacing:.2f}, sample_spacing={self.sample_spacing:.2f}, '
            f'prune_stride={self.prune_stride}'
        )
        self.get_logger().info(
            f'interlap_max_dist={self.interlap_max_dist:.2f}, '
            f'endpoint_to_sample_col_tolerance={self.endpoint_to_sample_col_tolerance:.2f}, '
            f'max_interlap_bridges={self.max_interlap_bridges_per_segment_pair}'
        )
        self.get_logger().info(
            f'node_spacing_filter={self.enable_node_spacing_filter}, '
            f'min_node_keep_distance={self.min_node_keep_distance:.2f}, '
            f'edge_intersection_check={self.enable_edge_intersection_check}'
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
        assert self.free_msg is not None
        assert self.free_arr is not None

        info = self.free_msg.info
        res = info.resolution
        h = info.height
        w = info.width

        free = self.free_arr.copy()

        if self.obstacle_arr is not None:
            obstacle = self.obstacle_arr.copy()
        else:
            # obstacle_map 没来时，非 free 都按真实不可通行处理。
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

        # 正常采样用的安全自由区：严格避开 obstacle_buffer、unknown_buffer 和 map_border_buffer。
        safe = free.copy()
        safe[obstacle_buffer_mask] = False
        safe[unknown_buffer_mask] = False

        if border_rad > 0:
            safe[:border_rad, :] = False
            safe[h - border_rad:, :] = False
            safe[:, :border_rad] = False
            safe[:, w - border_rad:] = False

        # 特殊连边用的 relaxed mask：
        # 只用于“靠近地图边界/障碍物的端点连边”。它不检查 obstacle_buffer 和 map_border_buffer，
        # 但仍然禁止穿过真实障碍、未知区缓冲，避免把边直接穿墙或穿进未探索区。
        relaxed = free.copy()
        relaxed[obstacle] = False
        relaxed[unknown_buffer_mask] = False
        self.relaxed_edge_arr = relaxed

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
        assert self.free_msg is not None

        info = self.free_msg.info
        res = info.resolution

        lap_step = max(1, int(round(self.lap_spacing / res)))
        sample_step = max(1, int(round(self.sample_spacing / res)))
        frontier_rad = max(1, int(round(self.frontier_keep_radius / res)))

        interlap_max_cells = max(1.0, self.interlap_max_dist / res)
        endpoint_col_tol_cells = max(
            1, int(round(self.endpoint_to_sample_col_tolerance / res))
        )

        nodes: List[GraphNode] = []
        edges: Set[Tuple[int, int]] = set()

        lap_seg_to_nodes: Dict[Tuple[int, int], List[GraphNode]] = {}
        lap_seg_bounds: Dict[Tuple[int, int], Tuple[int, int]] = {}

        lap_rows = self.collect_lap_rows_from_safe_mask(lap_step)

        for lap_id, row in enumerate(lap_rows):
            runs = self.find_safe_runs_on_row(row)

            for seg_id, (start_col, end_col) in enumerate(runs):
                sample_cols = self.sample_segment(start_col, end_col, sample_step)

                candidates: List[Candidate] = []

                for i, col in enumerate(sample_cols):
                    if not self.is_safe_cell(row, col):
                        continue

                    near_frontier = self.has_frontier_near(row, col, frontier_rad)
                    endpoint = (i == 0 or i == len(sample_cols) - 1)

                    keep = endpoint or near_frontier or (i % self.prune_stride == 0)
                    if not keep:
                        continue

                    candidates.append((col, endpoint, near_frontier, i))

                candidates = self.filter_close_candidates(candidates)

                kept_nodes: List[GraphNode] = []
                for col, endpoint, near_frontier, _ in candidates:
                    x, y = self.cell_to_world(col, row)
                    node = GraphNode(
                        idx=len(nodes),
                        row=row,
                        col=col,
                        x=x,
                        y=y,
                        lap_id=lap_id,
                        seg_id=seg_id,
                        essential=(endpoint or near_frontier),
                        endpoint=endpoint,
                        near_frontier=near_frontier
                    )

                    nodes.append(node)
                    kept_nodes.append(node)

                # same-lap：同一安全段内只连接相邻点
                for a, b in zip(kept_nodes[:-1], kept_nodes[1:]):
                    self.try_add_edge(edges, nodes, a.idx, b.idx)

                lap_seg_to_nodes[(lap_id, seg_id)] = kept_nodes
                lap_seg_bounds[(lap_id, seg_id)] = (start_col, end_col)

        # inter-lap：端点主动连接相邻 lap 上最近 sample；中间 sample 只作为被连接目标
        for lap_id in range(len(lap_rows) - 1):
            row_gap = abs(lap_rows[lap_id + 1] - lap_rows[lap_id])
            if row_gap > interlap_max_cells:
                continue

            curr_seg_ids = sorted(
                key[1] for key in lap_seg_to_nodes.keys() if key[0] == lap_id
            )
            next_seg_ids = sorted(
                key[1] for key in lap_seg_to_nodes.keys() if key[0] == lap_id + 1
            )

            for curr_seg_id in curr_seg_ids:
                seg_a = lap_seg_to_nodes.get((lap_id, curr_seg_id), [])
                if not seg_a:
                    continue

                a_start, a_end = lap_seg_bounds[(lap_id, curr_seg_id)]

                for next_seg_id in next_seg_ids:
                    seg_b = lap_seg_to_nodes.get((lap_id + 1, next_seg_id), [])
                    if not seg_b:
                        continue

                    b_start, b_end = lap_seg_bounds[(lap_id + 1, next_seg_id)]

                    if not self.segments_overlap_or_near(
                        a_start,
                        a_end,
                        b_start,
                        b_end,
                        endpoint_col_tol_cells
                    ):
                        continue

                    self.add_endpoint_to_nearest_sample_edges(
                        edges=edges,
                        nodes=nodes,
                        seg_a=seg_a,
                        seg_b=seg_b,
                        interlap_max_cells=interlap_max_cells,
                        endpoint_col_tol_cells=endpoint_col_tol_cells,
                    )

        self.get_logger().info(
            f'RCG rebuilt: nodes={len(nodes)}, edges={len(edges)}, laps={len(lap_rows)}'
        )
        return nodes, edges

    def segments_overlap_or_near(
        self,
        a_start: int,
        a_end: int,
        b_start: int,
        b_end: int,
        tolerance_cells: int
    ) -> bool:
        return (
            min(a_end, b_end) + tolerance_cells >=
            max(a_start, b_start) - tolerance_cells
        )

    def get_segment_endpoints(
        self,
        seg_nodes: List[GraphNode]
    ) -> Tuple[Optional[GraphNode], Optional[GraphNode]]:
        if not seg_nodes:
            return None, None

        ordered = sorted(seg_nodes, key=lambda n: n.col)
        return ordered[0], ordered[-1]

    def add_endpoint_to_nearest_sample_edges(
        self,
        edges: Set[Tuple[int, int]],
        nodes: List[GraphNode],
        seg_a: List[GraphNode],
        seg_b: List[GraphNode],
        interlap_max_cells: float,
        endpoint_col_tol_cells: int
    ) -> int:
        """
        核心逻辑：
        - seg_a 的左右端点，主动连接 seg_b 中最近的 sample；
        - seg_b 的左右端点，主动连接 seg_a 中最近的 sample；
        - lap 中间 sample 不主动发起跨 lap 连接，但可以被端点连接；
        - 这样能处理“一长一短 lap”的情况。
        """
        a_left, a_right = self.get_segment_endpoints(seg_a)
        b_left, b_right = self.get_segment_endpoints(seg_b)

        if a_left is None or a_right is None or b_left is None or b_right is None:
            return 0

        anchors: List[Tuple[GraphNode, List[GraphNode]]] = [
            (a_left, seg_b),
            (a_right, seg_b),
            (b_left, seg_a),
            (b_right, seg_a),
        ]

        candidates: List[Tuple[float, float, int, GraphNode, GraphNode]] = []

        for anchor_index, (anchor, target_seg) in enumerate(anchors):
            target = self.find_best_target_for_endpoint(
                anchor=anchor,
                target_seg=target_seg,
                interlap_max_cells=interlap_max_cells,
                endpoint_col_tol_cells=endpoint_col_tol_cells,
            )

            if target is None:
                continue

            dcol = abs(target.col - anchor.col)
            dist = math.hypot(float(target.row - anchor.row), float(target.col - anchor.col))

            # 排序：先保证每个端点自己的近竖直连接，再按距离
            candidates.append((float(dcol), dist, anchor_index, anchor, target))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        added = 0
        used_anchors: Set[int] = set()
        used_edges: Set[Tuple[int, int]] = set()

        for _, _, anchor_index, anchor, target in candidates:
            if anchor_index in used_anchors:
                continue

            e = (min(anchor.idx, target.idx), max(anchor.idx, target.idx))
            if e in used_edges:
                continue

            if self.try_add_edge(edges, nodes, anchor.idx, target.idx):
                added += 1
                used_anchors.add(anchor_index)
                used_edges.add(e)

            if added >= self.max_interlap_bridges_per_segment_pair:
                return added

        if added > 0 or not self.enable_endpoint_fallback_bridge:
            return added

        # fallback：如果端点到最近 sample 全失败，再在端点和对侧 sample 中找一条最短安全边
        fallback_candidates: List[Tuple[float, float, GraphNode, GraphNode]] = []

        for anchor, target_seg in anchors:
            for target in target_seg:
                if anchor.idx == target.idx:
                    continue

                drow = float(target.row - anchor.row)
                dcol = float(target.col - anchor.col)
                dist = math.hypot(drow, dcol)

                if dist > interlap_max_cells:
                    continue

                if not self.edge_line_is_acceptable(anchor, target):
                    continue

                fallback_candidates.append((abs(dcol), dist, anchor, target))

        fallback_candidates.sort(key=lambda item: (item[0], item[1]))

        for _, _, anchor, target in fallback_candidates:
            if self.try_add_edge(edges, nodes, anchor.idx, target.idx):
                added += 1
                break

        return added

    def find_best_target_for_endpoint(
        self,
        anchor: GraphNode,
        target_seg: List[GraphNode],
        interlap_max_cells: float,
        endpoint_col_tol_cells: int
    ) -> Optional[GraphNode]:
        best: Optional[GraphNode] = None
        best_key = (float('inf'), float('inf'))

        for target in target_seg:
            if target.idx == anchor.idx:
                continue

            drow = float(target.row - anchor.row)
            dcol = float(target.col - anchor.col)
            dist = math.hypot(drow, dcol)

            if dist > interlap_max_cells:
                continue

            if abs(target.col - anchor.col) > endpoint_col_tol_cells:
                continue

            if not self.edge_line_is_acceptable(anchor, target):
                continue

            key = (abs(dcol), dist)
            if key < best_key:
                best_key = key
                best = target

        return best

    def filter_close_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda item: item[0])

        if not self.enable_node_spacing_filter:
            return candidates

        assert self.free_msg is not None
        res = self.free_msg.info.resolution
        min_dist_cells = max(1, int(round(self.min_node_keep_distance / res)))

        # 太短的 segment 用中点，避免左右端点贴得太近
        if len(candidates) <= 2:
            first = candidates[0]
            last = candidates[-1]
            c0 = first[0]
            c1 = last[0]

            if abs(c1 - c0) < max(2, int(0.85 * min_dist_cells)):
                mid = int(round((c0 + c1) / 2.0))
                near = any(c[2] for c in candidates)
                return [(mid, True, near, 0)]

        kept: List[Candidate] = []

        for cand in candidates:
            col, endpoint, near_frontier, _ = cand

            if not kept:
                kept.append(cand)
                continue

            prev = kept[-1]
            prev_col, prev_endpoint, prev_near, _ = prev

            close = abs(col - prev_col) < min_dist_cells

            if not close:
                kept.append(cand)
                continue

            cand_priority = self.candidate_priority(endpoint, near_frontier)
            prev_priority = self.candidate_priority(prev_endpoint, prev_near)

            if cand_priority > prev_priority:
                kept[-1] = cand

        # 确保最后端点没有被过度压掉，便于端点跨 lap 连接
        last = candidates[-1]
        if last not in kept:
            if not kept or abs(last[0] - kept[-1][0]) >= min_dist_cells:
                kept.append(last)

        return kept

    def candidate_priority(self, endpoint: bool, near_frontier: bool) -> int:
        if endpoint and near_frontier:
            return 3
        if endpoint:
            return 2
        if near_frontier:
            return 1
        return 0

    def try_add_edge(
        self,
        edges: Set[Tuple[int, int]],
        nodes: List[GraphNode],
        i: int,
        j: int
    ) -> bool:
        if i == j:
            return False

        a, b = min(i, j), max(i, j)

        if (a, b) in edges:
            return False

        n1 = nodes[a]
        n2 = nodes[b]

        if not self.edge_line_is_acceptable(n1, n2):
            return False

        if self.enable_edge_intersection_check:
            p1 = (n1.x, n1.y)
            p2 = (n2.x, n2.y)

            for e1, e2 in edges:
                # 共享端点不算交叉
                if a in (e1, e2) or b in (e1, e2):
                    continue

                p3 = (nodes[e1].x, nodes[e1].y)
                p4 = (nodes[e2].x, nodes[e2].y)

                if self.segments_intersect(p1, p2, p3, p4):
                    return False

        edges.add((a, b))
        return True

    def segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> bool:
        eps = 1e-9

        def orient(a, b, c) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on_segment(a, b, c) -> bool:
            return (
                min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
                min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps and
                abs(orient(a, b, c)) <= eps
            )

        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)

        if (o1 * o2 < -eps) and (o3 * o4 < -eps):
            return True

        # 共线或端点落在线段上时，也按交叉处理；共享端点已在 try_add_edge 中排除。
        if abs(o1) <= eps and on_segment(p1, p2, p3):
            return True
        if abs(o2) <= eps and on_segment(p1, p2, p4):
            return True
        if abs(o3) <= eps and on_segment(p3, p4, p1):
            return True
        if abs(o4) <= eps and on_segment(p3, p4, p2):
            return True

        return False

    def find_safe_runs_on_row(self, row: int) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        inside = False
        start = 0

        arr = self.safe_free_arr
        if arr is None:
            return runs

        assert self.free_msg is not None
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

        length = end_col - start_col + 1

        # 短 segment 用中点，避免过短安全段生成两个贴得很近的端点
        if length <= max(2, int(0.75 * step)):
            return [(start_col + end_col) // 2]

        cols = list(range(start_col, end_col + 1, step))
        if cols[-1] != end_col:
            cols.append(end_col)

        return sorted(set(cols))

    def has_frontier_near(self, row: int, col: int, rad: int) -> bool:
        assert self.frontier_arr is not None
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

    def is_relaxed_edge_cell(self, row: int, col: int) -> bool:
        if self.relaxed_edge_arr is None:
            return False

        if row < 0 or row >= self.relaxed_edge_arr.shape[0]:
            return False
        if col < 0 or col >= self.relaxed_edge_arr.shape[1]:
            return False

        return bool(self.relaxed_edge_arr[row, col])

    def border_line_is_safe(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """
        边界/障碍物附近端点的放松连边检查：
        - 不检查 map_border_buffer；
        - 不检查 obstacle_buffer；
        - 仍然要求整条线在 free 区域内；
        - 仍然禁止穿过真实 obstacle 和 unknown_buffer。
        """
        if self.relaxed_edge_arr is None:
            return False

        n = max(abs(r1 - r0), abs(c1 - c0)) + 1

        for i in range(n + 1):
            t = 0.0 if n == 0 else i / n
            rr = int(round((1.0 - t) * r0 + t * r1))
            cc = int(round((1.0 - t) * c0 + t * c1))

            if not self.is_relaxed_edge_cell(rr, cc):
                return False

        return True

    def is_near_map_border(self, node: GraphNode) -> bool:
        if self.free_msg is None:
            return False

        info = self.free_msg.info
        res = info.resolution
        border_rad = max(1, int(math.ceil(self.map_border_buffer / res)))

        return (
            node.row <= border_rad or
            node.col <= border_rad or
            node.row >= info.height - 1 - border_rad or
            node.col >= info.width - 1 - border_rad
        )

    def is_near_obstacle(self, node: GraphNode) -> bool:
        if self.obstacle_arr is None or self.free_msg is None:
            return False

        res = self.free_msg.info.resolution
        rad = max(1, int(math.ceil(self.obstacle_buffer / res)))

        r0 = max(0, node.row - rad)
        r1 = min(self.obstacle_arr.shape[0], node.row + rad + 1)
        c0 = max(0, node.col - rad)
        c1 = min(self.obstacle_arr.shape[1], node.col + rad + 1)

        return bool(np.any(self.obstacle_arr[r0:r1, c0:c1]))

    def is_relaxable_endpoint(self, node: GraphNode) -> bool:
        # 只对端点开放放松检查，普通中间 sample 仍使用严格 safe_free_arr。
        if not node.endpoint:
            return False

        return self.is_near_map_border(node) or self.is_near_obstacle(node)

    def edge_line_is_acceptable(self, n1: GraphNode, n2: GraphNode) -> bool:
        # 优先使用严格检查，保证大多数边仍然遵守正常 buffer。
        if self.line_is_safe(n1.row, n1.col, n2.row, n2.col):
            return True

        # 只有靠近地图边界/障碍物的端点，才允许使用放松检查。
        if not (self.is_relaxable_endpoint(n1) or self.is_relaxable_endpoint(n2)):
            return False

        return self.border_line_is_safe(n1.row, n1.col, n2.row, n2.col)

    def cell_to_world(self, col: int, row: int) -> Tuple[float, float]:
        assert self.free_msg is not None

        info = self.free_msg.info
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def collect_lap_rows_from_safe_mask(self, lap_step: int) -> List[int]:
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

            if band_height <= max(2, lap_step // 2):
                rows.add((start + end) // 2)
                continue

            local_rows: List[int] = []
            r = start
            while r <= end:
                local_rows.append(r)
                r += lap_step

            if local_rows:
                rows.update(local_rows)
                if end - local_rows[-1] >= max(2, lap_step // 2):
                    rows.add(end)

        return sorted(rows)

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
