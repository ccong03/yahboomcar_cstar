#!/usr/bin/env python3
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


GridCell = Tuple[int, int]  # row, col


class CStarHoleDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_hole_detector_node')

        self.declare_parameter('free_map_topic', '/cstar/free_map')
        self.declare_parameter('covered_map_topic', '/cstar/covered_map')
        self.declare_parameter('obstacle_map_topic', '/cstar/obstacle_map')
        self.declare_parameter('unknown_map_topic', '/cstar/unknown_map')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        self.declare_parameter('publish_period', 0.5)

        # 小车至少累计走过这么远之后，才开始检测 hole
        self.declare_parameter('start_after_robot_moved_distance', 0.50)

        # 只检测机器人附近的 hole，避免一启动就全图乱报
        self.declare_parameter('local_detection_radius', 1.50)

        # hole 面积过滤，单位 m^2
        self.declare_parameter('min_hole_area', 0.05)
        self.declare_parameter('max_hole_area', 4.00)

        # unknown 附近不要判定为 hole，避免把 frontier 当成 hole
        self.declare_parameter('unknown_buffer', 0.15)

        # covered 轻微膨胀，用于封住 covered_map 里的小缝隙
        # 如果门口明明被扫过但栅格有小洞，这个参数能提高 hole 检测稳定性
        self.declare_parameter('covered_seal_buffer', 0.06)

        # 边界封闭性要求。越大越严格。
        # 这里默认不要太高，否则仿真地图锯齿多时容易漏检。
        self.declare_parameter('min_enclosed_boundary_ratio', 0.35)

        # 要求 hole 边界附近至少有少量 covered 区域，否则可能只是未探索大区域
        self.declare_parameter('min_covered_boundary_cells', 3)

        self.free_map_topic = self.get_parameter('free_map_topic').value
        self.covered_map_topic = self.get_parameter('covered_map_topic').value
        self.obstacle_map_topic = self.get_parameter('obstacle_map_topic').value
        self.unknown_map_topic = self.get_parameter('unknown_map_topic').value

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.publish_period = float(self.get_parameter('publish_period').value)

        self.start_after_robot_moved_distance = float(
            self.get_parameter('start_after_robot_moved_distance').value
        )
        self.local_detection_radius = float(self.get_parameter('local_detection_radius').value)

        self.min_hole_area = float(self.get_parameter('min_hole_area').value)
        self.max_hole_area = float(self.get_parameter('max_hole_area').value)

        self.unknown_buffer = float(self.get_parameter('unknown_buffer').value)
        self.covered_seal_buffer = float(self.get_parameter('covered_seal_buffer').value)

        self.min_enclosed_boundary_ratio = float(
            self.get_parameter('min_enclosed_boundary_ratio').value
        )
        self.min_covered_boundary_cells = int(
            self.get_parameter('min_covered_boundary_cells').value
        )

        self.free_msg: Optional[OccupancyGrid] = None
        self.covered_msg: Optional[OccupancyGrid] = None
        self.obstacle_msg: Optional[OccupancyGrid] = None
        self.unknown_msg: Optional[OccupancyGrid] = None

        self.free_arr: Optional[np.ndarray] = None
        self.covered_arr: Optional[np.ndarray] = None
        self.obstacle_arr: Optional[np.ndarray] = None
        self.unknown_arr: Optional[np.ndarray] = None

        self.robot_moved_distance = 0.0
        self.last_robot_xy: Optional[Tuple[float, float]] = None
        self.detection_started = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            OccupancyGrid,
            self.free_map_topic,
            self.free_callback,
            10
        )
        self.create_subscription(
            OccupancyGrid,
            self.covered_map_topic,
            self.covered_callback,
            10
        )
        self.create_subscription(
            OccupancyGrid,
            self.obstacle_map_topic,
            self.obstacle_callback,
            10
        )
        self.create_subscription(
            OccupancyGrid,
            self.unknown_map_topic,
            self.unknown_callback,
            10
        )

        self.hole_map_pub = self.create_publisher(
            OccupancyGrid,
            '/cstar/hole_map',
            10
        )

        self.hole_candidate_pub = self.create_publisher(
            OccupancyGrid,
            '/cstar/hole_candidate_map',
            10
        )

        self.local_hole_candidate_pub = self.create_publisher(
            OccupancyGrid,
            '/cstar/local_hole_candidate_map',
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/cstar/hole_markers',
            10
        )

        self.timer = self.create_timer(self.publish_period, self.on_timer)

        self.get_logger().info('CStarHoleDetectorNode started.')
        self.get_logger().info(f'free_map_topic={self.free_map_topic}')
        self.get_logger().info(f'covered_map_topic={self.covered_map_topic}')
        self.get_logger().info(f'obstacle_map_topic={self.obstacle_map_topic}')
        self.get_logger().info(f'unknown_map_topic={self.unknown_map_topic}')
        self.get_logger().info(
            f'start_after_robot_moved_distance={self.start_after_robot_moved_distance:.2f} m, '
            f'local_detection_radius={self.local_detection_radius:.2f} m'
        )
        self.get_logger().info(
            f'min_hole_area={self.min_hole_area:.2f}, '
            f'max_hole_area={self.max_hole_area:.2f}, '
            f'unknown_buffer={self.unknown_buffer:.2f}, '
            f'covered_seal_buffer={self.covered_seal_buffer:.2f}'
        )

    def free_callback(self, msg: OccupancyGrid) -> None:
        self.free_msg = msg
        self.free_arr = self.grid_to_bool(msg)

    def covered_callback(self, msg: OccupancyGrid) -> None:
        self.covered_msg = msg
        self.covered_arr = self.grid_to_bool(msg)

    def obstacle_callback(self, msg: OccupancyGrid) -> None:
        self.obstacle_msg = msg
        self.obstacle_arr = self.grid_to_bool(msg)

    def unknown_callback(self, msg: OccupancyGrid) -> None:
        self.unknown_msg = msg
        self.unknown_arr = self.grid_to_bool(msg)

    def grid_to_bool(self, msg: OccupancyGrid) -> np.ndarray:
        h = msg.info.height
        w = msg.info.width
        arr = np.asarray(msg.data, dtype=np.int16).reshape((h, w))
        return arr > 50

    def maps_ready(self) -> bool:
        if self.free_msg is None or self.covered_msg is None:
            return False
        if self.free_arr is None or self.covered_arr is None:
            return False

        base = self.free_msg.info

        msgs = [self.covered_msg]
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

    def update_robot_distance(self, robot_xy: Optional[Tuple[float, float]]) -> None:
        if robot_xy is None:
            return

        if self.last_robot_xy is None:
            self.last_robot_xy = robot_xy
            return

        lx, ly = self.last_robot_xy
        rx, ry = robot_xy

        d = math.hypot(rx - lx, ry - ly)

        # 避免 TF 抖动导致距离乱累计
        if 0.01 <= d <= 0.30:
            self.robot_moved_distance += d
            self.last_robot_xy = robot_xy
        elif d > 0.30:
            # 出现跳变时只更新位置，不累计距离
            self.last_robot_xy = robot_xy

        if (
            not self.detection_started and
            self.robot_moved_distance >= self.start_after_robot_moved_distance
        ):
            self.detection_started = True
            self.get_logger().info(
                f'Hole detection enabled after robot moved '
                f'{self.robot_moved_distance:.2f} m.'
            )

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

    def build_local_gate(self, robot_xy: Tuple[float, float]) -> np.ndarray:
        assert self.free_msg is not None

        h = self.free_msg.info.height
        w = self.free_msg.info.width
        res = self.free_msg.info.resolution

        robot_cell = self.world_to_cell(robot_xy[0], robot_xy[1])

        gate = np.zeros((h, w), dtype=bool)

        if robot_cell is None:
            return gate

        rr, cc = robot_cell
        rad = max(1, int(math.ceil(self.local_detection_radius / res)))
        rad2 = rad * rad

        r0 = max(0, rr - rad)
        r1 = min(h - 1, rr + rad)
        c0 = max(0, cc - rad)
        c1 = min(w - 1, cc + rad)

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                dr = r - rr
                dc = c - cc

                if dr * dr + dc * dc <= rad2:
                    gate[r, c] = True

        return gate

    def build_candidate_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        返回：
        candidate:
            全图 hole 候选区域。
        local_candidate:
            机器人附近的 hole 候选区域。
        covered_seal:
            轻微膨胀后的 covered，用于封闭小缝隙。
        obstacle:
            obstacle mask。
        """
        assert self.free_msg is not None
        assert self.free_arr is not None
        assert self.covered_arr is not None

        info = self.free_msg.info
        res = info.resolution

        free = self.free_arr.copy()
        covered = self.covered_arr.copy()

        if self.obstacle_arr is not None and self.obstacle_arr.shape == free.shape:
            obstacle = self.obstacle_arr.copy()
        else:
            obstacle = np.logical_not(free)

        if self.unknown_arr is not None and self.unknown_arr.shape == free.shape:
            unknown = self.unknown_arr.copy()
        else:
            unknown = np.zeros_like(free, dtype=bool)

        unknown_rad = max(0, int(math.ceil(self.unknown_buffer / res)))
        covered_rad = max(0, int(math.ceil(self.covered_seal_buffer / res)))

        unknown_block = self.dilate_bool(unknown, unknown_rad)
        covered_seal = self.dilate_bool(covered, covered_rad)

        candidate = free.copy()
        candidate[obstacle] = False
        candidate[covered_seal] = False
        candidate[unknown_block] = False

        robot_xy = self.get_robot_pose()
        if robot_xy is None:
            local_candidate = np.zeros_like(candidate, dtype=bool)
        else:
            local_gate = self.build_local_gate(robot_xy)
            local_candidate = candidate & local_gate

        return candidate, local_candidate, covered_seal, obstacle

    def connected_components(self, mask: np.ndarray) -> List[List[GridCell]]:
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)

        components: List[List[GridCell]] = []

        for row in range(h):
            for col in range(w):
                if not mask[row, col] or visited[row, col]:
                    continue

                comp: List[GridCell] = []
                q = deque()
                q.append((row, col))
                visited[row, col] = True

                while q:
                    r, c = q.popleft()
                    comp.append((r, c))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr = r + dr
                        nc = c + dc

                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue

                        if visited[nr, nc]:
                            continue

                        if not mask[nr, nc]:
                            continue

                        visited[nr, nc] = True
                        q.append((nr, nc))

                components.append(comp)

        return components

    def component_touches_border(self, comp: List[GridCell], h: int, w: int) -> bool:
        for r, c in comp:
            if r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1:
                return True
        return False

    def component_has_local_overlap(
        self,
        comp: List[GridCell],
        local_candidate: np.ndarray
    ) -> bool:
        for r, c in comp:
            if local_candidate[r, c]:
                return True
        return False

    def component_boundary_stats(
        self,
        comp: List[GridCell],
        covered_seal: np.ndarray,
        obstacle: np.ndarray
    ) -> Tuple[float, int]:
        h, w = covered_seal.shape

        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True

        dilated = self.dilate_bool(comp_mask, 1)
        boundary = dilated & np.logical_not(comp_mask)

        boundary_count = int(np.count_nonzero(boundary))
        if boundary_count <= 0:
            return 0.0, 0

        closed_boundary = boundary & (covered_seal | obstacle)
        covered_boundary = boundary & covered_seal

        closed_count = int(np.count_nonzero(closed_boundary))
        covered_count = int(np.count_nonzero(covered_boundary))

        ratio = float(closed_count) / float(boundary_count)
        return ratio, covered_count

    def component_centroid(self, comp: List[GridCell]) -> Tuple[float, float]:
        row_mean = sum(r for r, _ in comp) / float(len(comp))
        col_mean = sum(c for _, c in comp) / float(len(comp))

        return self.cell_to_world((int(round(row_mean)), int(round(col_mean))))

    def detect_holes(
        self,
        candidate: np.ndarray,
        local_candidate: np.ndarray,
        covered_seal: np.ndarray,
        obstacle: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
        """
        返回：
        hole_mask:
            检测到的 hole 区域。
        hole_infos:
            [(cx, cy, area), ...]
        """
        assert self.free_msg is not None

        info = self.free_msg.info
        res = info.resolution
        h = info.height
        w = info.width

        hole_mask = np.zeros((h, w), dtype=np.int8)
        hole_infos: List[Tuple[float, float, float]] = []

        components = self.connected_components(candidate)

        for comp in components:
            cell_count = len(comp)
            area = cell_count * res * res

            if area < self.min_hole_area:
                continue

            if area > self.max_hole_area:
                continue

            if self.component_touches_border(comp, h, w):
                continue

            # 关键：只保留机器人附近的 hole，防止全图提前报 hole
            if not self.component_has_local_overlap(comp, local_candidate):
                continue

            boundary_ratio, covered_boundary_cells = self.component_boundary_stats(
                comp,
                covered_seal,
                obstacle
            )

            if boundary_ratio < self.min_enclosed_boundary_ratio:
                continue

            if covered_boundary_cells < self.min_covered_boundary_cells:
                continue

            for r, c in comp:
                hole_mask[r, c] = 100

            cx, cy = self.component_centroid(comp)
            hole_infos.append((cx, cy, area))

        return hole_mask, hole_infos

    def build_grid_from_mask(self, mask: np.ndarray) -> OccupancyGrid:
        assert self.free_msg is not None

        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.map_frame
        grid.info = self.free_msg.info

        if mask.dtype == bool:
            out = np.where(mask, 100, 0).astype(np.int8)
        else:
            out = mask.astype(np.int8)

        grid.data = out.reshape(-1).tolist()
        return grid

    def publish_empty_outputs(self) -> None:
        if self.free_msg is None:
            self.publish_delete_markers()
            return

        h = self.free_msg.info.height
        w = self.free_msg.info.width
        empty = np.zeros((h, w), dtype=np.int8)

        self.hole_map_pub.publish(self.build_grid_from_mask(empty))
        self.hole_candidate_pub.publish(self.build_grid_from_mask(empty))
        self.local_hole_candidate_pub.publish(self.build_grid_from_mask(empty))
        self.publish_delete_markers()

    def publish_delete_markers(self) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL

        ma.markers.append(delete_all)
        self.marker_pub.publish(ma)

    def publish_markers(self, hole_infos: List[Tuple[float, float, float]]) -> None:
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        centers = Marker()
        centers.header.stamp = self.get_clock().now().to_msg()
        centers.header.frame_id = self.map_frame
        centers.ns = 'hole_centers'
        centers.id = 0
        centers.type = Marker.SPHERE_LIST
        centers.action = Marker.ADD
        centers.scale.x = 0.16
        centers.scale.y = 0.16
        centers.scale.z = 0.16
        centers.color.r = 1.0
        centers.color.g = 0.0
        centers.color.b = 1.0
        centers.color.a = 0.95
        centers.pose.orientation.w = 1.0

        for i, (x, y, area) in enumerate(hole_infos):
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.10
            centers.points.append(p)

            text = Marker()
            text.header.stamp = self.get_clock().now().to_msg()
            text.header.frame_id = self.map_frame
            text.ns = 'hole_labels'
            text.id = i + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.25
            text.pose.orientation.w = 1.0
            text.scale.z = 0.16
            text.color.r = 1.0
            text.color.g = 0.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f'hole\n{area:.2f}m2'
            ma.markers.append(text)

        ma.markers.append(centers)
        self.marker_pub.publish(ma)

    def on_timer(self) -> None:
        if not self.maps_ready():
            return

        robot_xy = self.get_robot_pose()
        self.update_robot_distance(robot_xy)

        if not self.detection_started:
            self.publish_empty_outputs()
            return

        candidate, local_candidate, covered_seal, obstacle = self.build_candidate_mask()

        hole_mask, hole_infos = self.detect_holes(
            candidate,
            local_candidate,
            covered_seal,
            obstacle
        )

        self.hole_candidate_pub.publish(self.build_grid_from_mask(candidate))
        self.local_hole_candidate_pub.publish(self.build_grid_from_mask(local_candidate))
        self.hole_map_pub.publish(self.build_grid_from_mask(hole_mask))
        self.publish_markers(hole_infos)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CStarHoleDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()