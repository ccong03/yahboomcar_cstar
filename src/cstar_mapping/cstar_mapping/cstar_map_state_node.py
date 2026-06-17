#!/usr/bin/env python3
import math
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class CStarMapStateNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_map_state_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('coverage_radius', 0.15) #原始0.20
        self.declare_parameter('obstacle_threshold', 50)
        self.declare_parameter('publish_period', 0.5)

        # 轨迹显示参数
        self.declare_parameter('trajectory_topic', '/cstar/trajectory_marker')
        self.declare_parameter('trajectory_min_step', 0.03)
        self.declare_parameter('trajectory_width', 0.03)
        self.declare_parameter('trajectory_alpha', 0.45)
        self.declare_parameter('trajectory_max_points', 20000)

        self.map_topic = self.get_parameter('map_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.coverage_radius = float(self.get_parameter('coverage_radius').value)
        self.obstacle_threshold = int(self.get_parameter('obstacle_threshold').value)
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.trajectory_topic = self.get_parameter('trajectory_topic').value
        self.trajectory_min_step = float(self.get_parameter('trajectory_min_step').value)
        self.trajectory_width = float(self.get_parameter('trajectory_width').value)
        self.trajectory_alpha = float(self.get_parameter('trajectory_alpha').value)
        self.trajectory_max_points = int(self.get_parameter('trajectory_max_points').value)

        self.latest_map: Optional[OccupancyGrid] = None
        self.map_array: Optional[np.ndarray] = None
        self.covered_mask: Optional[np.ndarray] = None

        self.last_width: Optional[int] = None
        self.last_height: Optional[int] = None
        self.last_resolution: Optional[float] = None
        self.last_origin_x: Optional[float] = None
        self.last_origin_y: Optional[float] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )

        self.unknown_pub = self.create_publisher(OccupancyGrid, '/cstar/unknown_map', 10)
        self.free_pub = self.create_publisher(OccupancyGrid, '/cstar/free_map', 10)
        self.obstacle_pub = self.create_publisher(OccupancyGrid, '/cstar/obstacle_map', 10)
        self.covered_pub = self.create_publisher(OccupancyGrid, '/cstar/covered_map', 10)

        self.trajectory_pub = self.create_publisher(Marker, self.trajectory_topic, 10)

        self.timer = self.create_timer(self.publish_period, self.on_timer)

        self.no_tf_warn_count = 0

        self.trajectory_points: List[Point] = []
        self.last_traj_xy: Optional[Tuple[float, float]] = None

        self.get_logger().info('CStarMapStateNode started.')
        self.get_logger().info(f'map_topic={self.map_topic}')
        self.get_logger().info(f'map_frame={self.map_frame}, base_frame={self.base_frame}')
        self.get_logger().info(f'coverage_radius={self.coverage_radius:.3f} m')
        self.get_logger().info(f'trajectory_topic={self.trajectory_topic}')

    def map_callback(self, msg: OccupancyGrid) -> None:
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        metadata_changed = (
            self.last_width != width or
            self.last_height != height or
            self.last_resolution != resolution or
            self.last_origin_x != origin_x or
            self.last_origin_y != origin_y
        )

        old_covered = None if self.covered_mask is None else self.covered_mask.copy()
        old_width = self.last_width
        old_height = self.last_height
        old_resolution = self.last_resolution
        old_origin_x = self.last_origin_x
        old_origin_y = self.last_origin_y

        self.latest_map = msg
        self.map_array = np.asarray(msg.data, dtype=np.int16).reshape((height, width))

        if self.covered_mask is None:
            self.covered_mask = np.zeros((height, width), dtype=np.int8)
            self.get_logger().info(
                f'Initialized covered mask: {width}x{height}, '
                f'resolution={resolution:.3f}, origin=({origin_x:.3f}, {origin_y:.3f})'
            )
        elif metadata_changed:
            self.covered_mask = self.remap_old_covered_mask(
                old_mask=old_covered,
                old_width=old_width,
                old_height=old_height,
                old_resolution=old_resolution,
                old_origin_x=old_origin_x,
                old_origin_y=old_origin_y,
                new_width=width,
                new_height=height,
                new_resolution=resolution,
                new_origin_x=origin_x,
                new_origin_y=origin_y
            )
            covered_count = int(np.count_nonzero(self.covered_mask > 0))
            self.get_logger().info(
                f'Map metadata changed -> remapped covered mask to {width}x{height}, '
                f'covered_cells={covered_count}'
            )

        self.last_width = width
        self.last_height = height
        self.last_resolution = resolution
        self.last_origin_x = origin_x
        self.last_origin_y = origin_y

        self.sanitize_covered_mask()

    def remap_old_covered_mask(
        self,
        old_mask: Optional[np.ndarray],
        old_width: Optional[int],
        old_height: Optional[int],
        old_resolution: Optional[float],
        old_origin_x: Optional[float],
        old_origin_y: Optional[float],
        new_width: int,
        new_height: int,
        new_resolution: float,
        new_origin_x: float,
        new_origin_y: float
    ) -> np.ndarray:
        new_mask = np.zeros((new_height, new_width), dtype=np.int8)

        if (
            old_mask is None or
            old_width is None or
            old_height is None or
            old_resolution is None or
            old_origin_x is None or
            old_origin_y is None
        ):
            return new_mask

        covered_indices = np.argwhere(old_mask > 0)
        if covered_indices.size == 0:
            return new_mask

        old_rows = covered_indices[:, 0].astype(np.float64)
        old_cols = covered_indices[:, 1].astype(np.float64)

        # 旧格子中心 -> 世界坐标
        world_x = old_origin_x + (old_cols + 0.5) * old_resolution
        world_y = old_origin_y + (old_rows + 0.5) * old_resolution

        # 世界坐标 -> 新格子
        new_cols = np.floor((world_x - new_origin_x) / new_resolution).astype(np.int32)
        new_rows = np.floor((world_y - new_origin_y) / new_resolution).astype(np.int32)

        valid = (
            (new_cols >= 0) & (new_cols < new_width) &
            (new_rows >= 0) & (new_rows < new_height)
        )

        new_mask[new_rows[valid], new_cols[valid]] = 100
        return new_mask

    def get_robot_pose_in_map(self) -> Optional[Tuple[float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.1)
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            return x, y
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.no_tf_warn_count += 1
            if self.no_tf_warn_count % 20 == 1:
                self.get_logger().warn(
                    f'Cannot get TF: {self.map_frame} -> {self.base_frame}'
                )
            return None

    def world_to_map(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        if self.latest_map is None:
            return None

        info = self.latest_map.info
        mx = int((wx - info.origin.position.x) / info.resolution)
        my = int((wy - info.origin.position.y) / info.resolution)

        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None

        return mx, my

    def mark_covered(self, wx: float, wy: float) -> None:
        if self.latest_map is None or self.map_array is None or self.covered_mask is None:
            return

        map_xy = self.world_to_map(wx, wy)
        if map_xy is None:
            return

        mx, my = map_xy
        resolution = self.latest_map.info.resolution
        radius_cells = max(1, int(math.ceil(self.coverage_radius / resolution)))

        height, width = self.covered_mask.shape

        for dy in range(-radius_cells, radius_cells + 1):
            cy = my + dy
            if cy < 0 or cy >= height:
                continue
            for dx in range(-radius_cells, radius_cells + 1):
                cx = mx + dx
                if cx < 0 or cx >= width:
                    continue

                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue

                cell_value = int(self.map_array[cy, cx])

                # 只把“已知自由空间”记为 covered
                if 0 <= cell_value < self.obstacle_threshold:
                    self.covered_mask[cy, cx] = 100

    def sanitize_covered_mask(self) -> None:
        if self.map_array is None or self.covered_mask is None:
            return

        invalid = (self.map_array < 0) | (self.map_array >= self.obstacle_threshold)
        self.covered_mask[invalid] = 0

    def update_trajectory(self, wx: float, wy: float) -> None:
        if self.last_traj_xy is not None:
            dx = wx - self.last_traj_xy[0]
            dy = wy - self.last_traj_xy[1]
            dist = math.hypot(dx, dy)
            if dist < self.trajectory_min_step:
                return

        p = Point()
        p.x = float(wx)
        p.y = float(wy)
        p.z = 0.03
        self.trajectory_points.append(p)
        self.last_traj_xy = (wx, wy)

        if len(self.trajectory_points) > self.trajectory_max_points:
            overflow = len(self.trajectory_points) - self.trajectory_max_points
            self.trajectory_points = self.trajectory_points[overflow:]

    def build_grid_from_mask(self, mask: np.ndarray) -> OccupancyGrid:
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = self.map_frame
        grid.info = self.latest_map.info
        grid.data = mask.reshape(-1).astype(np.int8).tolist()
        return grid

    def build_trajectory_marker(self) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'cstar_trajectory'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = self.trajectory_width

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = self.trajectory_alpha

        marker.points = self.trajectory_points
        return marker

    def on_timer(self) -> None:
        if self.latest_map is None or self.map_array is None or self.covered_mask is None:
            return

        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is not None:
            rx, ry = robot_pose
            self.mark_covered(rx, ry)
            self.update_trajectory(rx, ry)

        self.sanitize_covered_mask()

        unknown_mask = np.where(self.map_array == -1, 100, 0).astype(np.int8)
        obstacle_mask = np.where(self.map_array >= self.obstacle_threshold, 100, 0).astype(np.int8)
        free_mask = np.where(
            (self.map_array >= 0) & (self.map_array < self.obstacle_threshold),
            100,
            0
        ).astype(np.int8)

        self.unknown_pub.publish(self.build_grid_from_mask(unknown_mask))
        self.free_pub.publish(self.build_grid_from_mask(free_mask))
        self.obstacle_pub.publish(self.build_grid_from_mask(obstacle_mask))
        self.covered_pub.publish(self.build_grid_from_mask(self.covered_mask))
        self.trajectory_pub.publish(self.build_trajectory_marker())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CStarMapStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
