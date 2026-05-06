#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CStarSimpleControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('cstar_simple_controller_node')

        self.declare_parameter('goal_topic', '/cstar/goal')
        self.declare_parameter('escape_path_topic', '/cstar/escape_path')
        self.declare_parameter('cmd_vel_topic', '/mecanum_controller/reference_unstamped')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        self.declare_parameter('control_rate', 20.0)

        # 普通目标点到达阈值
        self.declare_parameter('goal_tolerance', 0.15) #建议0.16

        # escape path 最后一个点的到达阈值
        self.declare_parameter('escape_final_tolerance', 0.18) #建议0.08

        # 普通目标点控制参数
        self.declare_parameter('heading_tolerance', 0.35)
        self.declare_parameter('max_linear_speed_normal', 0.08) #建议0.06
        self.declare_parameter('max_linear_speed_escape', 0.07) #建议0.05
        self.declare_parameter('max_angular_speed', 0.25) #建议0.25

        self.declare_parameter('min_linear_speed', 0.015)
        self.declare_parameter('min_angular_speed', 0.06)

        self.declare_parameter('k_linear', 0.45)
        self.declare_parameter('k_angular', 0.75)

        # Pure Pursuit 参数
        self.declare_parameter('lookahead_distance', 0.25) #建议0.25
        self.declare_parameter('turn_in_place_angle', 0.75)
        self.declare_parameter('slowdown_angle', 0.45)
        self.declare_parameter('pure_pursuit_gain', 1.0)

        # escape path 超时，防止旧路径一直被跟踪
        self.declare_parameter('path_timeout', 2.0)

        self.goal_topic = self.get_parameter('goal_topic').value
        self.escape_path_topic = self.get_parameter('escape_path_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.control_rate = float(self.get_parameter('control_rate').value)

        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.escape_final_tolerance = float(self.get_parameter('escape_final_tolerance').value)

        self.heading_tolerance = float(self.get_parameter('heading_tolerance').value)
        self.max_linear_speed_normal = float(self.get_parameter('max_linear_speed_normal').value)
        self.max_linear_speed_escape = float(self.get_parameter('max_linear_speed_escape').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        self.min_linear_speed = float(self.get_parameter('min_linear_speed').value)
        self.min_angular_speed = float(self.get_parameter('min_angular_speed').value)

        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)

        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.turn_in_place_angle = float(self.get_parameter('turn_in_place_angle').value)
        self.slowdown_angle = float(self.get_parameter('slowdown_angle').value)
        self.pure_pursuit_gain = float(self.get_parameter('pure_pursuit_gain').value)

        self.path_timeout = float(self.get_parameter('path_timeout').value)

        self.current_goal: Optional[PoseStamped] = None

        # escape path 中只保存二维点
        self.escape_path: List[Tuple[float, float]] = []
        self.last_escape_path_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self.goal_callback,
            10
        )

        self.escape_path_sub = self.create_subscription(
            Path,
            self.escape_path_topic,
            self.escape_path_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )

        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.on_timer
        )

        self.get_logger().info('CStarSimpleControllerNode started.')
        self.get_logger().info(f'goal_topic={self.goal_topic}')
        self.get_logger().info(f'escape_path_topic={self.escape_path_topic}')
        self.get_logger().info(f'cmd_vel_topic={self.cmd_vel_topic}')
        self.get_logger().info(f'frame={self.map_frame} -> {self.base_frame}')
        self.get_logger().info(
            f'Pure Pursuit: lookahead={self.lookahead_distance:.2f}, '
            f'escape_v={self.max_linear_speed_escape:.2f}, '
            f'max_w={self.max_angular_speed:.2f}'
        )

    def goal_callback(self, msg: PoseStamped) -> None:
        self.current_goal = msg

    def escape_path_callback(self, msg: Path) -> None:
        path: List[Tuple[float, float]] = []

        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            y = pose_stamped.pose.position.y
            path.append((x, y))

        self.escape_path = path
        self.last_escape_path_time = self.get_clock().now()

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.05)
            )

            x = tf.transform.translation.x
            y = tf.transform.translation.y

            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            return x, y, yaw

        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def apply_min_angular_speed(self, w: float) -> float:
        if abs(w) < 1e-6:
            return 0.0

        sign = 1.0 if w > 0.0 else -1.0
        abs_w = max(abs(w), self.min_angular_speed)
        return sign * min(abs_w, self.max_angular_speed)

    def has_valid_escape_path(self) -> bool:
        if len(self.escape_path) < 2:
            return False

        if self.last_escape_path_time is None:
            return False

        if self.path_timeout <= 0.0:
            return True

        dt = self.get_clock().now() - self.last_escape_path_time
        if dt.nanoseconds / 1e9 > self.path_timeout:
            return False

        return True

    def world_to_robot_frame(
        self,
        rx: float,
        ry: float,
        ryaw: float,
        tx: float,
        ty: float
    ) -> Tuple[float, float]:
        dx = tx - rx
        dy = ty - ry

        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)

        # map/world -> robot/base frame
        x_r = cos_yaw * dx + sin_yaw * dy
        y_r = -sin_yaw * dx + cos_yaw * dy

        return x_r, y_r

    def find_lookahead_point(
        self,
        robot_xy: Tuple[float, float],
        path: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        if not path:
            return robot_xy

        # 找离机器人最近的 path 点
        nearest_i = 0
        nearest_d = float('inf')

        for i, p in enumerate(path):
            d = distance_xy(robot_xy, p)
            if d < nearest_d:
                nearest_d = d
                nearest_i = i

        # 从最近点开始，沿路径累计距离，找 lookahead 点
        target = path[-1]
        accumulated = distance_xy(robot_xy, path[nearest_i])

        if accumulated >= self.lookahead_distance:
            return path[nearest_i]

        for i in range(nearest_i, len(path) - 1):
            seg_len = distance_xy(path[i], path[i + 1])
            accumulated += seg_len

            if accumulated >= self.lookahead_distance:
                target = path[i + 1]
                break

        return target

    def track_escape_path(
        self,
        rx: float,
        ry: float,
        ryaw: float
    ) -> None:
        if len(self.escape_path) < 2:
            self.stop_robot()
            return

        final_goal = self.escape_path[-1]
        final_dist = distance_xy((rx, ry), final_goal)

        if final_dist <= self.escape_final_tolerance:
            # A* 路径已经基本跟踪完，但还不能直接停车。
            # 因为 escape_path 的终点可能只是 retreat_node 附近的 safe grid cell，
            # 不一定和真正的 /cstar/goal 完全重合。
            # 所以这里切换成普通 goal 跟踪，让小车继续贴近黄色 retreat_node。
            self.track_single_goal(rx, ry, ryaw)
            return

        target_x, target_y = self.find_lookahead_point((rx, ry), self.escape_path)

        x_r, y_r = self.world_to_robot_frame(
            rx, ry, ryaw,
            target_x, target_y
        )

        target_dist = math.hypot(x_r, y_r)

        if target_dist < 1e-6:
            self.stop_robot()
            return

        alpha = math.atan2(y_r, x_r)

        cmd = Twist()

        # 如果前视点在车后方，或者角度偏差太大，先原地慢慢转
        if x_r < -0.05 or abs(alpha) > self.turn_in_place_angle:
            w = clamp(
                self.k_angular * alpha,
                -self.max_angular_speed,
                self.max_angular_speed
            )
            cmd.linear.x = 0.0
            cmd.angular.z = self.apply_min_angular_speed(w)
            self.cmd_pub.publish(cmd)
            return

        # Pure Pursuit 曲率
        lookahead = max(target_dist, 1e-3)
        curvature = 2.0 * y_r / (lookahead * lookahead)

        # 转弯越大，线速度越低
        abs_alpha = abs(alpha)
        if abs_alpha >= self.slowdown_angle:
            speed_scale = 0.35
        else:
            speed_scale = 1.0 - 0.5 * (abs_alpha / max(self.slowdown_angle, 1e-3))

        v = clamp(
            self.k_linear * final_dist,
            self.min_linear_speed,
            self.max_linear_speed_escape
        )
        v *= speed_scale

        w = self.pure_pursuit_gain * v * curvature
        w = clamp(w, -self.max_angular_speed, self.max_angular_speed)

        cmd.linear.x = v
        cmd.linear.y = 0.0
        cmd.angular.z = w

        self.cmd_pub.publish(cmd)

    def track_single_goal(
        self,
        rx: float,
        ry: float,
        ryaw: float
    ) -> None:
        if self.current_goal is None:
            self.stop_robot()
            return

        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y

        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)

        if dist < self.goal_tolerance:
            self.stop_robot()
            return

        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - ryaw)

        cmd = Twist()

        if abs(yaw_error) > self.heading_tolerance:
            w = clamp(
                self.k_angular * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed
            )
            cmd.linear.x = 0.0
            cmd.angular.z = self.apply_min_angular_speed(w)
        else:
            v = clamp(
                self.k_linear * dist,
                self.min_linear_speed,
                self.max_linear_speed_normal
            )

            w = clamp(
                self.k_angular * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed
            )

            cmd.linear.x = v
            cmd.linear.y = 0.0
            cmd.angular.z = w

        self.cmd_pub.publish(cmd)

    def on_timer(self) -> None:
        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            self.stop_robot()
            return

        rx, ry, ryaw = robot_pose

        # escape path 优先级最高
        if self.has_valid_escape_path():
            self.track_escape_path(rx, ry, ryaw)
            return

        # 没有 escape path 时，走普通 C* goal
        self.track_single_goal(rx, ry, ryaw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CStarSimpleControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()