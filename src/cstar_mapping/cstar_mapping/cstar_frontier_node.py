#!/usr/bin/env python3
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker


class CStarFrontierNode(Node):
    def __init__(self):
        super().__init__('cstar_frontier_node')

        # ---------- parameters ----------
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('frontier_map_topic', '/cstar/frontier_map')
        self.declare_parameter('frontier_samples_topic', '/cstar/frontier_samples')
        self.declare_parameter('frontier_marker_topic', '/cstar/frontier_representatives')

        self.declare_parameter('occupied_thresh', 50)
        self.declare_parameter('min_cluster_size', 8)
        self.declare_parameter('connectivity', 8)
        self.declare_parameter('max_representatives', 100)
        self.declare_parameter('marker_scale', 0.10)

        map_topic = self.get_parameter('map_topic').value
        frontier_map_topic = self.get_parameter('frontier_map_topic').value
        frontier_samples_topic = self.get_parameter('frontier_samples_topic').value
        frontier_marker_topic = self.get_parameter('frontier_marker_topic').value

        # /map 常见场景下最好用 reliable + transient local
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        pub_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            map_qos
        )

        self.frontier_map_pub = self.create_publisher(
            OccupancyGrid,
            frontier_map_topic,
            pub_qos
        )

        self.frontier_samples_pub = self.create_publisher(
            PoseArray,
            frontier_samples_topic,
            10
        )

        self.frontier_marker_pub = self.create_publisher(
            Marker,
            frontier_marker_topic,
            10
        )

        self.get_logger().info('CStar frontier node started.')
        self.get_logger().info(f'subscribe map: {map_topic}')
        self.get_logger().info(f'publish frontier map: {frontier_map_topic}')
        self.get_logger().info(f'publish representatives (PoseArray): {frontier_samples_topic}')
        self.get_logger().info(f'publish representatives (Marker): {frontier_marker_topic}')

    # ---------------- utility ----------------

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def grid_to_world(self, row, col, info):
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        yaw = self.yaw_from_quaternion(info.origin.orientation)

        # cell center in map local coordinates
        lx = (col + 0.5) * res
        ly = (row + 0.5) * res

        # rotate by map origin yaw
        wx = ox + math.cos(yaw) * lx - math.sin(yaw) * ly
        wy = oy + math.sin(yaw) * lx + math.cos(yaw) * ly
        return wx, wy

    def rc_to_index(self, r, c, width):
        return r * width + c

    def in_bounds(self, r, c, height, width):
        return 0 <= r < height and 0 <= c < width

    def get_neighbors(self, r, c, height, width, connectivity):
        if connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]

        out = []
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc, height, width):
                out.append((nr, nc))
        return out

    # ---------------- core ----------------

    def compute_frontier_flags(self, data, height, width, occupied_thresh):
        """
        frontier 定义：
        当前栅格是 free(0)，并且邻域存在 unknown(-1)
        """
        frontier = [False] * (height * width)

        for r in range(height):
            for c in range(width):
                idx = self.rc_to_index(r, c, width)
                val = data[idx]

                # 只在 free 区里找 frontier
                if val != 0:
                    continue

                has_unknown_neighbor = False
                for nr, nc in self.get_neighbors(r, c, height, width, 8):
                    nidx = self.rc_to_index(nr, nc, width)
                    if data[nidx] == -1:
                        has_unknown_neighbor = True
                        break

                if has_unknown_neighbor:
                    frontier[idx] = True

        return frontier

    def cluster_frontiers(self, frontier_flags, height, width, connectivity):
        visited = [False] * (height * width)
        clusters = []

        for r in range(height):
            for c in range(width):
                idx = self.rc_to_index(r, c, width)
                if not frontier_flags[idx] or visited[idx]:
                    continue

                cluster = []
                q = deque()
                q.append((r, c))
                visited[idx] = True

                while q:
                    cr, cc = q.popleft()
                    cluster.append((cr, cc))

                    for nr, nc in self.get_neighbors(cr, cc, height, width, connectivity):
                        nidx = self.rc_to_index(nr, nc, width)
                        if frontier_flags[nidx] and not visited[nidx]:
                            visited[nidx] = True
                            q.append((nr, nc))

                clusters.append(cluster)

        return clusters

    def pick_representative(self, cluster):
        """
        每个簇选一个代表点：
        先算簇中心，再从簇内找离中心最近的真实 frontier cell
        """
        if not cluster:
            return None

        mean_r = sum(p[0] for p in cluster) / len(cluster)
        mean_c = sum(p[1] for p in cluster) / len(cluster)

        best = None
        best_dist = float('inf')

        for r, c in cluster:
            d = (r - mean_r) ** 2 + (c - mean_c) ** 2
            if d < best_dist:
                best_dist = d
                best = (r, c)

        return best

    # ---------------- callback ----------------

    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        data = list(msg.data)

        if width == 0 or height == 0 or len(data) != width * height:
            self.get_logger().warn('invalid map message, skip.')
            return

        occupied_thresh = int(self.get_parameter('occupied_thresh').value)
        min_cluster_size = int(self.get_parameter('min_cluster_size').value)
        connectivity = int(self.get_parameter('connectivity').value)
        max_representatives = int(self.get_parameter('max_representatives').value)
        marker_scale = float(self.get_parameter('marker_scale').value)

        # 1) frontier extraction
        frontier_flags = self.compute_frontier_flags(
            data, height, width, occupied_thresh
        )

        # 2) publish frontier map
        frontier_map = OccupancyGrid()
        frontier_map.header = msg.header
        frontier_map.info = msg.info
        frontier_map.data = [100 if flag else 0 for flag in frontier_flags]
        self.frontier_map_pub.publish(frontier_map)

        # 3) cluster
        clusters = self.cluster_frontiers(frontier_flags, height, width, connectivity)

        # 按簇大小从大到小排序
        clusters.sort(key=lambda x: len(x), reverse=True)

        # 4) filter + representative
        representatives = []
        kept_cluster_sizes = []

        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue

            rep = self.pick_representative(cluster)
            if rep is None:
                continue

            representatives.append(rep)
            kept_cluster_sizes.append(len(cluster))

            if len(representatives) >= max_representatives:
                break

        # 5) publish PoseArray
        poses = PoseArray()
        poses.header = msg.header

        for r, c in representatives:
            wx, wy = self.grid_to_world(r, c, msg.info)
            p = Pose()
            p.position.x = wx
            p.position.y = wy
            p.position.z = 0.0
            p.orientation.w = 1.0
            poses.poses.append(p)

        self.frontier_samples_pub.publish(poses)

        # 6) publish Marker (更适合 RViz 看)
        marker = Marker()
        marker.header = msg.header
        marker.ns = 'cstar_frontier_representatives'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker_scale
        marker.scale.y = marker_scale
        marker.scale.z = marker_scale
        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.2

        for r, c in representatives:
            wx, wy = self.grid_to_world(r, c, msg.info)
            pt = Point()
            pt.x = wx
            pt.y = wy
            pt.z = 0.03
            marker.points.append(pt)

        self.frontier_marker_pub.publish(marker)

        self.get_logger().info(
            f'frontier clusters={len(clusters)}, kept={len(representatives)}, '
            f'cluster_sizes={kept_cluster_sizes[:10]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CStarFrontierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
