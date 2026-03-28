import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class GlobalObstaclePublisher(Node):
    def __init__(self):
        super().__init__('global_obstacle_publisher')

        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_period', 0.2)

        # Each obstacle is defined by center (cx, cy) and half-sizes (hx, hy)
        self.declare_parameter('obstacle_cxs', [3.5])
        self.declare_parameter('obstacle_cys', [3.5])
        self.declare_parameter('obstacle_hxs', [0.5])
        self.declare_parameter('obstacle_hys', [0.5])

        self.frame_id = str(self.get_parameter('frame_id').value)
        self.publish_period = float(self.get_parameter('publish_period').value)

        self.pub = self.create_publisher(MarkerArray, '/detected_obstacles', 10)
        self.timer = self.create_timer(self.publish_period, self.publish_obstacles)

        self.get_logger().info("Global obstacle publisher started on /detected_obstacles")

    def get_obstacle_arrays(self):
        cxs = list(self.get_parameter('obstacle_cxs').value)
        cys = list(self.get_parameter('obstacle_cys').value)
        hxs = list(self.get_parameter('obstacle_hxs').value)
        hys = list(self.get_parameter('obstacle_hys').value)
        return cxs, cys, hxs, hys

    def publish_obstacles(self):
        cxs, cys, hxs, hys = self.get_obstacle_arrays()

        n = len(cxs)
        if not (len(cys) == n and len(hxs) == n and len(hys) == n):
            self.get_logger().error(
                "Obstacle parameter arrays must have the same length: "
                f"len(cxs)={len(cxs)}, len(cys)={len(cys)}, "
                f"len(hxs)={len(hxs)}, len(hys)={len(hys)}"
            )
            return

        msg = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        msg.markers.append(delete_all)

        stamp = self.get_clock().now().to_msg()

        for i, (cx, cy, hx, hy) in enumerate(zip(cxs, cys, hxs, hys)):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = stamp
            marker.ns = 'detected_obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(cx)
            marker.pose.position.y = float(cy)
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(2.0 * hx)
            marker.scale.y = float(2.0 * hy)
            marker.scale.z = 0.5

            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.35

            msg.markers.append(marker)

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalObstaclePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()