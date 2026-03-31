import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity


class DelayedGazeboObstacleSpawner(Node):
    def __init__(self):
        super().__init__('delayed_gazebo_obstacle_spawner')

        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

        self.spawned = False
        self.timer = self.create_timer(30.0, self.spawn_obstacle_once)

    def spawn_obstacle_once(self):
        if self.spawned:
            return

        self.spawned = True
        self.timer.cancel()

        sdf = """
        <sdf version="1.7">
          <model name="delayed_rect_obstacle">
            <static>true</static>
            <link name="link">
              <collision name="collision">
                <geometry>
                  <box>
                    <size>3.0 1.0 0.5</size>
                  </box>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <box>
                    <size>3.0 1.0 0.5</size>
                  </box>
                </geometry>
                <material>
                  <ambient>1 0.1 0.1 0.1 1</ambient>
                  <diffuse>1 0.2 0.2 0.2 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """

        req = SpawnEntity.Request()
        req.name = 'delayed_rect_obstacle'
        req.xml = sdf
        req.robot_namespace = ''
        req.reference_frame = 'world'

        req.initial_pose = Pose()
        req.initial_pose.position.x = 6.0
        req.initial_pose.position.y = 8.5
        req.initial_pose.position.z = 0.1
        req.initial_pose.orientation.w = 1.0

        future = self.client.call_async(req)
        future.add_done_callback(self.spawn_done)

    def spawn_done(self, future):
        try:
            result = future.result()
            self.get_logger().info(
                f"Spawn result: success={result.success}, message={result.status_message}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to spawn obstacle: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DelayedGazeboObstacleSpawner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()