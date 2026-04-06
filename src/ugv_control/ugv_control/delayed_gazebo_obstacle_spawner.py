import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.parameter import Parameter

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity


class DelayedGazeboObstacleSpawner(Node):
    def __init__(self):
        super().__init__('delayed_gazebo_obstacle_spawner')

        self.spawn_delay_sec = self.reqf('spawn_delay_sec')
        self.obstacle_name = self.reqs('obstacle_name')

        self.obstacle_size_x = self.reqf('obstacle_size_x')
        self.obstacle_size_y = self.reqf('obstacle_size_y')
        self.obstacle_size_z = self.reqf('obstacle_size_z')

        self.obstacle_pos_x = self.reqf('obstacle_pos_x')
        self.obstacle_pos_y = self.reqf('obstacle_pos_y')
        self.obstacle_pos_z = self.reqf('obstacle_pos_z')

        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

        self.spawned = False

        self.wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.timer = self.create_timer(
            self.spawn_delay_sec,
            self.spawn_obstacle_once,
            clock=self.wall_clock
        )

        self.get_logger().info(
            f"Obstacle '{self.obstacle_name}' will spawn "
            f"{self.spawn_delay_sec:.1f} real seconds after script launch."
        )

    def require_param(self, name):
        self.declare_parameter(name)
        param = self.get_parameter(name)

        if param.type_ == Parameter.Type.NOT_SET:
            raise RuntimeError(
                f"Required parameter '{name}' is missing for node "
                f"'{self.get_name()}'. Provide it in the YAML config or launch file."
            )

        return param.value

    def reqf(self, name):
        return float(self.require_param(name))

    def reqs(self, name):
        return str(self.require_param(name))

    def spawn_obstacle_once(self):
        if self.spawned:
            return

        self.spawned = True
        self.timer.cancel()

        sdf = f"""
        <sdf version="1.7">
          <model name="{self.obstacle_name}">
            <static>true</static>
            <link name="link">
              <collision name="collision">
                <geometry>
                  <box>
                    <size>{self.obstacle_size_x} {self.obstacle_size_y} {self.obstacle_size_z}</size>
                  </box>
                </geometry>
              </collision>
              <visual name="visual">
                <geometry>
                  <box>
                    <size>{self.obstacle_size_x} {self.obstacle_size_y} {self.obstacle_size_z}</size>
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
        req.name = self.obstacle_name
        req.xml = sdf
        req.robot_namespace = ''
        req.reference_frame = 'world'

        req.initial_pose = Pose()
        req.initial_pose.position.x = self.obstacle_pos_x
        req.initial_pose.position.y = self.obstacle_pos_y
        req.initial_pose.position.z = self.obstacle_pos_z
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