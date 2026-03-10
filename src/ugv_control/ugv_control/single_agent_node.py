import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from geometry_msgs.msg import Point

class DecentralizedAgent(Node):
    def __init__(self):
        super().__init__('single_agent_node')
        
        self.declare_parameter('robot_name', 'ugv1')
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)
        
        self.robot_name = self.get_parameter('robot_name').value
        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value

        self.force_pub = self.create_publisher(Point, f'/{self.robot_name}/force_vector', 10)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.others = {} 
        
       
        self.arrived = False

        self.cmd_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)

        all_robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        for bot in all_robots:
            if bot != self.robot_name:
                self.others[bot] = {'x': 0.0, 'y': 0.0}
                self.create_subscription(
                    Odometry, 
                    f'/{bot}/odom', 
                    lambda msg, name=bot: self.other_odom_callback(msg, name), 
                    10
                )

        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def other_odom_callback(self, msg, bot_name):
        self.others[bot_name]['x'] = msg.pose.pose.position.x
        self.others[bot_name]['y'] = msg.pose.pose.position.y

    def control_loop(self):
        cmd = Twist()

        if self.arrived:
            self.cmd_pub.publish(cmd)
            return

        distance_to_goal = math.hypot(self.goal_x - self.x, self.goal_y - self.y)

        # parking radius
        if distance_to_goal < 0.65:
            self.arrived = True
            self.get_logger().info(f'+++ {self.robot_name} has successfully parked! +++')
            self.cmd_pub.publish(cmd)
            return

        # APF parameters
        k_att = 0.5
        k_rep = 0.3      
        d_safe = 1.2     

        #  attractive force
        fx = k_att * (self.goal_x - self.x)
        fy = k_att * (self.goal_y - self.y)

        # repulsive force with TANGENTIAL routing
        for bot, pos in self.others.items():
            dx = self.x - pos['x']
            dy = self.y - pos['y']
            dist = math.hypot(dx, dy)

            if 0.01 < dist < d_safe: 
                #  radial repulsion magnitude
                rep_mag = k_rep * (1.0/dist - 1.0/d_safe) / (dist**2)
                
                # direct push away
                fx += rep_mag * (dx / dist)
                fy += rep_mag * (dy / dist)

                # tangential push 
                # If the other robot is closer to the goal than we are, it's blocking us.
                dist_other_to_goal = math.hypot(self.goal_x - pos['x'], self.goal_y - pos['y'])
                
                if dist_other_to_goal < distance_to_goal:
                    k_tangent = 1.5 # strength of the side step
                    tangent_mag = k_tangent * rep_mag
                    
                    # The perpendicular vector to (dx, dy) is (-dy, dx). 
                    # all robots slide counter-clockwise around obstacles
                    fx += tangent_mag * (-dy / dist)
                    fy += tangent_mag * (dx / dist)

        # antispin protection
        total_force = math.hypot(fx, fy)
        if total_force < 0.05:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        # Kinematics
        target_yaw = math.atan2(fy, fx)
        yaw_error = target_yaw - self.yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        cmd.angular.z = 2.0 * yaw_error
        cmd.linear.x = 0.5 * math.cos(yaw_error) * min(total_force, 1.0)
        
        if cmd.linear.x < 0:
            cmd.linear.x = 0.0
        

        
        force_msg = Point()
        force_msg.x = fx
        force_msg.y = fy
        force_msg.z = 0.0
        self.force_pub.publish(force_msg)

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DecentralizedAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()