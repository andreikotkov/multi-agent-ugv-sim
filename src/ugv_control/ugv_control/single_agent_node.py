import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
import math
import time

def get_yaw_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y - z * z)
    return math.atan2(t3, t4)

class SingleAgentController(Node):
    def __init__(self):
        super().__init__('single_agent_controller')
        
        self.declare_parameter('robot_name', 'ugv1')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        
        self.robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        self.other_robots_pos = {name: (0.0, 0.0) for name in self.robots if name != self.robot_name}
        
        # ---------------------------------------------------------
        # 1. THE "SUN" (Virtual Leader)
        # ---------------------------------------------------------
        # initial positions, goal and constant speed 
        self.vl_start_x = 0.0  
        self.vl_start_y = 0.0  
        self.vl_goal_x = 6.0
        self.vl_goal_y = 6.0
        self.vl_speed = 0.25 
        
        self.start_time = time.time() + 2.0 
        self.goal_yaw = math.atan2(self.vl_goal_y - self.vl_start_y, self.vl_goal_x - self.vl_start_x)
        
        # ---------------------------------------------------------
        # 2. DIAGONALLY SYMMETRIC ORBITS 
        # ---------------------------------------------------------
        # our 4 ugvs placed symmetrically
        self.shape_base = {
            'ugv1': [-1.0,  0.0],  
            'ugv2': [ 0.0,  1.0],  
            'ugv3': [ 0.0, -1.0],  
            'ugv4': [ 1.0,  0.0]   
        }
        
        self.evasion_vector = {
            'ugv1': [-1.0,  1.0],  
            'ugv2': [-1.0,  1.0],  
            'ugv3': [ 1.0, -1.0],  
            'ugv4': [ 1.0, -1.0]   
        }
        
        self.k_att = 2.0 # Increased for razor-sharp tracking
        
        # ---------------------------------------------------------
        # 3. INTER-PLANET & OBSTACLE REPULSION 
        # ---------------------------------------------------------
        self.d_safe_robot = 0.6  
        self.k_rep_robot = 1.0   
        self.obstacle = {'x': 3.5, 'y': 3.5}
        self.d_safe_obs = 1.0    
        self.k_rep_obs = 1.5     
        
        # Kinematics
        self.max_speed = 0.65    
        self.max_turn = 2.5      
        
        self.cmd_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        self.force_pub = self.create_publisher(Point, f'/{self.robot_name}/force_vector', 10)
        
        self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)
        for name in self.other_robots_pos.keys():
            self.create_subscription(Odometry, f'/{name}/odom', 
                lambda msg, n=name: self.other_odom_callback(msg, n), 10)
                
        self.timer = self.create_timer(0.1, self.control_loop)
        
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = get_yaw_from_quaternion(q.x, q.y, q.z, q.w)
        
    def other_odom_callback(self, msg, name):
        self.other_robots_pos[name] = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
    def get_target(self, travel, total_dist):
        """Helper function to calculate the exact mathematical target at any given travel distance"""
        # virtual leader trajectory
        vl_x = self.vl_start_x + (self.vl_goal_x - self.vl_start_x) * (travel / total_dist)
        vl_y = self.vl_start_y + (self.vl_goal_y - self.vl_start_y) * (travel / total_dist)

        L = vl_x 
        target_scale = 1.5 # Massive visual expansion
        scale = 0.0
        
        # COMPRESSED TIMINGS: Guaranteed to finish perfectly at L = 6.0
        if self.robot_name in ['ugv2', 'ugv4']:
            if L < 1.0: scale = 0.0
            elif L < 2.0: scale = target_scale * ((L - 1.0) / 1.0)
            elif L < 4.0: scale = target_scale
            elif L < 5.0: scale = target_scale * (1.0 - ((L - 4.0) / 1.0))
            else: scale = 0.0
        else:
            if L < 2.0: scale = 0.0
            elif L < 3.0: scale = target_scale * ((L - 2.0) / 1.0)
            elif L < 5.0: scale = target_scale
            elif L < 6.0: scale = target_scale * (1.0 - ((L - 5.0) / 1.0))
            else: scale = 0.0
            
        base_off = self.shape_base[self.robot_name]
        evasion_dir = self.evasion_vector[self.robot_name]
        
        # Affine transformation
        tx = vl_x + base_off[0] + (evasion_dir[0] * scale)
        ty = vl_y + base_off[1] + (evasion_dir[1] * scale)
        return tx, ty
        
    def control_loop(self):
        t = time.time() - self.start_time
        
        # Warm-up phase
        if t < 0: 
            self.current_yaw = self.goal_yaw 
            self.stop_robot()
            return 
            
        total_dist = math.hypot(self.vl_goal_x - self.vl_start_x, self.vl_goal_y - self.vl_start_y)
        
        # ---------------------------------------------------------
        # ANALYTICAL VELOCITY FEEDFORWARD 
        # ---------------------------------------------------------
        # 1. Where should I be right now?
        current_travel = min(self.vl_speed * t, total_dist)
        my_target_x, my_target_y = self.get_target(current_travel, total_dist)
        
        # 2. Where will my slot be in exactly 0.1 seconds?
        # analytical feedforward and P-controller
        next_travel = min(self.vl_speed * (t + 0.1), total_dist)
        next_tx, next_ty = self.get_target(next_travel, total_dist)
        
        # 3. Calculate the exact velocity of the slot
        ff_vx = (next_tx - my_target_x) / 0.1
        ff_vy = (next_ty - my_target_y) / 0.1
        
        # 4. Apply force: Target Velocity + P-Controller
        f_att_x = self.k_att * (my_target_x - self.current_x) + ff_vx
        f_att_y = self.k_att * (my_target_y - self.current_y) + ff_vy

        # ---------------------------------------------------------
        # THE HANDSHAKE (Safe Parking)
        # ---------------------------------------------------------
        dist_to_my_target = math.hypot(my_target_x - self.current_x, my_target_y - self.current_y)
        is_sun_at_goal = (current_travel >= total_dist)
        
        if is_sun_at_goal:
            everyone_at_targets = True
            for other_planet in self.robots:
                if other_planet == self.robot_name: continue
                other_off = self.shape_base[other_planet]
                other_t_x = self.vl_goal_x + other_off[0]
                other_t_y = self.vl_goal_y + other_off[1]
                other_curr_pos = self.other_robots_pos[other_planet]
                other_dist = math.hypot(other_t_x - other_curr_pos[0], other_t_y - other_curr_pos[1])
                if other_dist > 0.35: 
                    everyone_at_targets = False
                    break
            
            if everyone_at_targets:
                self.stop_robot()
                return
                
            
            if dist_to_my_target < 0.25:
                self.stop_robot()
                return
        
        # ---------------------------------------------------------
        # SEATBELT REPULSION 
        # ---------------------------------------------------------
        f_rep_x = 0.0
        f_rep_y = 0.0
        for name, (rx, ry) in self.other_robots_pos.items():
            dist = math.hypot(self.current_x - rx, self.current_y - ry)
            if 0.01 < dist < self.d_safe_robot:
                safe_dist = max(dist, 0.1) 
                rep_mag = self.k_rep_robot * (1.0/safe_dist - 1.0/self.d_safe_robot) / (safe_dist**2)
                f_rep_x += rep_mag * (self.current_x - rx) / dist
                f_rep_y += rep_mag * (self.current_y - ry) / dist
                
        f_obs_x = 0.0
        f_obs_y = 0.0
        dist_obs = math.hypot(self.current_x - self.obstacle['x'], self.current_y - self.obstacle['y'])
        
        if dist_obs < self.d_safe_obs:
            dist_safe = max(dist_obs, 0.1)
            rep_mag = self.k_rep_obs * (1.0/dist_safe - 1.0/self.d_safe_obs) / (dist_safe**2)
            dir_obs_x = (self.current_x - self.obstacle['x']) / dist_safe
            dir_obs_y = (self.current_y - self.obstacle['y']) / dist_safe
            f_obs_x = rep_mag * dir_obs_x
            f_obs_y = rep_mag * dir_obs_y

        total_fx = f_att_x + f_rep_x + f_obs_x
        total_fy = f_att_y + f_rep_y + f_obs_y
        
        force_msg = Point()
        force_msg.x = float(total_fx)
        force_msg.y = float(total_fy)
        self.force_pub.publish(force_msg)
        
        # ---------------------------------------------------------
        # KINEMATICS 
        # ---------------------------------------------------------
        target_yaw = math.atan2(total_fy, total_fx)
        yaw_error = target_yaw - self.current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        
        # Drive slow, turn sharp to avoid end-circles
        if is_sun_at_goal and dist_to_my_target < 0.6:
            current_max_speed = 0.25
            current_max_turn = 4.0
        else:
            current_max_speed = self.max_speed
            current_max_turn = self.max_turn
        
        cmd = Twist()
        if abs(yaw_error) < math.pi / 2: 
            speed = min(current_max_speed, math.hypot(total_fx, total_fy))
            cmd.linear.x = max(0.0, speed * math.cos(yaw_error)) 
        else:
            cmd.linear.x = 0.0 
            
        cmd.angular.z = max(-current_max_turn, min(current_max_turn, 2.0 * yaw_error))
        self.cmd_pub.publish(cmd)
        
    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.force_pub.publish(Point())

def main(args=None):
    rclpy.init(args=args)
    node = SingleAgentController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()