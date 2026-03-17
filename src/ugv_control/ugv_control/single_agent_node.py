import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def get_yaw_from_quaternion(x, y, z, w):
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def smoothstep01(u):
    u = clamp(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def raised_cosine_bump(x, center, half_width):
    """
    Smooth bump in [0,1]:
      1 at x=center
      0 at |x-center| >= half_width
    """
    if half_width <= 1e-6:
        return 0.0
    d = abs(x - center)
    if d >= half_width:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * d / half_width))


def closest_point_on_box(px, py, box):
    # calculating box obstacle dimensions
    cx = box['cx']
    cy = box['cy']
    hx = box['hx']
    hy = box['hy']

    qx = clamp(px, cx - hx, cx + hx)
    qy = clamp(py, cy - hy, cy + hy)
    return qx, qy


class SingleAgentController(Node):
    def __init__(self):
        super().__init__('single_agent_controller')

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter('robot_name', 'ugv1')

        self.declare_parameter('vl_start_x', 0.0)
        self.declare_parameter('vl_start_y', 0.0)
        self.declare_parameter('vl_goal_x', 6.0)
        self.declare_parameter('vl_goal_y', 6.0)
        self.declare_parameter('vl_speed', 0.25)

        self.declare_parameter('obstacle_cx', 3.5)
        self.declare_parameter('obstacle_cy', 3.5)
        self.declare_parameter('obstacle_hx', 1.0)
        self.declare_parameter('obstacle_hy', 1.0)

        self.declare_parameter('control_period', 0.1)
        self.declare_parameter('warmup_duration', 2.0)

        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        self.vl_start_x = float(self.get_parameter('vl_start_x').value)
        self.vl_start_y = float(self.get_parameter('vl_start_y').value)
        self.vl_goal_x = float(self.get_parameter('vl_goal_x').value)
        self.vl_goal_y = float(self.get_parameter('vl_goal_y').value)
        self.vl_speed = float(self.get_parameter('vl_speed').value)

        self.obstacle = {
            'cx': float(self.get_parameter('obstacle_cx').value),
            'cy': float(self.get_parameter('obstacle_cy').value),
            'hx': float(self.get_parameter('obstacle_hx').value),
            'hy': float(self.get_parameter('obstacle_hy').value),
        }

        self.control_period = float(self.get_parameter('control_period').value)
        self.warmup_duration = float(self.get_parameter('warmup_duration').value)

        self.robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        if self.robot_name not in self.robots:
            raise ValueError(f"Unknown robot_name '{self.robot_name}'. Expected one of {self.robots}")

        # ---------------------------------------------------------
        # Path / leader geometry
        # ---------------------------------------------------------
        dx = self.vl_goal_x - self.vl_start_x
        dy = self.vl_goal_y - self.vl_start_y
        self.total_dist = math.hypot(dx, dy)

        if self.total_dist < 1e-6:
            raise ValueError("Virtual leader start and goal are identical.")

        # Leader-aligned local frame:
        #   x_local = along the path
        #   y_local = lateral to the path
        self.path_ux = dx / self.total_dist
        self.path_uy = dy / self.total_dist
        self.path_nx = -self.path_uy
        self.path_ny = self.path_ux
        self.goal_yaw = math.atan2(dy, dx)

        # Project obstacle center to leader path
        rel_obs_x = self.obstacle['cx'] - self.vl_start_x
        rel_obs_y = self.obstacle['cy'] - self.vl_start_y
        self.obs_path_coord = rel_obs_x * self.path_ux + rel_obs_y * self.path_uy
        self.obs_path_coord = clamp(self.obs_path_coord, 0.0, self.total_dist)

        # ---------------------------------------------------------
        # Formation definition in leader-local coordinates
        # ---------------------------------------------------------
        # Cross formation around the ghost leader
        #   ugv1 = behind
        #   ugv2 = left/up lateral
        #   ugv3 = right/down lateral
        #   ugv4 = ahead
        self.shape_base = {
            'ugv1': (-1.0,  0.0),
            'ugv2': ( 0.0,  1.0),
            'ugv3': ( 0.0, -1.0),
            'ugv4': ( 0.7,  0.7),
        }

        # Side preference for obstacle bypass bias in the local lateral axis
        self.morph_bias_sign = {
            'ugv1':  1.0,
            'ugv2':  1.0,
            'ugv3': -1.0,
            'ugv4': -1.0,
        }

        formation_radius = max(math.hypot(px, py) for px, py in self.shape_base.values())
        obstacle_radius = math.hypot(self.obstacle['hx'], self.obstacle['hy'])

        # Smooth morph region around the obstacle projection on the path
        self.morph_half_width = obstacle_radius + formation_radius + 0.9

        # Affine morph gains
        self.lateral_stretch_gain = 1.35
        self.longitudinal_compression_gain = 0.10
        self.lateral_bias_gain = 0.85

        # ---------------------------------------------------------
        # Controller gains 
        # ---------------------------------------------------------
        self.k_att = 2.0
        self.k_yaw = 2.2

        self.d_safe_robot = 0.65
        self.k_rep_robot = 0.9

        self.d_safe_obs = 1.0
        self.k_rep_obs = 1.8

        self.max_speed = 0.65
        self.max_turn = 2.5

        self.final_max_speed = 0.22
        self.final_max_turn = 3.5

        self.final_tol = 0.08
        self.all_settled_tol = 0.12
        self.force_deadband = 1e-3

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.have_self_odom = False

        self.other_robots_pos = {
            name: None for name in self.robots if name != self.robot_name
        }

        self.motion_start_time = None
        self.last_control_time = None

        # ---------------------------------------------------------
        # ROS interfaces
        # ---------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        self.force_pub = self.create_publisher(Point, f'/{self.robot_name}/force_vector', 10)

        self.create_subscription(Odometry, f'/{self.robot_name}/odom', self.odom_callback, 10)

        for name in self.other_robots_pos.keys():
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda msg, n=name: self.other_odom_callback(msg, n),
                10
            )

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            f"{self.robot_name} controller started. "
            f"Leader path: ({self.vl_start_x:.2f},{self.vl_start_y:.2f}) -> "
            f"({self.vl_goal_x:.2f},{self.vl_goal_y:.2f}), obstacle box center "
            f"({self.obstacle['cx']:.2f},{self.obstacle['cy']:.2f}), "
            f"half-size ({self.obstacle['hx']:.2f},{self.obstacle['hy']:.2f})"
        )

    # ---------------------------------------------------------
    # Time helpers
    # ---------------------------------------------------------
    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.current_yaw = get_yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if not self.have_self_odom:
            self.have_self_odom = True
            self.motion_start_time = self.now_sec() + self.warmup_duration
            self.last_control_time = self.now_sec()

    def other_odom_callback(self, msg, name):
        self.other_robots_pos[name] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    # ---------------------------------------------------------
    # Geometry / target generation
    # ---------------------------------------------------------
    def leader_position(self, travel):
        travel = clamp(travel, 0.0, self.total_dist)
        x = self.vl_start_x + self.path_ux * travel
        y = self.vl_start_y + self.path_uy * travel
        return x, y

    def local_to_world(self, leader_x, leader_y, lx, ly):
        wx = leader_x + self.path_ux * lx + self.path_nx * ly
        wy = leader_y + self.path_uy * lx + self.path_ny * ly
        return wx, wy

    def morph_gain(self, travel):
        return raised_cosine_bump(travel, self.obs_path_coord, self.morph_half_width)

    def get_target_for_robot(self, robot_name, travel):
        """
        Affine morph in the leader-local frame:
            p_local_morph = A(s) * p_local_nominal + b_i(s)

        A(s) stretches laterally and slightly compresses longitudinally
        near the obstacle. b_i(s) gives a robot-dependent lateral bias
        so the group splits smoothly around the obstacle and reforms later.
        """
        leader_x, leader_y = self.leader_position(travel)

        base_x, base_y = self.shape_base[robot_name]
        m = self.morph_gain(travel)

        # Affine transform A(s)
        ax = 1.0 - self.longitudinal_compression_gain * m
        ay = 1.0 + self.lateral_stretch_gain * m

        # Bias b_i(s)
        bias_y = self.lateral_bias_gain * self.morph_bias_sign[robot_name] * m

        lx = ax * base_x
        ly = ay * base_y + bias_y

        tx, ty = self.local_to_world(leader_x, leader_y, lx, ly)
        return tx, ty

    def all_robots_settled(self):
        for name in self.robots:
            tx, ty = self.get_target_for_robot(name, self.total_dist)

            if name == self.robot_name:
                rx, ry = self.current_x, self.current_y
            else:
                pos = self.other_robots_pos[name]
                if pos is None:
                    return False
                rx, ry = pos

            if math.hypot(tx - rx, ty - ry) > self.all_settled_tol:
                return False
        return True

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    def control_loop(self):
        if not self.have_self_odom or self.motion_start_time is None:
            self.stop_robot()
            return

        now = self.now_sec()

        if self.last_control_time is None:
            dt = self.control_period
        else:
            dt = now - self.last_control_time
            dt = clamp(dt, 1e-3, 0.25)

        self.last_control_time = now

        t = now - self.motion_start_time

        # Warm-up phase
        if t < 0.0:
            self.stop_robot()
            return

        # Leader progress
        current_travel = min(self.vl_speed * t, self.total_dist)
        next_travel = min(current_travel + self.vl_speed * dt, self.total_dist)

        # Target and feedforward
        my_target_x, my_target_y = self.get_target_for_robot(self.robot_name, current_travel)
        next_target_x, next_target_y = self.get_target_for_robot(self.robot_name, next_travel)

        ff_vx = (next_target_x - my_target_x) / dt
        ff_vy = (next_target_y - my_target_y) / dt

        # Attractive tracking term + feedforward slot velocity
        ex = my_target_x - self.current_x
        ey = my_target_y - self.current_y

        f_att_x = self.k_att * ex + ff_vx
        f_att_y = self.k_att * ey + ff_vy

        dist_to_my_target = math.hypot(ex, ey)
        is_leader_at_goal = (current_travel >= self.total_dist - 1e-6)

        # ---------------------------------------------------------
        # Robot-robot repulsion
        # ---------------------------------------------------------
        f_rep_x = 0.0
        f_rep_y = 0.0

        for name, pos in self.other_robots_pos.items():
            if pos is None:
                continue

            rx, ry = pos
            dx = self.current_x - rx
            dy = self.current_y - ry
            dist = math.hypot(dx, dy)

            if 1e-6 < dist < self.d_safe_robot:
                safe_dist = max(dist, 0.08)
                rep_mag = self.k_rep_robot * (1.0 / safe_dist - 1.0 / self.d_safe_robot) / (safe_dist ** 2)
                f_rep_x += rep_mag * dx / safe_dist
                f_rep_y += rep_mag * dy / safe_dist

        # ---------------------------------------------------------
        # Box obstacle repulsion (safety layer)
        # ---------------------------------------------------------
        qx, qy = closest_point_on_box(self.current_x, self.current_y, self.obstacle)
        dxo = self.current_x - qx
        dyo = self.current_y - qy
        dist_obs = math.hypot(dxo, dyo)

        f_obs_x = 0.0
        f_obs_y = 0.0

        if dist_obs < self.d_safe_obs:
            # If exactly inside/on the box, push away from box center
            if dist_obs < 1e-6:
                dxo = self.current_x - self.obstacle['cx']
                dyo = self.current_y - self.obstacle['cy']
                norm = math.hypot(dxo, dyo)
                if norm < 1e-6:
                    # fallback direction: push along lateral side preference
                    dxo = self.path_nx * self.morph_bias_sign[self.robot_name]
                    dyo = self.path_ny * self.morph_bias_sign[self.robot_name]
                    norm = math.hypot(dxo, dyo)
                dxo /= norm
                dyo /= norm
                safe_dist = 0.08
            else:
                safe_dist = max(dist_obs, 0.08)
                dxo /= safe_dist
                dyo /= safe_dist

            rep_mag = self.k_rep_obs * (1.0 / safe_dist - 1.0 / self.d_safe_obs) / (safe_dist ** 2)
            f_obs_x = rep_mag * dxo
            f_obs_y = rep_mag * dyo

        # ---------------------------------------------------------
        # Total guidance force
        # ---------------------------------------------------------
        total_fx = f_att_x + f_rep_x + f_obs_x
        total_fy = f_att_y + f_rep_y + f_obs_y

        force_norm = math.hypot(total_fx, total_fy)

        force_msg = Point()
        force_msg.x = float(total_fx)
        force_msg.y = float(total_fy)
        force_msg.z = 0.0
        self.force_pub.publish(force_msg)

        # ---------------------------------------------------------
        # Goal settling logic
        # ---------------------------------------------------------
        if is_leader_at_goal:
            if self.all_robots_settled() and dist_to_my_target < self.final_tol:
                self.stop_robot()
                return

        # Near the end, slow down for clean parking
        if is_leader_at_goal and dist_to_my_target < 0.8:
            current_max_speed = self.final_max_speed
            current_max_turn = self.final_max_turn
        else:
            current_max_speed = self.max_speed
            current_max_turn = self.max_turn

        # ---------------------------------------------------------
        # Kinematic mapping
        # ---------------------------------------------------------
        cmd = Twist()

        if force_norm < self.force_deadband:
            self.cmd_pub.publish(cmd)
            return

        target_yaw = math.atan2(total_fy, total_fx)
        yaw_error = wrap_angle(target_yaw - self.current_yaw)

        # Forward-only differential drive style:
        # move only if target is roughly in front
        heading_factor = max(0.0, math.cos(yaw_error))
        speed_cmd = min(current_max_speed, force_norm) * heading_factor

        if abs(yaw_error) > math.pi / 2.0:
            speed_cmd = 0.0

        cmd.linear.x = speed_cmd
        cmd.angular.z = clamp(self.k_yaw * yaw_error, -current_max_turn, current_max_turn)

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.force_pub.publish(Point())


def main(args=None):
    rclpy.init(args=args)
    node = SingleAgentController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()