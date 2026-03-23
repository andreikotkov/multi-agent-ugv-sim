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
    # Correct yaw extraction
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def mat2_vec2_mul(M, v):
    return (
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1],
    )


def raised_cosine_bump(x, center, half_width):
    """
    Smooth bump:
      0 outside [center-half_width, center+half_width]
      1 at center
    """
    if half_width <= 1e-6:
        return 0.0
    d = abs(x - center)
    if d >= half_width:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * d / half_width))


def closest_point_on_box(px, py, box):
    """
    Closest point on an axis-aligned box.
    box = {'cx','cy','hx','hy'}
    """
    qx = clamp(px, box['cx'] - box['hx'], box['cx'] + box['hx'])
    qy = clamp(py, box['cy'] - box['hy'], box['cy'] + box['hy'])
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
            raise ValueError(f"Unknown robot_name '{self.robot_name}'")

        # ---------------------------------------------------------
        # Leader path geometry
        # ---------------------------------------------------------
        dx = self.vl_goal_x - self.vl_start_x
        dy = self.vl_goal_y - self.vl_start_y
        self.total_dist = math.hypot(dx, dy)
        if self.total_dist < 1e-6:
            raise ValueError("Leader start and goal cannot be identical.")

        # Path unit vector only for progress/morph timing
        self.path_ux = dx / self.total_dist
        self.path_uy = dy / self.total_dist

        # ---------------------------------------------------------
        # WORLD-FRAME RHOMBUS FORMATION
        # These are offsets from the leader in WORLD frame.
        # Since leader starts at (0,0), they match your Gazebo spawn.
        # The formation remains a rhombus in WORLD frame.
        # ---------------------------------------------------------
        self.shape_base = {
            'ugv1': (-1.0,  0.0),
            'ugv2': ( 0.0,  1.0),
            'ugv3': ( 0.0, -1.0),
            'ugv4': ( 1.0,  0.0),
        }

        # Morph side assignment near obstacle:
        # ugv1, ugv2 -> one side
        # ugv3, ugv4 -> other side
        self.morph_bias_sign = {
            'ugv1':  1.0,
            'ugv2':  1.0,
            'ugv3': -1.0,
            'ugv4': -1.0,
        }

        formation_radius = max(math.hypot(px, py) for px, py in self.shape_base.values())
        obstacle_radius = math.hypot(self.obstacle['hx'], self.obstacle['hy'])

        # Obstacle path coordinate for morph trigger
        rel_obs_x = self.obstacle['cx'] - self.vl_start_x
        rel_obs_y = self.obstacle['cy'] - self.vl_start_y
        self.obs_path_coord = clamp(
            rel_obs_x * self.path_ux + rel_obs_y * self.path_uy,
            0.0,
            self.total_dist
        )


        # ---------------------------------------------------------
        # Affine morph gains
        # WORLD-FRAME affine transform:
        #   p_target = p_leader + A(s) p_base + b_i(s)
        # A(s): stretches and slightly compresses the rhombus near obstacle
        # b_i(s): side-dependent bias for splitting around obstacle
        # ---------------------------------------------------------
        self.lateral_stretch_gain = 1.0
        self.longitudinal_compression_gain = 0.10
        self.lateral_bias_gain = 0.65

        # ---------------------------------------------------------
        # Controller gains / safety
        # ---------------------------------------------------------
        self.k_att = 1.4
        self.k_ff = 0.6
        self.k_yaw = 1.6

        self.filtered_fx = 0.0
        self.filtered_fy = 0.0
        self.force_filter_alpha = 0.25

        self.d_safe_robot = 0.65
        self.k_rep_robot = 0.7

        self.d_safe_obs = 1.0
        self.k_rep_obs = 1.4

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
            f"{self.robot_name} started. World-frame rhombus formation enabled."
        )

    # ---------------------------------------------------------
    # Time
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
    # Leader and formation target generation
    # ---------------------------------------------------------
    def leader_position(self, travel):
        travel = clamp(travel, 0.0, self.total_dist)
        ratio = travel / self.total_dist
        x = self.vl_start_x + (self.vl_goal_x - self.vl_start_x) * ratio
        y = self.vl_start_y + (self.vl_goal_y - self.vl_start_y) * ratio
        return x, y

    def morph_gain_for_robot(self, robot_name, travel):
        leader_x, leader_y = self.leader_position(travel)
        base_x, base_y = self.shape_base[robot_name]

        # nominal target before morph
        nominal_x = leader_x + base_x
        nominal_y = leader_y + base_y

        qx, qy = closest_point_on_box(nominal_x, nominal_y, self.obstacle)
        d = math.hypot(nominal_x - qx, nominal_y - qy)

        morph_range = 1.6
        if d >= morph_range:
            return 0.0

        u = 1.0 - d / morph_range
        return u * u * (3.0 - 2.0 * u)

    def get_target_for_robot(self, robot_name, travel):
        """
        WORLD-FRAME formation:
            p_target = p_leader + A(s) p_base + b_i(s)

        No rotation by leader yaw. This preserves the rhombus orientation
        in the world frame, matching your Gazebo spawn layout.
        """
        leader_x, leader_y = self.leader_position(travel)

        base_x, base_y = self.shape_base[robot_name]
        m = self.morph_gain_for_robot(robot_name, travel)

        # Affine deformation in WORLD frame
        # Compress slightly in x, stretch in y
        ax = 1.0 - self.longitudinal_compression_gain * m
        ay = 1.0 + self.lateral_stretch_gain * m

        # Side bias in WORLD y direction
        bias_y = self.lateral_bias_gain * self.morph_bias_sign[robot_name] * m

        A = (
            (ax, 0.0),
            (0.0, ay),
        )
        b = (0.0, bias_y)

        aff_x, aff_y = mat2_vec2_mul(A, (base_x, base_y))

        tx = leader_x + aff_x + b[0]
        ty = leader_y + aff_y + b[1]
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
    # Main control loop
    # ---------------------------------------------------------
    def control_loop(self):
        if not self.have_self_odom or self.motion_start_time is None:
            self.stop_robot()
            return

        now = self.now_sec()

        if self.last_control_time is None:
            dt = self.control_period
        else:
            dt = clamp(now - self.last_control_time, 1e-3, 0.25)

        self.last_control_time = now
        t = now - self.motion_start_time

        # Warm-up
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

        ex = my_target_x - self.current_x
        ey = my_target_y - self.current_y

        f_att_x = self.k_att * ex + self.k_ff * ff_vx
        f_att_y = self.k_att * ey + self.k_ff * ff_vy

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
        # Box obstacle repulsion
        # ---------------------------------------------------------
        qx, qy = closest_point_on_box(self.current_x, self.current_y, self.obstacle)
        dxo = self.current_x - qx
        dyo = self.current_y - qy
        dist_obs = math.hypot(dxo, dyo)

        f_obs_x = 0.0
        f_obs_y = 0.0

        if dist_obs < self.d_safe_obs:
            if dist_obs < 1e-6:
                # Inside or exactly on boundary: push using center-based fallback
                dxo = self.current_x - self.obstacle['cx']
                dyo = self.current_y - self.obstacle['cy']
                norm = math.hypot(dxo, dyo)

                if norm < 1e-6:
                    # fallback: push in world y based on side assignment
                    dxo = 0.0
                    dyo = self.morph_bias_sign[self.robot_name]
                    norm = 1.0

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
        # Total force
        # ---------------------------------------------------------
        total_fx = f_att_x + f_rep_x + f_obs_x
        total_fy = f_att_y + f_rep_y + f_obs_y

        # Low-pass filter
        self.filtered_fx = (
            (1.0 - self.force_filter_alpha) * self.filtered_fx
            + self.force_filter_alpha * total_fx
        )

        self.filtered_fy = (
            (1.0 - self.force_filter_alpha) * self.filtered_fy
            + self.force_filter_alpha * total_fy
        )

        force_x = self.filtered_fx
        force_y = self.filtered_fy

        force_norm = math.hypot(force_x, force_y)


        force_msg = Point()
        force_msg.x = float(total_fx)
        force_msg.y = float(total_fy)
        force_msg.z = 0.0
        self.force_pub.publish(force_msg)

        # ---------------------------------------------------------
        # Final settling
        # ---------------------------------------------------------
        if is_leader_at_goal:
            if self.all_robots_settled() and dist_to_my_target < self.final_tol:
                self.stop_robot()
                return

        if is_leader_at_goal and dist_to_my_target < 0.8:
            current_max_speed = self.final_max_speed
            current_max_turn = self.final_max_turn
        else:
            current_max_speed = self.max_speed
            current_max_turn = self.max_turn

        # ---------------------------------------------------------
        # Kinematic command
        # ---------------------------------------------------------
        cmd = Twist()

        if force_norm < self.force_deadband:
            self.cmd_pub.publish(cmd)
            return

        target_yaw = math.atan2(force_y, force_x)
        yaw_error = wrap_angle(target_yaw - self.current_yaw)

        if abs(yaw_error) < 0.03:
            speed_cmd = 0.0

        heading_factor = max(0.0, math.cos(yaw_error))
        speed_cmd = min(current_max_speed, force_norm) * heading_factor

        if abs(yaw_error) < 0.03:
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