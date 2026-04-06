import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Float64
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def get_yaw_from_quaternion(x, y, z, w):
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def closest_point_on_box(px, py, box):
    qx = clamp(px, box['cx'] - box['hx'], box['cx'] + box['hx'])
    qy = clamp(py, box['cy'] - box['hy'], box['cy'] + box['hy'])
    return qx, qy


def speed_cap(v, w, v_max, w_max):
    if v_max <= 1e-9 or w_max <= 1e-9:
        return 0.0, 0.0

    alpha = max(
        abs(v) / v_max,
        abs(w) / w_max,
        1.0
    )
    return v / alpha, w / alpha


class SingleAgentController(Node):
    def __init__(self):
        super().__init__('single_agent_controller')

        # ---------------------------------------------------------
        # Parameters (mandatory from YAML / launch)
        # ---------------------------------------------------------
        self.robot_name = self.reqs('robot_name')

        self.vl_start_x = self.reqf('vl_start_x')
        self.vl_start_y = self.reqf('vl_start_y')
        self.vl_goal_x = self.reqf('vl_goal_x')
        self.vl_goal_y = self.reqf('vl_goal_y')
        self.vl_speed = self.reqf('vl_speed')

        self.control_period = self.reqf('control_period')
        self.warmup_duration = self.reqf('warmup_duration')

        self.obstacle_topic = self.reqs('obstacle_topic')
        self.obstacle_timeout = self.reqf('obstacle_timeout')
        self.world_frame_id = self.reqs('world_frame_id')

        # These shape the deformation, but the manager decides
        # which mode to use and how strong the shift/split is.
        self.shift_longitudinal_scale = self.reqf('shift_longitudinal_scale')
        self.split_lateral_scale = self.reqf('split_lateral_scale')
        self.split_longitudinal_scale = self.reqf('split_longitudinal_scale')

        self.offset_blend_alpha_obstacle = self.reqf('offset_blend_alpha_obstacle')
        self.offset_blend_alpha_recover = self.reqf('offset_blend_alpha_recover')

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

        self.path_ux = dx / self.total_dist
        self.path_uy = dy / self.total_dist
        self.path_nx = -self.path_uy
        self.path_ny = self.path_ux

        # ---------------------------------------------------------
        # Nominal formation shape
        # ---------------------------------------------------------
        self.shape_base = {
            'ugv1': (-1.0,  0.0),
            'ugv2': ( 0.0,  1.0),
            'ugv3': ( 0.0, -1.0),
            'ugv4': ( 1.0,  0.0),
        }

        self.split_side_sign = self.compute_split_side_signs()

        # ---------------------------------------------------------
        # Controller gains / safety
        # ---------------------------------------------------------
        self.k_att = 1.4
        self.k_ff = 0.6
        self.k_yaw = 1.6

        self.filtered_fx = 0.0
        self.filtered_fy = 0.0
        self.force_filter_alpha = 0.18

        self.d_safe_robot = 0.65
        self.k_rep_robot = 0.7

        self.d_safe_obs = 1.0
        self.k_rep_obs = 1.4

        self.max_speed = 0.65
        self.max_turn = 2.5

        self.final_max_speed = 0.10
        self.final_max_turn = 0.8

        self.final_tol = 0.12
        self.all_settled_tol = 0.12
        self.force_deadband = 1e-3

        # Smoother turning than before
        self.rotate_in_place_angle = 0.65

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

        self.last_control_time = None

        self.detected_obstacles = []
        self.last_obstacle_update_time = None

        self.formation_mode = 'normal'
        self.mode_gain = 0.0
        self.active_obstacle = None
        self.dynamic_shift_amount = 0.0
        self.dynamic_split_extra = 0.0

        self.current_offsets = {
            name: self.shape_base[name] for name in self.robots
        }

        # Shared synchronized start
        self.have_odom_all = {name: False for name in self.robots}
        self.motion_start_time = None
        self.have_global_start_time = False
        self.logged_waiting_for_global_start = False

        # ---------------------------------------------------------
        # ROS interfaces
        # ---------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        self.force_pub = self.create_publisher(Point, f'/{self.robot_name}/force_vector', 10)

        self.mode_pub = self.create_publisher(String, f'/{self.robot_name}/formation_mode', 10)
        self.mode_gain_pub = self.create_publisher(Float32, f'/{self.robot_name}/formation_mode_gain', 10)
        self.active_obstacle_pub = self.create_publisher(
            Marker,
            f'/{self.robot_name}/active_obstacle_marker',
            10
        )

        self.create_subscription(
            Odometry,
            f'/{self.robot_name}/odom',
            self.odom_callback,
            10
        )

        for name in self.other_robots_pos.keys():
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda msg, n=name: self.other_odom_callback(msg, n),
                10
            )

        self.create_subscription(
            MarkerArray,
            self.obstacle_topic,
            self.obstacle_markers_callback,
            10
        )

        self.create_subscription(
            String,
            '/formation_mode_global',
            self.global_mode_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/formation_mode_gain_global',
            self.global_mode_gain_callback,
            10
        )

        self.create_subscription(
            Marker,
            '/active_obstacle_global',
            self.global_active_obstacle_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/formation_shift_amount_global',
            self.global_shift_amount_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/formation_split_extra_global',
            self.global_split_extra_callback,
            10
        )

        self.create_subscription(
            Float64,
            '/swarm_motion_start_time',
            self.global_motion_start_time_callback,
            10
        )

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            f"{self.robot_name} started. "
            f"Split side sign={self.split_side_sign[self.robot_name]:+.0f}. "
            f"Waiting for synchronized swarm start."
        )

    # ---------------------------------------------------------
    # Mandatory parameter helpers
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------
    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------
    # Geometry helpers
    # ---------------------------------------------------------
    def world_vec_to_path_frame(self, vx, vy):
        along = vx * self.path_ux + vy * self.path_uy
        lateral = vx * self.path_nx + vy * self.path_ny
        return along, lateral

    def path_frame_to_world(self, along, lateral):
        wx = along * self.path_ux + lateral * self.path_nx
        wy = along * self.path_uy + lateral * self.path_ny
        return wx, wy

    def compute_split_side_signs(self):
        signs = {}

        for name in self.robots:
            bx, by = self.shape_base[name]
            _, lateral = self.world_vec_to_path_frame(bx, by)

            if lateral > 1e-6:
                signs[name] = 1.0
            elif lateral < -1e-6:
                signs[name] = -1.0
            else:
                signs[name] = 1.0 if name in ('ugv1', 'ugv2') else -1.0

        return signs

    def current_behavior_scales(self):
        """
        Scale repulsion and smoothing based on manager-controlled formation mode.

        The main idea:
        - normal: full local repulsion, normal smoothing
        - shift: manager already steers formation, so reduce local repulsion somewhat
        - split: manager strongly prescribes geometry, so reduce local repulsion more
        """
        g = clamp(self.mode_gain, 0.0, 1.0)

        if self.formation_mode == 'split':
            obs_scale = 1.0 - 0.65 * g
            robot_scale = 1.0 - 0.45 * g
            force_alpha = 0.14
        elif self.formation_mode in ('shift_left', 'shift_right'):
            obs_scale = 1.0 - 0.40 * g
            robot_scale = 1.0 - 0.25 * g
            force_alpha = 0.16
        else:
            obs_scale = 1.0
            robot_scale = 1.0
            force_alpha = 0.18

        return obs_scale, robot_scale, force_alpha

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

        if not self.have_odom_all[self.robot_name]:
            self.have_odom_all[self.robot_name] = True
            self.get_logger().info(f"{self.robot_name}: received first odometry from self")

        if all(self.have_odom_all.values()) and not self.have_global_start_time and not self.logged_waiting_for_global_start:
            self.logged_waiting_for_global_start = True
            self.get_logger().info(
                f"{self.robot_name}: all odometry available, waiting for global start time."
            )

    def other_odom_callback(self, msg, name):
        self.other_robots_pos[name] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

        if not self.have_odom_all[name]:
            self.have_odom_all[name] = True
            self.get_logger().info(f"{self.robot_name}: received first odometry from {name}")

        if all(self.have_odom_all.values()) and not self.have_global_start_time and not self.logged_waiting_for_global_start:
            self.logged_waiting_for_global_start = True
            self.get_logger().info(
                f"{self.robot_name}: all odometry available, waiting for global start time."
            )

    def obstacle_markers_callback(self, msg):
        obstacles = []

        for marker in msg.markers:
            if marker.action in (Marker.DELETE, Marker.DELETEALL):
                continue
            if marker.type != Marker.CUBE:
                continue

            obstacles.append({
                'cx': marker.pose.position.x,
                'cy': marker.pose.position.y,
                'hx': 0.5 * abs(marker.scale.x),
                'hy': 0.5 * abs(marker.scale.y),
            })

        self.detected_obstacles = obstacles
        self.last_obstacle_update_time = self.now_sec()

    def global_mode_callback(self, msg):
        self.formation_mode = msg.data

    def global_mode_gain_callback(self, msg):
        self.mode_gain = float(msg.data)

    def global_active_obstacle_callback(self, msg):
        if msg.action in (Marker.DELETE, Marker.DELETEALL):
            self.active_obstacle = None
            return

        if msg.type != Marker.CUBE:
            return

        self.active_obstacle = {
            'cx': msg.pose.position.x,
            'cy': msg.pose.position.y,
            'hx': 0.5 * abs(msg.scale.x),
            'hy': 0.5 * abs(msg.scale.y),
        }

    def global_shift_amount_callback(self, msg):
        self.dynamic_shift_amount = float(msg.data)

    def global_split_extra_callback(self, msg):
        self.dynamic_split_extra = float(msg.data)

    def global_motion_start_time_callback(self, msg):
        new_start_time = float(msg.data)

        if not self.have_global_start_time:
            self.motion_start_time = new_start_time
            self.have_global_start_time = True
            self.last_control_time = self.now_sec()

            self.get_logger().info(
                f"{self.robot_name}: received global motion start time "
                f"{self.motion_start_time:.3f} s"
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

    def nominal_offset_in_path_frame(self, robot_name):
        base_x, base_y = self.shape_base[robot_name]
        return self.world_vec_to_path_frame(base_x, base_y)

    def apply_shift_deformation(self, along, lateral, shift_sign, gain):
        along *= (1.0 - gain * (1.0 - self.shift_longitudinal_scale))
        lateral += gain * shift_sign * self.dynamic_shift_amount
        return along, lateral

    def apply_split_deformation(self, robot_name, along, lateral, gain):
        along *= (1.0 - gain * (1.0 - self.split_longitudinal_scale))
        lateral *= (1.0 + gain * (self.split_lateral_scale - 1.0))
        lateral += gain * self.dynamic_split_extra * self.split_side_sign[robot_name]
        return along, lateral

    def get_desired_offset_for_robot(self, robot_name):
        along, lateral = self.nominal_offset_in_path_frame(robot_name)
        g = self.mode_gain

        if self.formation_mode == 'shift_left' and g > 1e-6:
            along, lateral = self.apply_shift_deformation(
                along,
                lateral,
                shift_sign=+1.0,
                gain=g
            )

        elif self.formation_mode == 'shift_right' and g > 1e-6:
            along, lateral = self.apply_shift_deformation(
                along,
                lateral,
                shift_sign=-1.0,
                gain=g
            )

        elif self.formation_mode == 'split' and g > 1e-6:
            along, lateral = self.apply_split_deformation(
                robot_name,
                along,
                lateral,
                gain=g
            )

        return self.path_frame_to_world(along, lateral)

    def update_smoothed_offsets(self):
        if self.formation_mode == 'normal':
            alpha = self.offset_blend_alpha_recover
        else:
            alpha = self.offset_blend_alpha_obstacle

        for name in self.robots:
            des_x, des_y = self.get_desired_offset_for_robot(name)
            cur_x, cur_y = self.current_offsets[name]

            new_x = (1.0 - alpha) * cur_x + alpha * des_x
            new_y = (1.0 - alpha) * cur_y + alpha * des_y

            self.current_offsets[name] = (new_x, new_y)

    def get_target_for_robot(self, robot_name, travel):
        leader_x, leader_y = self.leader_position(travel)
        off_x, off_y = self.current_offsets[robot_name]
        return leader_x + off_x, leader_y + off_y

    # ---------------------------------------------------------
    # Repulsion
    # ---------------------------------------------------------
    def obstacle_repulsion_force(self, px, py):
        f_obs_x = 0.0
        f_obs_y = 0.0

        for obs in self.detected_obstacles:
            qx, qy = closest_point_on_box(px, py, obs)
            dxo = px - qx
            dyo = py - qy
            dist_obs = math.hypot(dxo, dyo)

            if dist_obs < self.d_safe_obs:
                if dist_obs < 1e-6:
                    dxo = px - obs['cx']
                    dyo = py - obs['cy']
                    norm = math.hypot(dxo, dyo)

                    if norm < 1e-6:
                        dxo = self.path_nx * self.split_side_sign[self.robot_name]
                        dyo = self.path_ny * self.split_side_sign[self.robot_name]
                        norm = 1.0

                    dxo /= norm
                    dyo /= norm
                    safe_dist = 0.08
                else:
                    safe_dist = max(dist_obs, 0.08)
                    dxo /= safe_dist
                    dyo /= safe_dist

                rep_mag = self.k_rep_obs * (
                    (1.0 / safe_dist - 1.0 / self.d_safe_obs) / (safe_dist ** 2)
                )
                f_obs_x += rep_mag * dxo
                f_obs_y += rep_mag * dyo

        return f_obs_x, f_obs_y

    def all_robots_settled(self):
        goal_x, goal_y = self.leader_position(self.total_dist)

        for name in self.robots:
            off_x, off_y = self.current_offsets[name]
            tx = goal_x + off_x
            ty = goal_y + off_y

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
    # Publishers
    # ---------------------------------------------------------
    def publish_mode_state(self):
        mode_msg = String()
        mode_msg.data = self.formation_mode
        self.mode_pub.publish(mode_msg)

        gain_msg = Float32()
        gain_msg.data = float(self.mode_gain)
        self.mode_gain_pub.publish(gain_msg)

        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'active_obstacle'
        marker.id = 0
        marker.type = Marker.CUBE

        if self.active_obstacle is None:
            marker.action = Marker.DELETE
        else:
            marker.action = Marker.ADD
            marker.pose.position.x = float(self.active_obstacle['cx'])
            marker.pose.position.y = float(self.active_obstacle['cy'])
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(2.0 * self.active_obstacle['hx'])
            marker.scale.y = float(2.0 * self.active_obstacle['hy'])
            marker.scale.z = 0.5

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.55

        self.active_obstacle_pub.publish(marker)

    # ---------------------------------------------------------
    # Main control loop
    # ---------------------------------------------------------
    def control_loop(self):
        self.publish_mode_state()

        if not self.have_self_odom:
            self.stop_robot()
            return

        if not self.have_global_start_time or self.motion_start_time is None:
            self.stop_robot()
            return

        now = self.now_sec()

        if self.last_control_time is None:
            dt = self.control_period
        else:
            dt = clamp(now - self.last_control_time, 1e-3, 0.25)

        self.last_control_time = now
        t = now - self.motion_start_time

        if t < 0.0:
            self.stop_robot()
            return

        if (
            self.last_obstacle_update_time is not None and
            (now - self.last_obstacle_update_time) > self.obstacle_timeout
        ):
            self.detected_obstacles = []

        current_travel = min(self.vl_speed * t, self.total_dist)
        next_travel = min(current_travel + self.vl_speed * dt, self.total_dist)

        self.update_smoothed_offsets()

        my_target_x, my_target_y = self.get_target_for_robot(
            self.robot_name,
            current_travel
        )

        next_leader_x, next_leader_y = self.leader_position(next_travel)
        my_off_x, my_off_y = self.current_offsets[self.robot_name]
        next_target_x = next_leader_x + my_off_x
        next_target_y = next_leader_y + my_off_y

        ff_vx = (next_target_x - my_target_x) / dt
        ff_vy = (next_target_y - my_target_y) / dt

        ex = my_target_x - self.current_x
        ey = my_target_y - self.current_y
        
        dist_to_my_target = math.hypot(ex, ey)
        is_leader_at_goal = (current_travel >= self.total_dist - 1e-6)

        if is_leader_at_goal:
            f_att_x = self.k_att * ex
            f_att_y = self.k_att * ey
        else:
            f_att_x = self.k_att * ex + self.k_ff * ff_vx
            f_att_y = self.k_att * ey + self.k_ff * ff_vy

        

        obs_scale, robot_scale, force_alpha = self.current_behavior_scales()

        # Robot-robot repulsion
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
                rep_mag = self.k_rep_robot * (
                    (1.0 / safe_dist - 1.0 / self.d_safe_robot) / (safe_dist ** 2)
                )
                f_rep_x += rep_mag * dx / safe_dist
                f_rep_y += rep_mag * dy / safe_dist

        f_rep_x *= robot_scale
        f_rep_y *= robot_scale

        # Robot-obstacle repulsion
        f_obs_x, f_obs_y = self.obstacle_repulsion_force(self.current_x, self.current_y)
        f_obs_x *= obs_scale
        f_obs_y *= obs_scale

        # Total force
        total_fx = f_att_x + f_rep_x + f_obs_x
        total_fy = f_att_y + f_rep_y + f_obs_y

        # Smooth force
        if is_leader_at_goal and dist_to_my_target < 0.30:
            # Near the goal, remove filter lag so the robot does not
            # keep following an old curved force direction.
            self.filtered_fx = total_fx
            self.filtered_fy = total_fy
        else:
            self.filtered_fx = (
                (1.0 - force_alpha) * self.filtered_fx
                + force_alpha * total_fx
            )
            self.filtered_fy = (
                (1.0 - force_alpha) * self.filtered_fy
                + force_alpha * total_fy
            )

        force_x = self.filtered_fx
        force_y = self.filtered_fy
        force_norm = math.hypot(force_x, force_y)

        force_msg = Point()
        force_msg.x = float(force_x)
        force_msg.y = float(force_y)
        force_msg.z = 0.0
        self.force_pub.publish(force_msg)

        # ---------------------------------------------------------
        # Final stop logic
        # ---------------------------------------------------------
        if is_leader_at_goal:
        # Once this robot is close enough to its final target,
        # stop completely instead of chasing tiny force/yaw noise.
            if dist_to_my_target < self.final_tol:
                self.stop_robot()
                return

        # Safety net: if the whole formation is settled, also stop.
            if self.all_robots_settled():
                self.stop_robot()
                return

        if is_leader_at_goal and dist_to_my_target < 0.8:
            current_max_speed = self.final_max_speed
            current_max_turn = self.final_max_turn
        else:
            current_max_speed = self.max_speed
            current_max_turn = self.max_turn

        cmd = Twist()

        if force_norm < self.force_deadband:
            self.stop_robot()
            return

        target_yaw = math.atan2(force_y, force_x)
        yaw_error = wrap_angle(target_yaw - self.current_yaw)

        if is_leader_at_goal and dist_to_my_target < 0.30:
            w_raw = 0.5 * self.k_yaw * yaw_error
            v_raw = min(force_norm, current_max_speed)
        else:
            w_raw = self.k_yaw * yaw_error
            heading_factor = max(0.0, math.cos(yaw_error))
            v_raw = force_norm * heading_factor

        # Softer rotate-in-place transition than before
        if abs(yaw_error) > self.rotate_in_place_angle:
            excess = abs(yaw_error) - self.rotate_in_place_angle
            fade = clamp(1.0 - excess / 0.8, 0.0, 1.0)
            v_raw *= fade

        v_cmd, w_cmd = speed_cap(
            v_raw,
            w_raw,
            current_max_speed,
            current_max_turn
        )

        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
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