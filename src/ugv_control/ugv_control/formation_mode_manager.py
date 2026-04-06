import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Float64
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, low, high):
    return max(low, min(high, value))


class FormationModeManager(Node):
    def __init__(self):
        super().__init__('formation_mode_manager')

        # ---------------------------------------------------------
        # Parameters (mandatory from YAML / launch)
        # ---------------------------------------------------------
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

        self.mode_enter_lookahead = self.reqf('mode_enter_lookahead')
        self.mode_exit_lookahead = self.reqf('mode_exit_lookahead')
        self.corridor_margin = self.reqf('corridor_margin')
        self.split_center_threshold = self.reqf('split_center_threshold')
        self.split_center_hysteresis = self.reqf('split_center_hysteresis')

        self.obstacle_pass_clearance = self.reqf('obstacle_pass_clearance')
        self.recovery_hold_time = self.reqf('recovery_hold_time')

        self.formation_obstacle_margin = self.reqf('formation_obstacle_margin')
        self.max_shift_amount = self.reqf('max_shift_amount')
        self.max_split_extra = self.reqf('max_split_extra')
        self.min_shift_amount = self.reqf('min_shift_amount')
        self.min_split_extra = self.reqf('min_split_extra')

        # Robot body footprint (full dimensions)
        self.robot_length = self.reqf('robot_length')
        self.robot_width = self.reqf('robot_width')

        # ---------------------------------------------------------
        # Path geometry
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
        self.robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        self.shape_base = {
            'ugv1': (-1.0,  0.0),
            'ugv2': ( 0.0,  1.0),
            'ugv3': ( 0.0, -1.0),
            'ugv4': ( 1.0,  0.0),
        }

        (
            self.nominal_longitudinal_radius,
            self.nominal_lateral_radius
        ) = self.compute_nominal_path_frame_radii()

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.detected_obstacles = []
        self.last_obstacle_update_time = None

        self.formation_mode = 'normal'
        self.active_obstacle = None
        self.mode_gain = 0.0
        self.last_return_to_normal_time = None

        self.dynamic_shift_amount = 0.0
        self.dynamic_split_extra = 0.0

        # synchronized start
        self.have_odom = {name: False for name in self.robots}
        self.start_time = None
        self.vl_start_time = None
        self.global_motion_start_time = None

        # ---------------------------------------------------------
        # ROS interfaces
        # ---------------------------------------------------------
        self.mode_pub = self.create_publisher(String, '/formation_mode_global', 10)
        self.mode_gain_pub = self.create_publisher(Float32, '/formation_mode_gain_global', 10)
        self.active_obstacle_pub = self.create_publisher(Marker, '/active_obstacle_global', 10)

        self.shift_amount_pub = self.create_publisher(Float32, '/formation_shift_amount_global', 10)
        self.split_extra_pub = self.create_publisher(Float32, '/formation_split_extra_global', 10)
        self.motion_start_pub = self.create_publisher(Float64, '/swarm_motion_start_time', 10)

        self.create_subscription(
            MarkerArray,
            self.obstacle_topic,
            self.obstacle_markers_callback,
            10
        )

        for name in self.robots:
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda msg, n=name: self.odom_callback(msg, n),
                10
            )

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info(
            'formation_mode_manager started with mandatory YAML parameters loaded. '
            f'Footprint-aware formation radii enabled: '
            f'robot_length={self.robot_length:.3f}, '
            f'robot_width={self.robot_width:.3f}, '
            f'nominal_longitudinal_radius={self.nominal_longitudinal_radius:.3f}, '
            f'nominal_lateral_radius={self.nominal_lateral_radius:.3f}. '
            'Waiting for all robot odometry...'
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

    def body_half_extents_in_path_frame(self):
        hx = 0.5 * self.robot_length
        hy = 0.5 * self.robot_width

        along_half = abs(self.path_ux) * hx + abs(self.path_uy) * hy
        lateral_half = abs(self.path_nx) * hx + abs(self.path_ny) * hy

        return along_half, lateral_half

    def compute_nominal_path_frame_radii(self):
        max_along = 0.0
        max_lateral = 0.0

        body_along_half, body_lateral_half = self.body_half_extents_in_path_frame()

        for name in self.robots:
            bx, by = self.shape_base[name]
            along_c, lateral_c = self.world_vec_to_path_frame(bx, by)

            robot_along_extent = abs(along_c) + body_along_half
            robot_lateral_extent = abs(lateral_c) + body_lateral_half

            max_along = max(max_along, robot_along_extent)
            max_lateral = max(max_lateral, robot_lateral_extent)

        return max_along, max_lateral

    def leader_position(self, travel):
        travel = clamp(travel, 0.0, self.total_dist)
        ratio = travel / self.total_dist
        x = self.vl_start_x + (self.vl_goal_x - self.vl_start_x) * ratio
        y = self.vl_start_y + (self.vl_goal_y - self.vl_start_y) * ratio
        return x, y

    def obstacle_path_metrics(self, obs, travel):
        leader_x, leader_y = self.leader_position(travel)

        dx = obs['cx'] - leader_x
        dy = obs['cy'] - leader_y

        along_center = dx * self.path_ux + dy * self.path_uy
        lateral_center = dx * self.path_nx + dy * self.path_ny

        along_half = abs(self.path_ux) * obs['hx'] + abs(self.path_uy) * obs['hy']
        lateral_half = abs(self.path_nx) * obs['hx'] + abs(self.path_ny) * obs['hy']

        return along_center, lateral_center, along_half, lateral_half

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def odom_callback(self, msg, robot_name):
        if not self.have_odom[robot_name]:
            self.have_odom[robot_name] = True
            self.get_logger().info(f"Received first odometry from {robot_name}")

        if self.start_time is None and all(self.have_odom.values()):
            self.start_time = self.now_sec()
            self.vl_start_time = self.start_time + self.warmup_duration
            self.global_motion_start_time = self.vl_start_time

            self.get_logger().info(
                f"All robot odometry received. Global swarm motion start scheduled at "
                f"{self.global_motion_start_time:.3f} s."
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

    # ---------------------------------------------------------
    # Mode logic
    # ---------------------------------------------------------
    def find_relevant_obstacle(self, travel, lookahead):
        best_obs = None
        best_score = float('inf')

        for obs in self.detected_obstacles:
            along_c, lateral_c, along_h, lateral_h = self.obstacle_path_metrics(obs, travel)

            front_edge = along_c - along_h
            back_edge = along_c + along_h

            if back_edge < -0.4:
                continue
            if front_edge > lookahead:
                continue

            lateral_gap = max(0.0, abs(lateral_c) - lateral_h)
            score = max(front_edge, -0.5) + 0.35 * lateral_gap

            if score < best_score:
                best_score = score
                best_obs = obs

        return best_obs

    def classify_mode_from_obstacle(self, obs, travel):
        if obs is None:
            return 'normal'

        along_c, lateral_c, along_h, lateral_h = self.obstacle_path_metrics(obs, travel)

        front_edge = along_c - along_h
        back_edge = along_c + along_h

        lookahead = (
            self.mode_exit_lookahead
            if self.formation_mode != 'normal'
            else self.mode_enter_lookahead
        )

        if back_edge < -0.4 or front_edge > lookahead:
            return 'normal'

        corridor_half_width = self.nominal_lateral_radius + self.corridor_margin
        overlaps_corridor = abs(lateral_c) <= (lateral_h + corridor_half_width)

        if not overlaps_corridor:
            return 'normal'

        center_thresh = self.split_center_threshold
        if self.formation_mode == 'split':
            center_thresh += self.split_center_hysteresis

        if abs(lateral_c) <= (lateral_h + center_thresh):
            return 'split'

        if lateral_c > 0.0:
            return 'shift_right'

        return 'shift_left'

    def compute_mode_gain_for_mode(self, obs, travel, mode_name):
        if obs is None or mode_name == 'normal':
            return 0.0

        along_c, lateral_c, along_h, lateral_h = self.obstacle_path_metrics(obs, travel)

        front_edge = max(0.0, along_c - along_h)
        lookahead = (
            self.mode_exit_lookahead
            if mode_name != 'normal'
            else self.mode_enter_lookahead
        )

        proximity = 1.0 - clamp(front_edge / max(lookahead, 1e-3), 0.0, 1.0)
        proximity = proximity * proximity * (3.0 - 2.0 * proximity)

        corridor_half_width = self.nominal_lateral_radius + self.corridor_margin
        overlap = (lateral_h + corridor_half_width) - abs(lateral_c)
        overlap_norm = clamp(
            overlap / max(lateral_h + corridor_half_width, 1e-3),
            0.0,
            1.0
        )

        return proximity * overlap_norm

    def is_obstacle_clearly_passed(self, obs, travel):
        if obs is None:
            return True

        along_c, _, along_h, _ = self.obstacle_path_metrics(obs, travel)
        back_edge = along_c + along_h
        return back_edge < -self.obstacle_pass_clearance

    def recovery_hold_active(self):
        if self.last_return_to_normal_time is None:
            return False
        return (self.now_sec() - self.last_return_to_normal_time) < self.recovery_hold_time

    # ---------------------------------------------------------
    # Smarter deformation logic
    # ---------------------------------------------------------
    def compute_required_shift_amount(self, obs, travel, mode_name):
        if obs is None:
            return 0.0

        _, lateral_c, _, lateral_h = self.obstacle_path_metrics(obs, travel)
        margin = self.formation_obstacle_margin

        if mode_name == 'shift_right':
            req = lateral_c + lateral_h + margin + self.nominal_lateral_radius
            return clamp(req, self.min_shift_amount, self.max_shift_amount)

        if mode_name == 'shift_left':
            req = -lateral_c + lateral_h + margin + self.nominal_lateral_radius
            return clamp(req, self.min_shift_amount, self.max_shift_amount)

        return 0.0

    def compute_required_split_extra(self, obs, travel):
        if obs is None:
            return 0.0

        _, _, _, lateral_h = self.obstacle_path_metrics(obs, travel)
        margin = self.formation_obstacle_margin

        extra = lateral_h + margin - self.nominal_lateral_radius
        extra = max(0.0, extra)
        return clamp(extra, self.min_split_extra, self.max_split_extra)

    def update_dynamic_deformation(self, obs, travel, mode_name):
        if mode_name in ('shift_left', 'shift_right'):
            self.dynamic_shift_amount = self.compute_required_shift_amount(
                obs,
                travel,
                mode_name
            )
            self.dynamic_split_extra = 0.0
        elif mode_name == 'split':
            self.dynamic_shift_amount = 0.0
            self.dynamic_split_extra = self.compute_required_split_extra(obs, travel)
        else:
            self.dynamic_shift_amount = 0.0
            self.dynamic_split_extra = 0.0

    def update_formation_mode(self, travel):
        prev_mode = self.formation_mode
        prev_active = self.active_obstacle

        if prev_active is not None and not self.is_obstacle_clearly_passed(prev_active, travel):
            obs = prev_active
        else:
            if self.recovery_hold_active():
                obs = None
            else:
                lookahead = (
                    self.mode_enter_lookahead
                    if self.formation_mode == 'normal'
                    else self.mode_exit_lookahead
                )
                obs = self.find_relevant_obstacle(travel, lookahead)

        new_mode = self.classify_mode_from_obstacle(obs, travel)

        if self.recovery_hold_active():
            new_mode = 'normal'
            obs = None

        if new_mode == 'normal':
            new_gain = 0.0
            new_active = None
        else:
            new_gain = self.compute_mode_gain_for_mode(obs, travel, new_mode)
            new_active = obs

        self.update_dynamic_deformation(obs, travel, new_mode)

        if prev_mode != 'normal' and new_mode == 'normal':
            self.last_return_to_normal_time = self.now_sec()

        if new_mode != prev_mode:
            self.get_logger().info(
                f"Formation mode changed: {prev_mode} -> {new_mode} | "
                f"shift={self.dynamic_shift_amount:.2f}, "
                f"split_extra={self.dynamic_split_extra:.2f}"
            )

        self.formation_mode = new_mode
        self.mode_gain = new_gain
        self.active_obstacle = new_active

    # ---------------------------------------------------------
    # Publishers
    # ---------------------------------------------------------
    def publish_motion_start_time(self):
        if self.global_motion_start_time is None:
            return

        msg = Float64()
        msg.data = float(self.global_motion_start_time)
        self.motion_start_pub.publish(msg)

    def publish_mode_state(self):
        mode_msg = String()
        mode_msg.data = self.formation_mode
        self.mode_pub.publish(mode_msg)

        gain_msg = Float32()
        gain_msg.data = float(self.mode_gain)
        self.mode_gain_pub.publish(gain_msg)

        shift_msg = Float32()
        shift_msg.data = float(self.dynamic_shift_amount)
        self.shift_amount_pub.publish(shift_msg)

        split_msg = Float32()
        split_msg.data = float(self.dynamic_split_extra)
        self.split_extra_pub.publish(split_msg)

        marker = Marker()
        marker.header.frame_id = self.world_frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'active_obstacle_global'
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
    # Main loop
    # ---------------------------------------------------------
    def control_loop(self):
        now = self.now_sec()

        # Always republish shared start time so every robot eventually gets it
        self.publish_motion_start_time()

        if self.last_obstacle_update_time is not None:
            if (now - self.last_obstacle_update_time) > self.obstacle_timeout:
                self.detected_obstacles = []

        if self.vl_start_time is None:
            self.formation_mode = 'normal'
            self.mode_gain = 0.0
            self.active_obstacle = None
            self.dynamic_shift_amount = 0.0
            self.dynamic_split_extra = 0.0
            self.publish_mode_state()
            return

        t = now - self.vl_start_time
        if t < 0.0:
            self.formation_mode = 'normal'
            self.mode_gain = 0.0
            self.active_obstacle = None
            self.dynamic_shift_amount = 0.0
            self.dynamic_split_extra = 0.0
            self.publish_mode_state()
            return

        travel = min(self.vl_speed * t, self.total_dist)

        self.update_formation_mode(travel)
        self.publish_mode_state()


def main(args=None):
    rclpy.init(args=args)
    node = FormationModeManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()