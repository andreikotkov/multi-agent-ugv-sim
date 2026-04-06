import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32, Float64
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, low, high):
    return max(low, min(high, value))


class SwarmRvizVisualizer(Node):
    def __init__(self):
        super().__init__('swarm_rviz_visualizer')

        self.robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        self.frame_id = 'map'

        self.d_safe = 0.3
        self.traj_append_dist = 0.02
        self.force_scale = 0.8

        # ---------------------------------------------------------
        # Parameters (mandatory from YAML / launch)
        # ---------------------------------------------------------
        self.obstacle_timeout = self.reqf('obstacle_timeout')
        self.mode_timeout = self.reqf('mode_timeout')

        self.vl_start_x = self.reqf('vl_start_x')
        self.vl_start_y = self.reqf('vl_start_y')
        self.vl_goal_x = self.reqf('vl_goal_x')
        self.vl_goal_y = self.reqf('vl_goal_y')
        self.vl_speed = self.reqf('vl_speed')
        self.warmup_duration = self.reqf('warmup_duration')

        # Optional tuning params for RViz rendering
        self.robot_height = self.get_param_float('robot_height', 0.12)
        self.robot_label_z = self.get_param_float('robot_label_z', 0.35)
        self.force_z = self.get_param_float('force_z', 0.06)
        self.trail_width = self.get_param_float('trail_width', 0.04)
        self.ghost_trail_width = self.get_param_float('ghost_trail_width', 0.06)
        self.reference_path_width = self.get_param_float('reference_path_width', 0.05)
        self.obstacle_height = self.get_param_float('obstacle_height', 0.20)

        self.status_text_x = self.get_param_float('status_text_x', 11.0)
        self.status_text_y = self.get_param_float('status_text_y', 2.0)
        self.status_text_z = self.get_param_float('status_text_z', 1.0)
        self.status_line_spacing = self.get_param_float('status_line_spacing', 0.38)
        self.status_text_size = self.get_param_float('status_text_size', 0.30)

        # ---------------------------------------------------------
        # Leader path geometry
        # ---------------------------------------------------------
        dx = self.vl_goal_x - self.vl_start_x
        dy = self.vl_goal_y - self.vl_start_y
        self.total_dist = math.hypot(dx, dy)
        if self.total_dist < 1e-9:
            raise ValueError("Leader start and goal cannot be identical.")

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------
        self.current_pos = {name: [0.0, 0.0] for name in self.robots}
        self.current_force = {name: [0.0, 0.0] for name in self.robots}
        self.have_odom = {name: False for name in self.robots}

        self.trails = {name: [] for name in self.robots}
        self.ghost_trail = []

        self.detected_obstacles = []
        self.last_obstacle_update_time = None

        self.current_mode = 'unknown'
        self.current_mode_gain = 0.0
        self.last_mode_update_time = None

        self.active_obstacle = None
        self.last_active_obstacle_time = None

        self.start_time = None
        self.vl_start_time = None
        self.have_global_start_time = False

        # These histories are kept only because your old node subscribed to them.
        # Not visualized here, but harmless to retain.
        self.cmd_linear_history = {name: [] for name in self.robots}
        self.cmd_angular_history = {name: [] for name in self.robots}
        self.time_history = {name: [] for name in self.robots}

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------
        self.marker_pub = self.create_publisher(MarkerArray, '/swarm_rviz_markers', 10)

        # ---------------------------------------------------------
        # Subscriptions
        # ---------------------------------------------------------
        for name in self.robots:
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda msg, n=name: self.odom_callback(msg, n),
                10
            )
            self.create_subscription(
                Point,
                f'/{name}/force_vector',
                lambda msg, n=name: self.force_callback(msg, n),
                10
            )
            self.create_subscription(
                Twist,
                f'/{name}/cmd_vel',
                lambda msg, n=name: self.vel_callback(msg, n),
                10
            )

        self.create_subscription(
            MarkerArray,
            '/detected_obstacles',
            self.obstacle_markers_callback,
            10
        )

        self.create_subscription(
            String,
            '/formation_mode_global',
            self.mode_callback,
            10
        )

        self.create_subscription(
            Float32,
            '/formation_mode_gain_global',
            self.mode_gain_callback,
            10
        )

        self.create_subscription(
            Marker,
            '/active_obstacle_global',
            self.active_obstacle_callback,
            10
        )

        self.create_subscription(
            Float64,
            '/swarm_motion_start_time',
            self.global_motion_start_time_callback,
            10
        )

        # Publish loop
        self.timer = self.create_timer(0.05, self.publish_markers)

        self.get_logger().info('swarm_rviz_visualizer started.')

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

    def get_param_float(self, name, default_value):
        self.declare_parameter(name, default_value)
        return float(self.get_parameter(name).value)

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------
    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def leader_position(self, travel):
        travel = clamp(travel, 0.0, self.total_dist)
        ratio = travel / self.total_dist
        x = self.vl_start_x + (self.vl_goal_x - self.vl_start_x) * ratio
        y = self.vl_start_y + (self.vl_goal_y - self.vl_start_y) * ratio
        return x, y

    def same_obstacle(self, a, b, tol=1e-6):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return (
            abs(a['cx'] - b['cx']) < tol and
            abs(a['cy'] - b['cy']) < tol and
            abs(a['hx'] - b['hx']) < tol and
            abs(a['hy'] - b['hy']) < tol
        )

    def color_for_robot(self, idx):
        palette = [
            (0.10, 0.40, 1.00),
            (0.10, 0.80, 0.20),
            (1.00, 0.55, 0.10),
            (0.85, 0.15, 0.85),
        ]
        return palette[idx % len(palette)]

    def make_point(self, x, y, z=0.0):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def make_marker(self, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def append_trail_point_if_needed(self, trail, x, y):
        if not trail:
            trail.append(self.make_point(x, y, 0.0))
            return

        last = trail[-1]
        if math.hypot(x - last.x, y - last.y) > self.traj_append_dist:
            trail.append(self.make_point(x, y, 0.0))

    def trim_obsolete_ids(self, markers, ns, used_count, max_count):
        """
        Publish DELETE markers for IDs that may have existed previously but are no longer used.
        This avoids stale obstacle labels/cubes staying visible in RViz.
        """
        for mid in range(used_count, max_count):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            markers.markers.append(m)

    def build_status_lines(self, now, obstacles_are_fresh, mode_is_fresh, active_is_fresh):
        lines = []

        lines.append(f"Mode: {self.current_mode if mode_is_fresh else 'unknown'}")
        lines.append(f"Gain: {self.current_mode_gain:.2f}" if mode_is_fresh else "Gain: 0.00")
        lines.append(f"Obstacles: {len(self.detected_obstacles) if obstacles_are_fresh else 0}")

        if active_is_fresh and self.active_obstacle is not None:
            a = self.active_obstacle
            lines.append(f"Obs c=({a['cx']:.2f}, {a['cy']:.2f})")
            lines.append(f"Obs h=({a['hx']:.2f}, {a['hy']:.2f})")
        else:
            lines.append("Obs:")
            lines.append("")

        return lines

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def odom_callback(self, msg, name):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.current_pos[name] = [x, y]

        if not self.have_odom[name]:
            self.have_odom[name] = True
            self.get_logger().info(f"Received first odometry from {name}")

            if all(self.have_odom.values()) and not self.have_global_start_time:
                self.get_logger().info(
                    "All robot odometry available. Waiting for global swarm start time."
                )

        self.append_trail_point_if_needed(self.trails[name], x, y)

    def force_callback(self, msg, name):
        self.current_force[name] = [msg.x, msg.y]

    def vel_callback(self, msg, name):
        if self.start_time is None:
            curr_t = 0.0
        else:
            curr_t = self.now_sec() - self.start_time

        self.time_history[name].append(curr_t)
        self.cmd_linear_history[name].append(msg.linear.x)
        self.cmd_angular_history[name].append(msg.angular.z)

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

    def mode_callback(self, msg):
        self.current_mode = msg.data
        self.last_mode_update_time = self.now_sec()

    def mode_gain_callback(self, msg):
        self.current_mode_gain = float(msg.data)
        self.last_mode_update_time = self.now_sec()

    def active_obstacle_callback(self, msg):
        self.last_active_obstacle_time = self.now_sec()

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

    def global_motion_start_time_callback(self, msg):
        new_start_time = float(msg.data)

        if not self.have_global_start_time:
            self.vl_start_time = new_start_time
            self.start_time = new_start_time - self.warmup_duration
            self.have_global_start_time = True

            self.get_logger().info(
                f"Visualizer received global motion start time {self.vl_start_time:.3f} s"
            )

    # ---------------------------------------------------------
    # Marker publishing
    # ---------------------------------------------------------
    def publish_markers(self):
        now = self.now_sec()

        if self.vl_start_time is None:
            travel = 0.0
        else:
            t = self.now_sec() - self.vl_start_time
            if t < 0.0:
                t = 0.0
            travel = min(self.vl_speed * t, self.total_dist)

        vl_x, vl_y = self.leader_position(travel)
        self.append_trail_point_if_needed(self.ghost_trail, vl_x, vl_y)

        obstacles_are_fresh = (
            self.last_obstacle_update_time is not None and
            (now - self.last_obstacle_update_time) <= self.obstacle_timeout
        )

        mode_is_fresh = (
            self.last_mode_update_time is not None and
            (now - self.last_mode_update_time) <= self.mode_timeout
        )

        active_is_fresh = (
            self.last_active_obstacle_time is not None and
            (now - self.last_active_obstacle_time) <= self.mode_timeout
        )

        shown_active = self.active_obstacle if active_is_fresh else None

        markers = MarkerArray()

        # -----------------------------------------------------
        # Goal marker
        # -----------------------------------------------------
        m = self.make_marker('goal', 0, Marker.SPHERE)
        m.pose.position.x = self.vl_goal_x
        m.pose.position.y = self.vl_goal_y
        m.pose.position.z = 0.0
        m.scale.x = 0.35
        m.scale.y = 0.35
        m.scale.z = 0.35
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        markers.markers.append(m)

        # -----------------------------------------------------
        # Goal label
        # -----------------------------------------------------
        m = self.make_marker('goal_label', 0, Marker.TEXT_VIEW_FACING)
        m.pose.position.x = self.vl_goal_x
        m.pose.position.y = self.vl_goal_y + 0.5
        m.pose.position.z = 1.0
        m.scale.z = 0.35
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.text = 'GOAL'
        markers.markers.append(m)

        # -----------------------------------------------------
        # Virtual leader reference path
        # -----------------------------------------------------
        m = self.make_marker('virtual_leader_reference', 0, Marker.LINE_STRIP)
        m.scale.x = self.reference_path_width
        m.color.r = 0.55
        m.color.g = 0.55
        m.color.b = 0.55
        m.color.a = 0.85
        m.points = [
            self.make_point(self.vl_start_x, self.vl_start_y, 0.01),
            self.make_point(self.vl_goal_x, self.vl_goal_y, 0.01),
        ]
        markers.markers.append(m)

        # -----------------------------------------------------
        # Virtual leader traversed trajectory
        # -----------------------------------------------------
        m = self.make_marker('virtual_leader_traj', 0, Marker.LINE_STRIP)
        m.scale.x = self.ghost_trail_width
        m.color.r = 0.35
        m.color.g = 0.35
        m.color.b = 0.35
        m.color.a = 1.0
        m.points = self.ghost_trail
        markers.markers.append(m)

        # -----------------------------------------------------
        # Virtual leader current marker
        # -----------------------------------------------------
        m = self.make_marker('virtual_leader_body', 0, Marker.SPHERE)
        m.pose.position.x = vl_x
        m.pose.position.y = vl_y
        m.pose.position.z = 0.0
        m.scale.x = 0.25
        m.scale.y = 0.25
        m.scale.z = 0.25
        m.color.r = 0.45
        m.color.g = 0.45
        m.color.b = 0.45
        m.color.a = 1.0
        markers.markers.append(m)

        # -----------------------------------------------------
        # Robots
        # -----------------------------------------------------
        for idx, name in enumerate(self.robots):
            x, y = self.current_pos[name]
            fx, fy = self.current_force[name]
            r, g, b = self.color_for_robot(idx)

            # Robot body
            m = self.make_marker('robots_body', idx, Marker.CYLINDER)
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.0
            m.scale.x = self.d_safe
            m.scale.y = self.d_safe
            m.scale.z = self.robot_height
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.85
            markers.markers.append(m)

            # Robot label
            m = self.make_marker('robots_label', idx, Marker.TEXT_VIEW_FACING)
            m.pose.position.x = x
            m.pose.position.y = y + 0.5
            m.pose.position.z = self.robot_label_z
            m.scale.z = 0.35
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.text = name
            markers.markers.append(m)

            # Robot trajectory
            m = self.make_marker('robots_traj', idx, Marker.LINE_STRIP)
            m.scale.x = self.trail_width
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0
            m.points = self.trails[name]
            markers.markers.append(m)

            # Force vector arrow
            m = self.make_marker('robots_force', idx, Marker.ARROW)
            m.points = [
                self.make_point(x, y, self.force_z),
                self.make_point(x + self.force_scale * fx, y + self.force_scale * fy, self.force_z),
            ]
            m.scale.x = 0.03
            m.scale.y = 0.07
            m.scale.z = 0.10
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            markers.markers.append(m)

        # -----------------------------------------------------
        # Detected obstacles
        # -----------------------------------------------------
        if obstacles_are_fresh:
            for idx, obs in enumerate(self.detected_obstacles):
                m = self.make_marker('obstacles', idx, Marker.CUBE)
                m.pose.position.x = obs['cx']
                m.pose.position.y = obs['cy']
                m.pose.position.z = 0.0
                m.scale.x = 2.0 * obs['hx']
                m.scale.y = 2.0 * obs['hy']
                m.scale.z = self.obstacle_height
                m.color.r = 1.0
                m.color.g = 0.60
                m.color.b = 0.00
                m.color.a = 0.35
                markers.markers.append(m)

                m = self.make_marker('obstacles_label', idx, Marker.TEXT_VIEW_FACING)
                m.pose.position.x = obs['cx']
                m.pose.position.y = obs['cy']
                m.pose.position.z = self.obstacle_height * 0.7 + 0.15
                m.scale.z = 0.35
                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 1.0
                m.text = f'O{idx+1}'
                markers.markers.append(m)

        # Delete stale obstacle markers that might remain from a previous larger set
        max_obstacle_slots = 64
        used_obstacles = len(self.detected_obstacles) if obstacles_are_fresh else 0
        self.trim_obsolete_ids(markers, 'obstacles', used_obstacles, max_obstacle_slots)
        self.trim_obsolete_ids(markers, 'obstacles_label', used_obstacles, max_obstacle_slots)

        # -----------------------------------------------------
        # Active obstacle highlight
        # -----------------------------------------------------
        if shown_active is not None:
            # translucent box
            m = self.make_marker('active_obstacle_fill', 0, Marker.CUBE)
            m.pose.position.x = shown_active['cx']
            m.pose.position.y = shown_active['cy']
            m.pose.position.z = 0.01
            m.scale.x = 2.0 * shown_active['hx'] + 0.04
            m.scale.y = 2.0 * shown_active['hy'] + 0.04
            m.scale.z = self.obstacle_height + 0.04
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.18
            markers.markers.append(m)

            # outline using LINE_STRIP
            x0 = shown_active['cx'] - shown_active['hx'] - 0.02
            x1 = shown_active['cx'] + shown_active['hx'] + 0.02
            y0 = shown_active['cy'] - shown_active['hy'] - 0.02
            y1 = shown_active['cy'] + shown_active['hy'] + 0.02
            z = self.obstacle_height * 0.55

            m = self.make_marker('active_obstacle_outline', 0, Marker.LINE_STRIP)
            m.scale.x = 0.05
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.points = [
                self.make_point(x0, y0, z),
                self.make_point(x1, y0, z),
                self.make_point(x1, y1, z),
                self.make_point(x0, y1, z),
                self.make_point(x0, y0, z),
            ]
            markers.markers.append(m)
        else:
            # Remove active highlight markers when no active obstacle
            for ns in ('active_obstacle_fill', 'active_obstacle_outline'):
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = ns
                m.id = 0
                m.action = Marker.DELETE
                markers.markers.append(m)

        # -----------------------------------------------------
        # Status text in RViz (2-column layout)
        # -----------------------------------------------------
        status_lines = self.build_status_lines(
            now=now,
            obstacles_are_fresh=obstacles_are_fresh,
            mode_is_fresh=mode_is_fresh,
            active_is_fresh=active_is_fresh
        )

        label_x = self.status_text_x
        value_x = self.status_text_x + 1.0
        line_spacing = self.status_line_spacing

        max_status_lines = 8

        for i, line in enumerate(status_lines):
            if ':' in line:
                label, value = line.split(':', 1)
            else:
                label, value = line, ''

            y = self.status_text_y - i * line_spacing

            # label marker
            m = self.make_marker('status_label', i, Marker.TEXT_VIEW_FACING)
            m.pose.position.x = label_x
            m.pose.position.y = y
            m.pose.position.z = self.status_text_z
            m.scale.z = self.status_text_size
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.text = label + ':'
            markers.markers.append(m)

            # value marker
            m = self.make_marker('status_value', i, Marker.TEXT_VIEW_FACING)
            m.pose.position.x = value_x
            m.pose.position.y = y
            m.pose.position.z = self.status_text_z
            m.scale.z = self.status_text_size
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.text = value.strip()
            markers.markers.append(m)

        # delete old unused markers
        for i in range(len(status_lines), max_status_lines):
            for ns in ('status_label', 'status_value'):
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = ns
                m.id = i
                m.action = Marker.DELETE
                markers.markers.append(m)

        

        # Publish
        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = SwarmRvizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()