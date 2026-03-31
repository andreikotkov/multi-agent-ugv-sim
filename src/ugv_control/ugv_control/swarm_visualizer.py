import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, low, high):
    return max(low, min(high, value))


class SwarmTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('swarm_visualizer')

        self.robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        self.d_safe = 0.3

        self.declare_parameter('obstacle_timeout', 1.0)
        self.declare_parameter('mode_timeout', 1.0)

        self.obstacle_timeout = float(self.get_parameter('obstacle_timeout').value)
        self.mode_timeout = float(self.get_parameter('mode_timeout').value)

        self.traj_x = {name: [] for name in self.robots}
        self.traj_y = {name: [] for name in self.robots}
        self.current_pos = {name: [0.0, 0.0] for name in self.robots}
        self.current_force = {name: [0.0, 0.0] for name in self.robots}
        self.traj_append_dist = 0.02

        self.vl_start_x = 0.0
        self.vl_start_y = 0.0
        self.vl_goal_x = 10.0
        self.vl_goal_y = 10.0
        self.vl_speed = 0.25
        self.warmup_duration = 0.0

        self.ghost_traj_x = []
        self.ghost_traj_y = []

        self.cmd_linear_history = {name: [] for name in self.robots}
        self.cmd_angular_history = {name: [] for name in self.robots}
        self.time_history = {name: [] for name in self.robots}

        self.detected_obstacles = []
        self.last_obstacle_update_time = None

        self.current_mode = 'unknown'
        self.current_mode_gain = 0.0
        self.last_mode_update_time = None

        self.active_obstacle = None
        self.last_active_obstacle_time = None

        self.event_log = []
        self.max_event_lines = 12
        self.last_logged_mode = None
        self.last_logged_active_obstacle = None
        self.last_logged_obstacle_count = None

        dx = self.vl_goal_x - self.vl_start_x
        dy = self.vl_goal_y - self.vl_start_y
        self.total_dist = math.hypot(dx, dy)
        if self.total_dist < 1e-9:
            raise ValueError("Leader start and goal cannot be identical.")

        # synchronized start with all robots
        self.start_time = None
        self.vl_start_time = None
        self.have_odom = {name: False for name in self.robots}

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

        plt.ion()
        self.fig_traj, self.ax_traj = plt.subplots(figsize=(16, 12))
        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry("800x600+30+30")
        except Exception:
            pass

        self.robot_colors = {}
        self.lines = {}
        self.circles = {}
        self.robot_labels = {}

        self.ghost_line = None
        self.ghost_marker = None

        self.status_text = None
        self.obstacle_list_text = None
        self.event_text = None

        self.obstacle_rects = []
        self.obstacle_texts = []
        self.active_obstacle_rect = None
        self.quivers = {}

        self.setup_plot()
        self.timer = self.create_timer(0.01, self.update_plot)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------
    # Plot setup
    # ---------------------------------------------------------
    def setup_plot(self):
        self.ax_traj.set_xlim(-4, 16)
        self.ax_traj.set_ylim(-1, 12)
        self.ax_traj.set_aspect('equal')

        self.ax_traj.xaxis.set_major_locator(MultipleLocator(1.0))
        self.ax_traj.yaxis.set_major_locator(MultipleLocator(1.0))
        self.ax_traj.grid(True, which='major', linewidth=0.8)

        self.ax_traj.set_title("Real-Time Swarm Trajectory, Forces, Obstacles, and Exact Controller Mode")

        self.ax_traj.plot(
            self.vl_goal_x, self.vl_goal_y,
            'rX', markersize=12, label='Goal'
        )

        self.ax_traj.plot(
            [self.vl_start_x, self.vl_goal_x],
            [self.vl_start_y, self.vl_goal_y],
            ':', color='gray', alpha=0.6, label='Leader path'
        )

        self.ghost_line, = self.ax_traj.plot(
            [], [], '--', color='gray', alpha=0.7, linewidth=2, label='Virtual Leader'
        )
        self.ghost_marker = patches.Circle((0.0, 0.0), 0.12, color='gray', alpha=0.7)
        self.ax_traj.add_patch(self.ghost_marker)

        for name in self.robots:
            line, = self.ax_traj.plot([], [], linewidth=1.6, label=f'{name} Path')
            self.lines[name] = line
            self.robot_colors[name] = line.get_color()

        for name in self.robots:
            circle = patches.Circle(
                (0.0, 0.0), self.d_safe / 2.0,
                color=self.robot_colors[name], alpha=0.45
            )
            self.ax_traj.add_patch(circle)
            self.circles[name] = circle

            self.robot_labels[name] = self.ax_traj.text(0.0, 0.0, name, fontsize=9)
            self.quivers[name] = None

        self.status_text = self.ax_traj.text(
            0.02, 0.98, "",
            transform=self.ax_traj.transAxes,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.88)
        )

        self.obstacle_list_text = self.ax_traj.text(
            0.02, 0.62, "",
            transform=self.ax_traj.transAxes,
            ha='left', va='top',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.88)
        )

        self.event_text = self.ax_traj.text(
            0.50, 0.06, "",
            transform=self.ax_traj.transAxes,
            ha='left', va='bottom',
            fontsize=9,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.88)
        )

        self.ax_traj.legend(loc='upper right', fontsize=8, framealpha=0.9)
        self.fig_traj.tight_layout()
        self.fig_traj.canvas.draw()
        self.fig_traj.canvas.flush_events()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def add_event(self, text):
        if self.start_time is None:
            stamp = 0.0
        else:
            stamp = self.now_sec() - self.start_time
        self.event_log.append(f"[{stamp:6.2f}s] {text}")
        if len(self.event_log) > self.max_event_lines:
            self.event_log = self.event_log[-self.max_event_lines:]

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

    def leader_position(self, travel):
        travel = clamp(travel, 0.0, self.total_dist)
        ratio = travel / self.total_dist
        x = self.vl_start_x + (self.vl_goal_x - self.vl_start_x) * ratio
        y = self.vl_start_y + (self.vl_goal_y - self.vl_start_y) * ratio
        return x, y

    def log_events(self, obstacle_status, mode_status):
        obstacle_count = len(self.detected_obstacles) if obstacle_status != "STALE" else 0
        mode_label = self.current_mode if mode_status != "STALE" else "stale"

        if self.last_logged_mode is None:
            self.last_logged_mode = mode_label
            self.add_event(f"Initial mode: {mode_label}")
        elif mode_label != self.last_logged_mode:
            self.add_event(f"Mode changed: {self.last_logged_mode} -> {mode_label}")
            self.last_logged_mode = mode_label

        if self.last_logged_obstacle_count is None:
            self.last_logged_obstacle_count = obstacle_count
            self.add_event(f"Detected obstacles: {obstacle_count}")
        elif obstacle_count != self.last_logged_obstacle_count:
            self.add_event(f"Detected obstacles changed: {self.last_logged_obstacle_count} -> {obstacle_count}")
            self.last_logged_obstacle_count = obstacle_count

        active = self.active_obstacle if mode_status != "STALE" else None
        if not self.same_obstacle(active, self.last_logged_active_obstacle):
            if active is None:
                self.add_event("Active obstacle cleared")
            else:
                self.add_event("Active obstacle:")
                self.add_event(
                    f"x={active['cx']:.2f},y={active['cy']:.2f}, "
                    f"hx={active['hx']:.2f},hy={active['hy']:.2f}"
                )
            self.last_logged_active_obstacle = None if active is None else dict(active)

    def clear_obstacle_artists(self):
        for rect in self.obstacle_rects:
            rect.remove()
        for txt in self.obstacle_texts:
            txt.remove()

        self.obstacle_rects = []
        self.obstacle_texts = []

        if self.active_obstacle_rect is not None:
            self.active_obstacle_rect.remove()
            self.active_obstacle_rect = None

    def rebuild_obstacle_artists(self, shown_active, obstacles_are_fresh):
        self.clear_obstacle_artists()

        if obstacles_are_fresh:
            for idx, obs in enumerate(self.detected_obstacles):
                rect = patches.Rectangle(
                    (obs['cx'] - obs['hx'], obs['cy'] - obs['hy']),
                    2.0 * obs['hx'],
                    2.0 * obs['hy'],
                    facecolor='orange',
                    edgecolor='black',
                    linewidth=1.2,
                    alpha=0.35
                )
                self.ax_traj.add_patch(rect)
                self.obstacle_rects.append(rect)

                txt = self.ax_traj.text(
                    obs['cx'],
                    obs['cy'],
                    f"O{idx+1}",
                    ha='center',
                    va='center',
                    fontsize=9,
                    weight='bold'
                )
                self.obstacle_texts.append(txt)

        if shown_active is not None:
            self.active_obstacle_rect = patches.Rectangle(
                (shown_active['cx'] - shown_active['hx'], shown_active['cy'] - shown_active['hy']),
                2.0 * shown_active['hx'],
                2.0 * shown_active['hy'],
                fill=False,
                edgecolor='red',
                linewidth=3.0,
                linestyle='--'
            )
            self.ax_traj.add_patch(self.active_obstacle_rect)

    def rebuild_quivers(self):
        for name in self.robots:
            if self.quivers[name] is not None:
                self.quivers[name].remove()

            cx, cy = self.current_pos[name]
            fx, fy = self.current_force[name]

            self.quivers[name] = self.ax_traj.quiver(
                cx, cy, fx, fy,
                angles='xy',
                scale_units='xy',
                scale=1.0,
                color='black',
                width=0.004
            )

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def odom_callback(self, msg, name):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if not self.have_odom[name]:
            self.have_odom[name] = True
            self.get_logger().info(f"Received first odometry from {name}")

        if self.start_time is None and all(self.have_odom.values()):
            self.start_time = self.now_sec()
            self.vl_start_time = self.start_time + self.warmup_duration
            self.get_logger().info(
                f"All robot odometry received. Visualizer virtual leader will start after {self.warmup_duration:.2f}s warmup."
            )

        if not self.traj_x[name] or math.hypot(x - self.traj_x[name][-1], y - self.traj_y[name][-1]) > self.traj_append_dist:
            self.traj_x[name].append(x)
            self.traj_y[name].append(y)

        self.current_pos[name] = [x, y]

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

    # ---------------------------------------------------------
    # Plot update
    # ---------------------------------------------------------
    def update_plot(self):
        if self.vl_start_time is None:
            t = 0.0
            travel = 0.0
        else:
            t = self.now_sec() - self.vl_start_time
            if t < 0.0:
                t = 0.0
            travel = min(self.vl_speed * t, self.total_dist)

        vl_x, vl_y = self.leader_position(travel)

        if not self.ghost_traj_x or math.hypot(vl_x - self.ghost_traj_x[-1], vl_y - self.ghost_traj_y[-1]) > self.traj_append_dist:
            self.ghost_traj_x.append(vl_x)
            self.ghost_traj_y.append(vl_y)

        now = self.now_sec()

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

        if obstacles_are_fresh and len(self.detected_obstacles) > 0:
            obstacle_status = "DETECTED"
        elif obstacles_are_fresh:
            obstacle_status = "NO OBSTACLE"
        else:
            obstacle_status = "STALE"

        mode_status = "FRESH" if mode_is_fresh else "STALE"
        shown_mode = self.current_mode if mode_is_fresh else "unknown"
        shown_gain = self.current_mode_gain if mode_is_fresh else 0.0
        shown_active = self.active_obstacle if active_is_fresh else None

        self.log_events(obstacle_status, mode_status)

        current_signature = (
            tuple((o['cx'], o['cy'], o['hx'], o['hy']) for o in self.detected_obstacles) if obstacles_are_fresh else tuple(),
            None if shown_active is None else (shown_active['cx'], shown_active['cy'], shown_active['hx'], shown_active['hy'])
        )

        if not hasattr(self, '_last_obstacle_signature') or self._last_obstacle_signature != current_signature:
            self._last_obstacle_signature = current_signature
            self.rebuild_obstacle_artists(shown_active, obstacles_are_fresh)

        status_lines = [
            "Mode source: global",
            f"Mode: {shown_mode}",
            f"Mode gain: {shown_gain:.2f}",
            f"Detected obstacles: {len(self.detected_obstacles) if obstacles_are_fresh else 0}",
        ]

        if self.vl_start_time is None:
            status_lines.append("Virtual leader: waiting for odom")
        else:
            wait_left = self.vl_start_time - now
            if wait_left > 0.0:
                status_lines.append(f"Virtual leader start in: {wait_left:.2f}s")

        if obstacles_are_fresh and self.detected_obstacles:
            obstacle_lines = ["Detected obstacle boxes:"]
            for i, obs in enumerate(self.detected_obstacles[:6]):
                mark = "*" if self.same_obstacle(obs, shown_active) else " "
                obstacle_lines.append(
                    f"{mark} O{i+1}: c=({obs['cx']:.2f},{obs['cy']:.2f}) "
                    f"h=({obs['hx']:.2f},{obs['hy']:.2f})"
                )
        else:
            obstacle_lines = ["Detected obstacle boxes:", "none"]

        event_lines = ["Recent events:"] + self.event_log[-self.max_event_lines:]

        self.status_text.set_text("\n".join(status_lines))
        self.obstacle_list_text.set_text("\n".join(obstacle_lines))
        self.event_text.set_text("\n".join(event_lines))

        self.ghost_line.set_data(self.ghost_traj_x, self.ghost_traj_y)
        self.ghost_marker.center = (vl_x, vl_y)

        for name in self.robots:
            self.lines[name].set_data(self.traj_x[name], self.traj_y[name])

            cx, cy = self.current_pos[name]
            self.circles[name].center = (cx, cy)
            self.robot_labels[name].set_position((cx + 0.10, cy + 0.10))

        self.rebuild_quivers()

        self.fig_traj.canvas.draw_idle()
        self.fig_traj.canvas.flush_events()

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------
    def plot_final_results(self):
        plt.ioff()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_name = f"test_run_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        traj_path = os.path.join(folder_name, 'trajectory_map.png')
        self.fig_traj.savefig(traj_path, dpi=200, bbox_inches='tight')
        print(f"[Visualizer] Trajectory map saved to {traj_path}")

        fig_cmd, axs = plt.subplots(len(self.robots), 1, figsize=(10, 12), sharex=True)
        fig_cmd.suptitle(f'Command Profiles - {timestamp}', fontsize=16)

        for i, name in enumerate(self.robots):
            axs[i].plot(
                self.time_history[name],
                self.cmd_linear_history[name],
                label='cmd.linear.x',
                color='blue',
                alpha=0.85
            )
            axs[i].plot(
                self.time_history[name],
                self.cmd_angular_history[name],
                label='cmd.angular.z',
                color='red',
                alpha=0.85
            )
            axs[i].set_ylabel(name)
            axs[i].grid(True)
            axs[i].legend(loc='upper right')

        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        cmd_path = os.path.join(folder_name, 'command_profile.png')
        fig_cmd.savefig(cmd_path, dpi=200, bbox_inches='tight')
        print(f"[Visualizer] Command profile saved to {cmd_path}")

        plt.show()


def main():
    rclpy.init()
    node = SwarmTrajectoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Visualizer] Simulation stopped. Generating final report...")
        node.plot_final_results()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()