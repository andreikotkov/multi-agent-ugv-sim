import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class SwarmPlotLogger(Node):
    def __init__(self):
        super().__init__('swarm_plot_logger')

        self.declare_parameter('robots', ['ugv1', 'ugv2', 'ugv3', 'ugv4'])
        self.declare_parameter('output_root', '.')
        self.declare_parameter('figure_dpi', 180)

        # Trimming thresholds:
        # trailing interval is removed once ALL robots stay below these limits
        self.declare_parameter('stop_linear_eps', 0.02)
        self.declare_parameter('stop_angular_eps', 0.03)
        self.declare_parameter('stop_hold_time', 1.0)

        self.robots = list(self.get_parameter('robots').value)
        self.output_root = str(self.get_parameter('output_root').value)
        self.figure_dpi = int(self.get_parameter('figure_dpi').value)

        self.stop_linear_eps = float(self.get_parameter('stop_linear_eps').value)
        self.stop_angular_eps = float(self.get_parameter('stop_angular_eps').value)
        self.stop_hold_time = float(self.get_parameter('stop_hold_time').value)

        ts = datetime.now().strftime('test_run_%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(self.output_root, ts)
        os.makedirs(self.output_dir, exist_ok=True)

        self.start_time = self.now_sec()
        self.saved_once = False

        self.data = {}
        for robot in self.robots:
            self.data[robot] = {
                't_cmd': [],
                'v_cmd': [],
                'w_cmd': [],
                't_odom': [],
                'x': [],
                'y': [],
            }

        for robot in self.robots:
            self.create_subscription(
                Twist,
                f'/{robot}/cmd_vel',
                lambda msg, r=robot: self.cmd_callback(msg, r),
                500
            )

            self.create_subscription(
                Odometry,
                f'/{robot}/odom',
                lambda msg, r=robot: self.odom_callback(msg, r),
                500
            )

        self.get_logger().info('SwarmPlotLogger started')
        self.get_logger().info('Stores all samples and saves once on shutdown')
        self.get_logger().info(f'Output folder: {self.output_dir}')

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------
    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def rel_time(self):
        return self.now_sec() - self.start_time

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    def cmd_callback(self, msg, robot):
        t = self.rel_time()
        self.data[robot]['t_cmd'].append(t)
        self.data[robot]['v_cmd'].append(float(msg.linear.x))
        self.data[robot]['w_cmd'].append(float(msg.angular.z))

    def odom_callback(self, msg, robot):
        t = self.rel_time()
        self.data[robot]['t_odom'].append(t)
        self.data[robot]['x'].append(float(msg.pose.pose.position.x))
        self.data[robot]['y'].append(float(msg.pose.pose.position.y))

    # ---------------------------------------------------------
    # Trimming logic
    # ---------------------------------------------------------
    def get_trim_end_time(self):
        """
        Find the end time for plotting by removing the trailing interval where
        all robots are stopped continuously for at least stop_hold_time.

        Returns:
            trim_end_time (float or None)
            If None, use full data.
        """
        event_times = sorted({
            t
            for robot in self.robots
            for t in self.data[robot]['t_cmd']
        })

        if not event_times:
            return None

        latest_cmd_idx = {robot: -1 for robot in self.robots}
        latest_v = {robot: None for robot in self.robots}
        latest_w = {robot: None for robot in self.robots}

        stopped_since = None
        trim_end_time = None

        def robot_stopped(robot):
            v = latest_v[robot]
            w = latest_w[robot]
            if v is None or w is None:
                return False
            return (
                abs(v) <= self.stop_linear_eps and
                abs(w) <= self.stop_angular_eps
            )

        for t in event_times:
            for robot in self.robots:
                times = self.data[robot]['t_cmd']
                next_idx = latest_cmd_idx[robot] + 1

                while next_idx < len(times) and times[next_idx] <= t:
                    latest_cmd_idx[robot] = next_idx
                    latest_v[robot] = self.data[robot]['v_cmd'][next_idx]
                    latest_w[robot] = self.data[robot]['w_cmd'][next_idx]
                    next_idx += 1

            all_known = all(latest_v[r] is not None and latest_w[r] is not None for r in self.robots)
            all_stopped = all_known and all(robot_stopped(r) for r in self.robots)

            if all_stopped:
                if stopped_since is None:
                    stopped_since = t
                elif (t - stopped_since) >= self.stop_hold_time:
                    trim_end_time = stopped_since
                    break
            else:
                stopped_since = None

        return trim_end_time

    def trim_series(self, times, values, end_time):
        if end_time is None:
            return times, values

        trimmed_t = []
        trimmed_v = []

        for t, v in zip(times, values):
            if t <= end_time:
                trimmed_t.append(t)
                trimmed_v.append(v)
            else:
                break

        return trimmed_t, trimmed_v

    def trim_xy_series(self, times, xs, ys, end_time):
        if end_time is None:
            return xs, ys

        trimmed_x = []
        trimmed_y = []

        for t, x, y in zip(times, xs, ys):
            if t <= end_time:
                trimmed_x.append(x)
                trimmed_y.append(y)
            else:
                break

        return trimmed_x, trimmed_y

    # ---------------------------------------------------------
    # Saving
    # ---------------------------------------------------------
    def save_velocity_profiles(self, trim_end_time):
        n = len(self.robots)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for i, robot in enumerate(self.robots):
            ax = axes[i]

            t_full = self.data[robot]['t_cmd']
            v_full = self.data[robot]['v_cmd']
            w_full = self.data[robot]['w_cmd']

            t_v, v = self.trim_series(t_full, v_full, trim_end_time)
            t_w, w = self.trim_series(t_full, w_full, trim_end_time)

            if len(t_v) > 0:
                ax.plot(t_v, v, label='linear.x')
                ax.plot(t_w, w, label='angular.z')

            ax.set_title(robot)
            ax.set_ylabel('Velocity')
            ax.grid(True)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel('Time [s]')
        fig.suptitle('UGV Command Velocities', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(self.output_dir, 'velocities_all_ugvs.png')
        fig.savefig(out_path, dpi=self.figure_dpi)
        plt.close(fig)

        self.get_logger().info(f'Saved velocity plot: {out_path}')

    def save_trajectory_map(self, trim_end_time):
        plt.figure(figsize=(9, 9))

        for robot in self.robots:
            t_odom = self.data[robot]['t_odom']
            xs_full = self.data[robot]['x']
            ys_full = self.data[robot]['y']

            xs, ys = self.trim_xy_series(t_odom, xs_full, ys_full, trim_end_time)

            if len(xs) == 0:
                continue

            plt.plot(xs, ys, label=robot)
            plt.scatter(xs[0], ys[0], marker='o', s=35)
            plt.scatter(xs[-1], ys[-1], marker='x', s=50)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('UGV Trajectories')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, 'trajectories_all_ugvs.png')
        plt.savefig(out_path, dpi=self.figure_dpi)
        plt.close()

        self.get_logger().info(f'Saved trajectory plot: {out_path}')

    def save_all_once(self):
        if self.saved_once:
            return
        self.saved_once = True

        for robot in self.robots:
            self.get_logger().info(
                f'{robot}: cmd samples={len(self.data[robot]["t_cmd"])}, '
                f'odom samples={len(self.data[robot]["t_odom"])}'
            )

        trim_end_time = self.get_trim_end_time()

        if trim_end_time is None:
            self.get_logger().info('No trailing all-stopped interval detected. Using full data.')
        else:
            self.get_logger().info(
                f'Trimming trailing zero-motion interval. Plot end time = {trim_end_time:.3f} s'
            )

        self.save_velocity_profiles(trim_end_time)
        self.save_trajectory_map(trim_end_time)

    def destroy_node(self):
        try:
            self.get_logger().info('Shutting down logger, saving final plots...')
            self.save_all_once()
        except Exception as e:
            self.get_logger().error(f'Failed to save plots on shutdown: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SwarmPlotLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()