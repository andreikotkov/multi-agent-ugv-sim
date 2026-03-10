import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import math
import time
import os
from datetime import datetime

class SwarmTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('swarm_visualizer')
        
        self.robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']
        self.d_safe = 0.3 
        
        
        self.traj_x = {name: [] for name in self.robots}
        self.traj_y = {name: [] for name in self.robots}
        self.current_pos = {name: [0.0, 0.0] for name in self.robots}
        self.current_force = {name: [0.0, 0.0] for name in self.robots}

        # Velocity history for post-simulation plotting
        self.vel_history_x = {name: [] for name in self.robots}
        self.vel_history_y = {name: [] for name in self.robots}
        self.time_history = {name: [] for name in self.robots}
        self.start_time = time.time()

        # subscribers
        for name in self.robots:
            self.create_subscription(Odometry, f'/{name}/odom', 
                lambda msg, n=name: self.odom_callback(msg, n), 10)
            self.create_subscription(Point, f'/{name}/force_vector', 
                lambda msg, n=name: self.force_callback(msg, n), 10)
            self.create_subscription(Twist, f'/{name}/cmd_vel',
                lambda msg, n=name: self.vel_callback(msg, n), 10)

        # Matplotlib Live Setup
        plt.ion()
        self.fig_traj, self.ax_traj = plt.subplots(figsize=(10, 10))
        self.ax_traj.plot(5.0, 5.0, 'rX', markersize=12, label='Goal')
        self.ax_traj.set_xlim(-1, 7); self.ax_traj.set_ylim(-1, 7)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True)
        self.ax_traj.set_title("Real-Time Swarm Trajectory & APF Forces")
        
        # Initialize Animated Elements for Blitting
        self.lines = {name: self.ax_traj.plot([], [], label=f'{name} Path', animated=True)[0] for name in self.robots}
        self.circles = {}
        self.quivers = {}

        for name in self.robots:
            color = self.lines[name].get_color()
            # Safety Circle
            circle = patches.Circle((0, 0), self.d_safe/2, color=color, alpha=0.50, animated=True)
            self.ax_traj.add_patch(circle)
            self.circles[name] = circle
            
            # Force Arrows (Black for visibility) - scale=5.0 for moderate length
            q = self.ax_traj.quiver(0, 0, 0, 0, color='black', scale=20.0, width=0.006, headwidth=4, animated=True)
            self.quivers[name] = q

        self.ax_traj.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Capture background for blitting
        self.fig_traj.canvas.draw()
        self.bg = self.fig_traj.canvas.copy_from_bbox(self.ax_traj.bbox)

        # Update timer (0.1s provides the best balance of smoothness and stability)
        self.timer = self.create_timer(0.1, self.update_plot)

    def odom_callback(self, msg, name):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        # Downsample: Only add point if it moved > 5cm to prevent lag
        if not self.traj_x[name] or math.hypot(x - self.traj_x[name][-1], y - self.traj_y[name][-1]) > 0.05:
            self.traj_x[name].append(x)
            self.traj_y[name].append(y)
        self.current_pos[name] = [x, y]

    def force_callback(self, msg, name):
        self.current_force[name] = [msg.x, msg.y]

    def vel_callback(self, msg, name):
        curr_t = time.time() - self.start_time
        self.time_history[name].append(curr_t)
        # We log the force vector components as the intended Cartesian velocities
        fx, fy = self.current_force[name]
        self.vel_history_x[name].append(fx)
        self.vel_history_y[name].append(fy)

    def update_plot(self):
        if self.bg is None: return
        
        # Restore the clean background
        self.fig_traj.canvas.restore_region(self.bg)

        for name in self.robots:
            if self.traj_x[name]:
                # Update trajectory path
                self.lines[name].set_data(self.traj_x[name], self.traj_y[name])
                
                # Update safety circle position
                cx, cy = self.current_pos[name]
                self.circles[name].center = (cx, cy)
                
                # Update force arrow direction and magnitude
                fx, fy = self.current_force[name]
                self.quivers[name].set_offsets([cx, cy])
                self.quivers[name].set_UVC(fx, fy)
                
                # Redraw artists manually for blitting performance
                self.ax_traj.draw_artist(self.lines[name])
                self.ax_traj.draw_artist(self.circles[name])
                self.ax_traj.draw_artist(self.quivers[name])
        
        # Blit the updated region
        self.fig_traj.canvas.blit(self.ax_traj.bbox)
        self.fig_traj.canvas.flush_events()

    def plot_final_results(self):
        """Saves trajectory and velocity analysis to a new timestamped folder"""
        plt.ioff()
        
        # Create Folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_name = f"test_run_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        print(f"\n[Visualizer] Folder created: {folder_name}")

        # 1. Save Final Trajectory Map
        traj_path = os.path.join(folder_name, 'trajectory_map.png')
        self.fig_traj.savefig(traj_path)
        print(f"[Visualizer] Trajectory map saved to {traj_path}")

        # 2. Generate and Save Velocity Analysis Profiles
        fig_vel, axs = plt.subplots(len(self.robots), 1, figsize=(10, 12), sharex=True)
        fig_vel.suptitle(f'Velocity Profiles - {timestamp}', fontsize=16)

        for i, name in enumerate(self.robots):
            axs[i].plot(self.time_history[name], self.vel_history_x[name], label='Vx', color='blue', alpha=0.8)
            axs[i].plot(self.time_history[name], self.vel_history_y[name], label='Vy', color='red', alpha=0.8)
            axs[i].set_ylabel(f'{name} (m/s)')
            axs[i].set_ylim(-1.5, 1.5)
            axs[i].grid(True)
            axs[i].legend(loc='upper right')

        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        vel_path = os.path.join(folder_name, 'velocity_profile.png')
        plt.savefig(vel_path)
        print(f"[Visualizer] Velocity profile saved to {vel_path}")
        
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