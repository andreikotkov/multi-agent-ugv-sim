import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

        # ---------------------------------------------------------
        # GHOST TRACKING VARIABLES
        # ---------------------------------------------------------
        self.ghost_traj_x = []
        self.ghost_traj_y = []
        self.vl_start_x = 0.0  
        self.vl_start_y = 0.0  
        self.vl_goal_x = 6.0
        self.vl_goal_y = 6.0
        self.vl_speed = 0.25 
        
        self.vel_history_x = {name: [] for name in self.robots}
        self.vel_history_y = {name: [] for name in self.robots}
        self.time_history = {name: [] for name in self.robots}
        
        # Sync timer with the controllers
        self.start_time = time.time()
        self.vl_start_time = self.start_time + 2.0 

        # Subscribers
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
        try:
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry("800x600+50+50")
        except Exception:
            pass 
            
        self.ax_traj.plot(6.0, 6.0, 'rX', markersize=12, label='Goal')
        
        # Static Obstacle
        obs_patch = patches.Rectangle((3.0, 3.0), 1.0, 1.0, color='gray', alpha=0.7, label='Obstacle')
        self.ax_traj.add_patch(obs_patch)

        self.ax_traj.set_xlim(-1, 8); self.ax_traj.set_ylim(-1, 8)
        self.ax_traj.set_aspect('equal')
        self.ax_traj.grid(True)
        self.ax_traj.set_title("Real-Time Swarm Trajectory & APF Forces")
        
        # Initialize Animated Elements for Blitting
        self.lines = {name: self.ax_traj.plot([], [], label=f'{name} Path', animated=True)[0] for name in self.robots}
        self.circles = {}
        self.quivers = {}
        
        # --- NEW: Initialize Ghost Graphics ---
        self.ghost_line, = self.ax_traj.plot([], [], '--', color='gray', alpha=0.5, linewidth=2, label='Virtual Leader', animated=True)
        self.ghost_marker = patches.Circle((0, 0), 0.15, color='gray', alpha=0.5, animated=True)
        self.ax_traj.add_patch(self.ghost_marker)

        for name in self.robots:
            color = self.lines[name].get_color()
            circle = patches.Circle((0, 0), self.d_safe/2, color=color, alpha=0.50, animated=True)
            self.ax_traj.add_patch(circle)
            self.circles[name] = circle
            
            q = self.ax_traj.quiver(0, 0, 0, 0, color='black', scale=20.0, width=0.006, headwidth=4, animated=True)
            self.quivers[name] = q

        self.ax_traj.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Capture background for blitting
        self.fig_traj.canvas.draw()
        self.bg = self.fig_traj.canvas.copy_from_bbox(self.ax_traj.bbox)

        self.timer = self.create_timer(0.1, self.update_plot)

    def odom_callback(self, msg, name):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if not self.traj_x[name] or math.hypot(x - self.traj_x[name][-1], y - self.traj_y[name][-1]) > 0.05:
            self.traj_x[name].append(x)
            self.traj_y[name].append(y)
        self.current_pos[name] = [x, y]

    def force_callback(self, msg, name):
        self.current_force[name] = [msg.x, msg.y]

    def vel_callback(self, msg, name):
        curr_t = time.time() - self.start_time
        self.time_history[name].append(curr_t)
        fx, fy = self.current_force[name]
        self.vel_history_x[name].append(fx)
        self.vel_history_y[name].append(fy)

    def update_plot(self):
        if self.bg is None: return
        self.fig_traj.canvas.restore_region(self.bg)

        # ---------------------------------------------------------
        # UPDATE GHOST POSITION
        # ---------------------------------------------------------
        t = time.time() - self.vl_start_time
        if t < 0: t = 0.0
        
        total_dist = math.hypot(self.vl_goal_x - self.vl_start_x, self.vl_goal_y - self.vl_start_y)
        travel = min(self.vl_speed * t, total_dist)
        
        vl_x = self.vl_start_x + (self.vl_goal_x - self.vl_start_x) * (travel / total_dist)
        vl_y = self.vl_start_y + (self.vl_goal_y - self.vl_start_y) * (travel / total_dist)

        # Record ghost path
        if not self.ghost_traj_x or math.hypot(vl_x - self.ghost_traj_x[-1], vl_y - self.ghost_traj_y[-1]) > 0.05:
            self.ghost_traj_x.append(vl_x)
            self.ghost_traj_y.append(vl_y)
            
        # Draw Ghost
        self.ghost_line.set_data(self.ghost_traj_x, self.ghost_traj_y)
        self.ghost_marker.center = (vl_x, vl_y)
        self.ax_traj.draw_artist(self.ghost_line)
        self.ax_traj.draw_artist(self.ghost_marker)

        # Draw Robots
        for name in self.robots:
            if self.traj_x[name]:
                self.lines[name].set_data(self.traj_x[name], self.traj_y[name])
                
                cx, cy = self.current_pos[name]
                self.circles[name].center = (cx, cy)
                
                fx, fy = self.current_force[name]
                self.quivers[name].set_offsets([cx, cy])
                self.quivers[name].set_UVC(fx, fy)
                
                self.ax_traj.draw_artist(self.lines[name])
                self.ax_traj.draw_artist(self.circles[name])
                self.ax_traj.draw_artist(self.quivers[name])
        
        self.fig_traj.canvas.blit(self.ax_traj.bbox)
        self.fig_traj.canvas.flush_events()

    def plot_final_results(self):
        plt.ioff()
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_name = f"test_run_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        traj_path = os.path.join(folder_name, 'trajectory_map.png')
        self.fig_traj.savefig(traj_path)
        print(f"[Visualizer] Trajectory map saved to {traj_path}")

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