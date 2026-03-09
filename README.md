# ==========================================================
# ROS 2 MULTI-UGV LAPTOP SETUP SCRIPT
# ==========================================================

# 1. Install System Dependencies
# These are required for Gazebo and Xacro to work
sudo apt update
sudo apt install -y ros-foxy-xacro ros-foxy-gazebo-ros-pkgs ros-foxy-gazebo-ros

# 2. Clone the Repository
cd ~
git clone https://github.com/andreikotkov/multi-agent-ugv-sim.git multi_ugv_ws

# 3. Recreate Missing Directories
# Git ignores empty folders, so we must recreate them for CMake
mkdir -p ~/multi_ugv_ws/src/ugv_description/launch
mkdir -p ~/multi_ugv_ws/src/ugv_description/urdf
mkdir -p ~/multi_ugv_ws/src/ugv_description/meshes
mkdir -p ~/multi_ugv_ws/src/ugv_description/rviz
mkdir -p ~/multi_ugv_ws/src/ugv_gazebo/launch
mkdir -p ~/multi_ugv_ws/src/ugv_gazebo/worlds
mkdir -p ~/multi_ugv_ws/src/ugv_control/launch

# 4. Add Permanent Aliases to .bashrc
# This makes 'cb' build your workspace and 'sws' source it
echo "
# --- ROS 2 Multi-UGV Aliases ---
source /opt/ros/foxy/setup.bash
alias cb='cd ~/multi_ugv_ws && colcon build --symlink-install && source install/setup.bash && echo \"Workspace Built!\"'
alias sws='source ~/multi_ugv_ws/install/setup.bash && echo \"Workspace Sourced!\"'
" >> ~/.bashrc

# Reload bashrc to activate changes
source ~/.bashrc

# 5. First Build
# We use the new 'cb' alias we just created
cd ~/multi_ugv_ws
cb

# ==========================================================
# SETUP COMPLETE
# ==========================================================
# To run the simulation now:
# Terminal 1: ros2 launch ugv_gazebo multi_spawn.launch.py
# Terminal 2: ros2 launch ugv_control control.launch.py
