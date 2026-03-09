# Multi-Agent UGV Simulation

A decentralized Artificial Potential Field (APF) swarm simulation using 4 Unicycle UGVs in ROS 2 Foxy and Gazebo.

## Prerequisites
* ROS 2 Foxy
* Gazebo 11 (`ros-foxy-gazebo-ros-pkgs`)
* Xacro (`ros-foxy-xacro`)

## How to Build
```bash
# Clone the repo into a workspace src folder, then run:
colcon build --symlink-install
source install/setup.bash
