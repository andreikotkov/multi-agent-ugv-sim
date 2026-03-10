import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # configuration
    shared_goal_x = 5.0
    shared_goal_y = 5.0
    robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']

    # add 4 individual robot controllers
    for robot_name in robots:
        node = Node(
            package='ugv_control',
            executable='single_agent_node',
            name=f'{robot_name}_controller',
            parameters=[
                {'robot_name': robot_name},
                {'goal_x': shared_goal_x},
                {'goal_y': shared_goal_y}
            ],
            output='screen'
        )
        ld.add_action(node)

    # warm visualizer (plots)
    visualizer_node = Node(
        package='ugv_control',
        executable='swarm_visualizer',
        name='swarm_visualizer',
        output='screen'
    )
    ld.add_action(visualizer_node)

    return ld