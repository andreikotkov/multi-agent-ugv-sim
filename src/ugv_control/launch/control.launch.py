import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # configuration
    shared_goal_x = 10.0
    shared_goal_y = 10.0
    robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']

    # global obstacle publisher
    obstacle_publisher_node = Node(
        package='ugv_control',
        executable='global_obstacle_publisher',
        name='global_obstacle_publisher',
        output='screen'
    )
    ld.add_action(obstacle_publisher_node)

    # add 4 individual robot controllers
    for robot_name in robots:
        node = Node(
            package='ugv_control',
            executable='single_agent_node',
            name=f'{robot_name}_controller',
            parameters=[
                {'robot_name': robot_name},
                {'vl_goal_x': shared_goal_x},
                {'vl_goal_y': shared_goal_y}
            ],
            output='screen'
        )
        ld.add_action(node)

    # visualizer
    visualizer_node = Node(
        package='ugv_control',
        executable='swarm_visualizer',
        name='swarm_visualizer',
        parameters=[
            {'mode_source_robot': 'ugv1'}
        ],
        output='screen'
    )
    ld.add_action(visualizer_node)

    return ld