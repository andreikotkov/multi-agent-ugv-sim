from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    shared_goal_x = 10.0
    shared_goal_y = 10.0
    robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']

    ld.add_action(Node(
        package='ugv_control',
        executable='global_obstacle_publisher',
        name='global_obstacle_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    ))

    ld.add_action(Node(
        package='ugv_control',
        executable='formation_mode_manager',
        name='formation_mode_manager',
        output='screen',
        parameters=[
            {'vl_goal_x': shared_goal_x},
            {'vl_goal_y': shared_goal_y},
            {'use_sim_time': True}
        ]
    ))

    ld.add_action(Node(
        package='ugv_control',
        executable='delayed_gazebo_obstacle_spawner',
        name='delayed_gazebo_obstacle_spawner',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    ))

    for robot_name in robots:
        ld.add_action(Node(
            package='ugv_control',
            executable='single_agent_node',
            name=f'{robot_name}_controller',
            parameters=[
                {'robot_name': robot_name},
                {'vl_goal_x': shared_goal_x},
                {'vl_goal_y': shared_goal_y},
                {'use_sim_time': True}
            ],
            output='screen'
        ))

    ld.add_action(Node(
        package='ugv_control',
        executable='swarm_visualizer',
        name='swarm_visualizer',
        parameters=[
            {'mode_source_robot': 'ugv1'},
            {'use_sim_time': True}
        ],
        output='screen'
    ))

    return ld