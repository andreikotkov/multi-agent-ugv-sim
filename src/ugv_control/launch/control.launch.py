import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def load_params(config_path, node_key):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    if node_key not in data:
        raise RuntimeError(
            f"Missing '{node_key}' section in YAML file: {config_path}"
        )

    if 'ros__parameters' not in data[node_key]:
        raise RuntimeError(
            f"Missing 'ros__parameters' under '{node_key}' in YAML file: {config_path}"
        )

    return dict(data[node_key]['ros__parameters'])


def generate_launch_description():
    pkg_share = get_package_share_directory('ugv_control')
    config_file = os.path.join(pkg_share, 'config', 'swarm_params.yaml')

    rviz_config = os.path.join(
        get_package_share_directory('ugv_description'),
        'rviz',
        'swarm_topdown.rviz'
    )

    robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']

    single_agent_params = load_params(config_file, 'single_agent_node')
    formation_manager_params = load_params(config_file, 'formation_mode_manager')
    visualizer_params = load_params(config_file, 'swarm_visualizer')
    spawner_params = load_params(config_file, 'delayed_gazebo_obstacle_spawner')

    ld = LaunchDescription()

    ld.add_action(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    ))

    ld.add_action(Node(
        package='ugv_control',
        executable='global_obstacle_publisher',
        name='global_obstacle_publisher',
        output='screen',
        parameters=[
            spawner_params,
            {'use_sim_time': True}
        ]
    ))

    ld.add_action(Node(
        package='ugv_control',
        executable='formation_mode_manager',
        name='formation_mode_manager',
        output='screen',
        parameters=[
            formation_manager_params,
            {'use_sim_time': True}
        ]
    ))

    ld.add_action(Node(
        package='ugv_control',
        executable='delayed_gazebo_obstacle_spawner',
        name='delayed_gazebo_obstacle_spawner',
        output='screen',
        parameters=[
            spawner_params,
            {'use_sim_time': True}
        ]
    ))

    for robot_name in robots:
        controller_params = dict(single_agent_params)
        controller_params['robot_name'] = robot_name

        ld.add_action(Node(
            package='ugv_control',
            executable='single_agent_node',
            name=f'{robot_name}_controller',
            output='screen',
            parameters=[
                controller_params,
                {'use_sim_time': True}
            ]
        ))

    ld.add_action(Node(
        package='ugv_control',
        executable='swarm_visualizer',
        name='swarm_visualizer',
        output='screen',
        parameters=[
            visualizer_params,
            {'use_sim_time': True}
        ]
    ))


    ld.add_action(Node(
        package='ugv_control',
        executable='swarm_plot_logger',
        name='swarm_plot_logger',
        output='screen',
        parameters=[{
            'robots': robots,
            'output_root': os.getcwd(),
            'figure_dpi': 180,
            'stop_linear_eps': 0.02,
            'stop_angular_eps': 0.03,
            'stop_hold_time': 1.0,
            'use_sim_time': True
        }]
    ))

    

    return ld