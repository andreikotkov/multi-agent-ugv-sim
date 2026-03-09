import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    pkg_ugv_description = get_package_share_directory('ugv_description')
    xacro_file = os.path.join(pkg_ugv_description, 'urdf', 'ugv.urdf.xacro')

    robot_name = LaunchConfiguration('robot_name')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    z_pose = LaunchConfiguration('z_pose')
    yaw_pose = LaunchConfiguration('yaw_pose')

    # Foxy handles Command directly without the ParameterValue wrapper
    robot_description_content = Command(['xacro ', xacro_file, ' robot_name:=', robot_name])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=robot_name,
        output='screen',
        parameters=[{
            'robot_description': robot_description_content, 
            'use_sim_time': True 
        }]
    )

    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        namespace=robot_name,
        output='screen',
        arguments=[
            '-entity', robot_name,
            '-topic', 'robot_description',
            '-x', x_pose,
            '-y', y_pose,
            '-z', z_pose,
            '-Y', yaw_pose
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='ugv1'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('z_pose', default_value='0.1'),
        DeclareLaunchArgument('yaw_pose', default_value='0.0'),
        robot_state_publisher_node,
        spawn_entity_node
    ])