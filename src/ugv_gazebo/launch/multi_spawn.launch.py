import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_ugv_gazebo = get_package_share_directory('ugv_gazebo')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        )
    )

    spawn_ugv_launch_file = os.path.join(pkg_ugv_gazebo, 'launch', 'spawn_ugv.launch.py')

    
    robots = [
        {'name': 'ugv1', 'x': '0.0', 'y': '0.0'},
        {'name': 'ugv2', 'x': '0.0', 'y': '1.0'},
        {'name': 'ugv3', 'x': '0.0', 'y': '-1.0'},
        {'name': 'ugv4', 'x': '1.0', 'y': '0.0'},
    ]

    spawn_actions = []
    for robot in robots:
        spawn_actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(spawn_ugv_launch_file),
                
                launch_arguments=[
                    ('robot_name', robot['name']),
                    ('x_pose', robot['x']),
                    ('y_pose', robot['y']),
                    ('z_pose', '0.1'),
                    ('yaw_pose', '0.0')
                ]
            )
        )


# SPAWN THE FLAG 
    pkg_ugv_description = get_package_share_directory('ugv_description')
    goal_urdf = os.path.join(pkg_ugv_description, 'urdf', 'goal.urdf')

    spawn_goal_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_goal',
        arguments=[
            '-entity', 'target_goal',
            '-file', goal_urdf,
            '-x', '5.0',
            '-y', '5.0',
            '-z', '0.0'
        ],
        output='screen'
    )
    # ----------------------------------

    ld = LaunchDescription()
    ld.add_action(gazebo)
    ld.add_action(spawn_goal_node) 

    
    for action in spawn_actions:
        ld.add_action(action)

    return ld