from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # goal for all 4 UGVs to meet at
    shared_goal_x = 5.0
    shared_goal_y = 5.0

    robots = ['ugv1', 'ugv2', 'ugv3', 'ugv4']

    for robot in robots:
        node = Node(
            package='ugv_control',
            executable='single_agent_node',
            name=f'control_{robot}',
            output='screen',
            parameters=[{
                'robot_name': robot,
                'goal_x': shared_goal_x,
                'goal_y': shared_goal_y
            }]
        )
        ld.add_action(node)

    return ld