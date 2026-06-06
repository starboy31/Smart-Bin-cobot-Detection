#!/usr/bin/env python3

import csv
import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import tf2_ros


class TwoStepCameraRobotRecorder(Node):
    def __init__(self):
        super().__init__('two_step_camera_robot_recorder')

        self.latest_camera_point = None
        self.saved_camera_point = None
        self.point_number = 1

        self.camera_topic = '/aruco_position_camera'
        self.robot_base_frame = 'base_link'
        self.robot_tcp_frame = 'tool0'

        self.output_file = os.path.expanduser(
            '~/ros2_ws/camera_robot_points.csv'
        )

        self.create_subscription(
            PointStamped,
            self.camera_topic,
            self.camera_callback,
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'point_id',
                    'cam_x', 'cam_y', 'cam_z',
                    'base_x', 'base_y', 'base_z'
                ])

        self.get_logger().info('Two-step camera-robot recorder started.')
        self.get_logger().info(f'Camera topic: {self.camera_topic}')
        self.get_logger().info(f'Robot TF: {self.robot_base_frame} -> {self.robot_tcp_frame}')
        self.get_logger().info(f'CSV file: {self.output_file}')

        self.timer = self.create_timer(0.2, self.start_keyboard_loop)

    def camera_callback(self, msg):
        self.latest_camera_point = msg.point

    def start_keyboard_loop(self):
        self.timer.cancel()

        while rclpy.ok():
            print('\n========================================')
            print(f'POINT {self.point_number}')
            print('Step 1: Place ArUco marker where you want.')
            print('Make sure camera detects it clearly.')
            input('Press ENTER to FREEZE camera ArUco XYZ... ')

            if self.latest_camera_point is None:
                self.get_logger().warn(
                    f'No camera point received yet from {self.camera_topic}'
                )
                continue

            self.saved_camera_point = (
                self.latest_camera_point.x,
                self.latest_camera_point.y,
                self.latest_camera_point.z
            )

            print(
                f'Frozen CAM XYZ: '
                f'({self.saved_camera_point[0]:.4f}, '
                f'{self.saved_camera_point[1]:.4f}, '
                f'{self.saved_camera_point[2]:.4f})'
            )

            print('\nStep 2: Now move robot TCP/tool0 to the exact centre of that ArUco marker.')
            input('When TCP is exactly at marker centre, press ENTER to SAVE robot TCP XYZ... ')

            self.save_pair()

    def save_pair(self):
        if self.saved_camera_point is None:
            self.get_logger().warn('No frozen camera point available.')
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.robot_tcp_frame,
                rclpy.time.Time()
            )

            base_x = tf.transform.translation.x
            base_y = tf.transform.translation.y
            base_z = tf.transform.translation.z

            cam_x, cam_y, cam_z = self.saved_camera_point

            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.point_number,
                    cam_x, cam_y, cam_z,
                    base_x, base_y, base_z
                ])

            self.get_logger().info(
                f'SAVED POINT {self.point_number}: '
                f'CAM=({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f})  '
                f'BASE=({base_x:.4f}, {base_y:.4f}, {base_z:.4f})'
            )

            self.point_number += 1
            self.saved_camera_point = None

        except Exception as e:
            self.get_logger().error(f'Could not read robot TF: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TwoStepCameraRobotRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
