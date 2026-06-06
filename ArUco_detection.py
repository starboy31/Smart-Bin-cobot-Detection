#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped

from tf2_ros import Buffer, TransformListener


class ArucoDetectionNode(Node):

    def __init__(self):
        super().__init__('aruco_detection_node')

        self.bridge = CvBridge()

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('color_topic',       '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('aruco_dictionary',  'DICT_4X4_50')
        self.declare_parameter('base_frame',        'base_link')

        self.color_topic       = self.get_parameter('color_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.base_frame        = self.get_parameter('base_frame').value

        # Intrinsics — filled in from CameraInfo
        self.fx = self.fy = self.cx = self.cy = None

        # Frame name — updated from CameraInfo header so it always matches
        # what the driver actually publishes (avoids hardcoded mismatches)
        self.camera_frame = 'camera_color_optical_frame'

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── ArUco dictionary — version-agnostic ───────────────────────────────
        aruco_dict_map = {
            'DICT_4X4_50':  cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_5X5_50':  cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_6X6_50':  cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        }
        dict_id = aruco_dict_map.get(
            self.get_parameter('aruco_dictionary').value,
            cv2.aruco.DICT_4X4_50
        )

        # Support OpenCV 4.5.x (Humble apt), 4.7.x+, and legacy builds
        try:
            if hasattr(cv2.aruco, 'DetectorParameters_create'):
                # OpenCV 4.5.x
                self.aruco_dict   = cv2.aruco.getPredefinedDictionary(dict_id)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
            elif hasattr(cv2.aruco, 'DetectorParameters'):
                # OpenCV 4.7.x+
                self.aruco_dict   = cv2.aruco.getPredefinedDictionary(dict_id)
                self.aruco_params = cv2.aruco.DetectorParameters()
            else:
                # Older legacy builds
                self.aruco_dict   = cv2.aruco.Dictionary_get(dict_id)
                self.aruco_params = cv2.aruco.DetectorParameters_create()
        except Exception as e:
            self.get_logger().error(f'ArUco init error: {e}')
            raise

        # ── Publishers ────────────────────────────────────────────────────────
        self.image_pub      = self.create_publisher(Image,        '/aruco_pose_image',    10)
        self.id_pub         = self.create_publisher(Int32,        '/detected_aruco_id',   10)
        self.base_point_pub = self.create_publisher(PointStamped, '/base_link_in_camera', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image,      self.color_topic,       self.color_callback,      10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info('ArUco detection node started')
        self.get_logger().info(f'  colour topic : {self.color_topic}')
        self.get_logger().info(f'  camera info  : {self.camera_info_topic}')
        self.get_logger().info(f'  base frame   : {self.base_frame}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        # Use whatever frame the driver actually reports — avoids name mismatches
        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def color_callback(self, msg: Image):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Colour conversion error: {e}')
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        try:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        except Exception as e:
            self.get_logger().error(f'ArUco detection error: {e}')
            corners, ids = [], None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i][0]
                center_u = int(np.mean(c[:, 0]))
                center_v = int(np.mean(c[:, 1]))

                self.get_logger().info(
                    f'ArUco ID {marker_id} at pixel u={center_u}, v={center_v}'
                )

                id_msg      = Int32()
                id_msg.data = int(marker_id)
                self.id_pub.publish(id_msg)

                cv2.circle(color_image, (center_u, center_v), 5, (0, 0, 255), -1)
                cv2.putText(
                    color_image, f'ID {marker_id}',
                    (center_u - 40, center_v - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        # Overlay base_link origin projected into camera view
        self._draw_base_link_projection(color_image)

        out_msg        = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_base_link_projection(self, image):
        """
        Looks up base_link in the camera frame and projects it onto the image.
        This is purely a visual sanity check — a blue dot shows where the robot
        origin sits relative to the camera. If the dot is in roughly the right
        place, the TF chain is correct.
        """
        if self.fx is None:
            cv2.putText(image, 'Waiting for camera_info...',
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return

        try:
            # FIXED: look up base_link expressed IN the camera frame
            # i.e. "where is base_link as seen from the camera?"
            # Direction: target=camera_frame, source=base_frame
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,   # target frame  (we want coords in this frame)
                self.base_frame,     # source frame  (the frame we're locating)
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            # Publish the 3-D position for external use / debugging
            pt            = PointStamped()
            pt.header.stamp    = self.get_clock().now().to_msg()
            pt.header.frame_id = self.camera_frame
            pt.point.x, pt.point.y, pt.point.z = float(x), float(y), float(z)
            self.base_point_pub.publish(pt)

            # Overlay text
            cv2.putText(
                image,
                f'base_link in cam: X={x:.3f} Y={y:.3f} Z={z:.3f}',
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

            # Project onto image plane (only meaningful if z > 0)
            if z > 0.01:
                u = int((x * self.fx / z) + self.cx)
                v = int((y * self.fy / z) + self.cy)
                h, w = image.shape[:2]

                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(image, (u, v), 8, (255, 0, 0), -1)
                    cv2.putText(image, 'base_link', (u + 10, v),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(image, 'base_link outside camera FOV',
                                (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 165, 255), 2)
            else:
                # z <= 0 means base_link is behind the camera — transform is wrong
                cv2.putText(image, 'WARNING: base_link behind camera (check TF)',
                            (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

        except Exception as e:
            # Show the missing link so it's obvious in rqt which frame is absent
            cv2.putText(
                image,
                f'TF missing: {self.base_frame} -> {self.camera_frame}',
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
