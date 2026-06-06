#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float64MultiArray

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


class WasteDetectionNode(Node):

    def __init__(self):
        super().__init__("waste_detection_node")

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("confidence_threshold", 0.70)
        self.declare_parameter("depth_patch_size", 11)

        self.declare_parameter(
            "model_path",
            "/home/rohit/ros2_ws/training_dataset/Waste Detection Objects.yolov8/runs/detect/train/weights/best.pt"
        )

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.model_path = self.get_parameter("model_path").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.depth_patch_size = int(self.get_parameter("depth_patch_size").value)

        self.bridge = CvBridge()
        self.latest_depth = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = "camera_color_optical_frame"

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info("YOLO model loaded.")

        self.create_subscription(Image, self.color_topic, self.color_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.image_pub = self.create_publisher(Image, "/detection_image", 10)
        self.camera_point_pub = self.create_publisher(PointStamped, "/detected_object_point", 10)
        self.base_point_pub = self.create_publisher(PointStamped, "/detected_object_base", 10)
        self.class_pub = self.create_publisher(String, "/detected_object_class", 10)
        self.data_pub = self.create_publisher(Float64MultiArray, "/detected_object_data", 10)

        self.get_logger().info("Dataset-only waste detection node started.")

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def classify_waste(self, class_name):
        name = class_name.lower().strip()

        recyclable = [
            "metal-can", "metal_can", "can", "tin-can", "tin_can",
            "plastic-bottle", "plastic_bottle", "bottle"
        ]

        biodegradable = [
            "banana", "bannana", "ripe", "unripe", "overripe",
            "over-ripe", "rotten"
        ]

        general = [
            "glove", "gloves"
        ]

        if name in recyclable:
            return "recyclable", "Recyclable waste"

        if name in biodegradable:
            return "biodegradable", "Biodegradable waste"

        if name in general:
            return "general", "General waste"

        return "general", "General waste"

    def color_callback(self, msg):
        if self.latest_depth is None:
            self.get_logger().warn("Waiting for depth image...", throttle_duration_sec=2.0)
            return

        if self.fx is None:
            self.get_logger().warn("Waiting for camera info...", throttle_duration_sec=2.0)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        display = frame.copy()
        h, w = frame.shape[:2]

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO detection failed: {e}")
            return

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                box_w = x2 - x1
                box_h = y2 - y1

                if box_w <= 0 or box_h <= 0:
                    continue

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                bin_type, category = self.classify_waste(class_name)

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bin_type": bin_type,
                    "category": category,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": center_x,
                    "cy": center_y,
                    "box_w": box_w,
                    "box_h": box_h,
                    "score": confidence * box_w * box_h,
                    "cam_xyz": None,
                    "base_xyz": None
                })

        if not detections:
            self.draw_text(display, "No object detected", (20, 35), (0, 0, 255))
            self.publish_image(display, msg.header)
            return

        detections.sort(key=lambda d: d["score"], reverse=True)

        tf_transform = self.lookup_tf()

        for det in detections:
            depth = self.get_depth(det["cx"], det["cy"])

            if depth is None:
                continue

            cam_x = (det["cx"] - self.cx) * depth / self.fx
            cam_y = (det["cy"] - self.cy) * depth / self.fy
            cam_z = depth

            det["cam_xyz"] = (cam_x, cam_y, cam_z)

            if tf_transform is not None:
                cam_point = PointStamped()
                cam_point.header.stamp = self.get_clock().now().to_msg()
                cam_point.header.frame_id = self.camera_frame
                cam_point.point.x = float(cam_x)
                cam_point.point.y = float(cam_y)
                cam_point.point.z = float(cam_z)

                try:
                    base_point = do_transform_point(cam_point, tf_transform)
                    det["base_xyz"] = (
                        base_point.point.x,
                        base_point.point.y,
                        base_point.point.z
                    )
                except Exception as e:
                    self.get_logger().warn(
                        f"TF transform failed: {e}",
                        throttle_duration_sec=2.0
                    )

        valid_detections = [d for d in detections if d["cam_xyz"] is not None]

        if valid_detections:
            best = valid_detections[0]
            self.publish_best_detection(best)

        for det in detections:
            self.draw_detection(display, det)

        self.draw_text(display, f"Objects detected: {len(detections)}", (10, 30), (255, 255, 255))

        self.publish_image(display, msg.header)

    def lookup_tf(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed: {self.camera_frame} -> {self.base_frame}: {e}",
                throttle_duration_sec=2.0
            )
            return None

    def get_depth(self, u, v):
        depth = self.latest_depth

        if depth is None:
            return None

        h, w = depth.shape[:2]

        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        half = self.depth_patch_size // 2

        x1 = max(0, u - half)
        x2 = min(w, u + half + 1)
        y1 = max(0, v - half)
        y2 = min(h, v + half + 1)

        patch = depth[y1:y2, x1:x2]

        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0]

        if len(valid) == 0:
            return None

        z = float(np.median(valid))

        if depth.dtype == np.uint16:
            z = z / 1000.0

        return z

    def publish_best_detection(self, best):
        cam_x, cam_y, cam_z = best["cam_xyz"]

        cam_msg = PointStamped()
        cam_msg.header.stamp = self.get_clock().now().to_msg()
        cam_msg.header.frame_id = self.camera_frame
        cam_msg.point.x = float(cam_x)
        cam_msg.point.y = float(cam_y)
        cam_msg.point.z = float(cam_z)
        self.camera_point_pub.publish(cam_msg)

        class_msg = String()
        class_msg.data = best["bin_type"]
        self.class_pub.publish(class_msg)

        if best["base_xyz"] is not None:
            bx, by, bz = best["base_xyz"]

            base_msg = PointStamped()
            base_msg.header.stamp = self.get_clock().now().to_msg()
            base_msg.header.frame_id = self.base_frame
            base_msg.point.x = float(bx)
            base_msg.point.y = float(by)
            base_msg.point.z = float(bz)
            self.base_point_pub.publish(base_msg)

            data_msg = Float64MultiArray()
            data_msg.data = [
                float(best["cx"]),
                float(best["cy"]),
                float(best["box_w"]),
                float(best["box_h"]),
                float(best["confidence"]),
                float(cam_x),
                float(cam_y),
                float(cam_z),
                float(bx),
                float(by),
                float(bz)
            ]
            self.data_pub.publish(data_msg)

            self.get_logger().info(
                f"{best['category']} | {best['class_name']} | "
                f"bin={best['bin_type']} | "
                f"conf={best['confidence']:.2f} | "
                f"CAM=({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}) | "
                f"BASE=({bx:.3f}, {by:.3f}, {bz:.3f})",
                throttle_duration_sec=1.0
            )

    def draw_detection(self, image, det):
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]

        if det["bin_type"] == "recyclable":
            colour = (0, 255, 0)
        elif det["bin_type"] == "biodegradable":
            colour = (0, 255, 255)
        else:
            colour = (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(image, (det["cx"], det["cy"]), 5, (255, 0, 0), -1)

        label = f"{det['category']} | {det['class_name']} | {det['confidence']:.2f}"
        y_text = y1 - 8

        if y_text < 30:
            y_text = y2 + 20

        y_text += self.draw_text(image, label, (x1, y_text), colour)

        if det["cam_xyz"] is not None:
            cam_x, cam_y, cam_z = det["cam_xyz"]
            y_text += self.draw_text(
                image,
                f"CAM: {cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f} m",
                (x1, y_text),
                (255, 255, 255)
            )

        if det["base_xyz"] is not None:
            bx, by, bz = det["base_xyz"]
            self.draw_text(
                image,
                f"BASE: {bx:.3f}, {by:.3f}, {bz:.3f} m",
                (x1, y_text),
                (255, 0, 0)
            )

    def draw_text(self, image, text, position, colour):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 2
        x, y = position

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        cv2.rectangle(
            image,
            (x - 2, y - text_h - 2),
            (x + text_w + 2, y + baseline),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            colour,
            thickness,
            cv2.LINE_AA
        )

        return text_h + baseline + 5

    def publish_image(self, image, header):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            img_msg.header = header
            img_msg.header.frame_id = self.camera_frame
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WasteDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
