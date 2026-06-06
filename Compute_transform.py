#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


CSV_PATH = "camera_robot_points.csv"

CAMERA_CHILD_FRAME = "camera_camera_link"
ROBOT_BASE_FRAME = "base_link"


def compute_rigid_transform(camera_points, robot_points):
    cam_centroid = np.mean(camera_points, axis=0)
    robot_centroid = np.mean(robot_points, axis=0)

    cam_centered = camera_points - cam_centroid
    robot_centered = robot_points - robot_centroid

    H = cam_centered.T @ robot_centered

    U, S, Vt = np.linalg.svd(H)

    rotation_matrix = Vt.T @ U.T

    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = Vt.T @ U.T

    translation = robot_centroid - rotation_matrix @ cam_centroid

    return rotation_matrix, translation


def main():
    data = pd.read_csv(CSV_PATH)

    camera_points = data[["cam_x", "cam_y", "cam_z"]].to_numpy(dtype=float)
    robot_points = data[["base_x", "base_y", "base_z"]].to_numpy(dtype=float)

    if len(camera_points) < 4:
        raise ValueError("You need at least 4 point pairs. 8 or more is better.")

    rotation_matrix, translation = compute_rigid_transform(
        camera_points,
        robot_points
    )

    quat = R.from_matrix(rotation_matrix).as_quat()
    qx, qy, qz, qw = quat

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation

    print("\n==============================")
    print("T_base_camera")
    print("==============================")
    print(transform_matrix)

    print("\nTranslation:")
    print(f"x  = {translation[0]:.6f}")
    print(f"y  = {translation[1]:.6f}")
    print(f"z  = {translation[2]:.6f}")

    print("\nQuaternion:")
    print(f"qx = {qx:.6f}")
    print(f"qy = {qy:.6f}")
    print(f"qz = {qz:.6f}")
    print(f"qw = {qw:.6f}")

    print("\n==============================")
    print("Static Transform Publisher Command")
    print("==============================")

    print(f"""
ros2 run tf2_ros static_transform_publisher \\
    --x {translation[0]:.6f} \\
    --y {translation[1]:.6f} \\
    --z {translation[2]:.6f} \\
    --qx {qx:.6f} \\
    --qy {qy:.6f} \\
    --qz {qz:.6f} \\
    --qw {qw:.6f} \\
    --frame-id {ROBOT_BASE_FRAME} \\
    --child-frame-id {CAMERA_CHILD_FRAME}
""")

    print("\n==============================")
    print("Calibration Error Check")
    print("==============================")

    errors = []

    for i, cam_point in enumerate(camera_points):
        predicted_robot_point = rotation_matrix @ cam_point + translation
        actual_robot_point = robot_points[i]

        error = np.linalg.norm(predicted_robot_point - actual_robot_point)
        errors.append(error)

        print(
            f"Point {i+1}: "
            f"error = {error * 1000:.2f} mm | "
            f"predicted = {predicted_robot_point} | "
            f"actual = {actual_robot_point}"
        )

    print("\nMean error:", f"{np.mean(errors) * 1000:.2f} mm")
    print("Max error: ", f"{np.max(errors) * 1000:.2f} mm")


if __name__ == "__main__":
    main()
