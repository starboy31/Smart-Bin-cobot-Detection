#!/usr/bin/env python3

import cv2
import numpy as np
import glob
import yaml
import os


IMAGE_FOLDER = "/home/rohit/ros2_ws/src/my_smart_cobot/calib_images"

BOARD_WIDTH = 13
BOARD_HEIGHT = 9
SQUARE_SIZE = 0.020

OUTPUT_YAML = "/home/rohit/ros2_ws/src/my_smart_cobot/camera_calibration.yaml"


def main():
    pattern_size = (BOARD_WIDTH, BOARD_HEIGHT)

    objp = np.zeros((BOARD_WIDTH * BOARD_HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_WIDTH, 0:BOARD_HEIGHT].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
    )

    if len(image_paths) == 0:
        print("No images found.")
        print(f"Put calibration images inside: {IMAGE_FOLDER}")
        return

    print(f"Found {len(image_paths)} images.")

    image_size = None
    used_images = 0

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"[ERROR] Could not read: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = img.copy()

        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp.copy())
            imgpoints.append(corners2)
            used_images += 1

            cv2.drawChessboardCorners(display, pattern_size, corners2, found)
            print(f"[OK] Chessboard found: {path}")

        else:
            print(f"[FAILED] Chessboard not found: {path}")

        cv2.imshow("Calibration Image Check", display)
        cv2.waitKey(300)

    cv2.destroyAllWindows()

    if used_images < 10:
        print("Not enough valid chessboard images.")
        print("Use at least 15-20 good images.")
        return

    print("Running calibration...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    total_error = 0.0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)

    print("\n===== CALIBRATION COMPLETE =====")
    print(f"Valid images used: {used_images}/{len(image_paths)}")
    print(f"OpenCV ret value: {ret}")
    print(f"Mean reprojection error: {mean_error:.6f}")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs.ravel())

    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "board_width": BOARD_WIDTH,
        "board_height": BOARD_HEIGHT,
        "square_size_m": SQUARE_SIZE,
        "valid_images_used": used_images,
        "opencv_ret": float(ret),
        "mean_reprojection_error": float(mean_error),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist()
    }

    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(data, f)

    print(f"\nSaved calibration YAML to: {OUTPUT_YAML}")


if __name__ == "__main__":
    main()
