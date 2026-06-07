# Smart Bin Cobot – Perception and Mapping Subsystem

This repository contains the code developed for my contribution to the Smart Bin Cobot project in Robotics Studio 2. My main work focused on the perception and mapping subsystem, which allows the robot to detect waste objects, estimate their position, and provide coordinate information for motion planning.

The perception subsystem was developed to support the project MVP, where the robot detects at least one waste object using an RGB-D camera and attempts a simple pick-and-place task using the UR3e cobot.

## Project Overview

The Smart Bin Cobot is a robotic waste sorting system designed to detect waste objects in the workspace and sort them into the correct bin. The system combines perception, motion planning, robot control, and gripper operation.

My subsystem is responsible for:

* Capturing RGB-D camera data
* Detecting waste objects
* Classifying objects as recyclable, biodegradable, or general waste
* Estimating depth and 3D object position
* Detecting ArUco markers for coordinate testing and integration
* Supporting camera-to-robot transformation
* Providing coordinate outputs for motion planning

## Repository Contents

This repository includes the following main nodes:

### 1. Detection Node

The detection node uses a trained YOLOv8 model to detect waste objects such as tin cans, plastic bottles, gloves, and bananas. It outputs the detected class, confidence score, bounding box, depth value, and approximate object position.

### 2. ArUco Detection Node

The ArUco detection node detects ArUco markers in the camera image and estimates their position. This was used during integration testing to simulate object and bin locations and to help the robot move toward detected marker coordinates.

### 3. Gamma Correction Node

The gamma correction node improves image visibility under changing lighting conditions. It was developed to support more stable object detection by adjusting image brightness before detection.

### 4. Camera Calibration Node

The calibration node was used to calibrate the Intel RealSense D435i camera using a chessboard pattern. The calibration produced camera intrinsic parameters used for depth and 3D position estimation.

### 5. Camera-to-Robot Transformation Node

This node was used to record corresponding camera points and robot/gripper points. These points were used to support transformation between the camera frame and the robot frame.

### 6. Motion Planning Node

This node was developed to support integration between the perception subsystem and motion planning. It allowed the robot to move toward detected ArUco marker positions, demonstrating how perception outputs could be used for robot movement.

## Hardware Used

* UR3e robotic arm
* OnRobot RG2 gripper
* Intel RealSense D435i RGB-D camera
* Tripod-mounted camera setup
* ArUco markers
* Waste objects such as tin can, plastic bottle, gloves, and banana

## Software Used

* ROS 2 Humble
* Python
* OpenCV
* YOLOv8
* RealSense camera package
* MoveIt 2
* RViz
* Linux/Ubuntu

## Camera Calibration Results

The Intel RealSense D435i camera was calibrated using a printed chessboard pattern. The main intrinsic values obtained were:

```text
fx = 883.107
fy = 870.577
cx = 697.215
cy = 384.883
Mean reprojection error = 0.16898
```

These values were used to support object depth estimation and 3D coordinate calculation.

## How to Build

Place the package inside the ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone <your-github-repository-link>
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## How to Run

Start the RealSense camera:

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

Run the detection node:

```bash
ros2 run <package_name> detection_node
```

Run the ArUco detection node:

```bash
ros2 run <package_name> aruco_detection_node
```

Run the gamma correction node:

```bash
ros2 run <package_name> gamma_correction_node
```

Run the camera-to-robot transformation node:

```bash
ros2 run <package_name> camera_to_robot_transformation_node
```

Run the motion planning node:

```bash
ros2 run <package_name> motion_planning_node
```

Replace `<package_name>` with the actual ROS 2 package name used in the workspace.

## Main ROS Topics

The subsystem uses RGB and depth images from the RealSense camera:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
```

The detection output is published/displayed through:

```text
/detection_image
```

The subsystem also outputs object class, confidence, depth, and estimated position information for use by the motion planning subsystem.

## Testing Completed

The following tests were carried out during development:

### Object Detection and Confidence Test

A tin can was placed in the workspace and detected using the YOLOv8 detection model. The subsystem passed when the object was correctly detected in at least 2 out of 3 trials with confidence above the required threshold.

### Depth Accuracy Test

The depth value from the RGB-D camera was compared against manually measured object distance. Feedback showed that some measurement inaccuracy may have come from the test setup, so this test highlighted the need for better controlled measurement methods.

### Object 3D Position Estimation Test

The subsystem estimated the object’s x, y, and z coordinates using RGB and aligned depth images. The test passed when valid coordinates were produced in at least 2 out of 3 trials and changed consistently when the object was moved.

### ArUco-Based Integration Test

ArUco markers were used to simulate object and bin positions. The robot attempted to move toward detected marker coordinates, showing early integration between perception and motion planning.

## Project Status

By the final project handover, the perception subsystem was able to detect waste objects, classify them, estimate depth, and output coordinate information. The ArUco detection and motion planning integration also demonstrated that the robot could respond to detected target positions.

Further improvement is still required for more accurate camera-to-robot transformation, motion planning precision, and final grasp execution.

## Limitations and Future Improvements

Future improvements include:

* Improving dataset balance for all object classes
* Increasing banana detection reliability
* Improving camera-to-robot calibration accuracy
* Testing with more object positions and lighting conditions
* Improving motion planning accuracy during pick-and-place
* Adding clearer troubleshooting and launch files for first-time users

## Author

Rohit Kumar Sasikumar
Perception and Mapping Subsystem
Robotics Studio 2 – Smart Bin Cobot Project
