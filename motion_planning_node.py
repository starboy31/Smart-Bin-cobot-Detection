#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState


class SimpleRGBDMotion(Node):
    def __init__(self):
        super().__init__("simple_rgbd_motion")

        self.callback_group = ReentrantCallbackGroup()

        self.base_frame = "base_link"
        self.group_name = "ur_onrobot_manipulator"
        self.ik_link_name = "gripper_tcp"

        self.latest_joint_state = None
        self.latest_object_msg = None
        self.motion_busy = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=self.callback_group
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self.callback_group
        )

        self.create_subscription(
            PointStamped,
            "/detected_object_point",
            self.object_callback,
            10,
            callback_group=self.callback_group
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        self.timer = self.create_timer(
            1.0,
            self.motion_timer_callback,
            callback_group=self.callback_group
        )

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.get_logger().info("=== RGBD MOTION NODE - ROBUST IK TEST ===")
        self.get_logger().info("Checking /compute_ik...")
        self.ik_client.wait_for_service()
        self.get_logger().info("OK: /compute_ik exists")

        self.get_logger().info("Checking trajectory controller...")
        self.trajectory_client.wait_for_server()
        self.get_logger().info("OK: trajectory controller exists")

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    def object_callback(self, msg):
        if self.motion_busy:
            return
        self.latest_object_msg = msg

    def motion_timer_callback(self):
        if self.motion_busy:
            return

        if self.latest_object_msg is None:
            return

        if self.latest_joint_state is None:
            self.get_logger().warn("No /joint_states yet.")
            return

        msg = self.latest_object_msg
        self.latest_object_msg = None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )

            point_base = do_transform_point(msg, transform)

            raw_x = point_base.point.x
            raw_y = point_base.point.y
            raw_z = point_base.point.z

            # Use camera X/Y but keep it inside reachable UR3e zone
            x = max(0.25, min(raw_x, 0.50))
            y = max(-0.20, min(raw_y, 0.20))

            self.get_logger().info(
                f"Raw base XYZ: x={raw_x:.3f}, y={raw_y:.3f}, z={raw_z:.3f} | "
                f"Using XY: x={x:.3f}, y={y:.3f}"
            )

        except Exception as e:
            self.get_logger().error(f"TF failed: {e}")
            return

        self.motion_busy = True

        success = self.try_move_with_candidates(x, y)

        if success:
            self.get_logger().info("Motion complete.")
        else:
            self.get_logger().error("No IK solution found for candidate poses.")

        self.motion_busy = False

    def try_move_with_candidates(self, x, y):
        # Try easier higher poses first
        z_candidates = [0.60, 0.55, 0.50, 0.45, 0.40]

        # Valid normalized quaternions
        orientation_candidates = [
            # neutral, easiest IK
            (0.0, 0.0, 0.0, 1.0),

            # gripper-down-ish
            (-0.707, 0.0, 0.0, 0.707),

            # alternative wrist orientations
            (0.707, 0.0, 0.0, 0.707),
            (0.0, 0.707, 0.0, 0.707),
            (0.0, -0.707, 0.0, 0.707),
        ]

        for z in z_candidates:
            for quat in orientation_candidates:
                self.get_logger().info(
                    f"Trying IK: x={x:.3f}, y={y:.3f}, z={z:.3f}, quat={quat}"
                )

                positions = self.solve_ik(x, y, z, quat)

                if positions is not None:
                    self.get_logger().info("IK SUCCESS. Sending trajectory.")
                    self.send_trajectory(positions)
                    return True

        return False

    def solve_ik(self, x, y, z, quat):
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 2
        req.ik_request.avoid_collisions = False

        robot_state = RobotState()
        robot_state.joint_state = self.latest_joint_state
        req.ik_request.robot_state = robot_state

        future = self.ik_client.call_async(req)

        start_time = self.get_clock().now()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > 3.0:
                self.get_logger().warn("IK timeout for this candidate.")
                return None

        result = future.result()

        if result is None:
            return None

        if result.error_code.val != 1:
            self.get_logger().warn(f"IK failed code: {result.error_code.val}")
            return None

        js = result.solution.joint_state

        positions = []
        for joint in self.joint_names:
            if joint not in js.name:
                self.get_logger().error(f"Joint missing in IK result: {joint}")
                return None

            idx = js.name.index(joint)
            positions.append(js.position[idx])

        return positions

    def send_trajectory(self, positions):
        goal = FollowJointTrajectory.Goal()

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 4

        traj.points.append(point)
        goal.trajectory = traj

        send_future = self.trajectory_client.send_goal_async(goal)

        start_time = self.get_clock().now()
        while rclpy.ok() and not send_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > 6.0:
                self.get_logger().error("Trajectory goal timeout")
                return

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected.")
            return

        result_future = goal_handle.get_result_async()

        start_time = self.get_clock().now()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > 10.0:
                self.get_logger().error("Trajectory execution timeout")
                return

        self.get_logger().info("Trajectory executed.")


def main(args=None):
    rclpy.init(args=args)

    node = SimpleRGBDMotion()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
