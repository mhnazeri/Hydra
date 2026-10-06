import matplotlib.pyplot as plt

from scipy.special import comb
import time
import sys
import os
import threading
import queue
from collections import deque
import io

import rospy
import torch
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge
import cv2
import PIL
from torchvision import transforms
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from hydra.utilities.projection_utils import project_trajectory_on_image
from hydra.utilities.ros_utilities import ROSUtils

from omegaconf import OmegaConf
import safetensors.torch
from accelerate import Accelerator

from scipy.ndimage import uniform_filter1d

sys.path.append(os.getcwd())
from hydra.model.vertiformer import load_model as load_vertiformer
from path_follower_cfg import path_follower_policy
from hydra.utilities.helpers import get_conf


class VertiformerFollowerNode:
    def __init__(self, config_file, experiment_name, robot_name="jackal"):
        self.stop_event = threading.Event()
        self.publish_thread = None
        self.actions_lock = threading.Lock()
        self._new_actions = False
        self.video_queue = queue.Queue()
        self.video_thread = None
        self.experiment_video_writer = None
        self.experiment_name = experiment_name

        base_cfg = get_conf(config_file)
        robot_cfg_path = os.path.join(os.path.dirname(config_file), "robots", f"{robot_name}.yaml")
        robot_cfg = get_conf(robot_cfg_path)

        self.config_file = OmegaConf.merge(base_cfg, robot_cfg)
        self.sensor_config = self.config_file.ros

        architecture_yaml = [i for i in os.listdir(self.config_file.deployment.checkpoint_path) if i.endswith('.yaml')][
            0]
        self.model_yaml = OmegaConf.load(os.path.join(self.config_file.deployment.checkpoint_path, architecture_yaml))
        self.model_architecture = self.model_yaml

        img_size = self.model_yaml.model.encoder.image_size
        self.resize = (img_size[0], img_size[1])
        self.device = self.config_file.deployment.device

        self.goal_tolerance = self.config_file.deployment.goal_tolerance  # meters
        self.transform = transforms.Compose([
            transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.debug_visualize = self.config_file.deployment.get("debug_visualize", False)

        self.K = None

        self.latest_raw_img = None
        t_cam_base_cfg = self.config_file.platform.get("T_cam_base", None)
        if t_cam_base_cfg is not None:
            self.T_cam_base = np.array(t_cam_base_cfg, dtype=np.float32)
        else:
            self.T_cam_base = np.array([
                [2.2204460492503131e-16, -1.0, 0.0, 5.9999999999999984e-02],
                [-4.9979169270678137e-02, 0.0, -9.9875026039496617e-01, 0.75],
                [9.9875026039496617e-01, 2.2204460492503131e-16, -4.9979169270678137e-02, 1.0749687539060158e-02],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

        self.model = load_vertiformer(self.model_architecture, self.device).to(self.device)

        chkpt_dir = self.config_file.deployment.checkpoint_path
        if os.path.exists(os.path.join(chkpt_dir, "weights")):
            model_weights_path = os.path.join(chkpt_dir, "weights")
        else:
            model_weights_path = chkpt_dir

        model_weight_options = os.listdir(model_weights_path)

        torch.set_float32_matmul_precision('high')
        if 'vertiformer.pth' in model_weight_options:
            model_weights_path = os.path.join(model_weights_path, 'vertiformer.pth')
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.T = self.config_file.inference.pred_len
        self.chunk_size = self.config_file.deployment.chunk_size
        self.util = ROSUtils()
        self.multi_goal = False

        # ROS History Buffers
        self.context_length = self.model_yaml.model.encoder.transformer.block_size
        self.img = deque(maxlen=self.context_length)
        self.poses = np.array([(0.0, 0.0, 0.0) for _ in range(self.context_length + 1)],
                              dtype=np.float32)  # Buffer for last obs_len actions

        self.actions = np.array([(0.0, 0.0) for _ in range(self.context_length)], dtype=np.float32)

        self.actions_to_execute = torch.tensor([[0.0, 0.0] for _ in range(self.chunk_size)], device=self.device)

        self.br = CvBridge()
        self.global_path = None
        self.at_goal = False
        self.path_callback(self.config_file.waypoints)

        self.use_pose = self.config_file.path_follower.get("use_pose_head", True)

        self.num_viz_paths = 16
        # Publishers
        self.cmd_vel_pub = rospy.Publisher(self.sensor_config.publishers.cmd_vel, Twist, queue_size=20)
        self.predicted_pose_pub = rospy.Publisher(self.sensor_config.publishers.best_pose_predicted, Path, queue_size=20)
        self.uncond_pose_pub = rospy.Publisher(self.sensor_config.publishers.uncond_pose_predicted, Path, queue_size=20)
        self.cond_pose_pub = rospy.Publisher(self.sensor_config.publishers.cond_pose_predicted, Path, queue_size=20)
        self.all_goals_publisher = rospy.Publisher(self.sensor_config.publishers.all_goals, Path, queue_size=20)

        self.predicted_image_pub = rospy.Publisher(self.sensor_config.publishers.vertiformer_predicted_images, Image, queue_size=1)
        self.projected_traj_pub = rospy.Publisher(self.sensor_config.publishers.vertiformer_projected_trajectory_image, Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.sensor_config.subscribers.image_compressed, CompressedImage, self.img_compressed_callback,
                         queue_size=10)
        rospy.Subscriber(self.sensor_config.subscribers.odom, Odometry, self.poses_callback, queue_size=10)
        rospy.Subscriber(self.sensor_config.subscribers.camera_info, CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber(self.sensor_config.subscribers.cmd_vel, Twist, self.actions_callback, queue_size=10)

        self.MAX_V = self.config_file.platform.max_v
        self.MAX_W = self.config_file.platform.max_w
        self.poses_to_go_to = torch.tensor([(0.0, 0.0, 0.0) for _ in range(self.T)])
        self.poses_to_go_to = self.poses_to_go_to.expand(self.T, 3)
        self.global_target_poses = None

        pp_cfg = self.config_file.pure_pursuit
        self.Kp_v = pp_cfg.get("Kp_v", 1.0)
        self.Kp_w = pp_cfg.get("Kp_w", 2.0)
        self.lookahead_dist = pp_cfg.get("lookahead_dist", 0.3)
        self.ema_alpha_v = pp_cfg.get("ema_alpha_v", 0.05)
        self.ema_alpha_w = pp_cfg.get("ema_alpha_w", 0.05)
        self.poses_call_counter = 0
        self.img_counter = 0
        self.action_counter = 0

        self.current_v = 0.0
        self.current_w = 0.0

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
            rospy.loginfo("Camera intrinsics (K) received from camera_info.")

    def img_callback(self, img_msg):
        self.img_counter += 1
        if self.img_counter > 5:
            self.img_counter = 0
            img = self.br.imgmsg_to_cv2(img_msg)
            self.latest_raw_img = img.copy()  # Save raw BGR copy for OpenCV trajectory rendering
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            img = self.transform(img)
            self.img.append(img)

    def img_compressed_callback(self, msg):
        """
        Process image data from a topic that publishes sensor_msgs/CompressedImage
        to a PIL image
        """
        # convert sensor_msgs/CompressedImage to PIL image
        img = PIL.Image.open(io.BytesIO(msg.data))
        self.latest_raw_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        img = self.transform(img)
        self.img.append(img)

    def publish_predicted_images(self, decoded_frames):
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in decoded_frames]
        film_strip = np.concatenate(bgr_frames, axis=1)

        if film_strip.shape[1] > 2000:
            h, w = film_strip.shape[:2]
            film_strip = cv2.resize(film_strip, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        if not hasattr(self, 'predicted_image_pub'):
            return

        try:
            img_msg = self.br.cv2_to_imgmsg(film_strip, encoding="bgr8")
            self.predicted_image_pub.publish(img_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish predicted images: {e}")

    def poses_callback(self, poses_msg):
        self.poses_call_counter += 1
        if self.poses_call_counter >= 25:
            self.poses_call_counter = 0
            pose = poses_msg.pose.pose

            roll, pitch, yaw = euler_from_quaternion([
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ])
            self.poses = np.roll(self.poses, -1, axis=0)
            self.poses[-1] = (pose.position.x, pose.position.y, yaw)

    def actions_callback(self, actions_msg):
        self.actions = np.roll(self.actions, -1, axis=0)
        self.actions[-1] = (actions_msg.linear.x, actions_msg.angular.z)

    def path_callback(self, path_msg):
        """Stores the incoming global path waypoints."""
        pts = []
        for points in path_msg:
            x, y, z, qx, qy, qz, qw = points
            _, _, yaw = euler_from_quaternion([qx, qy, qz, qw])
            pts.append((x, y, yaw))
        self.global_path = torch.tensor(pts).to(self.device)

    def tensor_to_twist_background(self):
        """
        Continuously publish actions from self.actions_to_execute.
        """
        rate = rospy.Rate(self.config_file.sample_rates.desired_rate)  # Usually 10 Hz
        last_twist = Twist()

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            with self.actions_lock:
                actions = self.actions_to_execute.cpu().numpy()
                self._new_actions = False

            for v, omega in actions:
                if rospy.is_shutdown() or self.stop_event.is_set() or self._new_actions:
                    break

                last_twist = Twist()
                last_twist.linear.x = float(np.clip(v, 0.0, self.MAX_V))
                last_twist.angular.z = float(np.clip(omega, -self.MAX_W, self.MAX_W))
                self.cmd_vel_pub.publish(last_twist)
                rate.sleep()
            else:
                # Chunk finished and no fresh plan yet: safely hold the last command
                while not (rospy.is_shutdown() or self.stop_event.is_set() or self._new_actions):
                    self.cmd_vel_pub.publish(last_twist)
                    rate.sleep()

    def pd_controller(self):
        """Pure Pursuit Proportional Controller tracking an Odom-anchored path"""
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            if self.global_target_poses is None or len(self.global_target_poses) == 0 or len(self.poses) == 0:
                rospy.sleep(0.02)
                continue

            # 1. Get current real-time odometry
            curr_x, curr_y, curr_yaw = self.poses[-1]

            # 2. Find the Lookahead Point on the global path
            target_pt = None
            for pt in self.global_target_poses[0]:
                gx, gy = pt[0], pt[1]
                dist = np.hypot(gx.item() - curr_x, gy.item() - curr_y)
                if dist >= self.lookahead_dist:
                    target_pt = (gx.item(), gy.item())
                    break

            if target_pt is None:
                if len(self.global_target_poses[0]) == 0:
                    rospy.sleep(0.02)
                    continue
                target_pt = self.global_target_poses[0][-1][:2]

            gx, gy = target_pt

            # 3. Convert global target back to current local base frame for steering
            dx_global = gx - curr_x
            dy_global = gy - curr_y
            dx = dx_global * np.cos(-curr_yaw) - dy_global * np.sin(-curr_yaw)
            dy = dx_global * np.sin(-curr_yaw) + dy_global * np.cos(-curr_yaw)

            twist_msg = Twist()
            dist_to_target = np.hypot(dx, dy)

            if dist_to_target < 0.05:  # Arrival tolerance
                v_target = 0.0
                w_target = 0.0
            else:
                angle_error = np.arctan2(dy, dx)
                w_target = np.clip(self.Kp_w * angle_error, -self.MAX_W, self.MAX_W)

                v_target = self.Kp_v * dist_to_target * np.cos(angle_error)
                v_target = np.clip(max(0.0, v_target), -self.MAX_V, self.MAX_V)  # Prevent reversing

            # --- Physical Low-Pass Filter (EMA) for Motor Commands ---
            # Prevents jerky hardware movements when the predicted network path shifts between inferences
            self.current_v = (self.ema_alpha_v * v_target) + (1.0 - self.ema_alpha_v) * self.current_v
            self.current_w = (self.ema_alpha_w * w_target) + (1.0 - self.ema_alpha_w) * self.current_w

            twist_msg.linear.x = self.current_v
            twist_msg.angular.z = self.current_w

            self.cmd_vel_pub.publish(twist_msg)
            rospy.sleep(0.02)

    def publish_all_poses(self, all_poses, current_odom):
        frame_id = "robot/odom"
        if self.predicted_pose_pub.get_num_connections() > 0:
            num_to_draw = min(self.num_viz_paths, all_poses.shape[0])
            for i in range(num_to_draw):
                pa = Path()
                pa.header = self.util.make_header(frame_id)
                pa.poses = list(
                    map(self.util.particle_to_posestamped, all_poses[i, :, :], [frame_id] * all_poses.shape[1])
                )
                self.predicted_pose_pub.publish(pa)

    def publish_all_goals(self, all_goals):
        frame_id = "robot/odom"
        all_goals = all_goals.cpu()
        if self.all_goals_publisher.get_num_connections() > 0:
            pa = Path()
            pa.header = self.util.make_header(frame_id)
            pa.poses = list(
                map(self.util.particle_to_posestamped, all_goals, [frame_id] * len(all_goals))
            )
            self.all_goals_publisher.publish(pa)

    def publish_predicted_poses(self, all_poses):
        frame_id = "robot/odom"
        if self.predicted_pose_pub.get_num_connections() > 0:
            pa = Path()
            pa.header = self.util.make_header(frame_id)
            pa.poses = list(
                map(self.util.particle_to_posestamped, all_poses[:, :], [frame_id] * len(all_poses))
            )
            self.predicted_pose_pub.publish(pa)

    def publish_goal_pose(self):
        # Publish the goal pose for visualization
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "robot/odom"
        odom_msg.pose.pose.position.x = self.global_path[0, 0].item()
        odom_msg.pose.pose.position.y = self.global_path[0, 1].item()
        # Assuming no change in z and orientation for simplicity
        new_qx, new_qy, new_qz, new_qw = quaternion_from_euler(0.0, 0.0, self.global_path[0, 2].item())
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = new_qx
        odom_msg.pose.pose.orientation.y = new_qy
        odom_msg.pose.pose.orientation.z = new_qz
        odom_msg.pose.pose.orientation.w = new_qw
        self.goal_publisher.publish(odom_msg)

    def extract_target_deltas(self, current_odom):
        """
        Crucial Coordinate Transform: Global Path -> Local Frame -> Step-wise Deltas.
        Generates a sequence of T step-wise deltas
        tracking the sparse global path seamlessly across multiple waypoints.
        """
        if self.global_path is None or len(self.global_path) == 0:
            return None

        # Pop waypoints that are already reached without skipping the control cycle
        while len(self.global_path) > 0:
            dist_to_goal = torch.norm(current_odom[:2] - self.global_path[0, :2])
            if dist_to_goal < self.goal_tolerance:
                rospy.loginfo(f"Sub-Goal Reached! {len(self.global_path) - 1} waypoints remaining.")
                self.global_path = self.global_path[1:]
            else:
                break

        if len(self.global_path) == 0:
            return None

        odom_x, odom_y, odom_yaw = current_odom[0].item(), current_odom[1].item(), current_odom[2].item()
        cos_t = np.cos(-odom_yaw)
        sin_t = np.sin(-odom_yaw)

        # 1. Transform the remaining global path into the local base frame
        local_waypoints = [(0.0, 0.0, 0.0)] # Anchor at origin

        # Take up to the next few waypoints to ensure we have enough track to sample T steps
        for point in self.global_path[:5]:
            gx, gy, gyaw = point[0].item(), point[1].item(), point[2].item()
            dx = gx - odom_x
            dy = gy - odom_y

            # Transform Global coordinates into Local Base Frame
            lx = dx * cos_t - dy * sin_t
            ly = dx * sin_t + dy * cos_t

            # Normalize relative yaw difference
            lyaw = (gyaw - odom_yaw + np.pi) % (2 * np.pi) - np.pi
            local_waypoints.append((lx, ly, lyaw))

        # 2. Resample the piecewise path to get exactly T steps based on the robot's expected velocity
        rate = self.config_file.sample_rates.desired_rate
        max_step_dist = self.MAX_V / rate

        sampled_path = [(0.0, 0.0, 0.0)]
        wp_idx = 0
        curr_pt = np.array([0.0, 0.0])

        for _ in range(self.T):
            if wp_idx >= len(local_waypoints) - 1:
                # Reached the end of the provided waypoints, pad with the last known position
                sampled_path.append(local_waypoints[-1])
                continue

            target_d = max_step_dist
            while wp_idx < len(local_waypoints) - 1:
                next_wp = np.array([local_waypoints[wp_idx+1][0], local_waypoints[wp_idx+1][1]])
                dist_to_next = np.linalg.norm(next_wp - curr_pt)

                if target_d <= dist_to_next:
                    # The target step lies on this segment
                    ratio = target_d / dist_to_next
                    curr_pt = curr_pt + ratio * (next_wp - curr_pt)

                    # Approximate yaw pointing towards the next waypoint
                    yaw_approx = np.arctan2(next_wp[1] - curr_pt[1], next_wp[0] - curr_pt[0])
                    sampled_path.append((curr_pt[0], curr_pt[1], yaw_approx))
                    break
                else:
                    # Step exceeds this segment, move to the next waypoint and reduce remaining distance
                    target_d -= dist_to_next
                    curr_pt = next_wp
                    wp_idx += 1

            if wp_idx >= len(local_waypoints) - 1:
                sampled_path.append(local_waypoints[-1])

        sampled_tensor = torch.tensor(sampled_path, dtype=torch.float32, device=self.device)

        # 3. Compute step-wise deltas
        target_deltas = sampled_tensor[1:] - sampled_tensor[:-1]

        # Normalize yaw deltas (prevent 2*pi wrapping jumps)
        target_deltas[:, 2] = (target_deltas[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi

        return target_deltas

    def control_loop(self):
        """Main deployment loop running on the real robot."""
        if not self.use_pose:
            self.publish_thread = threading.Thread(target=self.tensor_to_twist_background, daemon=True)
            self.publish_thread.start()
            task_type = 'ikd'
        else:
            self.publish_thread = threading.Thread(target=self.pd_controller, daemon=True)
            self.publish_thread.start()
            task_type = 'fkd'

        rospy.loginfo("Vertiformer Follower Node initialized. Waiting for /global_plan...")

        total_time = 0
        iters = 0
        while not rospy.is_shutdown():
            start = time.time()
            self.publish_all_goals(self.global_path)

            if len(self.global_path) == 0:
                rospy.loginfo("Reached Final Waypoint!")
                self.stop_robot()
                break

            buffers = self.get_recent_history_buffers()
            if buffers is None:
                rospy.sleep(0.05)
                continue

            init_imgs, init_actions, init_poses, raw_poses = buffers
            current_odom = raw_poses[-1]

            # 1. Transform Path to Target Deltas
            extracted_target_deltas = self.extract_target_deltas(current_odom)
            if extracted_target_deltas is None:
                continue

            # Encode history once to save compute
            cached_obs = self.model.encode_history(init_imgs, init_actions, init_poses, pred_len=self.T)

            cfg_follower_cfg = self.config_file.path_follower.get("cfg", {})
            best_traj = path_follower_policy(
                model=self.model,
                init_imgs=init_imgs,
                init_actions=init_actions,
                init_poses=init_poses,
                future_poses=extracted_target_deltas.unsqueeze(0),
                T=self.T,
                sample_images=self.debug_visualize,
                task_type=task_type,
                guidance_scale=cfg_follower_cfg.get("guidance_scale", 0.5),
                cached_obs_tokens=cached_obs,
            )

            # 3. Action Chunking Execution
            self.previous_rate = time.time() - start

            # Get the real-time odometry right NOW
            current_real_odom = self.poses[-1]
            odom_x, odom_y, odom_yaw = current_real_odom[0].item(), current_real_odom[1].item(), current_real_odom[2].item()

            cos_y_inv, sin_y_inv = np.cos(odom_yaw), np.sin(odom_yaw)
            R_inv = torch.tensor([[cos_y_inv, -sin_y_inv],
                                  [sin_y_inv, cos_y_inv]]).float()

            # Reconstruct Global Path for Rviz & PD Controller
            full_local_deltas = best_traj.get("poses")
            local_path_se2 = None

            if full_local_deltas is not None:
                local_abs_path = torch.cumsum(full_local_deltas[..., :2], dim=0).float()

                # Safely define local_path_se2 for Camera Projection
                local_abs_yaw = torch.cumsum(full_local_deltas[..., 2], dim=0) if full_local_deltas.shape[-1] == 3 else torch.zeros_like(local_abs_path[:, 0])
                local_path_se2 = torch.cat([local_abs_path, local_abs_yaw.unsqueeze(-1)], dim=-1)

                global_path_xy = local_abs_path @ R_inv.T
                global_path_xy[:, 0] += odom_x
                global_path_xy[:, 1] += odom_y
                global_yaw = (local_abs_yaw + odom_yaw + torch.pi) % (2 * torch.pi) - torch.pi

                global_path_se2 = torch.cat([global_path_xy, global_yaw.unsqueeze(-1)], dim=-1)
                self.publish_predicted_poses(global_path_se2)

            # Publish Latent CFG Components
            def get_global_path(deltas):
                if deltas is None: return None
                l_path = torch.cumsum(deltas[..., :2], dim=0).float()
                l_yaw = torch.cumsum(deltas[..., 2], dim=0) if deltas.shape[-1] == 3 else torch.zeros_like(l_path[:, 0])
                g_xy = l_path @ R_inv.T
                g_xy[:, 0] += odom_x
                g_xy[:, 1] += odom_y
                g_yaw = (l_yaw + odom_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                return torch.cat([g_xy, g_yaw.unsqueeze(-1)], dim=-1)

            uncond_global = get_global_path(best_traj.get("poses_uncond"))
            if uncond_global is not None and self.uncond_pose_pub.get_num_connections() > 0:
                uncond_global_np = uncond_global.cpu().numpy()
                pa = Path(header=self.util.make_header("robot/odom"), poses=list(map(self.util.particle_to_posestamped, uncond_global_np, ["robot/odom"] * len(uncond_global_np))))
                self.uncond_pose_pub.publish(pa)

            cond_global = get_global_path(best_traj.get("poses_cond"))
            if cond_global is not None and self.cond_pose_pub.get_num_connections() > 0:
                cond_global_np = cond_global.cpu().numpy()
                pa = Path(header=self.util.make_header("robot/odom"), poses=list(map(self.util.particle_to_posestamped, cond_global_np, ["robot/odom"] * len(cond_global_np))))
                self.cond_pose_pub.publish(pa)

            if task_type == 'ikd':
                # apply temporal Ensembling (EMA) to smooth jitter
                new_actions = best_traj["actions"]
                if getattr(self, "ensembled_actions", None) is None:
                    self.ensembled_actions = new_actions
                else:
                    alpha = self.config_file.path_follower.temporal_ensembling.get("action_blend_momentum", 0.6)
                    self.ensembled_actions = (alpha * new_actions) + ((1.0 - alpha) * self.ensembled_actions)

                # Hand the fresh chunk to the publisher thread
                with self.actions_lock:
                    self.actions_to_execute = self.ensembled_actions[:self.chunk_size]
                    self._new_actions = True

            elif task_type == 'fkd':
                # Optional: Visualize the tracked path in RViz
                self.poses_to_go_to = best_traj["poses"][:self.chunk_size]

                # apply temporal Ensembling (EMA) to smooth jitter PD waypoints
                self.global_target_poses = [self.ensemble_trajectory(
                    new_local_deltas=best_traj["poses"],
                    current_odom=current_real_odom,
                    blend_momentum=self.config_file.path_follower.temporal_ensembling.get("pose_blend_momentum", 0.85)
                ).cpu().numpy()]

            # Save imagined trajectory video if debugging is enabled
            if self.debug_visualize and best_traj.get("image_latents") is not None:
                images_to_save = best_traj["image_latents"]
                h_prime, w_prime = self.resize[0] // 8, self.resize[1] // 8

                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    decoded_frames = self.model.transformer_encoder.unified_encoder.visual_encoder.decode(
                        images_to_save, hw_shape=(h_prime, w_prime)
                    )

                decoded_frames = ((decoded_frames + 1.0) / 2.0).clamp(0, 1).float()
                decoded_frames = (decoded_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

                os.makedirs("debug_videos", exist_ok=True)
                os.makedirs(os.path.join("debug_videos", self.experiment_name), exist_ok=True)
                video_path = os.path.join("debug_videos", self.experiment_name,
                                          f"planned_traj_{int(time.time())}.mp4")

                height, width, _ = decoded_frames[0].shape
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))

                for frame in decoded_frames:
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                video_writer.release()
                rospy.loginfo(f"Saved debug video to {video_path}")

                self.publish_predicted_images(decoded_frames)

            # Publish 3D Projected Trajectory over Real RGB Feed
            if self.K is not None and self.latest_raw_img is not None and best_traj.get("poses") is not None:

                projected_img = project_trajectory_on_image(
                    self.latest_raw_img, local_path_se2.detach().cpu().numpy(), self.K, self.T_cam_base
                )

                # Determine 3-channel version for video writer & ROS publishing
                if len(projected_img.shape) == 3 and projected_img.shape[2] == 4:
                    projected_img_bgr = cv2.cvtColor(projected_img, cv2.COLOR_BGRA2BGR)
                else:
                    projected_img_bgr = projected_img.copy()

                # Save Continuous Real-World Experiment Dashcam
                if self.debug_visualize:
                    if getattr(self, "experiment_video_writer", None) is None:
                        os.makedirs(os.path.join("debug_videos_follower", self.experiment_name), exist_ok=True)
                        video_path = os.path.join("debug_videos_follower", self.experiment_name, f"experiment_rollout_{int(time.time())}.mp4")
                        height, width = projected_img_bgr.shape[:2]
                        self.experiment_video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
                        self.video_thread = threading.Thread(target=self._video_write_worker, daemon=True)
                        self.video_thread.start()

                    self.video_queue.put(projected_img_bgr)

                if hasattr(self, "projected_traj_pub"):
                    try:
                        img_msg = self.br.cv2_to_imgmsg(projected_img_bgr, encoding="bgr8")
                        self.projected_traj_pub.publish(img_msg)
                    except Exception as e:
                        rospy.logwarn(f"Failed to publish projected trajectory: {e}")

    def get_recent_history_buffers(self):
        if len(self.img) < self.context_length or len(self.poses) < self.context_length + 1 or len(
                self.actions) < self.context_length:
            return None

        imgs_tensor = torch.stack(list(self.img)).to(self.device)
        actions_tensor = torch.tensor(self.actions).to(self.device)
        raw_poses = torch.tensor(self.poses).to(self.device)

        # 1. Compute Step-Wise Deltas
        pos_deltas = raw_poses[1:, :2] - raw_poses[:-1, :2]
        yaw_deltas = raw_poses[1:, 2] - raw_poses[:-1, 2]
        yaw_deltas = (yaw_deltas + torch.pi) % (2 * torch.pi) - torch.pi

        # 2. Ego-Centric Rotation (Anchor at newest state)
        ego_yaw = raw_poses[-1, 2]
        cos_y, sin_y = torch.cos(-ego_yaw), torch.sin(-ego_yaw)
        R = torch.stack([
            torch.stack([cos_y, -sin_y]),
            torch.stack([sin_y,  cos_y])
        ])

        pos_deltas = pos_deltas @ R.T
        poses_tensor = torch.cat([pos_deltas, yaw_deltas.unsqueeze(-1)], dim=-1)

        return(
            imgs_tensor.unsqueeze(0),
            actions_tensor.unsqueeze(0),
            poses_tensor.unsqueeze(0),
            raw_poses,
        )

    def _video_write_worker(self):
        """Background thread that safely empties the video queue to disk without blocking the control loop."""
        while not rospy.is_shutdown():
            try:
                frame = self.video_queue.get(timeout=0.5)
                if frame is None:
                    break
                if self.experiment_video_writer is not None:
                    self.experiment_video_writer.write(frame)
            except queue.Empty:
                if self.stop_event.is_set():
                    break

    def ensemble_trajectory(self, new_local_deltas, current_odom, blend_momentum=0.6):
        """
        Action Chunking with Temporal Ensembling (ALOHA style) for spatial waypoints.
        Fuses a newly predicted local trajectory with the existing global path.

        Args:
            new_local_deltas: Tensor of shape [T, 2] or [T, 3] representing the new plan (local deltas).
            current_odom: Tensor or array of shape [3] representing the robot's CURRENT real-time pose.
            blend_momentum: How much to trust the NEW trajectory (0.0 to 1.0).

        Returns:
            The fused, continuous global path as a Tensor.
        """
        if new_local_deltas.ndim == 3:
            new_local_deltas = new_local_deltas.squeeze(0)

        new_local_deltas = new_local_deltas.cpu()
        if isinstance(current_odom, np.ndarray):
            current_odom = torch.from_numpy(current_odom).float()
        current_odom = current_odom.detach().cpu()

        # Convert local step-wise deltas to global Odom frame
        local_abs_path = torch.cumsum(new_local_deltas[..., :2], dim=0).float()
        odom_x, odom_y, odom_yaw = current_odom[0].item(), current_odom[1].item(), current_odom[2].item()

        cos_y_inv, sin_y_inv = np.cos(odom_yaw), np.sin(odom_yaw)
        R_inv = torch.tensor([[cos_y_inv, -sin_y_inv],
                              [sin_y_inv,  cos_y_inv]]).float()

        global_path_xy = local_abs_path @ R_inv.T
        global_path_xy[:, 0] += odom_x
        global_path_xy[:, 1] += odom_y

        if new_local_deltas.shape[-1] == 3:
            local_abs_yaw = torch.cumsum(new_local_deltas[..., 2], dim=0)
            global_yaw = (local_abs_yaw + odom_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            new_traj_odom = torch.cat([global_path_xy, global_yaw.unsqueeze(-1)], dim=-1)
        else:
            new_traj_odom = global_path_xy

        # 1. Initialize if empty
        if getattr(self, "ensembled_global_path", None) is None or len(self.ensembled_global_path) < 2:
            self.ensembled_global_path = new_traj_odom
            return self._slice_path_from_current(self.ensembled_global_path, current_odom)

        # 2. Slice off waypoints the robot has already passed
        clean_old_path = self._slice_path_from_current(self.ensembled_global_path, current_odom)
        clean_new_path = self._slice_path_from_current(new_traj_odom, current_odom)

        if len(clean_old_path) == 0:
            self.ensembled_global_path = clean_new_path
            return self.ensembled_global_path
        if len(clean_new_path) == 0:
            self.ensembled_global_path = clean_old_path
            return self.ensembled_global_path

        # 3. Temporal Ensembling (Blend overlapping segments)
        overlap_len = min(len(clean_old_path), len(clean_new_path))
        blended_path = []

        for i in range(overlap_len):
            blended_pt = (blend_momentum * clean_new_path[i]) + ((1.0 - blend_momentum) * clean_old_path[i])
            # Normalize yaw to [-pi, pi] if tracking 3D
            if blended_pt.shape[-1] == 3:
                blended_pt[2] = torch.remainder(blended_pt[2] + torch.pi, 2 * torch.pi) - torch.pi
            blended_path.append(blended_pt)

        # 4. Extend the path (Append the future that doesn't overlap)
        if len(clean_new_path) > overlap_len:
            blended_path.extend(clean_new_path[overlap_len:])
        elif len(clean_old_path) > overlap_len:
            blended_path.extend(clean_old_path[overlap_len:])

        self.ensembled_global_path = torch.stack(blended_path)
        return self.ensembled_global_path

    def _slice_path_from_current(self, path, current_odom):
        """Discards waypoints that fall strictly behind the robot's current position."""
        if len(path) == 0:
            return path

        current_pos = current_odom[:2]
        distances = torch.norm(path[:, :2] - current_pos.unsqueeze(0), dim=-1)
        closest_idx = torch.argmin(distances).item()

        # If the closest point is the last point and > 0.5m away, the path is entirely exhausted
        if closest_idx == len(path) - 1 and distances[closest_idx] > 0.5:
            return torch.empty((0, path.shape[-1]), device=path.device)

        return path[closest_idx:]


    def stop_robot(self):
        # signal background threads to stop and publish zero velocity
        try:
            self.stop_event.set()
            if self.publish_thread is not None and self.publish_thread.is_alive():
                self.publish_thread.join(timeout=1.0)
        except Exception:
            pass
        self.cmd_vel_pub.publish(Twist())

        if getattr(self, "experiment_video_writer", None) is not None:
            self.video_queue.put(None)
            if self.video_thread is not None and self.video_thread.is_alive():
                self.video_thread.join(timeout=3.0)

            self.experiment_video_writer.release()
            rospy.loginfo("Saved full experiment projected video.")

    def _on_shutdown(self):
        # rospy shutdown handler: ensure background thread exits and robot stops
        rospy.loginfo("VertiformerFollowerNode shutting down: stopping publish thread")
        self.stop_robot()


def main(args):
    control_node = VertiformerFollowerNode(args.conf, args.exp_name, args.robot)
    rospy.init_node('Vertiformer_Path_Follower', anonymous=True)
    rospy.on_shutdown(control_node._on_shutdown)
    control_node.control_loop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="deployment/conf/vertiformer_path_follow.yaml", type=str)
    parser.add_argument(
        "--robot",
        default="jackal",
        type=str,
        help="Robot type. Add new robot specs under deployment/conf/robots/<robot>.yaml.",
    )
    parser.add_argument("--exp_name", default="vertiformer_following", type=str)
    args = parser.parse_args()
    main(args)