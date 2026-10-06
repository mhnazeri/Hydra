from datetime import datetime

import time
import rospy
import torch
import torch._dynamo
import io

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch._dynamo.config.suppress_errors = True  # Safe for deployment
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge
import cv2
from tf.transformations import euler_from_quaternion
from collections import deque
import queue
from dlp import HydraDLP
from hydra.utilities.projection_utils import project_trajectory_on_image
import PIL

import safetensors.torch

import sys
import os
from torchvision import transforms

from omegaconf import OmegaConf

sys.path.append(os.getcwd())
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import threading
import numpy as np

from hydra.model.hydra import Hydra
from hydra.utilities.helpers import get_conf

import argparse
from scipy.ndimage import uniform_filter1d

from hydra.utilities.ros_utilities import ROSUtils


class HydraControlNode:
    def __init__(self, config_file, experiment_name, robot_name="jackal", use_goal_conditioned_seed=False):
        # Event used to signal background threads to stop
        self.stop_event = threading.Event()
        self.publish_thread = None
        self.video_queue = queue.Queue()
        self.video_thread = None
        self.experiment_video_writer = None
        self.experiment_name = experiment_name
        self.use_goal_conditioned_seed = use_goal_conditioned_seed

        base_cfg = get_conf(config_file)
        robot_cfg_path = os.path.join(os.path.dirname(config_file), "robots", f"{robot_name}.yaml")
        robot_cfg = get_conf(robot_cfg_path)

        self.config_file = OmegaConf.merge(base_cfg, robot_cfg)
        self.sensor_config = self.config_file.ros

        architecture_yaml = [i for i in os.listdir(self.config_file.deployment.checkpoint_path) if i.endswith('.yaml')][
            0]
        self.model_yaml = OmegaConf.load(os.path.join(self.config_file.deployment.checkpoint_path, architecture_yaml))
        self.context_length = self.model_yaml.model.encoder.transformer.block_size

        self.model_architecture = self.model_yaml.model
        img_size = self.model_yaml.model.encoder.image_size
        self.resize = (img_size[0], img_size[1])

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

        self.device = self.config_file.deployment.device
        self.model = Hydra(self.model_architecture).to(self.device)

        chkpt_dir = self.config_file.deployment.checkpoint_path
        if os.path.exists(os.path.join(chkpt_dir, "weights")):
            model_weights_path = os.path.join(chkpt_dir, "weights")
        else:
            model_weights_path = chkpt_dir

        model_weight_options = os.listdir(model_weights_path)

        torch.set_float32_matmul_precision('high')
        if 'hydra.pth' in model_weight_options:
            model_weights_path = os.path.join(model_weights_path, 'hydra.pth')
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

        # Compile the flow matching heads
        self.model.fm_pose = torch.compile(self.model.fm_pose, mode="reduce-overhead")
        self.model.fm_image = torch.compile(self.model.fm_image, mode="reduce-overhead")
        self.model.fm_action = torch.compile(self.model.fm_action, mode="reduce-overhead")
        self.model.transformer_encoder = torch.compile(self.model.transformer_encoder, mode="reduce-overhead")

        waypoint_configs = self.config_file.deployment.waypoints
        final_wp = waypoint_configs[-1]
        self.goal_pose = torch.tensor(final_wp["pose"], dtype=torch.float32).to(self.device)
        self.goal_img = final_wp.get("image", None)

        if self.goal_img is not None:
            self.goal_img = self.transform(PIL.Image.open(self.goal_img).convert("RGB")).to(
                self.device) if self.goal_img is not None else None
            self.goal_img = self.goal_img.unsqueeze(
                0) if self.goal_img is not None and self.goal_img.ndim == 3 else self.goal_img
            self.goal_img = self.model.transformer_encoder.unified_encoder.visual_encoder.encode(self.goal_img)

        self.T = self.config_file.inference.pred_len  # e.g., 5
        self.chunk_size = self.config_file.deployment.chunk_size  # k steps to execute before replanning (k < T)
        self.goal_tolerance = self.config_file.deployment.goal_tolerance  # meters
        self.util = ROSUtils()

        self.executed_path_history = deque(maxlen=2000)

        self.planner = HydraDLP(
            model=self.model,
            T=self.T,
            use_pose_head=self.config_file.deployment.use_pose_head,
            use_goal_conditioned_seed=self.use_goal_conditioned_seed,
            **self.config_file.planner,
        )
        if self.use_goal_conditioned_seed:
            rospy.loginfo("HydraDLP: goal-conditioned search seed ENABLED.")

        # Load the debug flag
        self.debug_visualize = self.config_file.deployment.get("debug_visualize", False)

        #### ROS setup

        self.img = deque(maxlen=self.context_length)  # Buffer for last obs_len images

        self.poses = np.array([(0.0, 0.0, 0.0) for _ in range(self.context_length + 1)],
                              dtype=np.float32)  # Buffer for last obs_len poses + 1 for deltas
        self.br = CvBridge()
        self.at_goal = False

        self.actions = np.array([(0.0, 0.0) for _ in range(self.context_length)],
                                dtype=np.float32)  # Buffer for last obs_len actions
        self.action_lock = threading.Lock()
        self.action_idx = 0
        self.current_action_chunk = None
        self.previous_rate = 4.0
        self.use_pose = self.config_file.deployment.use_pose_head

        # Projection properties
        self.K = None

        self.latest_raw_img = None

        t_cam_base_cfg = self.config_file.platform.get("T_cam_base", None)
        if t_cam_base_cfg is not None:
            self.T_cam_base = np.array(t_cam_base_cfg, dtype=np.float32)
        else:
            # Fallback to default URDF extrinsics
            # you should calibrate your own camera and get the correct T_cam_base and use the config file to set it. This is just a placeholder for the Jackal with a front camera.
            self.T_cam_base = np.array([
                [2.2204460492503131e-16, -1.0, 0.0, 5.9999999999999984e-02],
                [-4.9979169270678137e-02, 0.0, -9.9875026039496617e-01, 0.75],
                [9.9875026039496617e-01, 2.2204460492503131e-16, -4.9979169270678137e-02, 1.0749687539060158e-02],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)

        self.num_viz_paths = 16
        self.cmd_vel_pub = rospy.Publisher(self.sensor_config.publishers.cmd_vel, Twist, queue_size=20)
        self.pose_publisher = rospy.Publisher(self.sensor_config.publishers.best_pose_predicted, Path, queue_size=20)
        self.goal_publisher = rospy.Publisher(self.sensor_config.publishers.goal_pose, Odometry, queue_size=1)
        self.all_pose_publisher = rospy.Publisher(self.sensor_config.publishers.all_predicted_pose, Path, queue_size=20)
        self.predicted_image_pub = rospy.Publisher(self.sensor_config.publishers.hydra_predicted_images, Image, queue_size=1)
        self.projected_traj_pub = rospy.Publisher(self.sensor_config.publishers.hydra_projected_trajectory_image, Image, queue_size=1)

        rospy.Subscriber(self.sensor_config.subscribers.image_compressed, CompressedImage, self.img_compressed_callback,
                         queue_size=10)
        rospy.Subscriber(self.sensor_config.subscribers.camera_info, CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber(self.sensor_config.subscribers.odom, Odometry, self.poses_callback, queue_size=10)
        rospy.Subscriber(self.sensor_config.subscribers.cmd_vel, Twist, self.actions_callback, queue_size=10)

        # FOR PD CONTROLLER
        self.MAX_V = self.config_file.platform.max_v
        self.MAX_W = self.config_file.platform.max_w
        self.poses_to_go_to = torch.tensor([(0.0, 0.0, 0.0) for _ in range(self.T)])
        self.poses_to_go_to = self.poses_to_go_to.expand(self.T, 3)
        self.global_target_poses = None

        self.Kp_v = self.config_file.pure_pursuit["Kp_v"]  # Linear velocity gain
        self.Kp_w = self.config_file.pure_pursuit["Kp_w"]  # Angular velocity gain
        self.lookahead_dist = self.config_file.pure_pursuit["lookahead_dist"]  # Lookahead distance in meters

        # topic counters for syncing rates
        self.img_counter = 0
        self.odom_counter = 0

    def pd_controller(self):
        """Pure Pursuit Proportional Controller tracking an Odom-anchored path"""
        dt = 0.02  # High-frequency control loop

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            if self.global_target_poses is None or len(self.global_target_poses) == 0 or len(self.poses) == 0:
                rospy.sleep(dt)
                continue

            # 1. Get current real-time odometry
            curr_x, curr_y, curr_yaw = self.poses[-1]

            path_pts = self.global_target_poses[0]

            if len(path_pts) == 0:
                rospy.sleep(dt)
                continue

            # 2. Find the closest point on the path to the robot
            distances = np.hypot(path_pts[:, 0] - curr_x, path_pts[:, 1] - curr_y)
            closest_idx = np.argmin(distances)

            # 3. Search FORWARD from the closest point for the lookahead
            target_pt = None
            for i in range(closest_idx, len(path_pts)):
                if distances[i] >= self.lookahead_dist:
                    target_pt = path_pts[i][:2]
                    break

            # If no point is far enough, lock onto the final waypoint
            if target_pt is None:
                target_pt = path_pts[-1][:2]

            gx, gy = target_pt

            # 4. Convert global target back to current local base frame for steering
            dx_global = gx - curr_x
            dy_global = gy - curr_y
            dx = dx_global * np.cos(-curr_yaw) - dy_global * np.sin(-curr_yaw)
            dy = dx_global * np.sin(-curr_yaw) + dy_global * np.cos(-curr_yaw)

            twist_msg = Twist()
            dist_to_target = np.hypot(dx, dy)

            if dist_to_target < 0.05:  # Arrival tolerance
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
            else:
                angle_error = np.arctan2(dy, dx)
                w = np.clip(self.Kp_w * angle_error, -self.MAX_W, self.MAX_W)

                # Slow down when making sharp turns to track curves faithfully
                # If angle is 90 degrees, cos(90) = 0, so forward velocity pauses to turn in place
                turn_slowdown = np.clip(np.cos(angle_error), 0.0, 1.0)
                v = self.Kp_v * dist_to_target * turn_slowdown
                v = np.clip(max(0.0, v), -self.MAX_V, self.MAX_V)  # Prevent reversing

                twist_msg.linear.x = float(v)
                twist_msg.angular.z = float(w)

            self.cmd_vel_pub.publish(twist_msg)
            rospy.sleep(dt)

    def img_compressed_callback(self, msg):
        """
        Process image data from a topic that publishes sensor_msgs/CompressedImage
        to a PIL image
        """
        # convert sensor_msgs/CompressedImage to PIL image
        img = PIL.Image.open(io.BytesIO(msg.data))
        # We must save a raw BGR copy so OpenCV has a canvas for trajectory projection.
        self.latest_raw_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = self.transform(img)
        self.img.append(img)

    def publish_all_poses(self, all_poses, current_odom):
        frame_id = "robot/odom"
        if self.all_pose_publisher.get_num_connections() > 0:
            num_to_draw = min(self.num_viz_paths, all_poses.shape[0])
            for i in range(num_to_draw):
                pa = Path()
                pa.header = self.util.make_header(frame_id)
                pa.poses = list(
                    map(self.util.particle_to_posestamped, all_poses[i, :, :], [frame_id] * all_poses.shape[1])
                )
                self.all_pose_publisher.publish(pa)

    def publish_predicted_poses(self, all_poses):
        frame_id = "robot/odom"
        if self.pose_publisher.get_num_connections() > 0:
            pa = Path()
            pa.header = self.util.make_header(frame_id)
            pa.poses = list(
                map(self.util.particle_to_posestamped, all_poses[:, :], [frame_id] * len(all_poses))
            )
            self.pose_publisher.publish(pa)

    def publish_goal_pose(self):
        # Publish the goal pose for visualization
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "robot/odom"
        odom_msg.pose.pose.position.x = self.goal_pose[0].item()
        odom_msg.pose.pose.position.y = self.goal_pose[1].item()
        # Assuming no change in z and orientation for simplicity
        new_qx, new_qy, new_qz, new_qw = quaternion_from_euler(0.0, 0.0, self.goal_pose[2].item())
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = new_qx
        odom_msg.pose.pose.orientation.y = new_qy
        odom_msg.pose.pose.orientation.z = new_qz
        odom_msg.pose.pose.orientation.w = new_qw
        self.goal_publisher.publish(odom_msg)

    def camera_info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
            rospy.loginfo("Camera intrinsics (K) received from camera_info.")

    def img_callback(self, img_msg):

        self.img_counter += 1
        if self.img_counter > 5:
            self.img_counter = 0
            head = img_msg.header.stamp
            img = self.br.imgmsg_to_cv2(img_msg)

            # Save raw BGR copy for OpenCV trajectory rendering
            self.latest_raw_img = img.copy()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            img = self.transform(img)
            self.img.append(img)

    def poses_callback(self, poses_msg):

        self.odom_counter += 1
        if self.odom_counter > 25:
            self.odom_counter = 0
            pose = poses_msg.pose

            position = pose.pose.position
            orientation = pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

            self.poses = np.roll(self.poses, -1, axis=0)
            self.poses[-1] = (position.x, position.y, yaw)

    def actions_callback(self, actions_msg):
        linear = actions_msg.linear
        angular = actions_msg.angular

        self.actions = np.roll(self.actions, -1, axis=0)
        self.actions[-1] = (linear.x, angular.z)

    def tensor_to_twist_background(self):
        """Continuously publish actions from self.current_action_chunk.
        This runs in a background thread and executes step-by-step."""
        desired_rate = self.config_file.sample_rates.desired_rate
        dt = 1.0 / desired_rate

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            v, w = 0.0, 0.0
            with self.action_lock:
                if self.current_action_chunk is not None and self.action_idx < len(self.current_action_chunk):
                    v, w = self.current_action_chunk[self.action_idx]
                    self.action_idx += 1

            v = np.clip(v, 0.0, self.MAX_V)
            omega = np.clip(w, -self.MAX_W, self.MAX_W)

            twist_msg = Twist()
            twist_msg.linear.x = float(v)
            twist_msg.angular.z = float(omega)
            self.cmd_vel_pub.publish(twist_msg)

            rospy.sleep(dt)

    def publish_predicted_images(self, decoded_frames):
        """
        Publishes the imagined trajectory as a single 'film strip' image for RViz.
        """
        # Convert RGB to BGR for OpenCV and concatenate horizontally
        bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in decoded_frames]
        film_strip = np.concatenate(bgr_frames[:2], axis=1)

        # Scale down if the strip is too wide for RViz to render efficiently
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

    def control_loop(self):
        """Main deployment loop running on the real robot."""

        # Start background publisher thread and keep reference so we can stop it cleanly
        if not self.use_pose:  # Running command velocities
            self.publish_thread = threading.Thread(target=self.tensor_to_twist_background, daemon=True)
            self.publish_thread.start()
        else:
            self.publish_thread = threading.Thread(target=self.pd_controller, daemon=True)
            self.publish_thread.start()

        total_time = 0
        iters = 0
        while not rospy.is_shutdown():
            start = time.time()
            self.publish_goal_pose()
            # 1. Fetch REAL WORLD state from ROS buffers
            buffers = self.get_recent_history_buffers()
            if buffers is None:
                rospy.sleep(0.05)
                continue
            else:
                init_imgs, init_actions, init_poses, raw_poses = buffers

            current_odom = raw_poses[-1]

            # 2. Check terminating condition
            dist_to_goal = torch.norm(current_odom[:2] - self.goal_pose[:2])
            if dist_to_goal < self.goal_tolerance:
                print(f"Average time per iteration: {total_time / iters}")
                rospy.loginfo("Goal Reached!")
                self.stop_robot()
                break

            # Transform global goal_pose into local ego-centric frame for the planner
            odom_x, odom_y, odom_yaw = current_odom[0].item(), current_odom[1].item(), current_odom[2].item()
            gx, gy, gyaw = self.goal_pose[0].item(), self.goal_pose[1].item(), self.goal_pose[2].item()
            dx, dy = gx - odom_x, gy - odom_y
            cos_y, sin_y = np.cos(-odom_yaw), np.sin(-odom_yaw)

            local_gx = dx * cos_y - dy * sin_y
            local_gy = dx * sin_y + dy * cos_y
            local_gyaw = (gyaw - odom_yaw + np.pi) % (2 * np.pi) - np.pi
            local_goal_pose = torch.tensor([local_gx, local_gy, local_gyaw], dtype=torch.float32, device=self.device)

            # 3. Query HydraDLP for the best trajectory given the current state and goal
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                best_traj_dict = self.planner.plan(
                    init_imgs=init_imgs,
                    init_actions=init_actions,
                    init_poses=init_poses,
                    goal_pose=local_goal_pose,
                    goal_latent=self.goal_img
                )

            # 4. Action Chunking Execution
            self.previous_rate = time.time() - start
            print(f"Iteration time: {self.previous_rate}")
            total_time += self.previous_rate
            iters += 1

            new_planned_traj = best_traj_dict["poses"]

            current_real_odom = self.poses[-1]
            odom_x, odom_y, odom_yaw = current_real_odom[0].item(), current_real_odom[1].item(), current_real_odom[2].item()

            # Project new local plan to global frame to compare against the cache
            cos_y_inv, sin_y_inv = np.cos(odom_yaw), np.sin(odom_yaw)
            R_inv = torch.tensor([[cos_y_inv, -sin_y_inv],
                                  [sin_y_inv,  cos_y_inv]], dtype=torch.float32, device=self.device)

            local_poses = new_planned_traj[..., :2].to(self.device)
            global_new_xy = local_poses @ R_inv.T
            global_new_xy[..., 0] += odom_x
            global_new_xy[..., 1] += odom_y

            blend_momentum = 0.6
            accept_new_plan = True

            if self.planner.global_path is not None:
                clean_path = self.planner._slice_path_from_current(self.planner.global_path, torch.tensor(current_real_odom))

                if len(clean_path) > 3:
                    compare_len = min(len(clean_path), len(global_new_xy))
                    cache_segment = clean_path[:compare_len, :2]
                    new_segment = global_new_xy[:compare_len, :2].cpu()

                    # Calculate Euclidean deviation between the new plan and our current locked cache
                    deviation = torch.norm(new_segment - cache_segment, dim=-1).mean().item()
                    deviation_thresh = self.config_file.deployment.get("plan_deviation_threshold", 0.25)
                    min_path_len = self.chunk_size // 2

                    if deviation > deviation_thresh:
                        rospy.loginfo(f"Scene Shift! Plan deviation {deviation:.2f}m. Snapping to new path.")
                        blend_momentum = 0.95  # Snap quickly! Breaks the sluggish obstacle curve memory!
                        accept_new_plan = True
                    elif len(clean_path) > min_path_len:
                        # Scene is static. The new plan agrees with the cache. Track the cache perfectly!
                        if not self.use_pose:
                            with self.action_lock:
                                actions_left = len(self.current_action_chunk) - self.action_idx if self.current_action_chunk is not None else 0
                            if actions_left <= min_path_len:
                                accept_new_plan = True
                                blend_momentum = 0.1 # 0.6
                            else:
                                accept_new_plan = False
                        else:
                            accept_new_plan = False
                    else:
                        # Cache is running low, extend smoothly
                        blend_momentum = 0.6
                        accept_new_plan = True

            # Always record history for RViz
            self.executed_path_history.append(np.array([current_real_odom[0], current_real_odom[1], current_real_odom[2]]))

            # 5. Conditional Execution (Only update control buffers if we accepted the plan)
            if accept_new_plan:
                if not self.use_pose:
                    new_planned_actions = best_traj_dict["actions"]
                    with self.action_lock:
                        if self.current_action_chunk is None:
                            self.current_action_chunk = new_planned_actions[:self.chunk_size].cpu().numpy()
                        else:
                            # Apply the dynamic blend momentum!
                            self.current_action_chunk = (blend_momentum * new_planned_actions[:self.chunk_size].cpu().numpy()) + ((1 - blend_momentum) * self.current_action_chunk)
                        self.action_idx = 0

                self.poses_to_go_to = new_planned_traj[:self.chunk_size]

                # Fuse, Blend, and Extend the path! (transforms to Global Odom)
                self.global_target_poses = [self.planner.ensemble_trajectory(
                    new_local_deltas=new_planned_traj,
                    current_odom=current_real_odom,
                    blend_momentum=blend_momentum  # Dynamic blend breaks sluggishness!
                ).cpu().numpy()]

            # Visualize All Candidate Trajectories
            if hasattr(self.planner, 'latest_all_poses') and self.planner.latest_all_poses is not None:
                # Get the Odom orientation to project local tracks back to global space
                odom_x, odom_y, odom_yaw = current_real_odom[0].item(), current_real_odom[1].item(), current_real_odom[2].item()
                cos_y_inv, sin_y_inv = np.cos(odom_yaw), np.sin(odom_yaw)
                R_inv = torch.tensor([[cos_y_inv, -sin_y_inv],
                                      [sin_y_inv,  cos_y_inv]], dtype=torch.float32, device=self.device)

                all_poses_local = self.planner.latest_all_poses.to(self.device)
                global_all_poses = all_poses_local[..., :2] @ R_inv.T
                global_all_poses[..., 0] += odom_x
                global_all_poses[..., 1] += odom_y
                self.publish_all_poses(global_all_poses.cpu().numpy(), current_real_odom)

            # Save imagined trajectory video if debugging is enabled
            if self.debug_visualize:
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    images_to_save = self.planner.get_imagination()

                if images_to_save is not None:
                    h_prime, w_prime = self.resize[0] // 8, self.resize[1] // 8

                    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                        decoded_frames = self.model.transformer_encoder.unified_encoder.visual_encoder.decode(
                            images_to_save, hw_shape=(h_prime, w_prime)
                        )

                    # Denormalize [-1, 1] -> [0, 255]
                    decoded_frames = ((decoded_frames + 1.0) / 2.0).clamp(0, 1).float()
                    decoded_frames = (decoded_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

                    os.makedirs("debug_videos", exist_ok=True)
                    os.makedirs(os.path.join("debug_videos", self.experiment_name), exist_ok=True)
                    video_path = os.path.join("debug_videos", self.experiment_name, f"planned_traj_{int(time.time())}.mp4")

                    height, width, _ = decoded_frames[0].shape
                    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))

                    for frame in decoded_frames:
                        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    video_writer.release()
                    rospy.loginfo(f"Saved debug video to {video_path}")

                    # Broadcast real-time film strip to RViz
                    self.publish_predicted_images(decoded_frames)

            # Publish 3D Projected Trajectory over Real RGB Feed
            if self.K is not None and self.latest_raw_img is not None:
                # We plot the full sequence new_planned_traj, not just the chunk, to see the full plan!
                full_planned_deltas = new_planned_traj
                # T_cam_base projection matrix expects absolute paths in the local ego-frame, not step-wise deltas
                local_abs_path_se2 = torch.cumsum(full_planned_deltas[..., :2], dim=0)
                local_abs_yaw_se2 = torch.cumsum(full_planned_deltas[..., 2], dim=0) if full_planned_deltas.shape[-1] == 3 else torch.zeros_like(local_abs_path_se2[:, 0])
                local_path_se2 = torch.cat([local_abs_path_se2, local_abs_yaw_se2.unsqueeze(-1)], dim=-1)

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
                        os.makedirs(os.path.join("debug_videos", self.experiment_name), exist_ok=True)
                        video_path = os.path.join("debug_videos", self.experiment_name, f"experiment_rollout_{int(time.time())}.mp4")
                        height, width = projected_img_bgr.shape[:2]
                        self.experiment_video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
                        self.video_thread = threading.Thread(target=self._video_write_worker, daemon=True)
                        self.video_thread.start()

                    self.video_queue.put(projected_img_bgr)

                if hasattr(self, 'projected_traj_pub'):
                    try:
                        img_msg = self.br.cv2_to_imgmsg(projected_img_bgr, encoding="bgr8")
                        self.projected_traj_pub.publish(img_msg)
                    except Exception as e:
                        rospy.logwarn(f"Failed to publish projected trajectory: {e}")

            # Publish the single best predicted path to RViz
            global_path_xy = self.global_target_poses[0]

            # Stitch the historical trail with the current future cache
            if len(self.executed_path_history) > 0:
                history_arr = np.array(self.executed_path_history)
                history_arr = history_arr[:, :global_path_xy.shape[1]]
                full_rviz_path = np.vstack([history_arr, global_path_xy])
            else:
                full_rviz_path = global_path_xy

            self.publish_predicted_poses(full_rviz_path)

    def get_recent_history_buffers(self):
        if len(self.img) < self.context_length or len(self.poses) < self.context_length + 1 or len(
                self.actions) < self.context_length:
            return None  # Not enough data yet

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

        return (
            imgs_tensor.unsqueeze(0),
            actions_tensor.unsqueeze(0),
            poses_tensor.unsqueeze(0),
            raw_poses
        )

    def _video_write_worker(self):
        """Background thread that empties the video queue to disk without blocking the control loop."""
        while not rospy.is_shutdown():
            try:
                frame = self.video_queue.get(timeout=0.5)
                if frame is None:
                    break
                if getattr(self, "experiment_video_writer", None) is not None:
                    self.experiment_video_writer.write(frame)
            except queue.Empty:
                if self.stop_event.is_set():
                    break

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
            if getattr(self, "video_thread", None) is not None and self.video_thread.is_alive():
                self.video_thread.join(timeout=3.0)
            self.experiment_video_writer.release()
            rospy.loginfo("Saved full experiment projected video.")

    def _on_shutdown(self):
        # rospy shutdown handler: ensure background thread exits and robot stops
        rospy.loginfo("HydraControlNode shutting down: stopping publish thread")
        self.stop_robot()


def main(args):
    control_node = HydraControlNode(
        args.conf, args.exp_name, args.robot, use_goal_conditioned_seed=args.goal_conditioned
    )
    rospy.init_node('Hydra_Command_Publisher', anonymous=True)
    # Ensure our node's shutdown handler is called on ROS shutdown
    rospy.on_shutdown(control_node._on_shutdown)

    control_node.control_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--conf", default="deployment/conf/hydra_plan.yaml", type=str)
    parser.add_argument(
        "--robot",
        default="jackal",
        type=str,
        help="Robot type. Add new robot specs under deployment/conf/robots/<robot>.yaml.",
    )
    parser.add_argument("--exp_name", default=f"hydra_planning_{date_str}", type=str)
    parser.add_argument(
        "--goal_conditioned",
        action="store_true",
        default=False,
        help="Seed the latent MPC search toward the goal direction (normalized, scaled by the "
             "robot's recent step magnitude) instead of the unbiased model-queried prior. "
             "See HydraLatentMPC.use_goal_conditioned_seed in dlp.py.",
    )
    args = parser.parse_args()
    main(args)