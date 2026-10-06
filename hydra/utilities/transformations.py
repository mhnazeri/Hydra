import math

import cv2
import numpy as np
import torch


K_AZURE_COLOR_1080P = np.array([
    [975.0,   0.0, 960.0],   # fx, 0, cx
    [  0.0, 975.0, 540.0],   # 0, fy, cy
    [  0.0,   0.0,   1.0]
], dtype=np.float32)


def homography_crop_resize(org_img_size=(1920, 1080), crop_y=1, resize_img_size=(320, 240)):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                        [0, ratio_y, -ratio_y * crop_y],
                        [0, 0, 1]])
    return H_c


def world2pixel(xy):
    camera_height = 0.55  # meters
    camera_x_offset = 0.35  # distance between the center of the robot and the forward facing camera

    if len(xy) > 2:
        xy = xy[:, :2]

    horizon = xy.shape[0]

    camera_matrix = np.array([[975.0,   0.0, 960.0],
                              [0.0, 975.0, 540.0],
                              [0., 0., 1.]])
    camera_matrix = homography_crop_resize() @ camera_matrix


    dist_coeffs = np.array([0.6671934723854065, -3.1600592136383057, 0.0004678548430092633,
    -0.0004770300292875618, 1.8223274946212769, 0.5343461036682129, -2.9515035152435303, 1.7315576076507568]).astype(float)

    xyz = np.hstack([xy, -camera_height * np.ones(xy.shape[0]).reshape(horizon, 1)])

    tvec = (0, 0, 0)

    rvec = (0, 0, 0)
    xyz[..., 0] += camera_x_offset
    # opencv’s assumption is x right, y down (screen/camera), z out.
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)

    uv, _ = cv2.projectPoints(
        xyz_cv, rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(horizon, 2)
    u, v = uv[:, 0], uv[:, 1]
    return u, v


def rotation_matrix(translation, rotation):
    if translation.numel() < 3:
        temp = torch.zeros(3, dtype=translation.dtype, device=translation.device)
        temp[:2] = translation
        translation = temp

    cos_r, sin_r = torch.cos(rotation), torch.sin(rotation)
    rot = torch.zeros((3, 3), dtype=translation.dtype, device=translation.device)
    rot[0, 0] = cos_r
    rot[0, 1] = -sin_r
    rot[1, 0] = sin_r
    rot[1, 1] = cos_r
    rot[2, 2] = 1.0

    rot_matrix = torch.eye(4, dtype=translation.dtype, device=translation.device)
    rot_matrix[:3, :3] = rot
    rot_matrix[:3, 3] = translation
    return rot_matrix

def transform_robot_centric(past_positions, past_yaw, future_positions, future_yaw):
    """
    Transforms world-centric past and future positions to be robot-centric.
    """
    psi_odom_current = rotation_matrix(past_positions[0][:3], past_yaw[0])
    psi_current_odom = torch.linalg.inv(psi_odom_current)

    future_positions_transformed = [rotation_matrix(future_positions[i][:3], future_yaw[i])
                                    for i in range(len(future_positions))]
    past_positions_transformed = [rotation_matrix(past_positions[i][:3], past_yaw[i])
                                  for i in range(len(past_positions))]

    future_positions_transformed = torch.stack(future_positions_transformed, dim=0)
    past_positions_transformed = torch.stack(past_positions_transformed, dim=0)

    future_positions_robot_centric = psi_current_odom @ future_positions_transformed
    past_positions_robot_centric = psi_current_odom @ past_positions_transformed

    future_positions_final = future_positions_robot_centric[:, :2, 3]
    past_positions_final = past_positions_robot_centric[:, :2, 3]

    return past_positions_final, future_positions_final, # TODO: return yaw as well


def cartesian_to_polar(x: float, y: float) -> torch.Tensor:
    """Convert cartesian coordinates to polar coordinates
    x: (float) x direction
    y: (float) y direction
    """
    # Calculating radius
    radius = math.sqrt(x * x + y * y)
    # Calculating angle (theta) in radian
    theta = math.atan2(y, x)
    return torch.Tensor([radius, theta]).float()


def world_to_robot_centric(world_point: np.ndarray, robot_pose_se3: np.ndarray) -> np.ndarray:
    """
    Transforms a single point from the world frame to the robot's local frame.

    Args:
        world_point (np.ndarray): A 2D or 3D point in the world frame, e.g., [x, y].
        robot_pose_se3 (np.ndarray): A 4x4 SE(3) transformation matrix representing
                                     the robot's pose in the world frame.

    Returns:
        np.ndarray: The point transformed into the robot's local reference frame.
    """
    # Create a 4D homogeneous coordinate for the point
    point_h = np.ones(4)
    point_h[:len(world_point)] = world_point
    # Apply the inverse transformation
    return (np.linalg.inv(robot_pose_se3) @ point_h)[:len(world_point)]

def _project_points_xy_to_pixels(xy_base_m, K, T_cam_from_base, image_width, image_height):
    """
    Projects robot-centric (x,y) points to pixel coordinates, filtering for those
    that are both in front of the camera and within the image boundaries.

    Returns:
        np.ndarray: An (M,2) array of visible integer pixel coordinates.
    """
    N = xy_base_m.shape[0]
    xyz1 = np.column_stack([xy_base_m[:,0], xy_base_m[:,1], np.zeros(N, dtype=np.float32), np.ones(N, dtype=np.float32)])  # (N,4)
    xyz_cam = (T_cam_from_base @ xyz1.T).T[:, :3]  # (N,3)
    Z = xyz_cam[:, 2]
    front_mask = Z > 1e-6
    if not np.any(front_mask):
        return np.empty((0,2), dtype=int)

    # Project only points in front of the camera
    x = xyz_cam[front_mask, 0] / Z[front_mask]
    y = xyz_cam[front_mask, 1] / Z[front_mask]
    u = K[0,0]*x + K[0,2]
    v = K[1,1]*y + K[1,2]

    # Filter for points within image boundaries
    boundary_mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    visible_pts = np.stack([u[boundary_mask], v[boundary_mask]], axis=1).round().astype(int)
    return visible_pts

def _scale_K_to_image(K_base: np.ndarray, base_size: tuple, image_size: tuple) -> np.ndarray:
    """
    Scale intrinsics from base_size (W0,H0) to image_size (W,H).
    """
    W0, H0 = base_size
    W,  H  = image_size
    sx, sy = W / float(W0), H / float(H0)
    K = K_base.copy()
    K[0,0] *= sx  # fx
    K[1,1] *= sy  # fy
    K[0,2] *= sx  # cx
    K[1,2] *= sy  # cy
    return K

def get_K_azure_color(image_width: int, image_height: int) -> np.ndarray:
    """Ready-to-use K for Azure Kinect color camera at (image_width, image_height)."""
    return _scale_K_to_image(K_AZURE_COLOR_1080P, (1920, 1080), (image_width, image_height))


def make_T_cam_from_base(height_m=0.90, x_forward_m=0.20, y_left_m=0.00, pitch_deg=0.0):
    """
    Base frame:  x forward, y left, z up
    Camera frame: x right,  y down, z forward (OpenCV)
    Rotation maps base->cam, then an optional pitch about camera x.
    Translation is in CAMERA coords: [x_cam, y_cam, z_cam] = [y_left, height, x_forward]
    """
    R0 = np.array([[0.,  -1.,  0.],
                   [0.,  0., -1.],
                   [1.,  0.,  0.]], dtype=np.float32)  # x_cam=y_base, y_cam=-z_base, z_cam=x_base
    th = math.radians(pitch_deg)  # negative -> pitch camera downward
    Rx = np.array([[1., 0.,        0.       ],
                   [0., math.cos(th), -math.sin(th)],
                   [0., math.sin(th),  math.cos(th)]], dtype=np.float32)
    R = Rx @ R0
    t = np.array([y_left_m, height_m, x_forward_m], dtype=np.float32)
    T = np.zeros((3,4), dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = t
    return T


def positions_to_deltas(positions):
    """
    Convert an (T,2) or (T,3) sequence to per-step deltas.
    Keeps length by prepending a zero row.
    """
    deltas = positions[1:] - positions[:-1]
    zero = torch.zeros_like(positions[:1])
    return torch.cat([zero, deltas], dim=0)


def yaw_to_deltas(yaw):
    deltas = torch.zeros_like(yaw)
    deltas[0] = yaw[0]  # First element relative to the origin
    deltas[1:] = yaw[1:] - yaw[:-1]
    return (deltas + torch.pi) % (2 * torch.pi) - torch.pi