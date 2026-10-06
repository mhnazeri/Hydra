"""Utility functions related to vision"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image
import torchvision.utils as vutils
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
# from transforms3d.euler import euler2mat


def _motionvec_to_rotation(motionvec: np.ndarray) -> np.ndarray:
    """Create a rotation matrix from a motion vector

    Args:
        motionvec (np.ndarray): motion vector in (x, y, z, r, p, y)

    Returns:
        np.ndarray: rotation matrix
    """
    if isinstance(motionvec, torch.Tensor):
        motionvec = motionvec.numpy()

    translation = motionvec[:3]
    rot = R.from_rotvec(motionvec[3:])
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rot.as_matrix()
    rot_matrix[:3, 3] = translation
    return rot_matrix


def _motion_to_pose(motions):
    """
    Transforms a sequence of pose deltas into actual poses.

    Args:
        motions: List of pose deltas.

    Returns:
        List of poses (4x4 transformation matrices).
    """
    # initial pose is always 0, 0, 0 (robot has not moved yet)
    initial_pose = _motionvec_to_rotation(np.array([0, 0, 0, 0, 0, 0]))
    poses = [initial_pose]
    current_pose = initial_pose
    for delta in motions:
        current_pose = current_pose @ _motionvec_to_rotation(
            delta
        )  # Matrix multiplication
        poses.append(current_pose)

    return poses


def _extract_coordinates(poses: list) -> np.ndarray:
    """
    Extracts (x, y, z) coordinates from a list of 4x4 transformation matrices.

    Returns:
        np.ndarray: Array of shape (T, 3)
    """
    # Extract the translation component (last column, first three rows)
    coords = np.array([pose[:3, 3] for pose in poses])
    return coords


def visualize_trajectory(
    poses: torch.Tensor, gt_poses: torch.Tensor = None, orientation: bool = False
):
    """
    Visualizes a sequence of robot poses in 3D with color coding based on time.

    Args:
        poses: A list of tuples, each representing a robot predicted motion in the format (x, y, z, roll, pitch, yaw).
        gt_poses: A list of tuples, each representing a robot actual motion in the format (x, y, z, roll, pitch, yaw).
        orientation: whether to visualize the orientation or not
    Example:
        poses = [
                    (0, 0, 0, 0, 0, 0),
                    (1, 1, 0, 0, 0, np.pi/4),
                    (2, 2, 1, 0, np.pi/4, 0),
                    (3, 3, 2, np.pi/4, 0, 0)
                ]
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().detach().squeeze().tolist()
    if (gt_poses is not None) and isinstance(gt_poses, torch.Tensor):
        gt_poses = gt_poses.cpu().detach().squeeze().tolist()

    # transform motion to pose
    poses = _motion_to_pose(poses)

    coords = _extract_coordinates(poses)

    if gt_poses is not None:
        gt_poses = _motion_to_pose(gt_poses)
        coords_gt = _extract_coordinates(gt_poses)

    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c="red",
        marker="^",
        markersize=4,
        linestyle="dashed",
        linewidth=2,
        label="Predicted Trajectory",
    )

    if gt_poses is not None:
        ax.plot(
            coords_gt[:, 0],
            coords_gt[:, 1],
            coords_gt[:, 2],
            c="green",
            marker="v",
            markersize=4,
            linestyle="solid",
            linewidth=2,
            label="GroundTruth Trajectory",
        )

    # Optionally, visualize robot orientations at some key poses
    if orientation:
        for i in range(1, len(poses), 1):  # skip the initial pose
            pose = poses[i]
            u, v, w = pose[:, 0], pose[:, 1], pose[:, 2]  # Get axis vectors
            scale = 0.1  # Adjust scale as needed
            ax.quiver(
                pose[0, -1],
                pose[1, -1],
                pose[2, -1],
                u[0] * scale,
                u[1] * scale,
                u[2] * scale,
                color="blue",
            )
            ax.quiver(
                pose[0, -1],
                pose[1, -1],
                pose[2, -1],
                v[0] * scale,
                v[1] * scale,
                v[2] * scale,
                color="red",
            )
            ax.quiver(
                pose[0, -1],
                pose[1, -1],
                pose[2, -1],
                w[0] * scale,
                w[1] * scale,
                w[2] * scale,
                color="green",
            )

            if gt_poses is not None:
                gt_pose = gt_poses[i]
                # You might need to adjust the scale of the orientation visualization
                u_gt, v_gt, w_gt = (
                    gt_pose[:, 0],
                    gt_pose[:, 1],
                    gt_pose[:, 2],
                )  # Get axis vectors
                ax.quiver(
                    gt_pose[0, -1],
                    gt_pose[1, -1],
                    gt_pose[2, -1],
                    u_gt[0] * scale,
                    u_gt[1] * scale,
                    u_gt[2] * scale,
                    color="blue",
                )
                ax.quiver(
                    gt_pose[0, -1],
                    gt_pose[1, -1],
                    gt_pose[2, -1],
                    v_gt[0] * scale,
                    v_gt[1] * scale,
                    v_gt[2] * scale,
                    color="red",
                )
                ax.quiver(
                    gt_pose[0, -1],
                    gt_pose[1, -1],
                    gt_pose[2, -1],
                    w_gt[0] * scale,
                    w_gt[1] * scale,
                    w_gt[2] * scale,
                    color="green",
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Dynamically compute bounds to keep aspect ratio proportional
    all_coords = coords
    if gt_poses is not None:
        all_coords = np.concatenate([all_coords, coords_gt], axis=0)

    max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(),
                          all_coords[:, 1].max() - all_coords[:, 1].min(),
                          all_coords[:, 2].max() - all_coords[:, 2].min()]).max() / 2.0
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    
    max_range = max_range * 1.2 if max_range > 0 else 1.0
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    fig.tight_layout()
    return fig


def visualize_trajectory_2d(
    poses: torch.Tensor, gt_poses: torch.Tensor = None, orientation: bool = False
):
    """
    Visualizes a sequence of robot poses in 2D with color coding based on time.

    Args:
        poses: A list of tuples, each representing a robot predicted motion in the format (x, y, z, roll, pitch, yaw).
        gt_poses: A list of tuples, each representing a robot actual motion in the format (x, y, z, roll, pitch, yaw).
        orientation: whether to visualize the orientation or not
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    if isinstance(poses, torch.Tensor):
        poses = poses.cpu().detach().squeeze().tolist()
    if (gt_poses is not None) and isinstance(gt_poses, torch.Tensor):
        gt_poses = gt_poses.cpu().detach().squeeze().tolist()

    # transform motion to pose
    def get_xy(p):
        p = np.array(p)
        x = p[:, 0]
        y = p[:, 1]
        return x, y

    x, y = get_xy(poses)


    if gt_poses is not None:
        x_gt, y_gt = get_xy(gt_poses)

    ax.plot(
        x, y,
        c="red",
        marker="^",
        markersize=4,
        linestyle="dashed",
        linewidth=2,
        label="Predicted Trajectory",
    )

    if gt_poses is not None:
        ax.plot(
            x_gt, y_gt,
            c="green",
            marker="v",
            markersize=4,
            linestyle="solid",
            linewidth=2,
            label="GroundTruth Trajectory",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Dynamically compute bounds to keep aspect ratio proportional
    all_x = x
    all_y = y
    if gt_poses is not None:
        all_x = np.concatenate([all_x, x_gt])
        all_y = np.concatenate([all_y, y_gt])

    max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2.0
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5

    max_range = max_range * 1.2 if max_range > 0 else 1.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_aspect('equal', adjustable='box')

    ax.legend()
    fig.tight_layout()
    return fig


def visualize_hydra_attention(image_tensor, attention_weights, output_size=(224, 224)):
    """
    Visualizes the spatial attention of Hydra model on top of an image.

    Args:
        image_tensor: [3, H, W] tensor
        attention_weights: [49] or [H_feat, W_feat] tensor (e.g., 7x7)
        output_size: (width, height) for the resized heatmap overlay
    Returns:
        overlay: numpy array of the overlayed image in RGB
    """
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image_tensor

    # Map from [-1, 1] to [0, 1] if necessary to prevent dark, clipped backgrounds
    if img_np.min() < 0.0:
        img_np = (img_np + 1.0) / 2.0
    img_np = np.clip(img_np, 0.0, 1.0)

    img_cv = (img_np * 255).astype(np.uint8)

    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.cpu().numpy()
    else:
        attn = attention_weights

    grid_size = int(np.sqrt(attn.size))
    attn = attn.reshape(grid_size, grid_size)

    # Normalize attention
    if attn.max() > attn.min():
        attn = (attn - attn.min()) / (attn.max() - attn.min())
    attn = (attn * 255).astype(np.uint8)

    # Upsample
    heatmap = cv2.resize(attn, output_size, interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
    return overlay


def visualize_cross_attention(attn_weights: np.ndarray, title="Cross-Attention (Future vs Past)"):
    """
    Visualizes the cross-attention matrix between future tokens and past observations.
    Args:
        attn_weights: numpy array of shape [T_future, T_past]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(attn_weights, cmap="viridis", aspect="auto")
    fig.colorbar(cax, ax=ax, label="Attention Weight")

    ax.set_xlabel("Observation History Steps (Past)")
    ax.set_ylabel("Future Generation Steps")
    ax.set_title(title)

    ax.set_xticks(np.arange(attn_weights.shape[1]))
    ax.set_yticks(np.arange(attn_weights.shape[0]))

    return fig