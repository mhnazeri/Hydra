import torch

def compute_mae(pred, gt):
    """
    Mean Absolute Error (MAE) across all predicted dimensions and time steps.

    Args:
        pred: Predicted trajectories of shape (B, T, D)
        gt: Ground truth trajectories of shape (B, T, D)

    Returns:
        Tensor of shape (B,) with the MAE for each sample.
    """
    return torch.abs(pred - gt).mean(dim=(1, 2))

def compute_ade(pred, gt):
    """
    Average Displacement Error (ADE).
    Mean Euclidean distance between predicted and ground truth absolute positions (x, y)
    over all time steps.

    Args:
        pred: Predicted trajectories of shape (B, T, D) (absolute)
        gt: Ground truth trajectories of shape (B, T, D) (absolute)

    Returns:
        Tensor of shape (B,) with the ADE for each sample.
    """
    dist = torch.norm(pred[..., :2] - gt[..., :2], dim=-1) # (B, T)
    return dist.mean(dim=1)

def compute_fde(pred, gt):
    """
    Final Displacement Error (FDE).
    Euclidean distance between predicted and ground truth absolute positions (x, y)
    at the final time step.

    Args:
        pred: Predicted trajectories of shape (B, T, D) (absolute)
        gt: Ground truth trajectories of shape (B, T, D) (absolute)

    Returns:
        Tensor of shape (B,) with the FDE for each sample.
    """
    dist = torch.norm(pred[:, -1, :2] - gt[:, -1, :2], dim=-1) # (B,)
    return dist

def compute_ate(pred, gt):
    """
    Absolute Trajectory Error (ATE).
    Root Mean Squared Error of the Euclidean distances over all time steps.

    Args:
        pred: Predicted trajectories of shape (B, T, D) (absolute)
        gt: Ground truth trajectories of shape (B, T, D) (absolute)

    Returns:
        Tensor of shape (B,) with the ATE for each sample.
    """
    dist_sq = torch.sum((pred[..., :2] - gt[..., :2]) ** 2, dim=-1) # (B, T)
    rmse = torch.sqrt(dist_sq.mean(dim=1)) # (B,)
    return rmse

def compute_rpe(pred, gt):
    """
    Relative Pose Error (RPE).
    Measures local trajectory consistency. Since the inputs are ALREADY step-to-step
    displacements (deltas), RPE is simply the error directly on these inputs.

    Args:
        pred: Predicted trajectories of shape (B, T, D) (deltas)
        gt: Ground truth trajectories of shape (B, T, D) (deltas)

    Returns:
        Tensor of shape (B,) with the RPE for each sample.
    """
    dist = torch.norm(pred[..., :2] - gt[..., :2], dim=-1) # (B, T)
    return dist.mean(dim=1)