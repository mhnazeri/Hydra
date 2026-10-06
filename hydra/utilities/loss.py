import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LatentSpatialLoss(nn.Module):
    """
    Computes spatial gradients directly in the latent space.
    Forces the deterministic decoder to synthesize high-frequency structural edges,
    counteracting the blurriness of standard MSE.
    """
    def __init__(self, channels: int = 4):
        super().__init__()
        # Pre-compute Sobel operators for the latent grid
        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]])
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]])

        # Shape: [Channels, 1, 3, 3]
        self.register_buffer('weight_x', kernel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer('weight_y', kernel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    @torch.compiler.disable
    def forward(self, pred_latents: torch.Tensor, target_latents: torch.Tensor, h_patches: int, w_patches: int, patch_size: int = 2):
        # input shape: [B, T, K, D]
        b, t, k, d = pred_latents.shape
        c = d // (patch_size ** 2)

        pred = rearrange(pred_latents, 'b t (h w) (c p1 p2) -> (b t) c (h p1) (w p2)', h=h_patches, w=w_patches, p1=patch_size, p2=patch_size, c=c)
        targ = rearrange(target_latents, 'b t (h w) (c p1 p2) -> (b t) c (h p1) (w p2)', h=h_patches, w=w_patches, p1=patch_size, p2=patch_size, c=c)

        # Pad to prevent edge artifacts
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode='replicate')
        targ_pad = F.pad(targ, (1, 1, 1, 1), mode='replicate')

        pred_grad_x = F.conv2d(pred_pad, self.weight_x.to(pred.dtype), groups=c)
        pred_grad_y = F.conv2d(pred_pad, self.weight_y.to(pred.dtype), groups=c)

        targ_grad_x = F.conv2d(targ_pad, self.weight_x.to(targ.dtype), groups=c)
        targ_grad_y = F.conv2d(targ_pad, self.weight_y.to(targ.dtype), groups=c)

        loss_x = F.l1_loss(pred_grad_x, targ_grad_x)
        loss_y = F.l1_loss(pred_grad_y, targ_grad_y)

        return loss_x + loss_y


class MultiTaskCriterion(nn.Module):
    def __init__(self, num_tasks=3, use_auto_weighting=False, loss_weights=None):
        super().__init__()
        self.use_auto_weighting = use_auto_weighting

        # Learnable log variances for automated loss balancing
        if self.use_auto_weighting:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            loss_weights = loss_weights or {"actions": 1.0, "poses": 1.0, "images": 1.0}
            self.lambda_actions = loss_weights.get("actions", 1.0)
            self.lambda_poses = loss_weights.get("poses", 1.0)
            self.lambda_images = loss_weights.get("images", 1.0)

        # Loss components
        self.l1_loss = nn.L1Loss()  # Robust to outliers in actions
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets, vq_loss, aux=None):
        """
        preds: Tuple(pred_act, pred_pose, rec_img)
        targets: Tuple(target_act, target_pose, target_img)
        vq_loss: scalar VQ loss from the VQ-VAE bottleneck
        aux: auxiliary outputs from decoders (e.g., for flow)
        """
        p_act, p_pose, p_img = preds
        t_act, t_pose, t_img = targets

        zero = torch.tensor(0.0, device=vq_loss.device, dtype=vq_loss.dtype)

        # Action Loss (L1)
        has_act_loss = aux is not None and "fm_loss_act" in aux
        if has_act_loss:
            loss_act = aux["fm_loss_act"]
        else:
            loss_act = self.l1_loss(p_act, t_act) if p_act is not None and t_act is not None else zero

        # Pose Loss (MSE)
        has_pose_loss = aux is not None and "fm_loss_pose" in aux
        if has_pose_loss:
            loss_pose = aux["fm_loss_pose"]
        elif p_pose is not None and t_pose is not None:
            p_pos, p_yaw = p_pose[..., :2], p_pose[..., 2]
            t_pos, t_yaw = t_pose[..., :2], t_pose[..., 2]

            loss_pos = self.mse_loss(p_pos, t_pos)
            yaw_diff = (p_yaw - t_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            loss_yaw = torch.mean(yaw_diff ** 2)

            # Weighted average correctly: (x, y) = 2 parts, yaw = 1 part
            loss_pose = (loss_pos * 2.0 + loss_yaw) / 3.0
        else:
            loss_pose = zero

        # Image Loss
        has_img_loss = aux is not None and "fm_loss" in aux
        if has_img_loss:
            loss_img = aux["fm_loss"]
        elif p_img is not None and t_img is not None:
            # L1
            l1_loss = self.l1_loss(p_img, t_img)
            # Cosine Similarity forces the latent vector to point in the correct semantic direction
            cos_loss = 1.0 - torch.nn.functional.cosine_similarity(p_img, t_img, dim=-1).mean()
            # Variance matching
            std_p = torch.std(p_img, dim=-1)
            std_t = torch.std(t_img, dim=-1)
            std_loss = self.l1_loss(std_p, std_t)

            loss_img = l1_loss + 0.1 * cos_loss + 0.1 * std_loss
        else:
            loss_img = zero

        total_loss = vq_loss

        # Determine which branches successfully computed a valid loss
        valid_act = has_act_loss or (p_act is not None and t_act is not None)
        valid_pose = has_pose_loss or (p_pose is not None and t_pose is not None)
        valid_img = has_img_loss or (p_img is not None and t_img is not None)

        # Uncertainty Weighting
        # Task 0: Action, Task 1: Pose, Task 2: Image
        if self.use_auto_weighting:
            if valid_act:
                total_loss = total_loss + loss_act * torch.exp(-self.log_vars[0]) + self.log_vars[0]
            if valid_pose:
                total_loss = total_loss + loss_pose * torch.exp(-self.log_vars[1]) + self.log_vars[1]
            if valid_img:
                total_loss = total_loss + loss_img * torch.exp(-self.log_vars[2]) + self.log_vars[2]
        else:
            if valid_act:
                total_loss = total_loss + loss_act * self.lambda_actions
            if valid_pose:
                total_loss = total_loss + loss_pose * self.lambda_poses
            if valid_img:
                total_loss = total_loss + loss_img * self.lambda_images

        return total_loss, {
            "act_loss": loss_act,
            "pose_loss": loss_pose,
            "img_loss": loss_img,
            "vq_loss": vq_loss,
        }


class VertiFormerMultiTaskCriterion(nn.Module):
    def __init__(self, num_tasks=3, use_auto_weighting=False, loss_weights=None, cfg=None):
        super().__init__()
        self.use_auto_weighting = use_auto_weighting

        # Learnable log variances for automated loss balancing
        if self.use_auto_weighting:
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            loss_weights = loss_weights or {"actions": 1.0, "poses": 1.0, "images": 1.0}
            self.lambda_actions = loss_weights.get("actions", 1.0)
            self.lambda_poses = loss_weights.get("poses", 1.0)
            self.lambda_images = loss_weights.get("images", 1.0)

        self.cfg = cfg
        self.lambda_spatial = 0.5
        if self.cfg is not None:
            self.lambda_spatial = self.cfg.train_params.loss_weights.get("spatial", 0.5)
            self.patch_size = self.cfg.model.encoder.get("patch_size", 2)
            image_size = self.cfg.model.encoder.get("image_size", [224, 224])
            self.h_patches = image_size[0] // 8 // self.patch_size
            self.w_patches = image_size[1] // 8 // self.patch_size
        else:
            self.patch_size = 2
            self.h_patches = 14
            self.w_patches = 14

        # Loss components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.spatial_loss = LatentSpatialLoss(channels=4)

    def forward(self, preds, targets, vq_loss, aux=None):
        """
        preds: Tuple(pred_act, pred_pose, rec_img)
        targets: Tuple(target_act, target_pose, target_img)
        vq_loss: scalar VQ loss from the VQ-VAE bottleneck
        aux: auxiliary outputs from decoders (e.g., for flow)
        """
        p_act, p_pose, p_img = preds
        t_act, t_pose, t_img = targets

        zero = torch.tensor(0.0, device=vq_loss.device, dtype=vq_loss.dtype)

        # Action Loss
        has_act_loss = aux is not None and "fm_loss_act" in aux
        if has_act_loss:
            loss_act = aux["fm_loss_act"]
        else:
            loss_act = self.l1_loss(p_act, t_act) if p_act is not None and t_act is not None else zero

        # Pose Loss
        has_pose_loss = aux is not None and "fm_loss_pose" in aux
        if has_pose_loss:
            loss_pose = aux["fm_loss_pose"]
        elif p_pose is not None and t_pose is not None:
            p_pos, p_yaw = p_pose[..., :2], p_pose[..., 2]
            t_pos, t_yaw = t_pose[..., :2], t_pose[..., 2]

            loss_pos = self.mse_loss(p_pos, t_pos)
            yaw_diff = (p_yaw - t_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            loss_yaw = torch.mean(yaw_diff ** 2)

            # Weighted average correctly: (x, y) = 2 parts, yaw = 1 part
            loss_pose = (loss_pos * 2.0 + loss_yaw) / 3.0
        else:
            loss_pose = zero

        # Image Loss
        has_img_loss = aux is not None and "fm_loss" in aux
        spatial_loss_val = zero
        if has_img_loss:
            loss_img = aux["fm_loss"]
        elif p_img is not None and t_img is not None:
            l1_loss = self.l1_loss(p_img, t_img)
            cos_loss = 1.0 - torch.nn.functional.cosine_similarity(p_img, t_img, dim=-1).mean()
            std_p = torch.std(p_img, dim=-1)
            std_t = torch.std(t_img, dim=-1)
            std_loss = self.l1_loss(std_p, std_t)

            spatial_loss_val = self.spatial_loss(p_img, t_img, self.h_patches, self.w_patches, self.patch_size)

            loss_img = l1_loss + 0.1 * cos_loss + 0.1 * std_loss + self.lambda_spatial * spatial_loss_val
        else:
            loss_img = zero

        total_loss = vq_loss

        # Determine which branches successfully computed a valid loss
        valid_act = has_act_loss or (p_act is not None and t_act is not None)
        valid_pose = has_pose_loss or (p_pose is not None and t_pose is not None)
        valid_img = has_img_loss or (p_img is not None and t_img is not None)

        # Uncertainty Weighting
        # Task 0: Action, Task 1: Pose, Task 2: Image
        if self.use_auto_weighting:
            if valid_act:
                total_loss = total_loss + loss_act * torch.exp(-self.log_vars[0]) + self.log_vars[0]
            if valid_pose:
                total_loss = total_loss + loss_pose * torch.exp(-self.log_vars[1]) + self.log_vars[1]
            if valid_img:
                total_loss = total_loss + loss_img * torch.exp(-self.log_vars[2]) + self.log_vars[2]
        else:
            if valid_act:
                total_loss = total_loss + loss_act * self.lambda_actions
            if valid_pose:
                total_loss = total_loss + loss_pose * self.lambda_poses
            if valid_img:
                total_loss = total_loss + loss_img * self.lambda_images

        return total_loss, {
            "act_loss": loss_act,
            "pose_loss": loss_pose,
            "img_loss": loss_img,
            "vq_loss": vq_loss,
            "spatial_loss": spatial_loss_val,
        }