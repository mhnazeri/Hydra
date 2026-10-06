import torch
import torch.nn as nn

class RoboticsCurriculumAugmenter(nn.Module):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def _get_prob(self, epoch: int, max_prob: float) -> float:
        # Linear curriculum over the first 50% of training
        progress = min(1.0, epoch / (self.total_epochs * 0.5 + 1e-6))
        return max_prob * progress

    @torch.no_grad()
    def forward(self, past_frames, past_vw, past_positions, epoch: int):
        B, T, C, H, W = past_frames.shape
        device = past_frames.device

        # Clone to preserve original data
        past_frames = past_frames.clone()
        if past_vw is not None: past_vw = past_vw.clone()
        if past_positions is not None: past_positions = past_positions.clone()

        # Temporally-Consistent Color Jitter (Appearance)
        if torch.rand(1).item() < self._get_prob(epoch, 0.2):
            # Sample factors per batch, keep time dimension 1 for broadcast
            brightness = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.7, 1.3)
            contrast = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.7, 1.3)

            past_frames = past_frames * brightness
            mean = past_frames.mean(dim=(-3, -2, -1), keepdim=True)
            past_frames = (past_frames - mean) * contrast + mean
            past_frames = past_frames.clamp(-1.0, 1.0)

        # Spatial Tube Masking (Geometry)
        if torch.rand(1).item() < self._get_prob(epoch, 0.15):
            mask_h, mask_w = H // 4, W // 4
            top = torch.randint(0, H - mask_h, (B,), device=device)
            left = torch.randint(0, W - mask_w, (B,), device=device)

            mask = torch.ones(B, 1, 1, H, W, device=device)
            for i in range(B):
                mask[i, :, :, top[i]:top[i]+mask_h, left[i]:left[i]+mask_w] = 0.0
            past_frames = past_frames * mask  # 0.0 in [-1, 1] acts as mean/gray occlusion

        drop_mask_vw = None
        drop_mask_pos = None

        rand_kino = torch.rand(1).item()
        prob_drop = self._get_prob(epoch, 0.15)
        prob_noise = self._get_prob(epoch, 0.20)

        if rand_kino < prob_drop:
            drop_mask_vw = None
            drop_mask_pos = None

        elif rand_kino < prob_drop + prob_noise:
            # Cumulative Sensor Noise Injection (Odometry Drift Regularization)
            # Mutually exclusive from dropout to avoid conflicting physical states
            if past_vw is not None:
                past_vw = past_vw + torch.randn_like(past_vw) * 0.05
            if past_positions is not None:
                past_positions = past_positions + torch.randn_like(past_positions) * 0.02

        return past_frames, past_vw, past_positions, drop_mask_vw, drop_mask_pos