import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from hydra.model.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    SinusoidalPositionalEncoding3D,
    LearnablePositionalEncoding3D,
)
from diffusers.models import AutoencoderKL
from einops import rearrange

class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=512, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # We need output_dim // 2 frequencies because we use sin + cos
        self.num_frequencies = output_dim // 2

        # B is the random Gaussian matrix.
        # scale controls the "frequency spread". Higher scale = higher frequency sensitivity.
        b_matrix = torch.randn(self.num_frequencies, input_dim) * scale
        self.register_buffer("B", b_matrix)

        # A MLP on top to mix the features
        self.mlp = nn.Sequential(
            nn.Linear(self.num_frequencies * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        self.norm = nn.RMSNorm(output_dim)

    def forward(self, x):
        projection = 2 * math.pi * (x @ self.B.T.to(x.dtype))

        # Compute Sin and Cos
        # Concat along last dim: [B, T, C]
        fourier_feats = torch.cat(
            [torch.sin(projection), torch.cos(projection)], dim=-1
        )

        # map features
        out = self.mlp(fourier_feats)

        return self.norm(out)


class VisualEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        model_id = cfg.encoder.get("vae_model_id", "stabilityai/sd-vae-ft-ema")
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.patch_size = cfg.encoder.get("patch_size", 2)
        self.scale_factor = cfg.encoder.get("vae_scale_factor", 0.18215)
        self.latent_channels = 4
        self.out_channels = self.latent_channels * (self.patch_size ** 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.compiler.disable  # Compile gives error, don't know why
    def encode(self, x):
        is_training = torch.is_grad_enabled()
        with torch.no_grad():
            posterior = self.vae.encode(x).latent_dist
            if is_training:
                z = posterior.sample()
            else:
                z = posterior.mode()
            z = z * self.scale_factor

        B, C, H_prime, W_prime = z.shape
        p = self.patch_size

        # Patchify the latent sequence
        z = rearrange(z, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=p, p2=p)
        return z

    @torch.compiler.disable
    def decode(self, z, hw_shape, chunk_size=8):
        p = self.patch_size
        C = self.latent_channels
        H_prime, W_prime = hw_shape

        # Inverse patchification
        z = rearrange(z, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', h=H_prime // p, w=W_prime // p, p1=p, p2=p)

        z = z / self.scale_factor

        out_chunks = []
        with torch.no_grad():
            # Chunking prevents massive VRAM OOM spikes when decoding long autoregressive rollouts
            for i in range(0, z.shape[0], chunk_size):
                z_chunk = z[i : i + chunk_size]
                out_chunks.append(self.vae.decode(z_chunk).sample)

        return torch.cat(out_chunks, dim=0)


class UnifiedEncoder(nn.Module):
    def __init__(self, cfg: DictConfig = None):
        super().__init__()
        d_model = cfg.encoder.transformer_layer.d_model
        # Modality Encoders
        self.visual_encoder = VisualEncoder(cfg)

        # Calculate the number of visual tokens
        patch_size = cfg.encoder.get("patch_size", 2)
        image_size = cfg.encoder.get("image_size", [224, 224])
        # AutoencoderKL has a spatial compression factor of 8
        self.k = (image_size[0] // 8 // patch_size) * (image_size[1] // 8 // patch_size)

        self.backbone_dim = self.visual_encoder.out_channels
        if self.backbone_dim != d_model:
            self.proj = nn.Linear(self.backbone_dim, d_model)
        else:
            self.proj = nn.Identity()

        # Action and Pose encoders
        self.action_encoder = FourierFeatureEncoder(
            cfg.action_dim, d_model, scale=cfg.frequency_sensitivity
        )
        # TODO: currently pose and angles are treated the same, but it is better to treat them differently since they have different physical meanings.
        # Generally it is necessary when moving to 3D space (6DoF), for 2D navigation (3DoF) this is okay and does not add extra parameter and compute.
        self.pose_encoder = FourierFeatureEncoder(
            cfg.pose_dim, d_model, scale=cfg.frequency_sensitivity
        )

        # Learnable tokens to represent missing modalities, shared with the decoder
        self.action_mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pose_mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def train(self, mode: bool = True):
        """Override train to keep the visual encoder frozen in eval mode."""
        super().train(mode)
        self.visual_encoder.eval()
        return self

    def forward(self, images, actions, poses) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            images: [B, T, C, H, W] (Pixels) OR [B, T, K, D] (Latents)
            actions: [B, T, 2] or None
            poses: [B, T, 6] or None
        Returns:
            unified_tokens: [B, T, d_model]
        """
        b, t = images.shape[:2]

        # Handle both Pixel Space and Latent Space. Note, this is not ONNX friendly.
        if images.ndim == 5:
            c, h, w = images.shape[2:]
            emb_img = self.visual_encoder.encode(images.reshape(b * t, c, h, w))
        elif images.ndim == 4:
            k, d = images.shape[2:]
            emb_img = images.reshape(b * t, k, d)
        else:
            raise ValueError(f"Expected images to be 4D (latents) or 5D (pixels), got {images.ndim}D")

        emb_img = self.proj(emb_img)
        emb_img = emb_img.view(b, t * self.k, -1)

        # emb_act/pose are [B, T, d_model]
        emb_act = self.action_encoder(actions) if actions is not None else self.action_mask_token.expand(b, t, -1)
        emb_pose = self.pose_encoder(poses) if poses is not None else self.pose_mask_token.expand(b, t, -1)

        # We need to match the sequence length of the visual tokens (T*k)
        if self.k > 1:
            # Scale the embedding to preserve variance and prevent backward gradient explosion from K copies
            emb_act = emb_act.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)
            emb_pose = emb_pose.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)

        # Additive Fusion, only shifts the embedding
        # Forces modalities into a shared semantic subspace.
        # Masked modalities act as learned additive biases.
        unified_tokens = emb_img + emb_act + emb_pose

        return unified_tokens


class UnifiedTransformerEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.encoder.transformer.block_size
        self.ctx_len = cfg.pred_len
        self.emb_dim = cfg.encoder.transformer_layer.d_model
        ### modality encoders
        self.unified_encoder = UnifiedEncoder(cfg)
        self.k = self.unified_encoder.k
        self.ctx_token_len = self.ctx_len * self.k

        self.use_ctx_tokens = cfg.get("use_ctx_tokens", True)
        if self.use_ctx_tokens:
            # Learnable context query slots
            self.ctx_tokens = nn.Parameter(torch.randn(1, self.ctx_token_len, self.emb_dim) * 0.02)

        patch_size = cfg.encoder.get("patch_size", 2)
        image_size = cfg.encoder.get("image_size", [224, 224])
        h_patches = image_size[0] // 8 // patch_size
        w_patches = image_size[1] // 8 // patch_size
        max_t = self.block_size + self.ctx_len

        encoder_seq_len = self.block_size * self.k + self.ctx_token_len

        if cfg.pos_type == "learnable":
            self.pos = LearnablePositionalEncoding(
                max_len=encoder_seq_len,
                **cfg.encoder.pos_encoding,
            )
        elif cfg.pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(
                max_len=encoder_seq_len,
                **cfg.encoder.pos_encoding,
            )
        elif cfg.pos_type == "sinusoidal3d":
            self.pos = SinusoidalPositionalEncoding3D(
                max_t=max_t,
                h_patches=h_patches,
                w_patches=w_patches,
                **cfg.encoder.pos_encoding,
            )
        elif cfg.pos_type == "learnable3d":
            self.pos = LearnablePositionalEncoding3D(
                max_t=max_t,
                h_patches=h_patches,
                w_patches=w_patches,
                **cfg.encoder.pos_encoding,
            )
        else:
            raise TypeError(
                f"Provided positional encoding type {cfg.pos_type} is not supported."
            )
        transformer_encoder = nn.TransformerEncoderLayer(
            **cfg.encoder.transformer_layer
        )
        encoder_norm = nn.RMSNorm(self.emb_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder,
            cfg.encoder.transformer.num_layers,
            enable_nested_tensor=False,
            norm=encoder_norm,
        )

        # initialize weights
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.encoder.transformer.num_layers),
                )

    def _init_weights(self, module):
        if getattr(module, "_is_zero_init", False):
            return  # Safely skip modules that explicitly require zero-initialization
        if isinstance(module, nn.Linear):
            # Only initialize layers that are meant to be trained
            if module.weight.requires_grad:
                torch.nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None and module.bias.requires_grad:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self, imgs: torch.Tensor, actions: torch.Tensor, poses: torch.Tensor, pred_len: int = None
    ) -> torch.Tensor:
        """
        Args:
            imgs: patches of terrain in (B, T, C, H, W)
            actions: low-level actions in (B, T, 2) or None
            poses: pose of the robot on terrain in (B, T, D) or None
        Returns:
            pred_ctx_tokens: Predicted context tokens for the decoder of shape (B, ctx_len, D)
            last_features: Last feature map from the visual encoder for the skip connection of shape (B, C, H, W)
            ctx_pos_enc: Positional encodings for the context tokens of shape (1, ctx_len, D)
        """
        b, t = imgs.shape[:2]
        # concat tokens together
        tokens = self.unified_encoder(imgs, actions, poses)

        obs_len = tokens.shape[1]
        ctx_token_len = self.ctx_token_len if pred_len is None else pred_len * self.k

        if self.use_ctx_tokens:
            ctx_tokens = self.ctx_tokens[:, :ctx_token_len, :].repeat(b, 1, 1)
            # Append context tokens to the end of the sequence.
            tokens = torch.cat([tokens, ctx_tokens], dim=1)

        # Apply positional encoding to the full sequence
        tokens = self.pos(tokens)
        pred_tokens = self.transformer_encoder(tokens)

        # Extract the predicted context tokens from the end of the output sequence
        if self.use_ctx_tokens:
            out_tokens = pred_tokens[:, -ctx_token_len:, :]
        else:
            out_tokens = pred_tokens

        return out_tokens


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, t):
        # t: [B]
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.view(-1, 1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class AdaLNZeroBlock(nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # Modulation takes the dense conditioning and outputs scale, shift, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )

        # Zero-initialize the modulation and MLP output for identity mapping at init
        self.adaLN_modulation[-1]._is_zero_init = True
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c_time):
        b = x.shape[0]
        mods = self.adaLN_modulation(c_time)  # [B, 3 * hidden_dim]
        # Reshape mods to be broadcastable with x.
        while mods.ndim < x.ndim:
            mods = mods.unsqueeze(1)
        shift, scale, gate = mods.chunk(3, dim=-1)
        x_mod = self.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(x_mod)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # Modulation outputs 6 parameters: shift, scale, gate for both MSA and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        # Zero-initialize the modulation for identity mapping at init
        self.adaLN_modulation[-1]._is_zero_init = True
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c_time):
        orig_shape = x.shape
        b, d = orig_shape[0], orig_shape[-1]
        x_flat = x.reshape(b, -1, d)

        mods = self.adaLN_modulation(c_time).reshape(b, 1, -1) # [B, 1, 6 * hidden_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        # Multi-Head Self-Attention
        x_mod_1 = self.norm1(x_flat) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_mod_1, x_mod_1, x_mod_1, need_weights=False)
        x_flat = x_flat + gate_msa * attn_out

        # MLP
        x_mod_2 = self.norm2(x_flat) * (1 + scale_mlp) + shift_mlp
        x_flat = x_flat + gate_mlp * self.mlp(x_mod_2)

        return x_flat.reshape(*orig_shape)


class ConvFlowBlock(nn.Module):
    def __init__(self, hidden_dim, h_patches, w_patches, kernel_size=7, **kwargs):
        super().__init__()
        self.h_patches = h_patches
        self.w_patches = w_patches

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.dwconv_spatial = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=kernel_size // 2,
            groups=hidden_dim
        )
        self.dwconv_temporal = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1,
            groups=hidden_dim
        )

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        with torch.no_grad():
            w = self.dwconv_spatial.weight
            k_sq = kernel_size * kernel_size
            new_w = torch.empty_strided(w.size(), (k_sq, 1, kernel_size, 1), dtype=w.dtype, device=w.device)
            new_w.copy_(w)
            self.dwconv_spatial.weight = nn.Parameter(new_w)

        with torch.no_grad():
            w1d = self.dwconv_temporal.weight
            k1 = 3 # kernel_size
            new_w1d = torch.empty_strided(w1d.size(), (k1, 1, 1), dtype=w1d.dtype, device=w1d.device)
            new_w1d.copy_(w1d)
            self.dwconv_temporal.weight = nn.Parameter(new_w1d)

        # Modulation outputs parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )

        # Zero-initialize for identity mapping at init
        self.adaLN_modulation[-1]._is_zero_init = True
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c_time):
        b, t, k, d = x.shape

        mods = self.adaLN_modulation(c_time).reshape(b, 1, 1, -1)
        shift_conv, scale_conv, gate_conv, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)

        x_mod_1 = self.norm1(x) * (1 + scale_conv) + shift_conv

        x_mod_1_grid = rearrange(x_mod_1, 'b t (h w) d -> (b t) d h w', h=self.h_patches, w=self.w_patches)
        conv_spatial_out = self.dwconv_spatial(x_mod_1_grid)
        conv_spatial_out = rearrange(conv_spatial_out, '(b t) d h w -> b t (h w) d', b=b, t=t)

        if t > 1:
            x_temp = rearrange(conv_spatial_out, 'b t k d -> (b k) d t')
            conv_temp_out = self.dwconv_temporal(x_temp)
            conv_out_seq = rearrange(conv_temp_out, '(b k) d t -> b t k d', b=b, k=k)
        else:
            conv_out_seq = conv_spatial_out

        x = x + gate_conv * conv_out_seq

        x_mod_2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_mod_2)

        return x


class BaseFlowMatchingModule(nn.Module):
    """
    Base class that handles identical Flow Matching Mathematics (ODE solving & Loss formulation)
    for all modalities, regardless of the underlying network architecture.
    """
    def __init__(self):
        super().__init__()

    def compute_loss(self, x1, cond, t_sampling="logit_normal", weight_shift=3.0):
        """Optimal Transport / Rectified Flow Matching Loss (https://arxiv.org/abs/2209.03003)."""
        b = x1.shape[0]
        x0 = torch.randn_like(x1)  # Base distribution (Noise)

        t_shape = (b,) + (1,) * (x1.ndim - 1)
        if t_sampling == "logit_normal":
            t = torch.randn(t_shape, device=x1.device, dtype=x1.dtype)
            t = torch.sigmoid(t - math.log(weight_shift))
        else:
            # Standard uniform sampling (baseline)
            t = torch.rand(t_shape, device=x1.device, dtype=x1.dtype)

        xt = t * x1 + (1.0 - t) * x0

        v_target = x1 - x0
        v_pred = self.forward(xt, t.view(b), cond)
        return F.mse_loss(v_pred, v_target, reduction='none').mean()

    @torch.no_grad()
    def sample(self, cond, steps=10, schedule_shift=1.0, solver="heun"):
        b = cond.shape[0]

        x_shape = cond.shape[:-1] + (self.final_proj.out_features,)
        x = torch.randn(x_shape, device=cond.device)

        # Integrate noise->data: t goes 0 (noise) -> 1 (data)
        times = torch.linspace(0.0, 1.0, steps + 1, device=cond.device)

        # Apply Time-Shift Schedule
        times = (times * schedule_shift) / (1.0 + (schedule_shift - 1.0) * times)

        for i in range(steps):
            t_curr = times[i]
            t_next = times[i + 1]
            dt = t_next - t_curr

            t_tensor = torch.full((b,), t_curr.item(), device=cond.device, dtype=cond.dtype)
            v_pred = self.forward(x, t_tensor, cond)

            if solver == "heun" and i < steps - 1:
                x_next_euler = x + v_pred * dt
                t_next_tensor = torch.full((b,), t_next.item(), device=cond.device, dtype=cond.dtype)
                v_pred_next = self.forward(x_next_euler, t_next_tensor, cond)
                x = x + (v_pred + v_pred_next) / 2.0 * dt
            else:
                # Standard 1st Order Euler Fallback
                x = x + v_pred * dt

        return x


class ImageFlowMatchingModule(BaseFlowMatchingModule):
    """High-capacity generative backbone specifically for spatial visual latents."""
    def __init__(self, latent_dim=64, cond_dim=512, hidden_dim=512, num_layers=4, block_type='attention', h_patches=28, w_patches=28, **block_kwargs):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion_proj = nn.Linear(latent_dim + 2 + cond_dim, hidden_dim)
        self.fusion_norm = nn.RMSNorm(hidden_dim)

        # Create the fixed coordinate grid [-1, 1]. Trying to mimic coordconv.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h_patches),
            torch.linspace(-1, 1, w_patches),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(h_patches * w_patches, 2) # [K, 2]
        self.register_buffer("coord_grid", grid)

        if block_type == 'attention':
            block_class = DiTBlock
        else:
            block_class = ConvFlowBlock

        self.blocks = nn.ModuleList([
            block_class(hidden_dim, h_patches=h_patches, w_patches=w_patches, **block_kwargs) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim)

        self.final_modulation[-1]._is_zero_init = True
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, t, cond):
        # x and cond shape: [B, T, K, D]
        c_time = self.time_mlp(t * 1000.0)

        b, t_seq, k, d = x.shape

        # Expand absolute coordinates and concatenate (Latent CoordConv)
        grid = self.coord_grid.view(1, 1, k, 2).expand(b, t_seq, k, 2)

        # Fuse state, geometric prior, and history
        h_fused = torch.cat([x, grid, cond], dim=-1)
        h = self.fusion_norm(self.fusion_proj(h_fused))

        for block in self.blocks:
            h = block(h, c_time)

        mods = self.final_modulation(c_time)
        while mods.ndim < h.ndim:
            mods = mods.unsqueeze(1)
        shift, scale = mods.chunk(2, dim=-1)

        h = self.final_norm(h) * (1 + scale) + shift
        return self.final_proj(h)


class StateFlowMatchingModule(BaseFlowMatchingModule):
    """Fast, generative backbone specifically for 1D robotic states (actions/poses)."""
    def __init__(self, latent_dim=2, cond_dim=512, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fusion_proj = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fusion_norm = nn.RMSNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        self.final_proj = nn.Linear(hidden_dim, latent_dim)

        self.final_modulation[-1]._is_zero_init = True
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, t, cond):
        # x and cond shape: [B, T, D]
        c_time = self.time_mlp(t * 1000.0)

        h_fused = torch.cat([x, cond], dim=-1)
        h = self.fusion_norm(self.fusion_proj(h_fused))

        for block in self.blocks:
            h = block(h, c_time)

        mods = self.final_modulation(c_time).unsqueeze(1) # Broadcast precisely over Sequence length
        shift, scale = mods.chunk(2, dim=-1)

        h = self.final_norm(h) * (1 + scale) + shift
        return self.final_proj(h)


class ResidualMLPBlock(nn.Module):
    """A standard pre-norm residual MLP block with zero-initialized output projections."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

        self.fc2._is_zero_init = True
        # Zero-initialize the projection back to the residual stream
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class LatentImageDecoder(nn.Module):
    """Deep residual decoder to reconstruct VAE latents from transformer tokens."""
    def __init__(self, emb_dim: int, visual_latent_dim: int, hidden_dim: int, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(emb_dim, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=True, eps=1e-6)
        self.head = nn.Linear(emb_dim, visual_latent_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


class ConvLatentImageDecoder(nn.Module):
    def __init__(self, emb_dim: int, visual_latent_dim: int, hidden_dim: int, patch_size: int = 2, num_layers: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = visual_latent_dim // (patch_size ** 2)

        # Project Transformer embeddings back to patchified latent dimensions
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, visual_latent_dim)
        )

        # Lightweight Convolutional Refiner on the fully unpatchified grid
        def _build_refiner_block():
            block = nn.Sequential(
                nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=7, padding=3, groups=self.latent_channels),
                nn.GroupNorm(1, self.latent_channels),
                nn.Conv2d(self.latent_channels, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, self.latent_channels, kernel_size=1)
            )
            block[-1]._is_zero_init = True
            nn.init.zeros_(block[-1].weight)
            if block[-1].bias is not None:
                nn.init.zeros_(block[-1].bias)
            return block

        self.refiner_blocks = nn.ModuleList([_build_refiner_block() for _ in range(num_layers)])

    def forward(self, x, h_patches, w_patches):
        # x: [B*T, k, emb_dim]
        bt, k, d = x.shape

        # Map tokens back to the raw patchified latent dimension
        patchified_latents = self.proj(x)  # [B*T, k, visual_latent_dim]

        # Unpatchify into the true spatial arrangement (e.g., [B*T, 4, 28, 28])
        # Reverses the rearrange operation in VisualEncoder.encode
        grid = rearrange(
            patchified_latents,
            'bt (h w) (c p1 p2) -> bt c (h p1) (w p2)',
            h=h_patches,
            w=w_patches,
            p1=self.patch_size,
            p2=self.patch_size
        )

        # Apply convolutions on the native resolution to smooth patch boundaries.
        refined_grid = grid
        for block in self.refiner_blocks:
            refined_grid = refined_grid + block(refined_grid)

        # Re-patchify to match the sequence format for the loss criterion: [B*T, k, visual_latent_dim]
        out = rearrange(
            refined_grid,
            'bt c (h p1) (w p2) -> bt (h w) (c p1 p2)',
            p1=self.patch_size,
            p2=self.patch_size
        )
        return out