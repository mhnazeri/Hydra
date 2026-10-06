"""VertiFormer with Hydra-compatible encoder."""

import math

import torch
from torch import nn
from torch import optim

from hydra.model.modules import UnifiedTransformerEncoder, ConvLatentImageDecoder
from hydra.model.positional_encoding import (
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    SinusoidalPositionalEncoding3D,
    LearnablePositionalEncoding3D,
)
from hydra.utilities.loss import VertiFormerMultiTaskCriterion


class VertiFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer_encoder = UnifiedTransformerEncoder(cfg)
        self.emb_dim = cfg.decoder.transformer_layer.d_model
        self.k = self.transformer_encoder.k

        # Map to d_model for the transformer decoder
        self.f_mapping = nn.Linear(self.emb_dim * 2, self.emb_dim)

        # Learnable visual query mask to give the transformer a starting point for spatial decoding
        self.visual_mask_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        nn.init.normal_(self.visual_mask_token, std=0.02)

        patch_size = cfg.encoder.get("patch_size", 2)
        image_size = cfg.encoder.get("image_size", [224, 224])
        h_patches = image_size[0] // 8 // patch_size
        w_patches = image_size[1] // 8 // patch_size
        self.h_patches = h_patches
        self.w_patches = w_patches

        # Positional Encoding
        if cfg.pos_type == "learnable":
            self.pos = LearnablePositionalEncoding(max_len=self.cfg.pred_len * self.k, **cfg.decoder.pos_encoding)
        elif cfg.pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(max_len=self.cfg.pred_len * self.k, **cfg.decoder.pos_encoding)
        elif cfg.pos_type == "sinusoidal3d":
            self.pos = SinusoidalPositionalEncoding3D(max_t=self.cfg.pred_len, h_patches=h_patches, w_patches=w_patches, **cfg.decoder.pos_encoding)
        elif cfg.pos_type == "learnable3d":
            self.pos = LearnablePositionalEncoding3D(max_t=self.cfg.pred_len, h_patches=h_patches, w_patches=w_patches, **cfg.decoder.pos_encoding)
        else:
            raise TypeError(
                f"Provided positional encoding type {cfg.pos_type} is not supported."
            )

        # Decoder backbone (same as Hydra).
        decoder_layer = nn.TransformerDecoderLayer(**cfg.decoder.transformer_layer)
        decoder_norm = nn.RMSNorm(self.emb_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=cfg.decoder.transformer.num_layers,
            norm=decoder_norm,
        )

        # Non-VQ task heads.
        self.pose_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, cfg.pose_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, cfg.action_dim),
        )

        visual_latent_dim = self.cfg.encoder.get("visual_latent_dim", 16)

        num_img_layers = cfg.decoder.get("img_decoder_layers", 4)
        self.img_decoder = ConvLatentImageDecoder(
            emb_dim=self.emb_dim,
            visual_latent_dim=visual_latent_dim,
            hidden_dim=cfg.decoder.transformer_layer.dim_feedforward,
            patch_size=patch_size,
            num_layers=num_img_layers
        )

        # Block-Causal Mask
        pred_token_len = self.cfg.pred_len * self.k
        mask = torch.empty(pred_token_len, pred_token_len, dtype=torch.bool)
        mask.fill_(True)
        for i in range(self.cfg.pred_len):
            mask[i * self.k : (i + 1) * self.k, : (i + 1) * self.k] = False

        self.register_buffer("mask", mask)

        self.apply(self._init_weights)
        for name, param in self.transformer_decoder.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.decoder.transformer.num_layers),
                )

    def _init_weights(self, module):
        if getattr(module, "_is_zero_init", False):
            return # Safely skip modules that explicitly require zero-initialization
        if isinstance(module, nn.Linear):
            # Only initialize layers that are meant to be trained
            if module.weight.requires_grad:
                nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def encode_history(self, imgs, actions, poses, pred_len, markovian=None):
        """
        Pre-computes the unified observation tokens to save compute
        during iterative MPC search loops.
        """
        b = poses.shape[0]
        pred_token_len = pred_len * self.k
        markovian = markovian if markovian is not None else self.cfg.decoder.get("last_frame_conditioning", False)
        use_ctx_tokens = self.cfg.get("use_ctx_tokens", True)

        unified_obs_tokens = self.transformer_encoder(imgs, actions, poses, pred_len=pred_len)

        if not use_ctx_tokens and markovian:
            unified_obs_tokens = unified_obs_tokens[:, -self.k:, :]

        return unified_obs_tokens

    @torch.no_grad()
    def get_intent_tokens(
        self,
        cached_obs_tokens: torch.Tensor,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
        pred_len: int | None = None,
        mask_actions: bool = False,
        mask_poses: bool = False,
    ):
        """
        Fast-path to extract continuous intent tokens from the decoder.
        Used in Latent MPC to avoid running history encoding.
        """
        b = cached_obs_tokens.shape[0]
        pred_len = int(pred_len) if pred_len is not None else self.cfg.pred_len
        pred_token_len = pred_len * self.k

        if future_poses is not None and not mask_poses:
            query_future_poses = self.transformer_encoder.unified_encoder.pose_encoder(future_poses)
        else:
            mask_token = self.transformer_encoder.unified_encoder.pose_mask_token
            query_future_poses = mask_token.expand(b, pred_len, -1)

        if future_actions is not None and not mask_actions:
            query_future_actions = self.transformer_encoder.unified_encoder.action_encoder(future_actions)
        else:
            mask_token = self.transformer_encoder.unified_encoder.action_mask_token
            query_future_actions = mask_token.expand(b, pred_len, -1)

        query_base_act_pose = torch.cat([query_future_actions, query_future_poses], dim=-1)
        query_base = self.f_mapping(query_base_act_pose)

        query_base = query_base.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)
        query_base = query_base + self.visual_mask_token.expand(b, pred_token_len, -1)
        query_pe = self.pos(query_base)

        tgt_mask = self.mask[:pred_token_len, :pred_token_len]
        return self.transformer_decoder(query_pe, cached_obs_tokens, tgt_mask=tgt_mask, tgt_is_causal=False)

    def forward(
        self,
        imgs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        poses: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
        pred_len: int | None = None,
        task: str = "all",
        markovian: bool | None = None,
        mask_actions: bool = False,
        mask_poses: bool = False,
        cached_obs_tokens: torch.Tensor | None = None,
    ):
        if cached_obs_tokens is not None:
            b = cached_obs_tokens.shape[0]
        else:
            b = poses.shape[0]

        if pred_len is None:
            if future_actions is not None:
                pred_len = future_actions.shape[1]
            elif future_poses is not None:
                pred_len = future_poses.shape[1]
            else:
                pred_len = self.cfg.pred_len
        pred_len = int(pred_len)
        pred_token_len = pred_len * self.k

        markovian = markovian if markovian is not None else self.cfg.decoder.get("last_frame_conditioning", False)
        use_ctx_tokens = self.cfg.get("use_ctx_tokens", True)

        if cached_obs_tokens is not None:
            unified_obs_tokens = cached_obs_tokens
        else:
            unified_obs_tokens = self.transformer_encoder(imgs, actions, poses, pred_len=pred_len)
            if not use_ctx_tokens and markovian:
                unified_obs_tokens = unified_obs_tokens[:, -self.k:, :]

        if future_poses is not None and not mask_poses:
            query_future_poses = self.transformer_encoder.unified_encoder.pose_encoder(future_poses)
        else:
            mask_token = self.transformer_encoder.unified_encoder.pose_mask_token
            query_future_poses = mask_token.expand(b, pred_len, -1)

        if future_actions is not None and not mask_actions:
            query_future_actions = self.transformer_encoder.unified_encoder.action_encoder(future_actions)
        else:
            mask_token = self.transformer_encoder.unified_encoder.action_mask_token
            query_future_actions = mask_token.expand(b, pred_len, -1)

        query_base_act_pose = torch.cat([query_future_actions, query_future_poses], dim=-1)
        # Map concatenated action/pose queries to the transformer's expected d_model dimension
        query_base = self.f_mapping(query_base_act_pose)

        # Expand queries for sequence length (k tokens) & scale to prevent backward pass gradient explosion
        query_base = query_base.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)

        # Add the visual mask so the queries have visual capacity
        query_base = query_base + self.visual_mask_token.expand(b, pred_token_len, -1)

        # Extract Positional Encoding (PE) to re-inject before point-wise decoding
        query_pe = self.pos(query_base)
        pe_only = query_pe - query_base

        tgt_mask = self.mask[:pred_token_len, :pred_token_len]

        intent_tokens = self.transformer_decoder(
            query_pe, unified_obs_tokens, tgt_mask=tgt_mask, tgt_is_causal=False
        )

        pred_images, pred_actions, pred_poses = None, None, None
        aux = {}

        if task in ["all", "wm"]:
            intent_tokens_flat = (intent_tokens + pe_only).reshape(b * pred_len, self.k, -1)
            pred_images_flat = self.img_decoder(intent_tokens_flat, self.h_patches, self.w_patches)
            pred_images = pred_images_flat.reshape(b, pred_len, self.k, -1)

        if task in ["all", "ikd"]:
            z_act_q_agg = intent_tokens.reshape(b, pred_len, self.k, -1).mean(dim=2)
            pred_actions = self.action_decoder(z_act_q_agg)

        if task in ["all", "fkd"]:
            z_pose_q_agg = intent_tokens.reshape(b, pred_len, self.k, -1).mean(dim=2)
            pred_poses = self.pose_decoder(z_pose_q_agg)

        return pred_poses, pred_actions, pred_images, aux

    @torch.no_grad()
    def rollout(
        self,
        imgs: torch.Tensor | None,
        actions: torch.Tensor,
        poses: torch.Tensor,
        pred_len: int,
        autoregressive: bool = False,
        task: str = "all",
        markovian: bool | None = None,
        cached_obs_tokens: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
    ):
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}")

        if cached_obs_tokens is not None and autoregressive and pred_len > self.cfg.pred_len:
            raise RuntimeError(
                "Autoregressive rollout requires continuous history updating and is mathematically incompatible with a bypassed encoder (cached_obs_tokens)."
            )

        if not autoregressive or pred_len <= self.cfg.pred_len:
            chunk_len = min(pred_len, self.cfg.pred_len)
            curr_future_actions = future_actions[:, :chunk_len] if future_actions is not None else None
            curr_future_poses = future_poses[:, :chunk_len] if future_poses is not None else None
            return self(
                imgs=imgs,
                actions=actions,
                poses=poses,
                future_actions=curr_future_actions,
                future_poses=curr_future_poses,
                pred_len=chunk_len,
                task=task,
                markovian=markovian,
                cached_obs_tokens=cached_obs_tokens,
            )

        obs_len = actions.shape[1] if actions is not None else (poses.shape[1] if poses is not None else self.cfg.dataset.obs_len)
        remaining = pred_len
        hist_actions, hist_poses = actions, poses

        if cached_obs_tokens is not None:
            hist_imgs = None
        else:
            # Pre-encode observed image context once
            if imgs is not None and imgs.ndim == 5:
                b_sz, t_sz = imgs.shape[:2]
                visual_encoder = self.transformer_encoder.unified_encoder.visual_encoder
                z_img = visual_encoder.encode(imgs.reshape(b_sz * t_sz, *imgs.shape[2:]))
                hist_imgs = z_img.reshape(b_sz, t_sz, self.k, -1)
            else:
                hist_imgs = imgs

        pose_chunks, action_chunks, image_chunks = [], [], []
        img_aux = {}

        while remaining > 0:
            chunk_len = min(self.cfg.pred_len, remaining)
            current_step = pred_len - remaining

            curr_future_actions = future_actions[:, current_step : current_step + chunk_len] if future_actions is not None else None
            curr_future_poses = future_poses[:, current_step : current_step + chunk_len] if future_poses is not None else None

            pred_pose, pred_actions, pred_images, img_aux = self(
                imgs=hist_imgs,
                actions=hist_actions,
                poses=hist_poses,
                future_actions=curr_future_actions,
                future_poses=curr_future_poses,
                pred_len=chunk_len,
                task=task,
                markovian=markovian,
                cached_obs_tokens=cached_obs_tokens if remaining == pred_len else None,
            )

            if pred_pose is not None:
                pose_chunks.append(pred_pose)
            next_pose = curr_future_poses if curr_future_poses is not None else pred_pose
            if next_pose is not None:
                hist_poses = torch.cat([hist_poses, next_pose], dim=1)[:, -obs_len:, :]

            if pred_actions is not None:
                action_chunks.append(pred_actions)
            next_action = curr_future_actions if curr_future_actions is not None else pred_actions
            if next_action is not None:
                hist_actions = torch.cat([hist_actions, next_action], dim=1)[:, -obs_len:, :]

            if pred_images is not None:
                # Keep context strictly in latent space
                image_chunks.append(pred_images)
                if hist_imgs is not None:
                    hist_imgs = torch.cat([hist_imgs, pred_images], dim=1)[:, -obs_len:, :]
                else:
                    hist_imgs = pred_images[:, -obs_len:, :]

            remaining -= chunk_len

        pred_poses = torch.cat(pose_chunks, dim=1) if pose_chunks else None
        pred_actions = torch.cat(action_chunks, dim=1) if action_chunks else None
        pred_images = torch.cat(image_chunks, dim=1) if image_chunks else None

        # Decode at the very end of rollout
        if pred_images is not None and task in ["all", "wm"]:
            b_sz, t_sz, k_dim, d_dim = pred_images.shape
            z_img = pred_images.reshape(b_sz * t_sz, k_dim, d_dim)
            image_size = self.cfg.encoder.get("image_size", [224, 224])
            h_prime, w_prime = image_size[0] // 8, image_size[1] // 8

            visual_encoder = self.transformer_encoder.unified_encoder.visual_encoder
            p_img = visual_encoder.decode(z_img, hw_shape=(h_prime, w_prime))
            pred_images = p_img.reshape(b_sz, t_sz, *p_img.shape[1:])

        return pred_poses, pred_actions, pred_images, img_aux

    @torch.no_grad()
    def sample(
        self,
        imgs,
        actions,
        poses,
        pred_len,
        autoregressive=False,
        task="all",
        markovian=None,
        cached_obs_tokens=None,
        future_actions=None,
        future_poses=None,
    ):
        """Clean inference method that strips aux outputs and returns only the requested modality."""
        self.eval()
        pred_poses, pred_actions, pred_images, _ = self.rollout(
            imgs, actions, poses, pred_len, autoregressive, task, markovian,
            cached_obs_tokens=cached_obs_tokens, future_actions=future_actions, future_poses=future_poses
        )

        if task == "fkd":
            return pred_poses
        elif task == "ikd":
            return pred_actions
        elif task == "wm":
            return pred_images
        return pred_poses, pred_actions, pred_images

def load_model(cfg, device):
    """Initializes VertiFormer model with Hydra-compatible encoder."""

    model = VertiFormer(cfg.model).to(device)
    use_auto_weighting = cfg.train_params.get("auto_weight", True)
    loss_weights = cfg.train_params.get("loss_weights", {"actions": 1.0, "poses": 1.0, "images": 1.0})
    criterion = VertiFormerMultiTaskCriterion(
        num_tasks=3,
        use_auto_weighting=use_auto_weighting,
        loss_weights=loss_weights,
        cfg=cfg,
    ).to(device)

    opt_groups = [{"params": model.parameters()}]
    crit_params = [p for p in criterion.parameters() if p.requires_grad]
    if crit_params:
        opt_groups.append({"params": crit_params, "lr": cfg.train_params.loss_lr})

    optimizer = optim.AdamW(opt_groups, **cfg.adamw)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train_params.epochs
    )
    return model, optimizer, scheduler, criterion