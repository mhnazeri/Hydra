"""Hydra model definition"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from omegaconf import DictConfig

from hydra.model.modules import (
    UnifiedTransformerEncoder,
    ImageFlowMatchingModule,
    StateFlowMatchingModule
)
from hydra.model.vqvae import EMAVectorQuantizer
from hydra.model.positional_encoding import (
    get_positional_encoder,
)
from hydra.utilities.loss import MultiTaskCriterion


class Hydra(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer_encoder = UnifiedTransformerEncoder(cfg)
        self.emb_dim = cfg.decoder.transformer_layer.d_model
        # Weight for deterministic regularization (prevents codebook collapse)
        self.det_weight = self.cfg.decoder.get("det_loss_weight", 0.1)
        self.fm_image_steps = self.cfg.decoder.fm_image.get("sampling_steps", 10)
        self.fm_action_steps = self.cfg.decoder.fm_action.get("sampling_steps", 10)
        self.fm_pose_steps = self.cfg.decoder.fm_pose.get("sampling_steps", 10)
        self.f_mapping = nn.Linear(
            self.emb_dim * 2, self.emb_dim
        )  # Map to d_model for transformer decoder
        self.k = self.transformer_encoder.k

        patch_size = cfg.encoder.get("patch_size", 2)
        image_size = cfg.encoder.get("image_size", [224, 224])
        h_patches = image_size[0] // 8 // patch_size
        w_patches = image_size[1] // 8 // patch_size

        # Positional Encoding
        is_3d = "3d" in cfg.pos_type
        pos_cfg = cfg.decoder.pos_encoding

        # Main query PE (can be 1D or 3D)
        self.pos = get_positional_encoder(
            cfg.pos_type, is_3d, pos_cfg,
            max_len=self.cfg.pred_len * self.k,
            max_t=self.cfg.pred_len, h_patches=h_patches, w_patches=w_patches
        )
        # Image VQ PE (can be 1D or 3D)
        self.img_pos = get_positional_encoder(
            cfg.pos_type, is_3d, pos_cfg,
            max_len=self.cfg.pred_len * self.k,
            max_t=self.cfg.pred_len, h_patches=h_patches, w_patches=w_patches
        )
        # Action VQ PE (always 1D)
        self.act_pos = get_positional_encoder(
            cfg.pos_type, False, pos_cfg,
            max_len=self.cfg.pred_len
        )
        # Pose VQ PE (always 1D)
        self.pose_pos = get_positional_encoder(
            cfg.pos_type, False, pos_cfg,
            max_len=self.cfg.pred_len
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            **cfg.decoder.transformer_layer
        )
        # Use RMSNorm for the final layer normalization in the decoder stack
        decoder_norm = nn.RMSNorm(self.emb_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=cfg.decoder.transformer.num_layers,
            norm=decoder_norm,
        )

        # Split VQ Layers
        # High capacity for images, low capacity for control
        self.vq_image = EMAVectorQuantizer(**cfg.decoder.vq_image)
        self.vq_action = EMAVectorQuantizer(**cfg.decoder.vq_action)
        self.vq_pose = EMAVectorQuantizer(**cfg.decoder.vq_pose)

        # Decoders (accept VQ outputs)
        # Input to decoders is still emb_dim
        # Decode poses to 4D continuous SO(2) representation: [x, y, cos(yaw), sin(yaw)]
        self.pose_decoder = nn.Linear(self.emb_dim, 4, bias=False)
        self.action_decoder = nn.Linear(self.emb_dim, cfg.action_dim, bias=False)

        # Image decoder projection (to map VQ or tokens back to VAE patch latent dim)
        visual_latent_dim = self.cfg.encoder.get("visual_latent_dim", 16) # 4 channels * 2 * 2
        self.img_proj = nn.Linear(self.emb_dim, visual_latent_dim, bias=False)

        # Flow Matching Modules for Multi-Modal Continuous Generation
        self.fm_image = ImageFlowMatchingModule(
            latent_dim=visual_latent_dim,
            cond_dim=self.emb_dim,
            hidden_dim=cfg.decoder.fm_image.get("hidden_dim", 768),
            block_type=cfg.decoder.fm_image.get("block_type", "attention"),
            h_patches=h_patches,
            w_patches=w_patches,
        )
        self.fm_action = StateFlowMatchingModule(
            latent_dim=cfg.action_dim,
            cond_dim=self.emb_dim,
            hidden_dim=cfg.decoder.fm_action.get("hidden_dim", 256),
            num_heads=cfg.decoder.fm_action.get("num_heads", 8)
        )
        self.fm_pose = StateFlowMatchingModule(
            latent_dim=cfg.pose_dim,
            cond_dim=self.emb_dim,
            hidden_dim=cfg.decoder.fm_pose.get("hidden_dim", 256),
            num_heads=cfg.decoder.fm_pose.get("num_heads", 8)
        )

        # Block-Causal Mask:
        # We want patches within the same timestep to attend to each other bidirectionally
        # while preventing future timesteps from leaking.
        pred_token_len = self.cfg.pred_len * self.k
        mask = torch.empty(pred_token_len, pred_token_len, dtype=torch.bool)
        mask.fill_(True)
        for i in range(self.cfg.pred_len):
            # Timestep i can attend to all patches in timesteps j <= i
            mask[i * self.k : (i + 1) * self.k, : (i + 1) * self.k] = False

        self.register_buffer("mask", mask)
        # initialize weights
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
            return # skip modules that explicitly require zero-initialization
        if isinstance(module, nn.Linear):
            # Only initialize layers that are meant to be trained
            if module.weight.requires_grad:
                nn.init.normal_(module.weight, 0, 0.02)
            if module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)

    def _process_flow_branch(self, cond, target, fm_module, sample_flag, steps):
        fm_loss = torch.tensor(0.0, device=cond.device)
        if target is not None:
            fm_loss = fm_module.compute_loss(target, cond)

        # To keep training fast, skip sampling completely when not explicitly visualizing.
        if sample_flag or not self.training:
            final_pred = fm_module.sample(cond, steps=steps)
        else:
            final_pred = None

        return final_pred, fm_loss

    @torch.no_grad()
    def encode_history(self, imgs, actions, poses, pred_len, markovian=None):
        """
        Pre-computes the unified observation tokens to save compute
        during iterative DLP search loops.
        """
        b = imgs.shape[0]
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
        mask_actions: bool | torch.Tensor = False,
        mask_poses: bool | torch.Tensor = False,
    ):
        """
        Extract pre-VQ continuous intent tokens from the decoder.
        Used in DLP to avoid running full ODE generation or using hooks.
        """
        b = cached_obs_tokens.shape[0]
        pred_len = int(pred_len) if pred_len is not None else self.cfg.pred_len
        pred_token_len = pred_len * self.k

        if future_poses is not None:
            query_future_poses = self.transformer_encoder.unified_encoder.pose_encoder(future_poses)
            if isinstance(mask_poses, torch.Tensor) or mask_poses:
                mask_token = self.transformer_encoder.unified_encoder.pose_mask_token.expand(b, pred_len, -1)
                if isinstance(mask_poses, torch.Tensor):
                    query_future_poses = torch.where(mask_poses.view(-1, 1, 1), mask_token, query_future_poses)
                else:
                    query_future_poses = mask_token
        else:
            mask_token = self.transformer_encoder.unified_encoder.pose_mask_token
            query_future_poses = mask_token.expand(b, pred_len, -1)

        if future_actions is not None:
            query_future_actions = self.transformer_encoder.unified_encoder.action_encoder(future_actions)
            if isinstance(mask_actions, torch.Tensor) or mask_actions:
                mask_token = self.transformer_encoder.unified_encoder.action_mask_token.expand(b, pred_len, -1)
                if isinstance(mask_actions, torch.Tensor):
                    query_future_actions = torch.where(mask_actions.view(-1, 1, 1), mask_token, query_future_actions)
                else:
                    query_future_actions = mask_token
        else:
            mask_token = self.transformer_encoder.unified_encoder.action_mask_token
            query_future_actions = mask_token.expand(b, pred_len, -1)

        # Expand queries for the sequence length (k tokens per timestep)
        query_future_actions = query_future_actions.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)
        query_future_poses = query_future_poses.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)

        query = torch.cat([query_future_actions, query_future_poses], dim=-1)
        query = self.f_mapping(query)
        query = self.pos(query)

        tgt_mask = self.mask[:pred_token_len, :pred_token_len]
        return self.transformer_decoder(query, cached_obs_tokens, tgt_mask=tgt_mask, tgt_is_causal=False)

    def forward(
        self,
        imgs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        poses: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
        pred_len: int | None = None,
        task: str = "all",
        future_imgs_latent: torch.Tensor | None = None,
        sample_images: bool = False,
        markovian: bool | None = None,
        mask_actions: bool | torch.Tensor = False,
        mask_poses: bool | torch.Tensor = False,
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

        # 1. Encoding History
        if cached_obs_tokens is not None:
            # This caching is used during robot planning to save computation
            # by not recalculating history.
            unified_obs_tokens = cached_obs_tokens
        else:
            unified_obs_tokens = self.transformer_encoder(
                imgs,
                actions,
                poses,
                pred_len=pred_len
            )
            if not use_ctx_tokens and markovian:  # only use the last frame
                unified_obs_tokens = unified_obs_tokens[:, -self.k:, :]

        if future_poses is not None:
            query_future_poses = self.transformer_encoder.unified_encoder.pose_encoder(future_poses)
            if isinstance(mask_poses, torch.Tensor) or mask_poses:
                mask_token = self.transformer_encoder.unified_encoder.pose_mask_token.expand(b, pred_len, -1)
                if isinstance(mask_poses, torch.Tensor):
                    query_future_poses = torch.where(mask_poses.view(-1, 1, 1), mask_token, query_future_poses)
                else:
                    query_future_poses = mask_token
        else:
            # Use the shared mask token from the encoder, expanded to the prediction length
            mask_token = self.transformer_encoder.unified_encoder.pose_mask_token
            query_future_poses = mask_token.expand(b, pred_len, -1)

        if future_actions is not None:
            query_future_actions = self.transformer_encoder.unified_encoder.action_encoder(future_actions)
            if isinstance(mask_actions, torch.Tensor) or mask_actions:
                mask_token = self.transformer_encoder.unified_encoder.action_mask_token.expand(b, pred_len, -1)
                if isinstance(mask_actions, torch.Tensor):
                    query_future_actions = torch.where(mask_actions.view(-1, 1, 1), mask_token, query_future_actions)
                else:
                    query_future_actions = mask_token
        else:
            mask_token = self.transformer_encoder.unified_encoder.action_mask_token
            query_future_actions = mask_token.expand(b, pred_len, -1)

        # Expand queries for the sequence length (k tokens per timestep)
        query_future_actions = query_future_actions.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)
        query_future_poses = query_future_poses.repeat_interleave(self.k, dim=1) / math.sqrt(self.k)

        # 2. Encoding Intents
        query = torch.cat([query_future_actions, query_future_poses], dim=-1)
        query = self.f_mapping(query)  # Map to d_model

        # Add the positional encodings from the decoder to ensure temporal consistency
        query = self.pos(query)

        tgt_mask = self.mask[:pred_token_len, :pred_token_len]
        tokens = self.transformer_decoder(
            query, unified_obs_tokens, tgt_mask=tgt_mask, tgt_is_causal=False
        )

        # 3. Spliting via VQ-Bottlenecks
        # Conditional execution based on task
        pred_images, pred_actions, pred_poses = None, None, None
        vq_loss = 0
        metrics = {}

        # Image Branch (World Model)
        if task in ["all", "wm"]:
            z_img_q, loss_img, img_m = self.vq_image(tokens)

            z_img_q_spatial = self.img_pos(z_img_q)

            # Use VQ output (in emb_dim space) as FM conditioning
            cond = z_img_q_spatial.view(b, pred_len, self.k, self.emb_dim)  # [B, T, k, emb_dim]

            # Stabalize training with deterministic head
            if future_imgs_latent is not None:
                # Deterministic prediction: project VQ token -> visual latent space
                z_img_softvq = self.img_proj(z_img_q_spatial)
                det_pred_img = z_img_softvq.view(b, pred_len, self.k, -1)
                det_loss = torch.nn.functional.mse_loss(det_pred_img, future_imgs_latent)
                vq_loss = vq_loss + (self.det_weight * det_loss)

            pred_images, fm_loss = self._process_flow_branch(
                cond,  # [B, T*k, emb_dim] — conditioning signal
                future_imgs_latent,
                self.fm_image, sample_images, self.fm_image_steps
            )

            vq_loss = vq_loss + loss_img
            metrics.update({
                "images_perp": img_m["perplexity"],
                "images_usage": img_m["usage"],
                "fm_loss": fm_loss
            })

        # Action Branch (Inverse Kinematics / Policy)
        if task in ["all", "ikd"]:
            # Aggregate spatial patches first to enforce a strict 1D temporal bottleneck
            intent_tokens_act_agg = tokens.view(b, pred_len, self.k, -1).mean(dim=2)
            z_act_q_agg, loss_act, act_m = self.vq_action(intent_tokens_act_agg)

            # Re-inject 1D Temporal Positional Encoding
            z_act_q_agg_pos = self.act_pos(z_act_q_agg)

            if future_actions is not None:
                det_pred_act = self.action_decoder(z_act_q_agg_pos)
                det_loss_act = torch.nn.functional.l1_loss(det_pred_act, future_actions)
                vq_loss = vq_loss + (self.det_weight * det_loss_act)

            pred_actions, fm_loss_act = self._process_flow_branch(
                z_act_q_agg_pos, future_actions, self.fm_action, sample_images, self.fm_action_steps
            )

            vq_loss = vq_loss + loss_act
            metrics.update({
                "actions_perp": act_m["perplexity"],
                "actions_usage": act_m["usage"],
                "fm_loss_act": fm_loss_act if future_actions is not None else 0.0,
            })

        # Pose Branch (Trajectory Prediction / Forward Kinematics)
        if task in ["all", "fkd"]:
            # Aggregate spatial patches first to enforce a strict 1D temporal bottleneck
            intent_tokens_pose_agg = tokens.view(b, pred_len, self.k, -1).mean(dim=2)
            z_pose_q_agg, loss_pose, pose_m = self.vq_pose(intent_tokens_pose_agg)

            # Re-inject 1D Temporal Positional Encoding
            z_pose_q_agg_pos = self.pose_pos(z_pose_q_agg)

            if future_poses is not None:
                det_pred_pose = self.pose_decoder(z_pose_q_agg_pos)
                p_pos = det_pred_pose[..., :2]
                p_cos, p_sin = det_pred_pose[..., 2], det_pred_pose[..., 3]
                t_pos, t_yaw = future_poses[..., :2], future_poses[..., 2]

                t_cos, t_sin = torch.cos(t_yaw), torch.sin(t_yaw)
                loss_pos = torch.nn.functional.mse_loss(p_pos, t_pos)
                loss_rot = torch.nn.functional.mse_loss(p_cos, t_cos) + torch.nn.functional.mse_loss(p_sin, t_sin)

                # Integrate deltas to absolute paths to prevent compounding endpoint divergence
                abs_p_pos = torch.cumsum(p_pos, dim=1)
                abs_t_pos = torch.cumsum(t_pos, dim=1)
                loss_abs_pos = torch.nn.functional.mse_loss(abs_p_pos, abs_t_pos)

                # Anchor the final destination point heavily
                loss_terminal = torch.nn.functional.mse_loss(abs_p_pos[:, -1, :], abs_t_pos[:, -1, :])

                det_loss_pose = (loss_pos * 2.0 + loss_rot + loss_abs_pos * 2.0 + loss_terminal) / 6.0
                vq_loss = vq_loss + (self.det_weight * det_loss_pose)

            pred_poses, fm_loss_pose = self._process_flow_branch(
                z_pose_q_agg_pos, future_poses, self.fm_pose, sample_images, self.fm_pose_steps
            )

            vq_loss = vq_loss + loss_pose
            metrics.update({
                "poses_perp": pose_m["perplexity"],
                "poses_usage": pose_m["usage"],
                "fm_loss_pose": fm_loss_pose if future_poses is not None else 0.0,
            })

        # Orthogonal Subspace Regularization
        if task == "all" and self.cfg.decoder.get("orthogonal_reg_weight", 0.0) > 0.0:
            ortho_weight = self.cfg.decoder.orthogonal_reg_weight

            # Cosine similarity between the continuous quantized latents
            # Aggregate the image latents to 1D temporal to match action and pose shapes
            z_img_q_agg = z_img_q.view(b, pred_len, self.k, -1).mean(dim=2)
            sim_img_act = torch.nn.functional.cosine_similarity(z_img_q_agg, z_act_q_agg, dim=-1).abs().mean()
            sim_img_pose = torch.nn.functional.cosine_similarity(z_img_q_agg, z_pose_q_agg, dim=-1).abs().mean()
            sim_act_pose = torch.nn.functional.cosine_similarity(z_act_q_agg, z_pose_q_agg, dim=-1).abs().mean()

            loss_ortho = sim_img_act + sim_img_pose + sim_act_pose
            vq_loss = vq_loss + (ortho_weight * loss_ortho)

            metrics["ortho_loss"] = loss_ortho

        return pred_poses, pred_actions, pred_images, vq_loss, metrics

    @torch.no_grad()
    def rollout(
        self,
        imgs: torch.Tensor | None,
        actions: torch.Tensor,
        poses: torch.Tensor,
        pred_len: int,
        autoregressive: bool = False,
        task: str = "all",
        future_imgs_latent: torch.Tensor | None = None,
        sample_images: bool = False,
        markovian: bool | None = None,
        future_actions: torch.Tensor | None = None,
        future_poses: torch.Tensor | None = None,
    ):
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}")

        if not autoregressive:
            pred_len = min(pred_len, self.cfg.pred_len)

        obs_len = actions.shape[1]
        remaining = pred_len
        hist_actions, hist_poses = actions, poses

        # Pre-encode observed image context once
        if imgs is not None and imgs.ndim == 5:
            b_sz, t_sz = imgs.shape[:2]
            visual_encoder = self.transformer_encoder.unified_encoder.visual_encoder
            z_img = visual_encoder.encode(imgs.reshape(b_sz * t_sz, *imgs.shape[2:]))
            hist_imgs = z_img.reshape(b_sz, t_sz, self.k, -1)
        else:
            hist_imgs = imgs

        pose_chunks, action_chunks, image_chunks = [], [], []
        total_vq_loss = 0
        vq_metrics = {}

        while remaining > 0:
            chunk_len = min(self.cfg.pred_len, remaining)
            current_step = pred_len - remaining

            curr_future_actions = future_actions[:, current_step : current_step + chunk_len] if future_actions is not None else None
            curr_future_poses = future_poses[:, current_step : current_step + chunk_len] if future_poses is not None else None

            pred_pose, pred_actions, pred_images, vq_loss, metrics = self(
                imgs=hist_imgs,
                actions=hist_actions,
                poses=hist_poses,
                future_actions=curr_future_actions,
                future_poses=curr_future_poses,
                pred_len=chunk_len,
                task=task,
                future_imgs_latent=future_imgs_latent[:, current_step : current_step + chunk_len] if future_imgs_latent is not None else None,
                sample_images=sample_images,
                markovian=markovian,
            )
            total_vq_loss += vq_loss
            for key, value in metrics.items():
                val = value.item() if isinstance(value, torch.Tensor) else value
                vq_metrics[key] = vq_metrics.get(key, 0.0) + val

            if pred_pose is not None:
                pose_chunks.append(pred_pose)

            # Slide history using ground truth if available, preventing trajectory drift
            next_pose = curr_future_poses if curr_future_poses is not None else pred_pose
            if next_pose is not None:
                hist_poses = torch.cat([hist_poses, next_pose], dim=1)[:, -obs_len:, :]

            if pred_actions is not None:
                action_chunks.append(pred_actions)

            next_action = curr_future_actions if curr_future_actions is not None else pred_actions
            if next_action is not None:
                hist_actions = torch.cat([hist_actions, next_action], dim=1)[:, -obs_len:, :]

            if pred_images is not None:
                image_chunks.append(pred_images)

                # Keep context strictly in latent space
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

        return pred_poses, pred_actions, pred_images, total_vq_loss, vq_metrics

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
        future_actions=None,
        future_poses=None
    ):
        """Clean inference method that strips aux outputs and returns only the requested modality."""
        self.eval()
        pred_poses, pred_actions, pred_images, _, _ = self.rollout(
            imgs, actions, poses, pred_len, autoregressive, task, markovian=markovian,
            future_actions=future_actions, future_poses=future_poses
        )

        if task == "fkd":
            return pred_poses
        elif task == "ikd":
            return pred_actions
        elif task == "wm":
            return pred_images
        return pred_poses, pred_actions, pred_images

def load_model(cfg, device):
    """Initializes Hydra model"""
    model = Hydra(cfg.model).to(device)

    use_auto_weighting = cfg.train_params.get("auto_weight", True)
    # Allows manual tuning of multi-task scales when auto_weighting is disabled
    loss_weights = cfg.train_params.get("loss_weights", {"actions": 1.0, "poses": 1.0, "images": 1.0})

    criterion = MultiTaskCriterion(
        num_tasks=3,
        use_auto_weighting=use_auto_weighting,
        loss_weights=loss_weights,
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