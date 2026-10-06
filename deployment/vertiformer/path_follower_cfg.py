import math
import torch
from einops import repeat


@torch.no_grad()
def path_follower_policy(
        model,
        init_imgs,
        init_actions,
        init_poses,
        future_poses,
        T=None,
        sample_images=False,
        task_type="fkd",
        guidance_scale=0.5,  # 1.0 = Strict Path Follower, 0.0 = Pure Obstacle Avoidance Prior
        cached_obs_tokens=None
):
    """
    Classifier-Free Guidance (CFG) / Logit-Level Path Guidance Execution for VertiFormer.
    Blends the unconstrained safe expert prior with the requested
    geometric waypoint track in the continuous token space.
    """
    device = init_imgs.device

    if T is None:
        T = future_poses.shape[1]

    # Batch dims...
    if init_imgs.ndim == 4:
        init_imgs = init_imgs.unsqueeze(0)
    if init_actions is not None and init_actions.ndim == 2:
        init_actions = init_actions.unsqueeze(0)
    if init_poses is not None and init_poses.ndim == 2:
        init_poses = init_poses.unsqueeze(0)
    if future_poses.ndim == 2:
        future_poses = future_poses.unsqueeze(0)

    b = init_imgs.shape[0]

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        # 1. Encode Observation History (if not already provided)
        if cached_obs_tokens is None:
            cached_obs_tokens = model.encode_history(
                init_imgs, init_actions, init_poses, pred_len=T
            )

        # 2. Get Unconditional Prior Tokens
        # mask_poses=True hides the global path, forcing the model to guess safely
        tokens_uncond = model.get_intent_tokens(
            cached_obs_tokens,
            future_poses=None,
            pred_len=T,
            mask_actions=True,
            mask_poses=True
        )

        # 3. Get Conditional Tokens
        # mask_poses=False feeds the rigid global path to the model
        tokens_cond = model.get_intent_tokens(
            cached_obs_tokens,
            future_poses=future_poses,
            pred_len=T,
            mask_actions=True,
            mask_poses=False
        )

        # 4. Classifier-Free Guidance (Latent Blending)
        tokens_cfg = tokens_uncond + guidance_scale * (tokens_cond - tokens_uncond)

        # 5. Decode the blended latent intent into physical modalities
        pred_poses, pred_poses_uncond, pred_poses_cond = None, None, None
        pred_actions, pred_img_latents = None, None

        # Decode Poses
        if task_type in ["fkd", "all"]:
            def decode_pose(tokens):
                intent_pose = tokens.reshape(b, T, model.k, -1).mean(dim=2)
                p = model.pose_decoder(intent_pose)
                if p.shape[-1] == 4:
                    yaws = torch.atan2(p[..., 3], p[..., 2])
                    yaws = torch.remainder(yaws + torch.pi, 2 * torch.pi) - torch.pi
                    p = torch.cat([p[..., :2], yaws.unsqueeze(-1)], dim=-1)
                return p

            pred_poses = decode_pose(tokens_cfg)
            pred_poses_uncond = decode_pose(tokens_uncond)
            pred_poses_cond = decode_pose(tokens_cond)

        # Decode Actions
        if task_type in ["ikd", "all"]:
            intent_act = tokens_cfg.reshape(b, T, model.k, -1).mean(dim=2)
            pred_actions = model.action_decoder(intent_act)

        # Decode Imagination
        if sample_images or task_type in ["wm", "all"]:
            mask_act = model.transformer_encoder.unified_encoder.action_mask_token.expand(b, T, -1)
            mask_pose = model.transformer_encoder.unified_encoder.pose_mask_token.expand(b, T, -1)
            query_base_act_pose = torch.cat([mask_act, mask_pose], dim=-1)
            query_base = model.f_mapping(query_base_act_pose)
            query_base = query_base.repeat_interleave(model.k, dim=1) / math.sqrt(model.k)
            query_base = query_base + model.visual_mask_token.expand(b, T * model.k, -1)
            query_pe = model.pos(query_base)
            pe_only = query_pe - query_base

            intent_tokens_flat = (tokens_cfg + pe_only).reshape(b * T, model.k, -1)
            pred_images_flat = model.img_decoder(intent_tokens_flat, model.h_patches, model.w_patches)
            pred_img_latents = pred_images_flat.reshape(b, T, model.k, -1)

    return {
        "actions": pred_actions[0].cpu() if pred_actions is not None else None,
        "poses": pred_poses[0].cpu() if pred_poses is not None else None,
        "poses_uncond": pred_poses_uncond[0].cpu() if pred_poses_uncond is not None else None,
        "poses_cond": pred_poses_cond[0].cpu() if pred_poses_cond is not None else None,
        "image_latents": pred_img_latents[0] if pred_img_latents is not None else None,
    }