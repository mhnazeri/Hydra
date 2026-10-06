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
        guidance_scale=0.5  # 1.0 = Strict Path Follower, 0.0 = Pure Obstacle Avoidance Prior
):
    """
    Classifier-Free Guidance (CFG) / Logit-Level Path Guidance Execution.
    Blends the unconstrained expert prior with the rigidly requested
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
        # 1. Encode Observation History
        cached_obs_tokens = model.encode_history(
            init_imgs, init_actions, init_poses, pred_len=T
        )

        # 2. Get Unconditional Prior Tokens (Safe, obstacle-free manifold)
        # mask_poses=True hides the global path, forcing the model to guess the best path safely
        tokens_uncond = model.get_intent_tokens(
            cached_obs_tokens,
            future_poses=None,
            pred_len=T,
            mask_actions=True,
            mask_poses=True
        )

        # 3. Get Conditional Tokens (Goal-directed tracking)
        # mask_poses=False feeds the rigid global path to the model
        tokens_cond = model.get_intent_tokens(
            cached_obs_tokens,
            future_poses=future_poses,
            pred_len=T,
            mask_actions=True,
            mask_poses=False
        )

        # 4. Classifier-Free Guidance (Latent Blending)
        # guidance_scale=1.0 -> Follows path strictly.
        # guidance_scale=0.0 -> Ignores path, avoids obstacles purely.
        tokens_cfg = tokens_uncond + guidance_scale * (tokens_cond - tokens_uncond)

        # 5. Decode the blended latent intent into physical modalities
        pred_poses, pred_poses_uncond, pred_poses_cond = None, None, None
        pred_actions, pred_img_latents = None, None

        # --- Decode Poses ---
        if task_type in ["fkd", "all"]:
            def decode_pose(tokens):
                intent_pose = tokens.view(b, T, model.k, -1).mean(dim=2)
                z_pose, _, _ = model.vq_pose(intent_pose)
                z_pose_pos = model.pose_pos(z_pose)
                p = model.fm_pose.sample(z_pose_pos, steps=model.fm_pose_steps)
                if p.shape[-1] == 4:
                    yaws = torch.atan2(p[..., 3], p[..., 2])
                    yaws = torch.remainder(yaws + torch.pi, 2 * torch.pi) - torch.pi
                    p = torch.cat([p[..., :2], yaws.unsqueeze(-1)], dim=-1)
                return p

            pred_poses = decode_pose(tokens_cfg)
            pred_poses_uncond = decode_pose(tokens_uncond)
            pred_poses_cond = decode_pose(tokens_cond)

        # --- Decode Actions ---
        if task_type in ["ikd", "all"]:
            intent_act = tokens_cfg.view(b, T, model.k, -1).mean(dim=2)
            z_act, _, _ = model.vq_action(intent_act)
            z_act_pos = model.act_pos(z_act)
            pred_actions = model.fm_action.sample(z_act_pos, steps=model.fm_action_steps)

        # --- Decode Imagination (Images) ---
        if sample_images or task_type in ["wm", "all"]:
            z_img, _, _ = model.vq_image(tokens_cfg)
            z_img_pos = model.img_pos(z_img)
            cond_img = z_img_pos.view(b, T, model.k, model.emb_dim)
            pred_img_latents = model.fm_image.sample(cond_img, steps=model.fm_image_steps)

    return {
        "actions": pred_actions[0].cpu() if pred_actions is not None else None,
        "poses": pred_poses[0].cpu() if pred_poses is not None else None,
        "poses_uncond": pred_poses_uncond[0].cpu() if pred_poses_uncond is not None else None,
        "poses_cond": pred_poses_cond[0].cpu() if pred_poses_cond is not None else None,
        "image_latents": pred_img_latents[0] if pred_img_latents is not None else None,
    }