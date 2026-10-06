import math
import torch
from einops import repeat
import torch.nn.functional as F


@torch.no_grad()
def path_follower_policy(
        model,
        init_imgs,
        init_actions,
        init_poses,
        lookahead_subgoal,
        T=None,
        num_candidates=32,
        noise_scale=0.1,
        sample_images=False,
        task_type="fkd"
):
    """
    Sub-Goal Sampler for VertiFormer.
    Samples N trajectories from the unconstrained expert prior by injecting temporally
    smoothed Gaussian noise into the continuous intent manifold. Evaluates the endpoints
    against a local lookahead subgoal to find the most optimal maneuver.
    """
    device = init_imgs.device
    b = num_candidates

    if T is None:
        T = model.cfg.pred_len

    # Batch dims...
    if init_imgs.ndim == 4:
        init_imgs = init_imgs.unsqueeze(0)
    if init_actions is not None and init_actions.ndim == 2:
        init_actions = init_actions.unsqueeze(0)
    if init_poses is not None and init_poses.ndim == 2:
        init_poses = init_poses.unsqueeze(0)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        # 1. Encode Observation History
        cached_obs_tokens = model.encode_history(
            init_imgs, init_actions, init_poses, pred_len=T
        )

        # 2. Get Unbiased Expert Prior Tokens
        tokens_uncond = model.get_intent_tokens(
            cached_obs_tokens,
            future_poses=None,
            pred_len=T,
            mask_actions=True,
            mask_poses=True
        )

        # Expand to num_candidates batch size
        tokens = tokens_uncond.expand(b, -1, -1)

        # 3. Continuous Noise Injection for Diverse Sampling
        # Since VertiFormer is deterministic, we add scaled Gaussian noise to the latent intent
        noise = torch.randn_like(tokens)

        # Colored Noise (EMA smoothing) for coherent exploration
        alpha_sm = 0.7
        smoothed_noise = noise.clone()
        for t_idx in range(1, T):
            smoothed_noise[:, t_idx * model.k : (t_idx + 1) * model.k] = \
                alpha_sm * noise[:, t_idx * model.k : (t_idx + 1) * model.k] + \
                (1 - alpha_sm) * smoothed_noise[:, (t_idx - 1) * model.k : t_idx * model.k]

        noisy_tokens = tokens + (noise_scale * smoothed_noise)

        # 4. Decode Candidate Poses
        intent_pose = noisy_tokens.reshape(b, T, model.k, -1).mean(dim=2)
        fast_pred_poses = model.pose_decoder(intent_pose)

        if fast_pred_poses.shape[-1] == 4:
            yaws = torch.atan2(fast_pred_poses[..., 3], fast_pred_poses[..., 2])
            yaws = torch.remainder(yaws + torch.pi, 2 * torch.pi) - torch.pi
            fast_pred_poses = torch.cat([fast_pred_poses[..., :2], yaws.unsqueeze(-1)], dim=-1)

        # 5. Evaluate Trajectories against the Lookahead Subgoal
        abs_pred_poses = torch.cumsum(fast_pred_poses[..., :2], dim=1)

        dist_endpoints = torch.norm(abs_pred_poses[:, -1, :2] - lookahead_subgoal.reshape(1, 2), p=2, dim=-1)
        dist_traj = torch.norm(abs_pred_poses - lookahead_subgoal.reshape(1, 1, 2), p=2, dim=-1).mean(dim=1)
        costs = dist_endpoints + (0.5 * dist_traj)

        best_idx = torch.argmin(costs)

        # 6. Extract Best Trajectory
        best_tokens = noisy_tokens[best_idx:best_idx+1]
        best_pred_poses = fast_pred_poses[best_idx:best_idx+1]

        # Decode Best Actions
        pred_actions = None
        if task_type in ["ikd", "all"]:
            intent_act = best_tokens.reshape(1, T, model.k, -1).mean(dim=2)
            pred_actions = model.action_decoder(intent_act)

        # Decode Imagination (Images)
        pred_img_latents = None
        if sample_images or task_type in ["wm", "all"]:
            mask_act = model.transformer_encoder.unified_encoder.action_mask_token.expand(1, T, -1)
            mask_pose = model.transformer_encoder.unified_encoder.pose_mask_token.expand(1, T, -1)
            query_base_act_pose = torch.cat([mask_act, mask_pose], dim=-1)
            query_base = model.f_mapping(query_base_act_pose)
            query_base = query_base.repeat_interleave(model.k, dim=1) / math.sqrt(model.k)
            query_base = query_base + model.visual_mask_token.expand(1, T * model.k, -1)
            query_pe = model.pos(query_base)
            pe_only = query_pe - query_base

            intent_tokens_flat = (best_tokens + pe_only).reshape(1 * T, model.k, -1)
            pred_images_flat = model.img_decoder(intent_tokens_flat, model.h_patches, model.w_patches)
            pred_img_latents = pred_images_flat.reshape(1, T, model.k, -1)

    return {
        "actions": pred_actions[0].cpu() if pred_actions is not None else None,
        "poses": best_pred_poses[0].cpu() if best_pred_poses is not None else None,
        "image_latents": pred_img_latents[0] if pred_img_latents is not None else None,
        "all_poses_fast": abs_pred_poses.cpu()  # Pass back 32 candidates for RViz
    }