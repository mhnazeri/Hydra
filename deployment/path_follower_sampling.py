import rospy
import torch


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
        subgoal_cost_weight=1.0,
        sample_images=False,
        task_type="fkd",
):
    """
    Sub-Goal Sampler.

    Builds a diverse candidate set from the unconstrained expert prior and scores it
    against a local lookahead subgoal.
    """
    device = init_imgs.device

    if T is None:
        T = model.cfg.pred_len

    # Ensure history tensors are batched correctly
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

        # 2. Get Unbiased Expert Prior Tokens (Obstacle-Free Manifold)
        tokens_uncond = model.get_intent_tokens(
            cached_obs_tokens,
            future_poses=None,
            pred_len=T,
            mask_actions=True,
            mask_poses=True
        )

        intent_prior = tokens_uncond.view(1, T, model.k, -1).mean(dim=2)
        d = intent_prior.shape[-1]

        # 3. Build the Candidate Set
        codebook = model.vq_pose.embedding.weight  # [C, d] real VQ embeddings
        C = codebook.shape[0]
        budget = max(num_candidates - 1, 0)
        n_maneuver = min(C, (budget + 1) // 2)
        n_mix = budget - n_maneuver

        cand_list = [intent_prior]
        if n_maneuver > 0:
            if n_maneuver < C:
                code_idx = torch.randperm(C, device=device)[:n_maneuver]
            else:
                code_idx = torch.arange(C, device=device)
            pure = codebook[code_idx].unsqueeze(1).expand(-1, T, -1)
            cand_list.append(pure)
        if n_mix > 0:
            mix_idx = torch.randint(C, (n_mix,), device=device)
            lam = torch.rand(n_mix, 1, 1, device=device, dtype=intent_prior.dtype) * 0.75 + 0.25
            mixed = intent_prior + lam * (codebook[mix_idx].unsqueeze(1) - intent_prior)
            cand_list.append(mixed)

        intents = torch.cat(cand_list, dim=0)
        b = intents.shape[0]

        # Optional small jitter, drawn per (candidate, timestep)
        if noise_scale > 0:
            noise = torch.randn(b, T, 1, d, device=device, dtype=intents.dtype)
            alpha_sm = 0.7
            smoothed = noise.clone()
            for t_idx in range(1, T):
                smoothed[:, t_idx] = alpha_sm * noise[:, t_idx] + (1 - alpha_sm) * smoothed[:, t_idx - 1]
            token_rms = intents.float().pow(2).mean().sqrt().to(intents.dtype)
            intents = intents + noise_scale * token_rms * smoothed.reshape(b, T, d)

        # 4. Candidate Decoding for Scoring
        # The deterministic pose_decoder is used (not the flow-matching sampler):
        # it is deterministic, so selection is stable, and it is trained on
        # quantized intents + positional encoding. The FM sampler for maneuver-code candidates
        # has a large decode variance, which would make selection meaningless.
        z_pose, _, _ = model.vq_pose(intents)
        z_pose_pos = model.pose_pos(z_pose)
        cand_pred_poses = model.pose_decoder(z_pose_pos)

        if cand_pred_poses.shape[-1] == 4:
            yaws = torch.atan2(cand_pred_poses[..., 3], cand_pred_poses[..., 2])
            yaws = torch.remainder(yaws + torch.pi, 2 * torch.pi) - torch.pi
            cand_pred_poses = torch.cat([cand_pred_poses[..., :2], yaws.unsqueeze(-1)], dim=-1)
        elif cand_pred_poses.shape[-1] == 3:
            cand_pred_poses[..., 2] = torch.remainder(
                cand_pred_poses[..., 2] + torch.pi, 2 * torch.pi) - torch.pi

        # 5. Evaluate Trajectories against the Lookahead Subgoal
        abs_pred_poses = torch.cumsum(cand_pred_poses[..., :2], dim=1)

        dist_endpoints = torch.norm(abs_pred_poses[:, -1, :2] - lookahead_subgoal.reshape(1, 2), p=2, dim=-1)
        dist_traj = torch.norm(abs_pred_poses - lookahead_subgoal.reshape(1, 1, 2), p=2, dim=-1).mean(dim=1)
        costs = subgoal_cost_weight * (dist_endpoints + (0.5 * dist_traj))

        best_idx = torch.argmin(costs)

        rospy.loginfo(
            "[path_follower_policy] candidates=%d noise=%.3f | best_idx=%d | subgoal=(%.3f, %.3f) | "
            "endpoint std=(%.3f, %.3f) | sample endpoints: %s",
            b,
            noise_scale,
            int(best_idx.item()),
            float(lookahead_subgoal[0]),
            float(lookahead_subgoal[1]),
            float(abs_pred_poses[:, -1, 0].std().item()),
            float(abs_pred_poses[:, -1, 1].std().item()),
            " ".join([f"({p[0]:.2f},{p[1]:.2f})" for p in abs_pred_poses[:8, -1, :2].cpu().numpy()]),
        )

        # 6. Execute the Winner: reuse the deterministic decode
        best_pred_poses = cand_pred_poses[best_idx:best_idx + 1]
        best_intent = intents[best_idx:best_idx + 1]

        best_z_act, _, _ = model.vq_action(best_intent)
        best_z_act_pos = model.act_pos(best_z_act)
        pred_actions = model.action_decoder(best_z_act_pos)

        pred_img_latents = None
        if sample_images:
            best_tokens = best_intent.unsqueeze(2).expand(-1, -1, model.k, -1).reshape(1, T * model.k, -1)
            z_img_q, _, _ = model.vq_image(best_tokens)
            z_img_pos = model.img_pos(z_img_q)
            cond_img = z_img_pos.view(1, T, model.k, model.emb_dim)
            pred_img_latents = model.fm_image.sample(cond_img, steps=model.fm_image_steps)

    return {
        "actions": pred_actions[0].float().cpu() if pred_actions is not None else None,
        "poses": best_pred_poses[0].float().cpu() if best_pred_poses is not None else None,
        "image_latents": pred_img_latents[0] if pred_img_latents is not None else None,
        "all_poses_fast": abs_pred_poses.cpu()  # Pass back N candidates for RViz
    }