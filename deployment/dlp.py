import time

import torch
import torch.nn.functional as F
import numpy as np

class HydraDLP:
    """
    Discrete Latent Planning (DLP) Optimizer.
    Searches over the VQ codebook manifolds using a deterministic surrogate to rapidly
    evaluate the intents.

    `use_goal_conditioned_seed` flag controlling how the search's initial intent (`current_intent`) is seeded each time
    the receding-horizon cache is empty:
      - False (default): unchanged "unbiased expert prior" behavior -- query the model with
        masked future states and let it predict its own natural intent as the search base.
      - True: seed the search toward the goal instead.
        The intent is computed as the difference between the current position and the goal position
        (normalized and scaled by the number of remaining steps), this conditions only on the
        *normalized direction* to the goal, and scales it with a step magnitude drawn from
        the robot's own recent per-step motion (`init_poses`).
    """
    def __init__(
            self,
            model,
            use_pose_head=True,
            T=5,
            num_candidates=32,
            depth_iterations=3,
            temperature=0.1,
            lambda_img=1.0,
            lambda_prior=0.5,  # Weight for the Expert Prior
            temp_discount=0.85, # Temporal decay factor for generative uncertainty
            alpha_momentum=0.5,
            top_k_sampling=10,
            use_fm_for_search=True,
            max_turn_radians_per_step=0.35,
            fm_pose_steps=5,
            fm_image_steps=10,
            search_solver='euler',
            lambda_vq_energy=1.0,     # Weight for VQ error
            lambda_entropy=0.5,       # Weight for Predictive Entropy (Uncertainty)
            use_intent_cache = False,
            debug_visualize = False,
            use_goal_conditioned_seed = False,  # seed search toward goal direction instead of the unbiased model prior
    ):
        self.model = model
        self.use_pose_head = use_pose_head
        self.T = T
        self.num_candidates = num_candidates
        self.depth_iterations = depth_iterations
        self.temperature = temperature
        self.lambda_img = lambda_img
        self.lambda_prior = lambda_prior
        self.temp_discount = temp_discount
        self.alpha_momentum = alpha_momentum
        self.top_k_sampling = top_k_sampling
        self.use_fm_for_search = use_fm_for_search
        self.max_turn_radians_per_step = max_turn_radians_per_step
        self.fm_pose_steps = fm_pose_steps
        self.fm_image_steps = fm_image_steps
        self.search_solver = search_solver
        self.lambda_vq_energy = lambda_vq_energy
        self.lambda_entropy = lambda_entropy
        self.use_intent_cache = use_intent_cache
        self.debug_visualize = debug_visualize
        self.use_goal_conditioned_seed = use_goal_conditioned_seed

        self.k = self.model.k

        # Stateful cache for receding horizon intent
        self.intent_cache = None
        self.cached_goal_latent = None
        self.cached_goal_emb = None
        self.best_intent_tokens = None
        self.global_path = None
        self.latest_all_poses = None

    def reset(self):
        """Clears the intent cache. Call this when starting a new navigation episode."""
        self.intent_cache = None
        self.best_intent_tokens = None
        self.global_path = None
        self.latest_all_poses = None

    def _goal_conditioned_seed(self, init_poses, goal_pose, device):
        """
        Builds a [1, T, pose_dim] seed trajectory whose XY direction is the normalized
        unit vector toward `goal_pose` (in the same ego-local frame as `goal_pose` and
        `init_poses`), held constant across the horizon, with a per-step magnitude taken
        from the robot's own recent motion instead of an arbitrary constant or a raw
        distance/T split. Heading (yaw, or cos/sin) is left at identity -- Site 1 noise
        and the transformer are what shape any turning, this only biases translation.
        """
        pose_dim = init_poses.shape[-1]

        goal_xy = goal_pose[:2].to(device)
        goal_dist = torch.norm(goal_xy, p=2)
        direction = goal_xy / torch.clamp(goal_dist, min=1e-6)

        # Typical per-step displacement magnitude, estimated from the observed history.
        # init_poses are step-wise local deltas in the exact same units/frame the model
        # was trained on, so this tracks train-data statistics without hardcoding a
        # dataset-specific constant.
        hist_xy_deltas = init_poses[0, :, :2]
        step_mag = torch.norm(hist_xy_deltas, p=2, dim=-1).mean()
        step_mag = torch.clamp(step_mag, min=1e-3)

        seed_xy = (direction * step_mag).view(1, 1, 2).expand(1, self.T, 2)

        seed = torch.zeros(1, self.T, pose_dim, device=device)
        seed[..., :2] = seed_xy
        if pose_dim == 4:
            # cos/sin heading representation -> identity heading (cos=1, sin=0)
            seed[..., 2] = 1.0

        return seed

    @torch.no_grad()
    def plan(self, init_imgs, init_actions, init_poses, goal_pose, goal_latent=None, debug_visualize=None):
        """
        Executes iterative intent optimization to find the lowest-cost trajectory.
        """
        device = init_imgs.device
        b_size = self.num_candidates

        debug = self.debug_visualize if debug_visualize is None else debug_visualize

        # Ensure history tensors are batched correctly
        if init_imgs.ndim == 4:
            init_imgs = init_imgs.unsqueeze(0)
        if init_actions is not None and init_actions.ndim == 2:
            init_actions = init_actions.unsqueeze(0)
        if init_poses is not None and init_poses.ndim == 2:
            init_poses = init_poses.unsqueeze(0)

        # 1. Encode History Once
        cached_obs_tokens = self.model.encode_history(init_imgs, init_actions, init_poses, pred_len=self.T)

        # 2. Manage Receding Horizon Cache
        if self.intent_cache is not None and self.use_intent_cache:
            current_intent_poses = self.intent_cache.unsqueeze(0).expand(b_size, -1, -1)
        elif self.use_goal_conditioned_seed and self.use_pose_head and goal_pose is not None:
            # --- Goal-Conditioned Seed ---
            # Bias the search's starting point toward the goal direction (normalized,
            # scaled by the robot's own recent step magnitude) instead of the unbiased
            # model-queried prior below.
            current_intent = self._goal_conditioned_seed(init_poses, goal_pose, device).expand(b_size, -1, -1)
        else:
            # --- Unbiased Expert Prior ---
            # We query the model with masked future states. This forces the world model
            # to predict its natural, obstacle-avoiding intent as the search base!
            unbiased_tokens = self.model.get_intent_tokens(
                cached_obs_tokens,
                pred_len=self.T,
                mask_actions=True,
                mask_poses=True
            )

            # Decode the unbiased tokens into continuous intent depending on the chosen head
            if self.use_pose_head:
                intent_tokens_agg = unbiased_tokens.view(1, self.T, self.k, -1).mean(dim=2)
                z_q_agg, _, _ = self.model.vq_pose(intent_tokens_agg)
                z_q_pos = self.model.pose_pos(z_q_agg)

                if self.use_fm_for_search:
                    unbiased_intent = self.model.fm_pose.sample(z_q_pos, steps=self.fm_pose_steps, solver=self.search_solver)
                else:
                    unbiased_intent = self.model.pose_decoder(z_q_pos)
                    if unbiased_intent.shape[1] > 2:
                        smoothed = unbiased_intent.clone()
                        smoothed[:, 1:-1, :] = 0.25 * unbiased_intent[:, :-2, :] + 0.5 * unbiased_intent[:, 1:-1, :] + 0.25 * unbiased_intent[:, 2:, :]
                        unbiased_intent = smoothed
            else:
                intent_tokens_agg = unbiased_tokens.view(1, self.T, self.k, -1).mean(dim=2)
                z_q_agg, _, _ = self.model.vq_action(intent_tokens_agg)
                z_q_pos = self.model.act_pos(z_q_agg)

                if self.use_fm_for_search:
                    unbiased_intent = self.model.fm_action.sample(z_q_pos, steps=self.fm_pose_steps, solver=self.search_solver)
                else:
                    unbiased_intent = self.model.action_decoder(z_q_pos)
                    if unbiased_intent.shape[1] > 2:
                        smoothed = unbiased_intent.clone()
                        smoothed[:, 1:-1, :] = 0.25 * unbiased_intent[:, :-2, :] + 0.5 * unbiased_intent[:, 1:-1, :] + 0.25 * unbiased_intent[:, 2:, :]
                        unbiased_intent = smoothed

            current_intent = unbiased_intent.expand(b_size, -1, -1)

        # Expand cached tokens for the candidate batch evaluations
        cached_obs_tokens = cached_obs_tokens.expand(b_size, -1, -1)

        # 3. Latent Image Goal Mapping
        goal_emb = None
        if goal_latent is not None:
            if self.cached_goal_latent is None or not torch.equal(self.cached_goal_latent, goal_latent):
                self.cached_goal_latent = goal_latent.clone()
                # Project into the Transformer's embedding space (emb_dim) to avoid decoding
                self.cached_goal_emb = self.model.transformer_encoder.unified_encoder.proj(goal_latent)
            goal_emb = self.cached_goal_emb.expand(b_size, -1, -1)

        best_overall_cost = float('inf')
        best_overall_poses = None
        best_overall_actions = None
        all_samples_poses = []

        for i in range(self.depth_iterations):
            # --- A. Generate Temporal Noise for Exploration ---
            if i > 0:
                intent_noise = torch.randn_like(current_intent)

                # Colored Noise (EMA smoothing) for kinematically coherent exploration
                alpha_sm = 0.7
                smoothed_noise = intent_noise.clone()
                for t_idx in range(1, self.T):
                    smoothed_noise[:, t_idx] = alpha_sm * intent_noise[:, t_idx] + (1 - alpha_sm) * smoothed_noise[:, t_idx - 1]

                noise_scale = 0.05 * (0.5 ** i)
                noisy_intent = current_intent + (noise_scale * smoothed_noise)
            else:
                noisy_intent = current_intent

            # --- B. Get Continuous Intent ---
            if self.use_pose_head:
                tokens = self.model.get_intent_tokens(cached_obs_tokens, future_poses=noisy_intent, pred_len=self.T)
                codebook_intent = F.normalize(self.model.vq_pose.embedding.weight, p=2, dim=1)
            else:
                tokens = self.model.get_intent_tokens(cached_obs_tokens, future_actions=noisy_intent, pred_len=self.T)
                codebook_intent = F.normalize(self.model.vq_action.embedding.weight, p=2, dim=1)

            # --- C. Temporally Correlated Gumbel-Max Sampling ---
            flat_tokens = F.normalize(tokens.view(-1, self.model.emb_dim), p=2, dim=1)
            intent_logits = torch.matmul(flat_tokens, codebook_intent.t()).view(b_size, self.T * self.k, -1)

            curr_temp = self.temperature * (0.1 ** i)
            u = torch.rand_like(intent_logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8).view(b_size, self.T, self.k, -1)

            alpha_noise = min(0.4 + (0.15 * i), 1.0)
            smoothed_gumbel = gumbel_noise.clone()
            for t in range(1, self.T):
                smoothed_gumbel[:, t] = alpha_noise * gumbel_noise[:, t] + (1 - alpha_noise) * smoothed_gumbel[:, t - 1]

            noisy_logits = (intent_logits / curr_temp) + smoothed_gumbel.view(b_size, self.T * self.k, -1)

            _, top_k_indices = torch.topk(intent_logits, self.top_k_sampling, dim=-1)
            mask = torch.full_like(intent_logits, float('-inf'))
            mask.scatter_(dim=-1, index=top_k_indices, value=0.0)

            intent_idx = torch.argmax(noisy_logits + mask, dim=-1)

            # Expert Prior Costing ---
            # Compute the Negative Log-Likelihood of the sampled tokens under the expert distribution
            scaled_logits = intent_logits / self.temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)

            # Define Temporal Discounting: Penalize immediate frames heavily, decay future frames due to uncertainty
            discounts = (self.temp_discount ** torch.arange(self.T, device=device))

            sampled_log_probs = torch.gather(log_probs, -1, intent_idx.unsqueeze(-1)).squeeze(-1)
            sampled_log_probs_t = sampled_log_probs.view(b_size, self.T, self.k).mean(dim=-1)
            prior_cost = -(sampled_log_probs_t * discounts.unsqueeze(0)).mean(dim=-1)

            # Visual Predictive Entropy
            # Obstacles are primarily visual. If the robot drives into an unseen wall,
            # the visual distribution flattens (high uncertainty).
            codebook_img = F.normalize(self.model.vq_image.embedding.weight, p=2, dim=1)
            img_logits = torch.matmul(flat_tokens, codebook_img.t()).view(b_size, self.T, self.k, -1)

            img_scaled_logits = img_logits / self.temperature
            img_log_probs = F.log_softmax(img_scaled_logits, dim=-1)
            img_probs = F.softmax(img_scaled_logits, dim=-1)

            entropy_per_t = -(img_probs * img_log_probs).sum(dim=-1).mean(dim=2)
            entropy_cost = (entropy_per_t * discounts.unsqueeze(0)).mean(dim=-1)

            # Geometry Cost (Flow Matching or MLP Surrogate)
            if self.use_pose_head:
                z_pose_q = F.embedding(intent_idx, codebook_intent).view(b_size, self.T, self.k, -1).mean(dim=2)
                z_pose_q_pos = self.model.pose_pos(z_pose_q)
            else:
                # Query raw tokens to reconstruct the physical poses for spatial evaluations
                z_pose_q, _, _ = self.model.vq_pose(tokens)
                z_pose_q_pos = self.model.pose_pos(z_pose_q.view(b_size, self.T, self.k, -1).mean(dim=2))

            if self.use_fm_for_search:
                eval_pred_poses = self.model.fm_pose.sample(z_pose_q_pos, steps=self.fm_pose_steps, solver=self.search_solver)
            else:
                eval_pred_poses = self.model.pose_decoder(z_pose_q_pos)
                # Slight Temporal Smoothing (1D Convolution) for MLP outputs for positional smoothness
                if eval_pred_poses.shape[1] > 2:
                    smoothed_poses = eval_pred_poses.clone()
                    # Smooth all dimensions
                    smoothed_poses[:, 1:-1, :] = 0.25 * eval_pred_poses[:, :-2, :] + 0.5 * eval_pred_poses[:, 1:-1, :] + 0.25 * eval_pred_poses[:, 2:, :]
                    eval_pred_poses = smoothed_poses

            # Enforce Kinematic Feasibility
            if eval_pred_poses.shape[-1] == 3:
                # The FM head natively predicts [dx, dy, dyaw].
                delta_yaws = eval_pred_poses[..., 2]
                delta_yaws = torch.remainder(delta_yaws + torch.pi, 2 * torch.pi) - torch.pi
                clamped_deltas = torch.clamp(delta_yaws, -self.max_turn_radians_per_step, self.max_turn_radians_per_step)
                eval_pred_poses[..., 2] = clamped_deltas
                eval_poses_se2 = eval_pred_poses
            elif eval_pred_poses.shape[-1] == 4:
                # Fallback for continuous SO(2) representations
                # hydra.py definition maps index 2 -> cos(yaw), index 3 -> sin(yaw)
                yaws = torch.atan2(eval_pred_poses[..., 3], eval_pred_poses[..., 2])
                yaws = torch.remainder(yaws + torch.pi, 2 * torch.pi) - torch.pi
                clamped_deltas = torch.clamp(yaws, -self.max_turn_radians_per_step, self.max_turn_radians_per_step)

                # Write back normalized continuous rotations
                eval_pred_poses[..., 2] = torch.cos(clamped_deltas)
                eval_pred_poses[..., 3] = torch.sin(clamped_deltas)

                # Compress back to 3D SE(2) representation [x, y, yaw]
                eval_poses_se2 = torch.cat([eval_pred_poses[..., :2], clamped_deltas.unsqueeze(-1)], dim=-1)
            else:
                eval_poses_se2 = eval_pred_poses

            # Action Decoding (if required)
            eval_pred_actions = None
            if not self.use_pose_head:
                z_act_q = F.embedding(intent_idx, codebook_intent).view(b_size, self.T, self.k, -1).mean(dim=2)
                z_act_q_pos = self.model.act_pos(z_act_q)

                if self.use_fm_for_search:
                    eval_pred_actions = self.model.fm_action.sample(z_act_q_pos, steps=self.fm_pose_steps, solver=self.search_solver)
                else:
                    eval_pred_actions = self.model.action_decoder(z_act_q_pos)
                    # Smooth continuous actions
                    if eval_pred_actions.shape[1] > 2:
                        smoothed_actions = eval_pred_actions.clone()
                        smoothed_actions[:, 1:-1, :] = 0.25 * eval_pred_actions[:, :-2, :] + 0.5 * eval_pred_actions[:, 1:-1, :] + 0.25 * eval_pred_actions[:, 2:, :]
                        eval_pred_actions = smoothed_actions

            # Because the goal and the deltas are both in the local ego-centric frame,
            # we evaluate distance using a simple cumulative sum from the origin.
            p_pos = eval_pred_poses[..., :2]
            abs_p_pos = torch.cumsum(p_pos, dim=1)

            all_samples_poses.append(abs_p_pos)

            dist_endpoint = torch.norm(abs_p_pos[:, -1, :2] - goal_pose[:2].view(1, 2), p=2, dim=-1)

            dist_traj_per_t = torch.norm(abs_p_pos - goal_pose[:2].view(1, 1, 2), p=2, dim=-1)
            dist_traj = (dist_traj_per_t * discounts.unsqueeze(0)).mean(dim=-1)
            dist_cost = dist_endpoint + dist_traj

            # Image Cost (VQ Space Evaluation)
            img_cost = torch.zeros(b_size, device=device)

            if goal_emb is not None:
                z_img_q, _, _ = self.model.vq_image(tokens)
                z_img_q = z_img_q.view(b_size, self.T, self.k, -1)

                pred_last_img_emb = z_img_q[:, -1, :, :]
                img_cost = F.mse_loss(pred_last_img_emb, goal_emb, reduction='none').mean(dim=(1, 2))
                img_cost = img_cost * discounts[-1]  # Discount terminal visual goal state

            # VQ Quantization Error ---
            # Measures how far the continuous intent tokens fell into the "dead space"
            # of the visual codebook (i.e., hallucinating non-physical scenes).
            # We reuse the `img_logits` (cosine similarities) computed above
            max_sim_img, _ = torch.max(img_logits, dim=-1)
            vq_energy_per_t = (2.0 - 2.0 * max_sim_img).mean(dim=2)
            vq_energy_cost = (vq_energy_per_t * discounts.unsqueeze(0)).mean(dim=-1)

            # Dynamic Cost Normalization
            # Min-Max normalize each objective across the batch to [0, 1] so no single
            # metric's absolute scale dominates the others during candidate selection.
            def normalize_cost(c, min_range=1e-5):
                c_min, c_max = c.min(), c.max()
                return (c - c_min) / torch.clamp(c_max - c_min, min=min_range)

            norm_dist = normalize_cost(dist_cost)
            norm_img = normalize_cost(img_cost)
            norm_prior = normalize_cost(prior_cost)

            # Max Entropy for K=1024 is ~6.9. A variance of < 0.5 is meaningless model noise.
            norm_entropy = normalize_cost(entropy_cost, min_range=0.5)

            # Max VQ L2 distance is 4.0. A variance of < 0.1 is just standard embedding jitter.
            norm_vq_energy = normalize_cost(vq_energy_cost, min_range=0.1)

            # Iteration-local normalized objective for fair comparison
            costs = norm_dist + (self.lambda_img * norm_img) + \
                    (self.lambda_prior * norm_prior) + \
                    (self.lambda_entropy * norm_entropy) + (self.lambda_vq_energy * norm_vq_energy)

            best_idx = torch.argmin(costs)

            # Track absolute progress across iterations using the raw unnormalized sum
            # This ensures that an iteration with strictly better absolute distances wins overall
            absolute_cost_val = dist_cost[best_idx].item() + (self.lambda_img * img_cost[best_idx].item()) + \
                                (self.lambda_prior * prior_cost[best_idx].item()) + \
                                (self.lambda_entropy * entropy_cost[best_idx].item()) + (self.lambda_vq_energy * vq_energy_cost[best_idx].item())

            if absolute_cost_val < best_overall_cost:
                best_overall_cost = absolute_cost_val
                best_overall_poses = eval_poses_se2[best_idx]  # return 3D SE(2) local deltas
                if not self.use_pose_head:
                    best_overall_actions = eval_pred_actions[best_idx]

                # Cache the raw transformer tokens of the elite trajectory for optional visual decoding
                self.best_intent_tokens = tokens[best_idx].unsqueeze(0).clone()

            # Update Stateful Cache (Momentum Refinement)
            if self.use_pose_head:
                elite_traj = eval_poses_se2[best_idx]
            else:
                elite_traj = eval_pred_actions[best_idx]

            elite_traj_exp = elite_traj.unsqueeze(0).expand(b_size, -1, -1)
            current_intent = (self.alpha_momentum * elite_traj_exp) + ((1 - self.alpha_momentum) * current_intent)
            self.intent_cache = current_intent[0].detach().clone()

        # Cache all candidate paths from the final iteration (already in cumulative local space)
        self.latest_all_poses = all_samples_poses[-1].detach().cpu() if all_samples_poses else None

        return {
            "poses": best_overall_poses.cpu() if best_overall_poses is not None else None,
            "actions": best_overall_actions.cpu() if best_overall_actions is not None else None
        }

    @torch.no_grad()
    def get_imagination(self):
        """
        Decodes the visual imagination of the best planned trajectory.
        Call this after plan() when you want to visualize the robot's intended path.
        """
        if self.best_intent_tokens is None:
            return None

        # 1. Quantize the best tokens to enforce discrete structural validity
        z_img_q, _, _ = self.model.vq_image(self.best_intent_tokens)

        # 2. Add spatial positional encoding back in
        z_img_q_spatial = self.model.img_pos(z_img_q)
        cond = z_img_q_spatial.view(1, self.T, self.k, self.model.emb_dim)

        # 3. Sample continuous visual latents via the Image Flow Matching ODE
        img_latents = self.model.fm_image.sample(cond, steps=self.fm_image_steps, solver=self.search_solver)

        # Squeeze batch dimension -> Returns [T, k, visual_latent_dim]
        return img_latents.squeeze(0)

    def ensemble_trajectory(self, new_local_deltas, current_odom, blend_momentum=0.6):
        """
        Action Chunking with Temporal Ensembling (ALOHA style) for spatial waypoints.
        Fuses a newly predicted local trajectory with the existing global path.

        Args:
            new_local_deltas: Tensor of shape [T, 2] or [T, 3] representing the new plan (local deltas).
            current_odom: Tensor or array of shape [3] representing the robot's CURRENT real-time pose.
            blend_momentum: How much to trust the NEW trajectory (0.0 to 1.0).

        Returns:
            The fused, continuous global path as a Tensor.
        """
        if new_local_deltas.ndim == 3:
            new_local_deltas = new_local_deltas.squeeze(0)

        new_local_deltas = new_local_deltas.cpu()
        if isinstance(current_odom, np.ndarray):
            current_odom = torch.from_numpy(current_odom).float()
        current_odom = current_odom.cpu()

        # Convert local step-wise deltas to global Odom frame
        local_abs_path = torch.cumsum(new_local_deltas[..., :2], dim=0)
        odom_x, odom_y, odom_yaw = current_odom[0].item(), current_odom[1].item(), current_odom[2].item()

        cos_y_inv, sin_y_inv = np.cos(odom_yaw), np.sin(odom_yaw)
        R_inv = torch.tensor([[cos_y_inv, -sin_y_inv],
                              [sin_y_inv,  cos_y_inv]], dtype=torch.float32)

        global_path_xy = local_abs_path @ R_inv.T
        global_path_xy[:, 0] += odom_x
        global_path_xy[:, 1] += odom_y

        if new_local_deltas.shape[-1] == 3:
            local_abs_yaw = torch.cumsum(new_local_deltas[..., 2], dim=0)
            global_yaw = (local_abs_yaw + odom_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            new_traj_odom = torch.cat([global_path_xy, global_yaw.unsqueeze(-1)], dim=-1)
        else:
            new_traj_odom = global_path_xy

        # 1. Initialize if empty
        if self.global_path is None or len(self.global_path) < 2:
            self.global_path = new_traj_odom
            return self._slice_path_from_current(self.global_path, current_odom)

        # 2. Slice off waypoints the robot has already passed
        clean_old_path = self._slice_path_from_current(self.global_path, current_odom)
        clean_new_path = self._slice_path_from_current(new_traj_odom, current_odom)

        if len(clean_old_path) == 0:
            self.global_path = clean_new_path
            return self.global_path
        if len(clean_new_path) == 0:
            self.global_path = clean_old_path
            return self.global_path

        # 3. Temporal Ensembling (Blend overlapping segments)
        overlap_len = min(len(clean_old_path), len(clean_new_path))
        blended_path = []

        for i in range(overlap_len):
            blended_pt = (blend_momentum * clean_new_path[i]) + ((1.0 - blend_momentum) * clean_old_path[i])
            if blended_pt.shape[-1] == 3:
                yaw_old = clean_old_path[i, 2]
                yaw_new = clean_new_path[i, 2]
                yaw_diff = torch.remainder(yaw_new - yaw_old + torch.pi, 2 * torch.pi) - torch.pi
                blended_yaw = yaw_old + blend_momentum * yaw_diff
                blended_pt[2] = torch.remainder(blended_yaw + torch.pi, 2 * torch.pi) - torch.pi
            blended_path.append(blended_pt)

        # 4. Extend the path (Append the future that doesn't overlap)
        if len(clean_new_path) > overlap_len:
            blended_path.extend(clean_new_path[overlap_len:])
        elif len(clean_old_path) > overlap_len:
            blended_path.extend(clean_old_path[overlap_len:])

        self.global_path = torch.stack(blended_path)
        return self.global_path

    def _slice_path_from_current(self, path, current_odom):
        """Discards waypoints that fall behind the robot's current position."""
        if len(path) == 0:
            return path

        current_pos = current_odom[:2]
        distances = torch.norm(path[:, :2] - current_pos.unsqueeze(0), dim=-1)
        closest_idx = torch.argmin(distances).item()

        # If the closest point is the last point and > 0.5m away, the path is entirely exhausted
        if closest_idx == len(path) - 1 and distances[closest_idx] > 0.5:
            return torch.empty((0, path.shape[-1]), device=path.device)

        return path[closest_idx:]