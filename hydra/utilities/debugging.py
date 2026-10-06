import io
import inspect
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision.utils import make_grid, save_image
from sklearn.manifold import TSNE
from hydra.utilities.visualize import visualize_trajectory_2d, visualize_hydra_attention


def calculate_gradient_interference(model, shared_layer_name, loss_1, loss_2):
    """
    Calculates cosine similarity between the gradients of two tasks
    w.r.t shared weights.
    """
    # Ensure losses are differentiable tensors in the current graph
    if not isinstance(loss_1, torch.Tensor) or not loss_1.requires_grad:
        return 0.0
    if not isinstance(loss_2, torch.Tensor) or not loss_2.requires_grad:
        return 0.0

    # 1. Isolate the shared parameters (e.g., the last layer of the transformer)
    shared_params = [p for n, p in model.named_parameters() if shared_layer_name in n]
    if not shared_params:
        return 0.0

    # 2. Compute gradients for Task 1
    # Use retain_graph=True because we need to backprop the second loss next
    grad_1 = torch.autograd.grad(loss_1, shared_params, retain_graph=True, allow_unused=True)
    valid_grad_1 = [g.contiguous().view(-1) for g in grad_1 if g is not None]
    if not valid_grad_1:
        return 0.0
    grad_1_vec = torch.cat(valid_grad_1)

    # 3. Compute gradients for Task 2
    grad_2 = torch.autograd.grad(loss_2, shared_params, retain_graph=True, allow_unused=True)
    valid_grad_2 = [g.contiguous().view(-1) for g in grad_2 if g is not None]
    if not valid_grad_2:
        return 0.0
    grad_2_vec = torch.cat(valid_grad_2)

    # 4. Calculate Cosine Similarity
    cos_sim = torch.nn.functional.cosine_similarity(grad_1_vec, grad_2_vec, dim=0)

    return cos_sim.item()


def visualize_codebook(vq_layer, name, epoch, iteration, save_dir, logger=None, prefix="train", step=None, is_main_process=True):
    """
    Extracts the codebook weights and plots them in 2D.
    """
    if step is None:
        step = iteration

    if hasattr(vq_layer, "codebook"):
        codes = vq_layer.codebook.detach().cpu().numpy()  # [K, D]
    else:
        codes = vq_layer.embedding.weight.detach().cpu().numpy()  # [K, D]

    # Project 512D to 2D
    tsne = TSNE(n_components=2, perplexity=min(30, codes.shape[0] - 1))
    codes_2d = tsne.fit_transform(codes)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(codes_2d[:, 0], codes_2d[:, 1], alpha=0.6)
    ax.set_title(f"{name.capitalize()} Codebook Distribution (t-SNE) | Epoch {epoch:03d} | Iter {iteration:05d}")

    if is_main_process:
        save_path = Path(save_dir) / f"{prefix}_codebook" / f"epoch_{epoch:03d}_iter_{iteration:05d}"
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{name}.png", bbox_inches='tight', pad_inches=0.1)

        if logger is not None:
            logger.log_figure(
                figure_name=f"{prefix}/codebook_{name}/epoch_{epoch:03d}_iter_{iteration:05d}",
                figure=fig,
                step=step,
            )

    plt.close(fig)


def log_visualizations(
    m_model,
    cfg,
    logger,
    epoch,
    iteration,
    global_step,
    prefix,
    idx,
    data,
    past_frames_eval,
    pred_images_eval,
    img_gt_eval,
    abs_pred,
    abs_gt,
    curr_pred_len,
    future_vw_target,
    future_positions_target,
    image_denormalizer,
    is_main_process=True,
):
    """
    Consolidates trajectory, attention, codebook, reconstruction, and decoder diagnostics logging.
    """
    # Trajectory Visualization
    abs_pred_vis = torch.cat([torch.zeros_like(abs_pred[idx][:1]), abs_pred[idx]], dim=0).cpu().numpy()
    abs_gt_vis = torch.cat([torch.zeros_like(abs_gt[idx][:1]), abs_gt[idx]], dim=0).cpu().numpy()

    fig_traj = visualize_trajectory_2d(
        abs_pred_vis, abs_gt_vis, orientation=False
    )

    if is_main_process:
        save_dir_traj = Path(cfg.directory.save) / f"{prefix}_trajectory"
        save_dir_traj.mkdir(parents=True, exist_ok=True)
        save_path_traj = save_dir_traj / f"epoch_{epoch:03d}_iter_{iteration:05d}_sample_{idx:02d}.png"
        fig_traj.savefig(save_path_traj, bbox_inches="tight", dpi=120)

        buf = io.BytesIO()
        fig_traj.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)

        trajectory_name = 'trajectory_3d' if prefix == 'train' else 'trajectory'
        letter = 'T' if prefix == 'train' else 'V'
        logger.log_image(
            buf,
            name=f"{prefix}/{trajectory_name}/E{epoch:03d}_{letter}{global_step:05d}",
            step=global_step,
        )
    plt.close(fig_traj)


    # Codebook Visualization
    if hasattr(m_model, "vq_action"):
        visualize_codebook(m_model.vq_action, "actions", epoch, iteration, cfg.directory.save, logger, prefix=prefix, step=global_step, is_main_process=is_main_process)
        visualize_codebook(m_model.vq_pose, "poses", epoch, iteration, cfg.directory.save, logger, prefix=prefix, step=global_step, is_main_process=is_main_process)
        visualize_codebook(m_model.vq_image, "images", epoch, iteration, cfg.directory.save, logger, prefix=prefix, step=global_step, is_main_process=is_main_process)

    # Reconstruction & Decoder Diagnostics
    if cfg.train_params.task == "all" or cfg.train_params.task == "wm":
        if pred_images_eval is not None:
            recon_samples_hat = pred_images_eval[:8].detach().cpu()
            T = img_gt_eval.size(1)

            if is_main_process:
                save_dir = Path(
                    cfg.directory.save) / f"{prefix}_reconstruction" / f"epoch_{epoch:03d}_iter_{iteration:05d}"
                save_dir.mkdir(parents=True, exist_ok=True)

            # Frames are already denormalized to [0, 1] by caller.
            normalize_flag = False
            val_range = (0, 1)

            if is_main_process:
                for b, (gt_seq, pred_seq) in enumerate(zip(img_gt_eval[:8].cpu(), recon_samples_hat)):
                    save_image(
                        torch.cat([gt_seq, pred_seq], dim=0),
                        save_dir / f"sample_{b:02d}.png",
                        nrow=T,
                        value_range=val_range,
                        normalize=normalize_flag,
                        padding=2,
                    )

            log_samples = make_grid(
                [
                    make_grid(
                        [gt[i], pred[i]],
                        nrow=recon_samples_hat.size(0),
                        value_range=val_range,
                        normalize=normalize_flag,
                        padding=5,
                    )
                    for gt, pred in zip(img_gt_eval[:3].cpu(), recon_samples_hat)
                    for i in range(gt.size(0))
                ],
                nrow=T,
                padding=5,
            )

            fig, ax = plt.subplots(figsize=(max(10, T * 2.5), len(recon_samples_hat) * 2))
            ax.imshow(log_samples.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
            ax.set_title(
                f"{prefix.capitalize()} Reconstruction | Epoch {epoch:03d} | Iter {iteration:05d}\nRows: Batch Samples | Cols: Time Steps (Left: Ground Truth, Right: Prediction)",
                fontsize=14)
            ax.axis("off")

            if is_main_process:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)

                logger.log_image(
                    buf,
                    name=f"{prefix}/reconstruction/epoch_{epoch:03d}_iter_{iteration:05d}",
                    step=global_step,
                )
            plt.close(fig)