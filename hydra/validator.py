import argparse
import gc
import sys
from pathlib import Path
from datetime import timedelta
from operator import attrgetter

import torch.distributed as dist

try:
    sys.path.append(str(Path(".").resolve()))
except Exception:
    raise RuntimeError("Can't append root directory of the project to the path")

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, broadcast
from rich import print
from rich.table import Table
from rich.console import Console
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torcheval.metrics import PeakSignalNoiseRatio

from hydra.model.dataloader import NavigationDataset, custom_collate
from hydra.utilities.helpers import get_conf, fix_seed, timeit
from hydra.utilities.evals import compute_mae, compute_ade, compute_fde, compute_ate, compute_rpe
from hydra.utilities.tracker import MetricTracker
from hydra.utilities.debugging import log_visualizations

from dreamsim import dreamsim
import lpips

class Validator:
    def __init__(self, cfg, arch="hydra", full_psnr=False,
                 accelerator=None, model=None, val_data=None, criterion=None,
                 visual_encoder=None, k=None, image_denormalizer=None,
                 log_freq=0, logger=None, dataset=None, batch_size=None):
        self.cfg = cfg
        self.arch = arch.lower()
        self.full_psnr = full_psnr
        self.log_freq = log_freq
        self.logger = logger
        self.dataset = dataset
        self.batch_size = batch_size

        # Trainer Mode: Inherit active DDP states directly from train.py
        if accelerator is not None:
            self.accelerator = accelerator
            self.device = accelerator.device
            self.model = model
            self.val_data = val_data
            self.criterion = criterion
            self.visual_encoder = visual_encoder
            self.k = k
            self.image_denormalizer = image_denormalizer
        else:
            # Standalone Mode: Re-create the pipeline from scratch
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])
            self.device = self.accelerator.device

            fix_seed(self.cfg.train_params.seed)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

            if self.arch == "hydra":
                from model.hydra import load_model
            elif self.arch == "vertiformer":
                from model.vertiformer import load_model
            else:
                raise ValueError(f"Unknown architecture: {self.arch}")

            if self.accelerator.is_main_process:
                print(f"[bold green]INITIALIZING Standalone Validation for {self.arch.capitalize()}...[/bold green]")

            self.model, _, _, self.criterion = load_model(self.cfg, self.device)
            self.val_data = self.init_dataloader()

            self.model, self.val_data, self.criterion = self.accelerator.prepare(
                self.model, self.val_data, self.criterion
            )

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.visual_encoder = unwrapped_model.transformer_encoder.unified_encoder.visual_encoder
            self.k = unwrapped_model.k if hasattr(unwrapped_model, "k") else unwrapped_model.transformer_encoder.k

        self.psnr = PeakSignalNoiseRatio().to(self.device)

        if self.accelerator.is_main_process:
            print("[bold green]Loading Perceptual Metrics (DreamSim, LPIPS, PSNR)...[/bold green]")
        # dreamsim returns (model, preprocess_transform). We only need the model.
        self.dreamsim_model, _ = dreamsim(pretrained=True, device=self.device)
        self.dreamsim_model.eval()

        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_model.eval()

    def init_dataloader(self):
        if self.dataset is not None:
            self.cfg.val_dataset["root"] = str(Path(self.cfg.val_dataset["root"]) / self.dataset)

        val_dataset = NavigationDataset(**self.cfg.val_dataset)

        self.image_denormalizer = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5],
        )
        val_loader_cfg = dict(self.cfg.dataloader)
        val_loader_cfg.update({"shuffle": False})
        val_loader_cfg["collate_fn"] = custom_collate
        if self.batch_size is not None:
            val_loader_cfg["batch_size"] = self.batch_size

        if self.cfg.train_params.debug:
            debug_subset_size = max(1, int(self.cfg.train_params.get("debug_subset_size", 1000)))
            val_size_used = min(debug_subset_size, len(val_dataset))
            generator = torch.Generator().manual_seed(int(self.cfg.train_params.seed))
            val_indices = torch.randperm(len(val_dataset), generator=generator)[:val_size_used].tolist()
            val_loader_cfg.pop("shuffle", None)
            val_loader_cfg["sampler"] = SubsetRandomSampler(val_indices)

        val_data = DataLoader(val_dataset, **val_loader_cfg)
        return val_data

    def load_weights(self, ckpt_dir):
        if self.accelerator.is_main_process:
            print(f"Loading weights from {ckpt_dir}...")
        self.accelerator.load_state(str(ckpt_dir))

    @timeit(attrgetter("device"))
    @torch.no_grad()
    def validate(self, epoch=0, step=0, ema_manager=None, max_batches=None):
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if ema_manager is not None:
            ema_manager.apply_shadow(unwrapped_model)

        try:
            self.model.eval()
            self.psnr.reset()
            tracker = MetricTracker()
            dreamsim_scores = []
            lpips_scores = []

            bar = tqdm(
                self.val_data,
                total=len(self.val_data),
                unit="batch",
                desc=f"Epoch {epoch:03}, validating" if epoch > 0 else f"Validating {self.arch.capitalize()}",
                bar_format="{l_bar}{bar}|{r_bar}",
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process
            )

            for val_iter, data in enumerate(bar):
                if max_batches is not None and val_iter >= max_batches:
                    break

                B = data['past_positions'].size(0)
                pred_len = min(self.cfg.model.pred_len, data["future_frames"].size(1))

                data["future_frames"] = data["future_frames"][:, :pred_len]
                data["future_positions"] = data["future_positions"][:, :pred_len]
                if data["future_vw"] is not None:
                    data["future_vw"] = data["future_vw"][:, :pred_len]

                future_frames_latent = self.visual_encoder.encode(
                    data["future_frames"].reshape(-1, *data["future_frames"].shape[2:])
                )
                future_frames_latent = future_frames_latent.reshape(B, pred_len, self.k, -1)

                # Force sampling on the first batch to guarantee at least one visualization during validation
                sample_vis = (val_iter == 0) or (self.log_freq > 0 and val_iter % self.log_freq == 0)

                # Pass 1: FKD + WM
                if self.arch == "hydra":
                    pred_pose, _, pred_images, vq_loss_fkd, metrics_fkd = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"], data["future_positions"],
                        task=self.cfg.train_params.task,
                        future_imgs_latent=future_frames_latent,
                        sample_images=sample_vis,
                        mask_actions=False, mask_poses=True,
                    )
                else:
                    pred_pose, _, pred_images, img_aux = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"], data["future_positions"],
                        task=self.cfg.train_params.task,
                        mask_actions=False, mask_poses=True,
                    )
                    vq_loss_fkd = torch.tensor(0.0, device=self.device)
                    metrics_fkd = img_aux

                # Pass 2: IKD
                if self.arch == "hydra":
                    _, pred_actions, _, vq_loss_ikd, metrics_ikd = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"], data["future_positions"],
                        task=self.cfg.train_params.task,
                        mask_actions=True, mask_poses=False,
                    )
                else:
                    _, pred_actions, _, _ = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"], data["future_positions"],
                        task=self.cfg.train_params.task,
                        mask_actions=True, mask_poses=False,
                    )
                    vq_loss_ikd = torch.tensor(0.0, device=self.device)
                    metrics_ikd = {}

                # Pass 3: Navigation Policy / BC
                if self.arch == "hydra":
                    _, pred_actions_bc, _, vq_loss_bc, _ = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"], data["future_positions"],
                        task=self.cfg.train_params.task,
                        mask_actions=True, mask_poses=True,
                    )
                else:
                    _, pred_actions_bc, _, _ = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"], data["future_positions"],
                        task=self.cfg.train_params.task,
                        mask_actions=True, mask_poses=True,
                    )
                    vq_loss_bc = torch.tensor(0.0, device=self.device)

                preds = (pred_actions, pred_pose, pred_images)
                targets = (data["future_vw"], data["future_positions"], future_frames_latent)
                vq_val = vq_loss_fkd + vq_loss_ikd + vq_loss_bc

                aux_metrics = {**metrics_fkd, **metrics_ikd} if self.arch == "hydra" else metrics_fkd
                loss, loss_data = self.criterion(preds, targets, vq_loss=vq_val, aux=aux_metrics)

                if data["future_vw"] is not None:
                    loss_actions_bc = F.mse_loss(pred_actions_bc, data["future_vw"], reduction="none").mean(dim=1).mean()
                else:
                    loss_actions_bc = torch.tensor(0.0, device=self.device)

                loss_img = loss_data["img_loss"] if self.cfg.train_params.task in ["all", "wm"] else torch.tensor(0.0, device=self.device)
                loss = loss + loss_actions_bc

                pred_pose_eval = pred_pose
                future_positions_eval = data["future_positions"]
                pred_actions_eval = pred_actions
                future_vw_eval = data["future_vw"]
                past_frames_eval = self.image_denormalizer(data["past_frames"])
                future_frames_eval = self.image_denormalizer(data["future_frames"])

                # PSNR Evaluation
                decode_condition = True if self.full_psnr else sample_vis
                if decode_condition and pred_images is not None:
                    pred_images_first = pred_images[:, 0:1]
                    b, t, k_dim, d = pred_images_first.shape
                    z_img_softvq = pred_images_first.reshape(b * t, k_dim, d)
                    h_prime, w_prime = data["future_frames"].shape[-2] // 8, data["future_frames"].shape[-1] // 8

                    p_img = self.visual_encoder.decode(z_img_softvq, hw_shape=(h_prime, w_prime))
                    decoded_images = p_img.reshape(b, t, *p_img.shape[1:])
                    pred_images_eval = self.image_denormalizer(decoded_images)
                    future_frames_eval = self.image_denormalizer(data["future_frames"][:, 0:1])

                    self.psnr.update(pred_images_eval, future_frames_eval)

                    # Perceptual Metrics (DreamSim, LPIPS)
                    ds_pred = pred_images_eval.reshape(b * t, 3, pred_images_eval.shape[-2], pred_images_eval.shape[-1])
                    ds_real = future_frames_eval.reshape(b * t, 3, future_frames_eval.shape[-2], future_frames_eval.shape[-1])

                    ds_pred = ds_pred * 2.0 - 1.0
                    ds_real = ds_real * 2.0 - 1.0

                    lpips_dist = self.lpips_model(ds_pred, ds_real)
                    lpips_scores.extend(lpips_dist.flatten().tolist())

                    # Resize to 224x224 for DreamSim
                    if ds_pred.shape[-1] != 224 or ds_pred.shape[-2] != 224:
                        ds_pred = F.interpolate(ds_pred, size=(224, 224), mode="bicubic", antialias=True)
                        ds_real = F.interpolate(ds_real, size=(224, 224), mode="bicubic", antialias=True)

                    dsim_dist = self.dreamsim_model(ds_pred, ds_real)
                    dreamsim_scores.extend(dsim_dist.flatten().tolist())
                else:
                    pred_images_eval = None

                # Distance Metrics
                mae_per_step = torch.abs(pred_pose_eval - future_positions_eval).mean(dim=0, keepdim=True)
                gt_scale = future_positions_eval.abs().mean(dim=(0, 1), keepdim=True)
                error_rate = (mae_per_step / (gt_scale + 1e-6)).squeeze(0)

                abs_pred_pos = torch.cumsum(pred_pose_eval[..., :2], dim=1)
                abs_gt_pos = torch.cumsum(future_positions_eval[..., :2], dim=1)

                if pred_pose_eval.shape[-1] >= 3:
                    abs_pred_yaw = torch.cumsum(pred_pose_eval[..., 2:], dim=1)
                    abs_pred_yaw = (abs_pred_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                    abs_gt_yaw = torch.cumsum(future_positions_eval[..., 2:], dim=1)
                    abs_gt_yaw = (abs_gt_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                    abs_pred = torch.cat([abs_pred_pos, abs_pred_yaw], dim=-1)
                    abs_gt = torch.cat([abs_gt_pos, abs_gt_yaw], dim=-1)
                    yaw_err = torch.abs((abs_pred_yaw - abs_gt_yaw + torch.pi) % (2 * torch.pi) - torch.pi).mean().item()
                else:
                    abs_pred = abs_pred_pos
                    abs_gt = abs_gt_pos
                    yaw_err = 0.0

                mae_val = compute_mae(pred_pose_eval[..., :2], future_positions_eval[..., :2]).mean().item()
                ade_val = compute_ade(abs_pred_pos, abs_gt_pos).mean().item()
                fde_val = compute_fde(abs_pred_pos, abs_gt_pos).mean().item()
                ate_val = compute_ate(abs_pred_pos, abs_gt_pos).mean().item()
                rpe_val = compute_rpe(pred_pose_eval[..., :2], future_positions_eval[..., :2]).mean().item()

                if self.logger is not None and sample_vis:
                    idx = max(0, B - 5)
                    log_visualizations(
                        m_model=self.accelerator.unwrap_model(self.model),
                        cfg=self.cfg, logger=self.logger, epoch=epoch, iteration=val_iter,
                        global_step=step, prefix="val", idx=idx, data=data,
                        past_frames_eval=past_frames_eval, pred_images_eval=pred_images_eval,
                        img_gt_eval=future_frames_eval, abs_pred=abs_pred, abs_gt=abs_gt,
                        curr_pred_len=None, future_vw_target=data["future_vw"],
                        future_positions_target=data["future_positions"],
                        image_denormalizer=self.image_denormalizer,
                        is_main_process=self.accelerator.is_main_process,
                    )

                update_dict = {
                    "loss": loss, "loss_poses": loss_data["pose_loss"], "loss_actions": loss_data["act_loss"],
                    "loss_actions_bc": loss_actions_bc, "loss_img": loss_img, "error_rate": error_rate,
                    "mae": mae_val, "ade": ade_val, "fde": fde_val, "ate": ate_val, "rpe": rpe_val, "yaw_err": yaw_err
                }
                if self.arch == "hydra":
                    update_dict.update({
                        "act_p": metrics_ikd.get("actions_perp", 0.0), "act_u": metrics_ikd.get("actions_usage", 0.0),
                        "pos_p": metrics_fkd.get("poses_perp", 0.0), "pos_u": metrics_fkd.get("poses_usage", 0.0),
                        "img_p": metrics_fkd.get("images_perp", 0.0), "img_u": metrics_fkd.get("images_usage", 0.0),
                        "fm_loss": metrics_fkd.get("fm_loss", 0.0), "fm_loss_act": metrics_ikd.get("fm_loss_act", 0.0),
                        "fm_loss_pose": metrics_fkd.get("fm_loss_pose", 0.0)
                    })

                tracker.update(update_dict)
                latest = {k: v[-1] for k, v in tracker.metrics.items()}

                postfix_dict = {
                    "loss": latest["loss"], "ADE": latest["ade"], "FDE": latest["fde"],
                    "ATE": latest["ate"], "RPE": latest["rpe"], "YawErr": latest["yaw_err"]
                }
                if self.arch == "hydra":
                    postfix_dict.update({
                        "ActP": latest["act_p"], "ActU": latest["act_u"],
                        "PosP": latest["pos_p"], "PosU": latest["pos_u"],
                        "ImgP": latest["img_p"], "ImgU": latest["img_u"],
                        "FMLoss": latest["fm_loss"], "FMLossAct": latest["fm_loss_act"], "FMLossPose": latest["fm_loss_pose"]
                    })
                bar.set_postfix(**postfix_dict)

            bar.close()
            final_metrics = tracker.get_means()
            final_stds = {f"{k}_std": np.std(v) if len(v) > 0 else 0.0 for k, v in tracker.metrics.items()}
            final_metrics.update(final_stds)

            try:
                local_psnr = self.psnr.compute().item()
            except Exception:
                local_psnr = 0.0
            psnr_tensor = torch.tensor([local_psnr], device=self.device)
            final_metrics["psnr"] = self.accelerator.gather(psnr_tensor).mean().item()
            final_metrics["psnr_std"] = 0.0  # Torcheval PSNR is global, so std across batches is N/A

            if len(dreamsim_scores) > 0:
                local_dsim_tensor = torch.tensor(dreamsim_scores, device=self.device)
                global_dsim = self.accelerator.gather(local_dsim_tensor).cpu().numpy()
                if self.accelerator.is_main_process:
                    final_metrics["dreamsim"] = np.mean(global_dsim)
                    final_metrics["dreamsim_std"] = np.std(global_dsim)
                else:
                    final_metrics["dreamsim"] = 0.0
                    final_metrics["dreamsim_std"] = 0.0
                dsim_tensor = torch.tensor([final_metrics["dreamsim"], final_metrics["dreamsim_std"]], device=self.device)
                dsim_tensor = broadcast(dsim_tensor, from_process=0)
                final_metrics["dreamsim"] = dsim_tensor[0].item()
                final_metrics["dreamsim_std"] = dsim_tensor[1].item()
            else:
                final_metrics["dreamsim"] = 0.0
                final_metrics["dreamsim_std"] = 0.0

            if len(lpips_scores) > 0:
                local_lpips_tensor = torch.tensor(lpips_scores, device=self.device)
                global_lpips = self.accelerator.gather(local_lpips_tensor).cpu().numpy()
                if self.accelerator.is_main_process:
                    final_metrics["lpips"] = np.mean(global_lpips)
                    final_metrics["lpips_std"] = np.std(global_lpips)
                else:
                    final_metrics["lpips"] = 0.0
                    final_metrics["lpips_std"] = 0.0
                lpips_tensor = torch.tensor([final_metrics["lpips"], final_metrics["lpips_std"]], device=self.device)
                lpips_tensor = broadcast(lpips_tensor, from_process=0)
                final_metrics["lpips"] = lpips_tensor[0].item()
                final_metrics["lpips_std"] = lpips_tensor[1].item()
            else:
                final_metrics["lpips"] = 0.0
                final_metrics["lpips_std"] = 0.0

            return final_metrics
        finally:
            if ema_manager is not None:
                ema_manager.restore(unwrapped_model)

    @timeit(attrgetter("device"))
    @torch.no_grad()
    def evaluate_specific_metric(self, metric="dreamsim", ema_manager=None, max_batches=None, num_splits=5):
        """
        Fast, modular evaluation for isolated visual metrics (avoids calculating geometry/IK losses)
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if ema_manager is not None:
            ema_manager.apply_shadow(unwrapped_model)

        try:
            self.model.eval()
            dreamsim_scores = []
            lpips_scores = []

            bar = tqdm(
                self.val_data,
                total=len(self.val_data),
                unit="batch",
                desc=f"Evaluating Standalone {metric.upper()}",
                disable=not self.accelerator.is_main_process
            )

            for val_iter, data in enumerate(bar):
                if max_batches is not None and val_iter >= max_batches:
                    break

                B = data['past_positions'].size(0)
                pred_len = min(self.cfg.model.pred_len, data["future_frames"].size(1))

                # Stripped down forward pass exclusively focusing on "wm"
                if self.arch == "hydra":
                    _, _, pred_images, _, _ = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"][:, :pred_len] if data["future_vw"] is not None else None,
                        data["future_positions"][:, :pred_len],
                        task="wm", future_imgs_latent=None, sample_images=True,
                        mask_actions=False, mask_poses=True,
                    )
                else:
                    _, _, pred_images, _ = self.model(
                        data["past_frames"], data["past_vw"], data["past_positions"],
                        data["future_vw"][:, :pred_len] if data["future_vw"] is not None else None,
                        data["future_positions"][:, :pred_len],
                        task="wm", mask_actions=False, mask_poses=True,
                    )

                if pred_images is not None:
                    # Extract only the t+1 (first future) frame
                    pred_images_first = pred_images[:, 0:1]
                    b, t, k_dim, d = pred_images_first.shape
                    z_img_softvq = pred_images_first.reshape(b * t, k_dim, d)
                    h_prime, w_prime = data["future_frames"].shape[-2] // 8, data["future_frames"].shape[-1] // 8
                    p_img = self.visual_encoder.decode(z_img_softvq, hw_shape=(h_prime, w_prime))
                    decoded_images = p_img.reshape(b, t, *p_img.shape[1:])

                    pred_images_eval = self.image_denormalizer(decoded_images)
                    future_frames_eval = self.image_denormalizer(data["future_frames"][:, 0:1])

                    if metric in ["dreamsim", "lpips"]:
                        ds_pred = pred_images_eval.reshape(b * t, 3, pred_images_eval.shape[-2], pred_images_eval.shape[-1])
                        ds_real = future_frames_eval.reshape(b * t, 3, future_frames_eval.shape[-2], future_frames_eval.shape[-1])
                        ds_pred = ds_pred * 2.0 - 1.0
                        ds_real = ds_real * 2.0 - 1.0

                        if metric == "lpips":
                            lpips_dist = self.lpips_model(ds_pred, ds_real)
                            lpips_scores.extend(lpips_dist.flatten().tolist())
                        elif metric == "dreamsim":
                            if ds_pred.shape[-1] != 224 or ds_pred.shape[-2] != 224:
                                ds_pred = F.interpolate(ds_pred, size=(224, 224), mode="bicubic", antialias=True)
                                ds_real = F.interpolate(ds_real, size=(224, 224), mode="bicubic", antialias=True)
                            dsim_dist = self.dreamsim_model(ds_pred, ds_real)
                            dreamsim_scores.extend(dsim_dist.flatten().tolist())

            bar.close()

            # Reduce & Compute
            final_metrics = {}
            if metric == "dreamsim":
                if len(dreamsim_scores) > 0:
                    local_dsim_tensor = torch.tensor(dreamsim_scores, device=self.device)
                    global_dsim = self.accelerator.gather(local_dsim_tensor).cpu().numpy()
                    if self.accelerator.is_main_process:
                        final_metrics["dreamsim"] = np.mean(global_dsim)
                        final_metrics["dreamsim_std"] = np.std(global_dsim)
                    else:
                        final_metrics["dreamsim"] = 0.0
                        final_metrics["dreamsim_std"] = 0.0
                    dsim_tensor = torch.tensor([final_metrics["dreamsim"], final_metrics["dreamsim_std"]], device=self.device)
                    dsim_tensor = broadcast(dsim_tensor, from_process=0)
                    final_metrics["dreamsim"] = dsim_tensor[0].item()
                    final_metrics["dreamsim_std"] = dsim_tensor[1].item()
                else:
                    final_metrics["dreamsim"] = 0.0
                    final_metrics["dreamsim_std"] = 0.0

            elif metric == "lpips":
                if len(lpips_scores) > 0:
                    local_lpips_tensor = torch.tensor(lpips_scores, device=self.device)
                    global_lpips = self.accelerator.gather(local_lpips_tensor).cpu().numpy()
                    if self.accelerator.is_main_process:
                        final_metrics["lpips"] = np.mean(global_lpips)
                        final_metrics["lpips_std"] = np.std(global_lpips)
                    else:
                        final_metrics["lpips"] = 0.0
                        final_metrics["lpips_std"] = 0.0
                    lpips_tensor = torch.tensor([final_metrics["lpips"], final_metrics["lpips_std"]], device=self.device)
                    lpips_tensor = broadcast(lpips_tensor, from_process=0)
                    final_metrics["lpips"] = lpips_tensor[0].item()
                    final_metrics["lpips_std"] = lpips_tensor[1].item()
                else:
                    final_metrics["lpips"] = 0.0
                    final_metrics["lpips_std"] = 0.0

            return final_metrics

        finally:
            if ema_manager is not None:
                ema_manager.restore(unwrapped_model)

    def report(self, metrics):
        if not self.accelerator.is_main_process:
            return

        def fmt(key):
            val = metrics.get(key, 0.0)
            std = metrics.get(f"{key}_std", 0.0)
            if std > 0:
                return f"{val:.4f} ± {std:.4f}"
            return f"{val:.4f}"

        console = Console(record=True)
        console.print("\n")

        if self.dataset:
            console.print(f"[bold blue]Evaluation Report for Dataset: {self.dataset}[/bold blue]\n")

        # Trajectory Metrics
        traj_table = Table(title="Geometric Trajectory Metrics", show_header=True, header_style="bold magenta")
        traj_table.add_column("Metric", style="dim", width=20)
        traj_table.add_column("Value", justify="right")

        traj_table.add_row("ADE (m)", fmt('ade'))
        traj_table.add_row("FDE (m)", fmt('fde'))
        traj_table.add_row("ATE (m)", fmt('ate'))
        traj_table.add_row("RPE (m)", fmt('rpe'))
        traj_table.add_row("MAE (m)", fmt('mae'))
        traj_table.add_row("Yaw Error (rad)", fmt('yaw_err'))
        traj_table.add_row("Scale Error Rate", fmt('error_rate'))

        # Task Loss Metrics
        loss_table = Table(title="Criterion Tracking", show_header=True, header_style="bold cyan")
        loss_table.add_column("Metric", style="dim", width=20)
        loss_table.add_column("Value", justify="right")

        loss_table.add_row("Total Loss", fmt('loss'))
        loss_table.add_row("Poses Loss", fmt('loss_poses'))
        loss_table.add_row("Actions Loss", fmt('loss_actions'))
        loss_table.add_row("Actions (BC) Loss", fmt('loss_actions_bc'))
        loss_table.add_row("Images Loss", fmt('loss_img'))
        psnr_str = f"{metrics.get('psnr', 0.0):.2f} dB" + (" (Full Test Set)" if self.full_psnr else " (Sampled)")
        loss_table.add_row("PSNR (Reconstruction)", psnr_str)
        if "dreamsim" in metrics and metrics["dreamsim"] > 0:
            loss_table.add_row("DreamSim (Perceptual)", fmt('dreamsim'))
        if "lpips" in metrics and metrics["lpips"] > 0:
            loss_table.add_row("LPIPS (Perceptual)", fmt('lpips'))

        console.print(traj_table)
        console.print("\n")
        console.print(loss_table)

        # VQ Metrics (Hydra Only)
        if self.arch == "hydra":
            vq_table = Table(title="VQ Codebook Diagnostics", show_header=True, header_style="bold yellow")
            vq_table.add_column("Codebook", style="dim", width=15)
            vq_table.add_column("Perplexity", justify="right")
            vq_table.add_column("Usage (%)", justify="right")

            vq_table.add_row("Actions", f"{metrics['act_p']:.1f}", f"{metrics['act_u']:.1f}%")
            vq_table.add_row("Poses", f"{metrics['pos_p']:.1f}", f"{metrics['pos_u']:.1f}%")
            vq_table.add_row("Images", f"{metrics['img_p']:.1f}", f"{metrics['img_u']:.1f}%")

            console.print("\n")
            console.print(vq_table)

        console.print("\n[bold green]Validation Complete![/bold green]\n")

        # Save to text file next to the script
        output_name = f"validation_{self.arch}_report_{self.dataset}.txt" if self.dataset else "validation_report.txt"
        output_path = Path(__file__).resolve().parent / output_name

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(console.export_text(clear=False))
            f.write("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Validation Script for Hydra and VertiFormer")
    parser.add_argument("--conf", required=True, type=str, help="Path to the config file (e.g. conf/hydra.yaml)")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the checkpoint directory (e.g. hydra/checkpoint/model-name/best)")
    parser.add_argument("--arch", required=True, type=str, choices=["hydra", "vertiformer"], help="Model architecture to validate")
    parser.add_argument("--full-psnr", action="store_true", help="Decode EVERY frame to compute a rigorous PSNR (WARNING: Extremely slow)")
    parser.add_argument("--mode", default="all", type=str, choices=["all", "dreamsim", "lpips"], help="Which validation routine to run (defaults to all)")
    parser.add_argument("--dataset", default=None, type=str, help="Specific dataset to validate (appended to val_dataset.root)")
    parser.add_argument("--batch-size", default=None, type=int, help="Override the dataloader batch size")

    args = parser.parse_args()
    cfg = get_conf(args.conf)

    validator = Validator(
        cfg=cfg,
        arch=args.arch,
        full_psnr=args.full_psnr,
        dataset=args.dataset,
        batch_size=args.batch_size
    )
    validator.load_weights(args.ckpt)

    if args.mode == "all":
        results = validator.validate()
        # Handle the tuple return signature injected by the @timeit decorator
        if isinstance(results, tuple):
            results = results[0]
        validator.report(results)
    else:
        results = validator.evaluate_specific_metric(metric=args.mode)
        # Handle the tuple return signature injected by the @timeit decorator
        if isinstance(results, tuple):
            results = results[0]
        if validator.accelerator.is_main_process:
            console = Console()
            console.print(f"\n[bold green]Standalone {args.mode.upper()} Evaluation Complete![/bold green]")
            val = results.get(args.mode, 0.0)
            std = results.get(f"{args.mode}_std", 0.0)
            console.print(f"{args.mode.upper()}: {val:.4f} ± {std:.4f}\n")