import gc
import io
import sys
import math
import argparse
import json
import shutil
from pathlib import Path
from datetime import timedelta
from datetime import datetime
from operator import attrgetter
try:
    sys.path.append(str(Path(".").resolve()))
except Exception:
    raise RuntimeError("Can't append root directory of the project to the path")

from comet_ml.integration.pytorch import watch
from accelerate import Accelerator
from accelerate.utils import broadcast
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from rich import print
from rich.table import Table
from rich.console import Console
import numpy as np
from tqdm import tqdm
from icecream import ic, install
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torcheval.metrics import PeakSignalNoiseRatio
from PIL import Image
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.hydra import load_model
from model.dataloader import NavigationDataset, custom_collate
from utilities.debugging import calculate_gradient_interference, log_visualizations
from utilities.helpers import (
    get_conf,
    resolve_config_path,
    timeit,
    init_logger,
    fix_seed,
)
from utilities.tracker import MetricTracker
from utilities.nn import EMAManager
from utilities.evals import compute_mae, compute_ade, compute_fde, compute_ate, compute_rpe
from validator import Validator
from utilities.augmenter import RoboticsCurriculumAugmenter

class Learner:
    def __init__(self, cfg_dir: str):
        # load config file
        self.cfg = get_conf(cfg_dir)
        self.cfg_path = resolve_config_path(cfg_dir)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.train_params.accumulation_steps,
            kwargs_handlers=[ddp_kwargs, init_kwargs]
        )
        # set the name for the model
        date_str = datetime.now().strftime("%y%m%d_%H%M")  # YYMMDD_HHMM ensures alphabetical sorting is chronological
        exp_name = self.cfg.logger.experiment_name

        self.cfg.directory.model_name = f"{exp_name}/{date_str}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        self._snapshot_experiment_files()
        # if debugging True, set a few rules
        if self.cfg.train_params.debug:
            install()
            ic.enable()
            ic.configureOutput(prefix=lambda: f"{datetime.now():%H:%M:%S} |> ")
            torch.autograd.set_detect_anomaly(True)
            self.cfg.logger.disabled = True
            matplotlib.use("TkAgg")
            self.cfg.train_params.compile = False
            self.log_freq = self.cfg.train_params.log_freq
        else:
            matplotlib.use("Agg")
            ic.disable()
            torch.autograd.set_detect_anomaly(False)
            self.log_freq = self.cfg.train_params.log_freq

        # initialize the logger only on the main process to avoid empty ghost runs in DDP
        if self.accelerator.is_main_process:
            self.logger = init_logger(self.cfg)
            self.logger.disable_mp()
        else:
            self.logger = None

        self.device = self.accelerator.device
        if self.device.type == "cuda":
            # Handle instances where Accelerate sets CUDA_VISIBLE_DEVICES making the device index None
            torch.cuda.set_device(self.device.index if self.device.index is not None else 0)

        # Initialize Curriculum Augmenter
        self.augmenter = RoboticsCurriculumAugmenter(
            total_epochs=self.cfg.train_params.epochs
        ).to(self.device)

        # fix the seed for reproducibility
        fix_seed(self.cfg.train_params.seed)

        if self.cfg.train_params.get("variable_horizon", False):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True

        torch.set_float32_matmul_precision("high")
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        if self.accelerator.is_main_process:
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model for experiment {self.cfg.logger.experiment_name}!"
            )
        self.model, self.optimizer, self.scheduler, self.criterion = load_model(
            self.cfg, self.device
        )

        self.block_size = self.model.transformer_encoder.block_size

        # multi-gpu
        # Prepare criterion as well to ensure learnable loss weights are synced/checkpointed
        self.model, self.optimizer, self.data, self.val_data, self.scheduler, self.criterion = self.accelerator.prepare(
            self.model, self.optimizer, self.data, self.val_data, self.scheduler, self.criterion
        )

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.visual_encoder = unwrapped_model.transformer_encoder.unified_encoder.visual_encoder
        self.k = unwrapped_model.k

        # Initialize EMA Manager for FM heads
        self.ema = EMAManager(
            unwrapped_model,
            target_modules=("fm_image", "fm_action", "fm_pose"),
            decay=self.cfg.train_params.get("ema_decay", 0.9999)
        )

        # log the model gradients, weights, and activations in comet
        if self.accelerator.is_main_process:
            # watch(self.model, log_step_interval=5000)
            self.logger.log_code(folder="./hydra/model")  # log the models
        num_params = [x.numel() for x in self.model.parameters()]
        trainable_params = [
            x.numel() for x in self.model.parameters() if x.requires_grad
        ]
        if self.accelerator.is_main_process:
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of parameters: {sum(num_params) / 1e6:.2f}M"
            )
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of trainable parameters: {sum(trainable_params) / 1e6:.2f}M"
            )
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self.last_grad_norm = 0.0
        # if resuming, load the checkpoint
        self.epoch = 1
        self.iteration = 0
        self.best = np.inf
        self.e_loss = []
        self.val_loss = []

        if self.cfg.logger.resume:
            self.resume_checkpoint()

        if self.cfg.train_params.compile:  # Didn't get good performance out of this. Don't know why
            if self.accelerator.is_main_process:
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - COMPILING model with dynamic shapes...")
            compile_kwargs = {"mode": "default", "dynamic": True}
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.fm_pose = torch.compile(unwrapped_model.fm_pose, **compile_kwargs)
            unwrapped_model.fm_image = torch.compile(unwrapped_model.fm_image, **compile_kwargs)
            unwrapped_model.fm_action = torch.compile(unwrapped_model.fm_action, **compile_kwargs)
            unwrapped_model.transformer_encoder = torch.compile(unwrapped_model.transformer_encoder, **compile_kwargs)

        if self.accelerator.is_main_process and self.logger is not None:
            self.logger.set_epoch(self.epoch)

    def _snapshot_experiment_files(self):
        if not self.accelerator.is_main_process:
            return
        exp_dir = Path(self.cfg.directory.save)
        exp_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(__file__).resolve().parent / "model"
        shutil.copy2(self.cfg_path, exp_dir / self.cfg_path.name)
        shutil.copytree(model_dir, exp_dir / "model", dirs_exist_ok=True)

    def train(self):

        while self.epoch <= self.cfg.train_params.epochs:
            train_tracker = MetricTracker()

            bar = tqdm(
                self.data,
                total=len(self.data),
                unit="step",
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
                bar_format="{l_bar}{bar}|{r_bar}",
                dynamic_ncols=True,
            )
            for data in bar:
                self.iteration += 1
                loss_dict, t_train = self.forward_batch(data)
                t_train /= self.cfg.dataloader.batch_size
                loss_dict["time"] = t_train

                train_tracker.update(loss_dict)

                latest = {k: v[-1] for k, v in train_tracker.metrics.items()}

                bar.set_postfix(
                    loss=latest.get("loss", 0.0),
                    predLen=latest.get("pred_len", self.cfg.model.pred_len),
                    lossPoses=latest.get("loss_poses", 0.0),
                    lossActions=latest.get("loss_actions", 0.0),
                    lossImg=latest.get("loss_img", 0.0),
                    Grad_Norm=latest.get("grad_norm", 0.0),
                    Time=latest.get("time", 0.0),
                    ActP=latest.get("actions_perp", 0.0),
                    ActU=latest.get("actions_usage", 0.0),
                    PosP=latest.get("poses_perp", 0.0),
                    PosU=latest.get("poses_usage", 0.0),
                    ImgP=latest.get("images_perp", 0.0),
                    ImgU=latest.get("images_usage", 0.0),
                    FMLoss=latest.get("fm_loss", 0.0),
                    FMLossAct=latest.get("fm_loss_act", 0.0),
                    FMLossPose=latest.get("fm_loss_pose", 0.0),
                )

                cosine_data = loss_dict.get("cosine_data", {})

                if self.accelerator.is_main_process:
                    log_metrics = {k: v for k, v in latest.items() if k not in ["time", "cosine_data", "is_act_masked"]}
                    log_metrics.update(cosine_data)

                    if loss_dict.get("is_act_masked", False):
                        log_metrics["loss_actions_IKD_masked"] = loss_dict["loss_actions"]
                    else:
                        log_metrics["loss_actions_Auto_unmasked"] = loss_dict["loss_actions"]

                    self.logger.log_metrics(
                        log_metrics,
                        epoch=self.epoch,
                        step=self.iteration,
                    )

            bar.close()
            # Epoch-based schedulers step here
            self.scheduler.step()

            # average loss for an epoch
            train_means = train_tracker.get_means()
            self.e_loss.append(train_means["loss"])  # epoch loss

            run_val = (self.epoch % self.cfg.train_params.get("val_freq", 1) == 0) or (self.epoch == self.cfg.train_params.epochs)

            if run_val:
                # validate on val set
                validator = Validator(
                    cfg=self.cfg,
                    arch="hydra",
                    accelerator=self.accelerator,
                    model=self.model,
                    val_data=self.val_data,
                    criterion=self.criterion,
                    visual_encoder=self.visual_encoder,
                    k=self.k,
                    image_denormalizer=self.image_denormalizer,
                    full_psnr=False,
                    log_freq=self.log_freq,
                    logger=self.logger,
                )

                max_val_batches = self.cfg.train_params.get("val_max_batches", 50)
                val_means, t = validator.validate(
                    epoch=self.epoch,
                    step=self.iteration,
                    ema_manager=self.ema,
                    max_batches=max_val_batches
                )

                val_ade = val_means["ade"]
                # Sync val_ade across all processes to prevent deadlocks during save
                val_ade_tensor = torch.tensor([val_ade], device=self.device)
                val_ade = self.accelerator.gather(val_ade_tensor).mean().item()

                processed_samples = len(self.val_data.dataset)
                if max_val_batches is not None:
                    processed_samples = min(processed_samples, max_val_batches * self.cfg.dataloader.batch_size * self.accelerator.num_processes)
                t /= max(1, processed_samples)

                self.val_loss.append(val_means["loss"])

                psnr_val = self.psnr.compute().item()

            if self.accelerator.is_main_process:
                console = Console()
                table = Table(title=f"Epoch {self.epoch:03} Summary (Iter: {self.iteration:05d})", show_header=True, header_style="bold magenta")
                table.add_column("Metric", style="dim", width=25)
                table.add_column("Training", justify="right", style="green")
                if run_val:
                    table.add_column("Validation", justify="right", style="red")

                def add_row(metric, t_val, v_val=None, is_float=True, suffix=""):
                    t_str = f"{t_val:.4f}{suffix}" if is_float and t_val is not None else str(t_val)
                    if run_val:
                        v_str = f"{v_val:.4f}{suffix}" if is_float and v_val is not None else str(v_val)
                        table.add_row(metric, t_str, v_str)
                    else:
                        table.add_row(metric, t_str)

                add_row("Total Loss", train_means.get("loss"), val_means.get("loss") if run_val else None)
                add_row("Loss Poses", train_means.get("loss_poses"), val_means.get("loss_poses") if run_val else None)
                add_row("Loss Actions", train_means.get("loss_actions"), val_means.get("loss_actions") if run_val else None)
                add_row("Loss Image", train_means.get("loss_img"), val_means.get("loss_img") if run_val else None)

                table.add_section()
                add_row("ADE (m)", train_means.get("ade"), val_ade if run_val else None)
                add_row("FDE (m)", train_means.get("fde"), val_means.get("fde") if run_val else None)
                add_row("ATE (m)", train_means.get("ate"), val_means.get("ate") if run_val else None)
                add_row("RPE (m)", train_means.get("rpe"), val_means.get("rpe") if run_val else None)
                add_row("MAE (m)", train_means.get("mae"), val_means.get("mae") if run_val else None)
                add_row("Yaw Error (rad)", train_means.get("yaw_err"), val_means.get("yaw_err") if run_val else None)

                table.add_section()
                add_row("Act Codebook Perplexity", train_means.get("actions_perp", 0.0), val_means.get("act_p", 0.0) if run_val else None)
                add_row("Act Codebook Usage", train_means.get("actions_usage", 0.0), val_means.get("act_u", 0.0) if run_val else None, suffix="%")
                add_row("Pose Codebook Perplexity", train_means.get("poses_perp", 0.0), val_means.get("pos_p", 0.0) if run_val else None)
                add_row("Pose Codebook Usage", train_means.get("poses_usage", 0.0), val_means.get("pos_u", 0.0) if run_val else None, suffix="%")
                add_row("Img Codebook Perplexity", train_means.get("images_perp", 0.0), val_means.get("img_p", 0.0) if run_val else None)
                add_row("Img Codebook Usage", train_means.get("images_usage", 0.0), val_means.get("img_u", 0.0) if run_val else None, suffix="%")

                table.add_section()
                if run_val:
                    table.add_row("PSNR", "-", f"{psnr_val:.2f} dB")
                add_row("Grad Norm", train_means.get("grad_norm"), None)

                console.print("\n")
                console.print(table)
                t_train = train_means.get("time", 0.0)

                if run_val:
                    print(f"[dim]Time (Train): {t_train:.4f} s/batch | Time (Val): {t:.4f} s/sample[/dim]\n")
                    self.logger.log_metrics(
                        {
                            "train_loss": self.e_loss[-1],
                            "val_loss": val_means["loss"],
                            "loss_posesVal": val_means["loss_poses"],
                            "loss_actionsVal": val_means["loss_actions"],
                            "loss_actionsBCVal": val_means.get("loss_actions_bc", 0.0),
                            "loss_imgVal": val_means["loss_img"],
                            "actions_perpVal": val_means.get("act_p", 0.0),
                            "actions_usageVal": val_means.get("act_u", 0.0),
                            "poses_perpVal": val_means.get("pos_p", 0.0),
                            "poses_usageVal": val_means.get("pos_u", 0.0),
                            "images_perpVal": val_means.get("img_p", 0.0),
                            "images_usageVal": val_means.get("img_u", 0.0),
                            "error_rateVal": val_means["error_rate"],
                            "ADE_Val": val_ade,
                            "FDE_Val": val_means["fde"],
                            "ATE_Val": val_means["ate"],
                            "RPE_Val": val_means["rpe"],
                            "YawErr_Val": val_means["yaw_err"],
                            "MAE_Val": val_means["mae"],
                            "PSNR": psnr_val,
                            "time": t,
                        },
                        epoch=self.epoch,
                        step=self.iteration,
                    )
                else:
                    print(f"[dim]Time (Train): {t_train:.4f} s/batch[/dim]\n")
                    self.logger.log_metrics(
                        {
                            "train_loss": self.e_loss[-1],
                        },
                        epoch=self.epoch,
                        step=self.iteration,
                    )

            if self.epoch % self.cfg.train_params.save_every == 0 or (
                run_val and val_ade < self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save(name=f"epoch_{self.epoch:04d}", current_metric=val_ade if run_val else self.best)

            gc.collect()
            torch.cuda.empty_cache()
            self.epoch += 1
        if self.accelerator.is_main_process:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    @timeit(attrgetter("device"))
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        # forward, backward
        with self.accelerator.accumulate(self.model):
            B = data['past_positions'].size(0)
            pred_len = self.cfg.model.pred_len
            use_variable_horizon = self.cfg.train_params.get("variable_horizon", False)

            # sync curriculum randomness & horizons across DDP
            if use_variable_horizon:
                min_pred_len = int(self.cfg.train_params.get("min_pred_len", 1))
                max_pred_len = int(self.cfg.train_params.get("max_pred_len", pred_len))
                local_max_allowed = min(
                    pred_len, data["future_positions"].size(1), data["future_frames"].size(1),
                    data["future_vw"].size(1) if data["future_vw"] is not None else pred_len
                )

                # Get the safest max horizon across all active batches in the DDP group
                max_tensor = torch.tensor([min(max_pred_len, local_max_allowed)], device=self.device, dtype=torch.long)

                global_sampled_max = self.accelerator.gather(max_tensor).min().item()
            else:
                global_sampled_max = pred_len
                min_pred_len = pred_len

            decision_tensor = torch.zeros(2, device=self.device)
            if self.accelerator.is_main_process:
                decision_tensor[0] = torch.rand(1).item()
                if use_variable_horizon:
                    decision_tensor[1] = torch.randint(min_pred_len, global_sampled_max + 1, (1,)).item()

            decision_tensor = broadcast(decision_tensor, from_process=0)
            rand_val = decision_tensor[0].item()

            if use_variable_horizon:
                curr_pred_len = int(decision_tensor[1].item())
            else:
                curr_pred_len = pred_len

            future_vw = data["future_vw"][:, :curr_pred_len] if data["future_vw"] is not None else None
            future_positions = data["future_positions"][:, :curr_pred_len]
            future_frames = data["future_frames"][:, :curr_pred_len]

            # If the dataset returned a shorter trajectory (e.g., episode boundaries),
            # we must update curr_pred_len to match the actual sliced tensor shape before reshaping.
            curr_pred_len = future_frames.shape[1]

            # Curriculum Augmentation
            # Applied strictly to the observation history to prevent model from overfitting. Target features remain uncorrupted.
            if self.cfg.train_params.get("use_augmentation", True):
                past_frames_aug, past_vw_aug, past_positions_aug, past_mask_vw, past_mask_pos = self.augmenter(
                    data["past_frames"],
                    data["past_vw"],
                    data["past_positions"],
                    epoch=self.epoch
                )
            else:
                past_frames_aug = data["past_frames"]
                past_vw_aug = data["past_vw"]
                past_positions_aug = data["past_positions"]
                past_mask_vw = None
                past_mask_pos = None

            with torch.no_grad():
                future_frames_latent = self.visual_encoder.encode(
                    future_frames.reshape(-1, *future_frames.shape[2:])
                )
                future_frames_latent = future_frames_latent.reshape(B, curr_pred_len, self.k, -1)

            # Curriculum Modality Masking
            warmup_epochs = max(1, self.cfg.train_params.get("curriculum_epochs", 10))
            decay_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, (self.epoch - 1) / warmup_epochs)))
            p_joint = 0.1 + 0.4 * decay_factor
            p_fkd = (1.0 - p_joint) / 2.0
            p_ikd = 1.0 - p_joint - p_fkd
            p_fkd_threshold = p_joint + p_fkd

            sample_vis = (self.iteration % self.log_freq == 0)

            # Create tensor masks
            mask_actions = torch.tensor([rand_val >= p_fkd_threshold], device=self.device)
            mask_poses = torch.tensor([p_joint <= rand_val < p_fkd_threshold], device=self.device)
            is_act_masked = mask_actions.item()

            pred_pose, pred_actions, pred_images, vq_loss, metrics = self.model(
                past_frames_aug,
                past_vw_aug,
                past_positions_aug,
                future_vw,
                future_positions,
                pred_len=curr_pred_len,
                task=self.cfg.train_params.task,
                future_imgs_latent=future_frames_latent,
                sample_images=sample_vis,
                mask_actions=mask_actions,
                mask_poses=mask_poses,
            )
            # losses
            preds = (pred_actions, pred_pose, pred_images)
            targets = (future_vw, future_positions, future_frames_latent)
            total_loss, loss_data = self.criterion(preds, targets, vq_loss, aux=metrics)

            # Calculating the cosine similarity between gradients to identify interference
            cosine_data = {}
            if self.iteration % self.log_freq == 0:  # it is expensive, we don't need to do it every step
                shared_layer_name = "transformer_decoder_layers.5.multihead_attn.out_proj"
                unwrapped_model = self.accelerator.unwrap_model(self.model)

                actions_vs_imgs = calculate_gradient_interference(unwrapped_model, shared_layer_name, loss_data['act_loss'],
                                                                  loss_data["img_loss"])
                poses_vs_imgs = calculate_gradient_interference(unwrapped_model, shared_layer_name, loss_data['pose_loss'],
                                                                loss_data["img_loss"])
                actions_vs_poses = calculate_gradient_interference(unwrapped_model, shared_layer_name, loss_data['act_loss'],
                                                                   loss_data["pose_loss"])
                cosine_data = {
                    "cos_actions_vs_imgs": actions_vs_imgs,
                    "cos_poses_vs_imgs": poses_vs_imgs,
                    "cos_actions_vs_poses": actions_vs_poses,
                }

            self.accelerator.backward(total_loss)
            # gradient clipping
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train_params.grad_clipping
                )
                self.last_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            # update
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # EMA Update
            if self.accelerator.sync_gradients:
                self.ema.update(self.accelerator.unwrap_model(self.model))

        with torch.no_grad():
            future_positions_eval = future_positions
            future_vw_eval = future_vw
            future_frames_eval = self.image_denormalizer(future_frames)
            past_frames_eval = self.image_denormalizer(past_frames_aug)

            eval_metrics = {}

            # visualize a random pose and compute metrics periodically
            if self.iteration % self.log_freq == 0:
                # EMA inference for logging
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self.ema.apply_shadow(unwrapped_model)
                try:
                    # Switch to eval mode
                    unwrapped_model.eval()
                    # Re-run forward pass with EMA weights to get clean predictions
                    pred_pose_eval, pred_actions_eval, pred_images_eval_raw, _, _ = unwrapped_model(
                        data["past_frames"],
                        data["past_vw"],
                        data["past_positions"],
                        future_vw,
                        future_positions,
                        pred_len=curr_pred_len,
                        task=self.cfg.train_params.task,
                        future_imgs_latent=future_frames_latent,
                        sample_images=True,
                        mask_actions=False,
                        mask_poses=False,
                    )

                    if pred_images_eval_raw is not None:
                        b, t, k_dim, d = pred_images_eval_raw.shape
                        z_img_softvq = pred_images_eval_raw.reshape(b * t, k_dim, d)
                        h_prime, w_prime = future_frames.shape[-2] // 8, future_frames.shape[-1] // 8
                        p_img = self.visual_encoder.decode(z_img_softvq, hw_shape=(h_prime, w_prime))
                        decoded_images = p_img.reshape(b, t, *p_img.shape[1:])
                        pred_images_eval = self.image_denormalizer(decoded_images)
                    else:
                        pred_images_eval = None

                    mae_per_step = torch.abs(pred_pose_eval - future_positions_eval).mean(dim=0, keepdim=True)
                    gt_scale = future_positions_eval.abs().mean(dim=(0, 1), keepdim=True)
                    error_rate = mae_per_step / (gt_scale + 1e-6)
                    error_rate_val = error_rate.mean().item()

                    # Convert deltas to absolute trajectories for distance metrics (ADE, FDE, ATE)
                    abs_pred_pos = torch.cumsum(pred_pose_eval[..., :2], dim=1)
                    abs_gt_pos = torch.cumsum(future_positions_eval[..., :2], dim=1)

                    abs_pred_yaw = torch.cumsum(pred_pose_eval[..., 2:], dim=1)
                    abs_pred_yaw = (abs_pred_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                    abs_gt_yaw = torch.cumsum(future_positions_eval[..., 2:], dim=1)
                    abs_gt_yaw = (abs_gt_yaw + torch.pi) % (2 * torch.pi) - torch.pi

                    abs_pred = torch.cat([abs_pred_pos, abs_pred_yaw], dim=-1)
                    abs_gt = torch.cat([abs_gt_pos, abs_gt_yaw], dim=-1)

                    mae_val = compute_mae(pred_pose_eval[..., :2], future_positions_eval[..., :2]).mean().item()
                    ade_val = compute_ade(abs_pred_pos, abs_gt_pos).mean().item()
                    fde_val = compute_fde(abs_pred_pos, abs_gt_pos).mean().item()
                    ate_val = compute_ate(abs_pred_pos, abs_gt_pos).mean().item()
                    rpe_val = compute_rpe(pred_pose_eval[..., :2], future_positions_eval[..., :2]).mean().item()
                    yaw_err = torch.abs((abs_pred_yaw - abs_gt_yaw + torch.pi) % (2 * torch.pi) - torch.pi).mean().item()

                    eval_metrics = {
                        "error_rate": error_rate_val,
                        "mae": mae_val,
                        "ade": ade_val,
                        "fde": fde_val,
                        "ate": ate_val,
                        "rpe": rpe_val,
                        "yaw_err": yaw_err,
                    }

                    idx = max(0, B - 5)

                    log_visualizations(
                        m_model=self.accelerator.unwrap_model(self.model),
                        cfg=self.cfg,
                        logger=self.logger,
                        epoch=self.epoch,
                        iteration=self.iteration,
                        global_step=self.iteration,
                        prefix="train",
                        idx=idx,
                        data=data,
                        past_frames_eval=past_frames_eval,
                        pred_images_eval=pred_images_eval,
                        img_gt_eval=future_frames_eval,
                        abs_pred=abs_pred,
                        abs_gt=abs_gt,
                        curr_pred_len=curr_pred_len,
                        future_vw_target=future_vw,
                        future_positions_target=future_positions,
                        image_denormalizer=self.image_denormalizer,
                        is_main_process=self.accelerator.is_main_process,
                    )
                finally:
                    self.ema.restore(unwrapped_model)
                    # Restore training mode for the remainder of the iteration
                    unwrapped_model.train()

        out_dict = {
            "loss_poses": loss_data["pose_loss"],
            "loss_actions": loss_data["act_loss"],
            "loss_img": loss_data["img_loss"],
            "loss": total_loss,
            "grad_norm": self.last_grad_norm,
            "loss_vq": vq_loss,
            "cosine_data": cosine_data,
            "p_joint": p_joint,
            "p_fkd": p_fkd,
            "p_ikd": p_ikd,
            "actions_perp": metrics.get("actions_perp", 0.0),
            "actions_usage": metrics.get("actions_usage", 0.0),
            "poses_perp": metrics.get("poses_perp", 0.0),
            "poses_usage": metrics.get("poses_usage", 0.0),
            "images_perp": metrics.get("images_perp", 0.0),
            "images_usage": metrics.get("images_usage", 0.0),
            "fm_loss": metrics.get("fm_loss", 0.0),
            "fm_loss_act": metrics.get("fm_loss_act", 0.0),
            "fm_loss_pose": metrics.get("fm_loss_pose", 0.0),
            "var_loss": metrics.get("var_loss", 0.0),
            "ortho_loss": metrics.get("ortho_loss", 0.0),
            "pred_len": curr_pred_len,
            "is_act_masked": is_act_masked,
        }

        out_dict.update(eval_metrics)
        return out_dict

    def init_dataloader(self):
        """Initializes the dataloaders"""
        if self.accelerator.is_main_process:
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
            )

        with self.accelerator.main_process_first():
            Path(self.cfg.dataset.root).mkdir(parents=True, exist_ok=True)
            Path(self.cfg.val_dataset.root).mkdir(parents=True, exist_ok=True)

            # defining the dataset interface
            dataset = NavigationDataset(**self.cfg.dataset)
            val_dataset = NavigationDataset(**self.cfg.val_dataset)
        self.image_denormalizer = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5],
        )
        # creating dataloader
        train_loader_cfg = dict(self.cfg.dataloader)
        train_loader_cfg["collate_fn"] = custom_collate

        val_loader_cfg = dict(self.cfg.dataloader)
        val_loader_cfg.update({"shuffle": False})  # for val dataloader
        val_loader_cfg["collate_fn"] = custom_collate

        train_size_used = len(dataset)
        val_size_used = len(val_dataset)
        if self.cfg.train_params.debug:
            debug_subset_size = int(self.cfg.train_params.get("debug_subset_size", 1000))
            debug_subset_size = max(1, debug_subset_size)

            train_size_used = min(debug_subset_size, len(dataset))
            val_size_used = min(debug_subset_size, len(val_dataset))

            generator = torch.Generator()
            generator.manual_seed(int(self.cfg.train_params.seed))
            train_indices = torch.randperm(len(dataset), generator=generator)[:train_size_used].tolist()
            val_indices = torch.randperm(len(val_dataset), generator=generator)[:val_size_used].tolist()

            train_loader_cfg.pop("shuffle", None)
            val_loader_cfg.pop("shuffle", None)
            train_loader_cfg["sampler"] = SubsetRandomSampler(train_indices)
            val_loader_cfg["sampler"] = SubsetRandomSampler(val_indices)
            # Persistent workers require drop_last or infinite samplers, keep it safe during debug
            train_loader_cfg.pop("persistent_workers", None)

        data = DataLoader(dataset, **train_loader_cfg)
        val_data = DataLoader(val_dataset, **val_loader_cfg)

        # log dataset status
        if self.accelerator.is_main_process:
            self.logger.log_parameters(
                {
                    "train_len": len(dataset),
                    "val_len": len(val_dataset),
                    "train_len_used": train_size_used,
                    "val_len_used": val_size_used,
                }
            )
            print(
                f"Training consists of {train_size_used}/{len(dataset)} samples, "
                f"and validation consists of {val_size_used}/{len(val_dataset)} samples."
            )

        return data, val_data

    def resume_checkpoint(self):
        # load checkpoint
        if self.accelerator.is_main_process:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
        save_dir = Path(self.cfg.directory.load)

        if not save_dir.is_dir():
            if self.accelerator.is_main_process:
                print(f"[!] ERROR: Checkpoint path '{save_dir}' is not a valid directory.")
                print("[!] HuggingFace Accelerate requires a directory (e.g., 'checkpoint/best') to resume from, not a .tar/.pth file.")
            raise FileNotFoundError(f"Checkpoint directory {save_dir} does not exist. Cannot resume.")

        try:
            # 1. Accelerate Native Loading for DDP/FSDP
            _orig_load = torch.load
            def _safe_load(*args, **kwargs):
                if "map_location" not in kwargs:
                    # Accelerate will handle moving these to the correct local rank
                    kwargs["map_location"] = "cpu"
                kwargs["weights_only"] = False  # Bypasses unpickler security for OmegaConf
                return _orig_load(*args, **kwargs)

            try:
                torch.load = _safe_load
                self.accelerator.load_state(str(save_dir))
            finally:
                torch.load = _orig_load

            # Load custom training state
            custom_state_path = save_dir / "custom_state.pth"
            if custom_state_path.exists():
                # Load to CPU
                custom_state = torch.load(custom_state_path, map_location="cpu", weights_only=False)
                self.epoch = custom_state.get("epoch", 0) + 1
                self.iteration = custom_state.get("iteration", 0)  # loop increments it natively
                self.e_loss = custom_state.get("e_loss", [])
                self.val_loss = custom_state.get("val_loss", [])
                self.best = custom_state.get("best", np.inf)

                if "ema" in custom_state:
                    self.ema.load_state_dict(custom_state["ema"], self.device)
                else:
                    if self.accelerator.is_main_process:
                        print("[!] Warning: 'ema' not found in custom_state. Re-initializing EMA shadows from loaded weights.")
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    self.ema = EMAManager(unwrapped, target_modules=("fm_image", "fm_action", "fm_pose"), decay=self.cfg.train_params.get("ema_decay", 0.9999))

            if self.accelerator.is_main_process:
                print(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                    + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                    + f" and best metric {self.best:.4f}"
                )
            elif not custom_state_path.exists() and self.accelerator.is_main_process:
                print(
                    f"[!] WARNING: {custom_state_path.name} not found! Weights loaded, but Epoch/iteration "
                    + "counts will start from 0, which may corrupt learning rate schedules."
                )
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"[!] ERROR loading accelerator state: {e}")
            raise e

    def save(self, name=None, current_metric=None):
        # Accelerate Native Saving (handles DDP/FSDP/TPU sharding)
        save_dir = Path(self.cfg.directory.save)
        if name:
            save_dir = save_dir / name

        self.accelerator.save_state(str(save_dir), safe_serialization=True)

        # Save custom training metadata not handled by accelerate.save_state
        if current_metric is not None:
            val_to_check = current_metric
        elif len(self.val_loss) > 0:
            val_to_check = self.val_loss[-1].item() if hasattr(self.val_loss[-1], "item") else self.val_loss[-1]
        else:
            val_to_check = self.best

        val_tensor = torch.tensor([val_to_check], device=self.device, dtype=torch.float32)
        val_tensor = self.accelerator.gather(val_tensor)
        val_to_check = val_tensor.mean().item()

        is_best = False
        if val_to_check < self.best:
            self.best = val_to_check
            is_best = True

        # Safely extract EMA state and explicitly move shadow tensors to CPU
        ema_state = self.ema.state_dict()
        ema_state_cpu = {
            "shadow": {k: v.cpu() for k, v in ema_state["shadow"].items()},
            "decay": ema_state["decay"],
            "step": ema_state["step"]
        }

        custom_state = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "best": self.best,
            "e_loss": self.e_loss,
            "val_loss": self.val_loss,
            "time": str(datetime.now()),
            "ema": ema_state_cpu,
        }

        if self.accelerator.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(custom_state, save_dir / "custom_state.pth")

            # Export a deploy-ready model using the stabilized EMA weights explicitly applied
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.ema.apply_shadow(unwrapped_model)
            # Strip compiler prefixes to ensure the saved model can be loaded cleanly later
            clean_state_dict = {k.replace("_orig_mod.", ""): v.cpu() for k, v in unwrapped_model.state_dict().items()}
            torch.save(clean_state_dict, save_dir / "ema_model.pth")
            self.ema.restore(unwrapped_model)

        # If best, copy the directory or save a specific indicator
        if is_best:
            best_dir = Path(self.cfg.directory.save) / "best"
            self.accelerator.save_state(str(best_dir), safe_serialization=True)
            if self.accelerator.is_main_process:
                torch.save(custom_state, best_dir / "custom_state.pth")

                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self.ema.apply_shadow(unwrapped_model)
                clean_state_dict = {k.replace("_orig_mod.", ""): v.cpu() for k, v in unwrapped_model.state_dict().items()}
                torch.save(clean_state_dict, best_dir / "ema_model.pth")
                self.ema.restore(unwrapped_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/hydra", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    learner = Learner(cfg_path)
    learner.train()