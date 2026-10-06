# [Hydra: A Navigation World Action Model](https://arxiv.org/pdf/2608.28995)

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://robotixx.cs.gmu.edu/"><img src="https://img.shields.io/badge/Collection-RobotiXX-green" alt="Robotixx Open Source"></a>
  <a href="https://huggingface.co/mhnazeri/Hydra"><img src="https://img.shields.io/badge/Hugging%20Face-Weights-yellow.svg?logo=huggingface" alt="Pretrained weights on Hugging Face"></a>
  <a href="https://robotixx.github.io/hydra/"><img src="https://img.shields.io/badge/Website-Hydra-blue.svg?logo=firefox" alt="Hydra project website"></a>
  <a href="https://arxiv.org/pdf/2608.28995"><img src="https://img.shields.io/badge/arXiv-2608.2899-b31b1b.svg?logo=arxiv" alt="Hydra paper on arXiv"></a>
</p>

Hydra is a Navigation World Action Model (WAM) for robot navigation. It combines **discrete latent planning** with **continuous flow-matching execution** to predict future images, poses, and actions from an observation history.

## Installation

Hydra is intended to be installed from the repository so that the `hydra` package is importable by the training and deployment scripts.

```bash
git clone https://github.com/mhnazeri/Hydra.git
cd Hydra

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

The package dependencies are read from [`requirements.txt`](requirements.txt) by the project build configuration. Install a CUDA-compatible PyTorch build appropriate for your machine if the automatically selected build is not suitable. Training requires a CUDA-capable GPU in normal use; ROS deployment additionally requires a ROS 1 environment with the message packages used by the selected robot.

## Repository structure

```text
hydra/
├── model/              # Hydra, VertiFormer, VQ, transformer, and flow-matching modules
├── conf/               # Training and model configuration
├── utilities/          # Data, losses, evaluation, visualization, and ROS helpers
├── train.py            # Hydra training loop
├── train_former.py     # VertiFormer training loop

deployment/
├── conf/               # Runtime configs and robot-specific sensor definitions
├── hydra_plan_ros*.py  # Hydra latent planning ROS nodes
├── hydra_follower*.py  # Hydra path-following ROS nodes
└── vertiformer/        # VertiFormer deployment nodes

run.sh                 # Shell wrapper around Accelerate training
```

The main Hydra model is implemented in [`hydra/model/hydra.py`](hydra/model/hydra.py). [`hydra/model/dataloader.py`](hydra/model/dataloader.py) loads trajectories and creates observation/prediction windows. Training defaults are in [`hydra/conf/hydra.yaml`](hydra/conf/hydra.yaml), and ROS runtime defaults are in [`deployment/conf/`](deployment/conf/).

## Dataset preparation

Hydra uses trajectory data in the same general form as Navigation World Models (NWM). Follow the [NWM data instructions](https://github.com/facebookresearch/nwm/#data), which in turn point to the [NoMaD data-wrangling instructions](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling), to download and preprocess the source datasets. NWM describes the required bag/reconstruction preprocessing and the dataset organization.

Before training Hydra, configure the resulting data for [`NavigationDataset`](hydra/model/dataloader.py). Each trajectory directory must contain:

```text
<dataset-root>/
└── <dataset>/<trajectory>/
    ├── rgb/
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └── traj_data.pkl
```

`traj_data.pkl` must provide `position` and `yaw`; `vw` (linear/angular velocity) is optional. The loader expects the image files in `rgb/` and uses the trajectory metadata to compute local pose/action deltas. If the NWM/NoMaD preprocessing output uses a different image directory or metadata schema, convert it to this layout before training.

Set the train and validation roots in [`hydra/conf/hydra.yaml`](hydra/conf/hydra.yaml):

```yaml
dataset:
  root: /path/to/hydra_train
val_dataset:
  root: /path/to/hydra_val
```

Also check `obs_len`, `pred_len`, image resolution, sampling frequency, and `sample_stride` in the same file. The first dataset load creates a frequency-specific trajectory cache under each dataset root.

## Training

The default Hydra experiment uses [`hydra/conf/hydra.yaml`](hydra/conf/hydra.yaml). From the repository root, launch it with the Python wrapper:

```bash
./run.sh train
```

The wrapper invokes `accelerate launch hydra/train.py`.

For multi-GPU training, increase `--num-processes` (or edit `NUM_PROCESSES` in `run.sh`) and select the desired precision. The training script saves dated experiments under `hydra/checkpoint/`, including the resolved configuration and model code; checkpoints contain `ema_model.pth` and/or safetensors shards.

Additional training variants are available through the wrappers:

```bash
# VertiFormer
./run.sh train_former
```

For a new experiment, copy the relevant YAML configuration and change the dataset roots and optimization/model parameters rather than editing a saved checkpoint configuration.

## Download pretrained weights

The Hugging Face repository provides downloadable archives for both models:

| Model | Download |
| --- | --- |
| Hydra | [hydra.zip](https://huggingface.co/mhnazeri/Hydra/resolve/main/hydra.zip) |
| VertiFormer | [vertiformer.zip](https://huggingface.co/mhnazeri/Hydra/resolve/main/vertiformer.zip) |

Download and extract the archive into a local checkpoint directory:

```bash
mkdir -p checkpoints/hydra
curl -L https://huggingface.co/mhnazeri/Hydra/resolve/main/hydra.zip \
  -o /tmp/hydra.zip
unzip /tmp/hydra.zip -d checkpoints/hydra
```

Use the extracted directory itself as `deployment.checkpoint_path`; do not point the config at the zip file. The directory must contain the architecture YAML saved with the experiment `hydra.pth` or the `vertiformer.pth` weights. If the archive extracts into an extra nested directory, use that nested directory as the checkpoint path.

## ROS deployment

Deployment scripts expect a running ROS 1 master, the camera and odometry topics configured for the target robot, and a checkpoint directory downloaded as described above. Edit a copy of [`deployment/conf/hydra_plan.yaml`](deployment/conf/hydra_plan.yaml) or [`deployment/conf/hydra_path_follow.yaml`](deployment/conf/hydra_path_follow.yaml):

1. Set `deployment.checkpoint_path` to the extracted checkpoint directory.
2. Set `deployment.device` to `cuda` or a specific CUDA device.
3. Set the robot sensor topics in the merged robot config under [`deployment/conf/robots/`](deployment/conf/robots/).
4. Set `waypoints`, goal images, `pred_len`, and `chunk_size` for the mission.

All main deployment nodes accept `--robot {jackal,spot}` and `--conf`. Start a planner with:

```bash
python deployment/hydra_plan_ros.py \
  --conf deployment/conf/hydra_plan.yaml \
  --robot jackal
```

For a goal-conditioned planner or a multi-point mission:

```bash
python deployment/hydra_plan_ros_conditioned.py \
  --conf deployment/conf/hydra_plan.yaml \
  --robot jackal \
  --goal_conditioned
```

For path following, choose classifier-free guidance (`cfg`) or sampling-based execution:

```bash
python deployment/hydra_follower_ros_cfg.py \
  --conf deployment/conf/hydra_path_follow.yaml \
  --robot jackal

python deployment/hydra_follower_ros_sampling.py \
  --conf deployment/conf/hydra_path_follow.yaml \
  --robot jackal
```

These nodes publish velocity commands and visualization topics, so verify topic names, transforms, robot limits, and an emergency stop procedure before enabling motion. The more detailed deployment notes are in [`deployment/README.md`](deployment/README.md).

## Citation

```bibtex
@article{nazeri2026hydra,
  title={Hydra: A Navigation World Action Model with Discrete Latent Planning and Continuous Flow-Matching Execution},
  author={Nazeri, Mohammad and Card, Alexandyr and Huber, Samira and Pokhrel, Anuj and Wang, Yujun and Hammele, Ruben and Song, Daeun and Pirk, S{\"o}ren and Xiao, Xuesu},
  journal={arXiv preprint arXiv:2608.28995},
  year={2026}
}
```