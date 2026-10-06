# Hydra deployment

This directory contains the ROS 1 runtime nodes, planners, controllers, and
configuration files for running Hydra and VertiFormer on a
robot. The nodes recieve camera and odometry topics, load a checkpoint, plan in
the robot's local frame, and publish velocity commands and visualization
topics.

The repository must be installed from its root before running a node:

```bash
pip install -e .
```

Deployment requires a running ROS 1 environment and a CUDA-capable PyTorch installation. Download pretrained checkpoints from the
[Hydra Hugging Face repository](https://huggingface.co/mhnazeri/Hydra), extract
them, and set `deployment.checkpoint_path` to the extracted directory.

## Quick start

From the repository root, edit a copy of the relevant config and start one
node:

```bash
python deployment/hydra_plan_ros.py \
  --conf deployment/conf/hydra_plan.yaml \
  --robot jackal \
  --exp_name hydra_planning
```

Every primary ROS node accepts:

| Argument | Description |
| --- | --- |
| `--conf` | YAML deployment configuration |
| `--robot` | Robot profile, currently `jackal` or `spot` |
| `--exp_name` | Name used for logs and debug recordings |

The planner additionally accepts `--goal_conditioned` to condition seed generation on the goal.

Before running the scripts, verify the camera, odometry, TF, command velocity topic,
frame conventions, robot limits, and an emergency-stop procedure.

## Configuration

- [`conf/robots/jackal.yaml`](conf/robots/jackal.yaml) and
  [`conf/robots/spot.yaml`](conf/robots/spot.yaml) define robot-specific
  limits, transforms, and sensor settings.
- [`conf/hydra_plan.yaml`](conf/hydra_plan.yaml) configures Hydra latent
  planning, waypoint, planner costs, and action chunking.
- [`conf/hydra_plan_multipoint.yaml`](conf/hydra_plan_multipoint.yaml) same as `hydra_plan`, but configures
  multi-waypoint Hydra planning.
- [`conf/hydra_path_follow.yaml`](conf/hydra_path_follow.yaml) configures Hydra
  path following.
- [`conf/vertiformer_path_follow.yaml`](conf/vertiformer_path_follow.yaml)
  configure the original VertiFormer ROS nodes.

To add a robot, create `conf/robots/<robot_name>.yaml` and pass the new name
with `--robot`.

## Hydra ROS nodes

| Script | Description |
| --- | --- |
| [`hydra_plan_ros.py`](hydra_plan_ros.py) | Main Hydra latent-MPC planner. It searches discrete latent intents, decodes candidate trajectories, follows the selected path, and publishes commands and RViz/debug outputs. |
| [`hydra_plan_ros_multipoint.py`](hydra_plan_ros_multipoint.py) | Hydra planner for a sequence of global waypoints; it advances and resets planning state at each sub-goal. |
| [`hydra_follower_ros_cfg.py`](hydra_follower_ros_cfg.py) | Hydra path follower using classifier-free guidance to balance path tracking and obstacle-avoidance behavior. |
| [`hydra_follower_ros_sampling.py`](hydra_follower_ros_sampling.py) | Hydra path follower using candidate sampling and temporal action/pose blending. |
| [`dlp.py`](dlp.py) | Core Hydra discrete latent planning implementation. |
| [`path_follower_cfg.py`](path_follower_cfg.py) | Classifier-free-guidance path-following. |
| [`path_follower_sampling.py`](path_follower_sampling.py) | Sampling-based path-following. |

Run the sampling-based follower instead of the planner with:

```bash
python deployment/hydra_follower_ros_sampling.py \
  --conf deployment/conf/hydra_path_follow.yaml \
  --robot jackal
```

## VertiFormer nodes

| Script | Description |
| --- | --- |
| [`vertiformer/vertiformer_follower_ros_cfg.py`](vertiformer/vertiformer_follower_ros_cfg.py) | VertiFormer path follower with classifier-free guidance. |
| [`vertiformer/vertiformer_follower_ros_sampling.py`](vertiformer/vertiformer_follower_ros_sampling.py) | VertiFormer path follower using sampled candidate actions/poses. |
| [`vertiformer/path_follower_cfg.py`](vertiformer/path_follower_cfg.py) | VertiFormer CFG path-following utilities. |
| [`vertiformer/path_follower_sampling.py`](vertiformer/path_follower_sampling.py) | VertiFormer sampling-based path-following utilities. |

## Model weights

Download Hydra or VertiFormer archives from
[Hugging Face](https://huggingface.co/mhnazeri/Hydra), extract them, and point
the selected YAML's `deployment.checkpoint_path` at the directory containing
the architecture YAML and `hydra.pth` or `vertiformer.pth`. The runtime
loads the architecture from the YAML in that directory, so keep the matching
configuration with the weights.