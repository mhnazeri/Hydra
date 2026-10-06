from __future__ import annotations

from pathlib import Path
import argparse
import pickle

import numpy as np

from hydra.utilities.transformations import positions_to_deltas, transform_robot_centric
from hydra.utilities.normalization import IMAGENET_MEAN, IMAGENET_STD


def _iter_trajectory_dirs(root_path: Path):
    trajectory_dirs = [p for p in root_path.glob("*/*") if p.is_dir()]
    if not trajectory_dirs:
        trajectory_dirs = [p for p in root_path.glob("*") if p.is_dir()]
    return trajectory_dirs


def calculate_socialnav_stats(
    root: str,
    obs_len: int,
    pred_len: int,
    original_data_frequency_hz: int = 10,
    sampling_frequency_hz: int = 10,
    output_path: str | None = None,
):
    root_path = Path(root)
    step_size = max(1, round(original_data_frequency_hz / sampling_frequency_hz))
    seq_len_sampled = obs_len + pred_len
    seq_len_original = seq_len_sampled * step_size

    vw_chunks: list[np.ndarray] = []
    xy_chunks: list[np.ndarray] = []
    total_windows = 0

    for traj_dir in _iter_trajectory_dirs(root_path):
        try:
            metadata_path = next(traj_dir.glob("*.pkl"))
        except StopIteration:
            continue

        with metadata_path.open("rb") as f:
            traj_data = pickle.load(f)

        required = {"position", "yaw", "vw"}
        if not required.issubset(set(traj_data.keys())):
            continue

        traj_len = min(
            len(traj_data["position"]),
            len(traj_data["yaw"]),
            len(traj_data["vw"]),
        )
        if traj_len < seq_len_original:
            continue

        for i in range(traj_len - seq_len_original + 1):
            past_positions = traj_data["position"][i : i + obs_len * step_size : step_size].astype(np.float32)
            future_positions = traj_data["position"][i + obs_len * step_size : i + seq_len_original : step_size].astype(np.float32)
            past_yaw = traj_data["yaw"][i : i + obs_len * step_size : step_size].astype(np.float32)
            future_yaw = traj_data["yaw"][i + obs_len * step_size : i + seq_len_original : step_size].astype(np.float32)
            past_vw = traj_data["vw"][i : i + obs_len * step_size : step_size].astype(np.float32)
            future_vw = traj_data["vw"][i + obs_len * step_size : i + seq_len_original : step_size].astype(np.float32)

            past_positions, future_positions = transform_robot_centric(
                past_positions, past_yaw, future_positions, future_yaw
            )
            past_deltas = positions_to_deltas(past_positions).astype(np.float32)
            future_deltas = positions_to_deltas(future_positions).astype(np.float32)

            vw_chunks.append(np.concatenate([past_vw, future_vw], axis=0))
            xy_chunks.append(np.concatenate([past_deltas, future_deltas], axis=0))
            total_windows += 1

    if total_windows == 0:
        raise RuntimeError(f"No valid windows found under {root_path}")

    vw_all = np.concatenate(vw_chunks, axis=0)
    xy_all = np.concatenate(xy_chunks, axis=0)

    vw_mean = vw_all.mean(axis=0, dtype=np.float64).astype(np.float32)
    vw_std = vw_all.std(axis=0, dtype=np.float64).astype(np.float32)
    xy_mean = xy_all.mean(axis=0, dtype=np.float64).astype(np.float32)
    xy_std = xy_all.std(axis=0, dtype=np.float64).astype(np.float32)
    vw_std = np.maximum(vw_std, 1e-6)
    xy_std = np.maximum(xy_std, 1e-6)

    stats = {
        "version": 1,
        "dataset_root": str(root_path),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "sampling": {
            "original_data_frequency_hz": int(original_data_frequency_hz),
            "sampling_frequency_hz": int(sampling_frequency_hz),
            "step_size": int(step_size),
            "num_windows": int(total_windows),
        },
        "motion": {
            "vw": {"mean": vw_mean.tolist(), "std": vw_std.tolist()},
            "xy": {"mean": xy_mean.tolist(), "std": xy_std.tolist()},
        },
        "image": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "source": "imagenet",
        },
    }

    output = Path(output_path) if output_path else root_path / "normalization_stats.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(stats, f)

    print(f"Saved normalization stats to: {output}")
    print(f"vw mean/std: {stats['motion']['vw']['mean']} / {stats['motion']['vw']['std']}")
    print(f"xy mean/std: {stats['motion']['xy']['mean']} / {stats['motion']['xy']['std']}")
    print(f"image mean/std: {stats['image']['mean']} / {stats['image']['std']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--obs-len", type=int, default=16)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--original-hz", type=int, default=10)
    parser.add_argument("--sampling-hz", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    calculate_socialnav_stats(
        root=args.root,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        original_data_frequency_hz=args.original_hz,
        sampling_frequency_hz=args.sampling_hz,
        output_path=args.output,
    )