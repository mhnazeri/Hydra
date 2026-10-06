from pathlib import Path
import pickle
from typing import Union

import numpy as np
import torch
import torchvision.io as tv_io
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import default_collate

def imread(address: str):
    return tv_io.read_image(address, mode=tv_io.ImageReadMode.RGB)


def custom_collate(batch):
    elem = batch[0]
    collated = {}
    for key in elem:
        # If any sample in the batch is missing the modality, use None
        if any(d[key] is None for d in batch):
            collated[key] = None
        else:
            collated[key] = default_collate([d[key] for d in batch])
    return collated


class NavigationDataset(Dataset):
    def __init__(
            self,
            root: str,
            obs_len: int,
            pred_len: int,
            train: bool = True,
            resize: Union[list, tuple] = (224, 224),
            original_data_frequency_hz: int = 10,
            sampling_frequency_hz: int = 10,
            sample_stride: int = 1,
            **kwargs,
    ):
        self.resize = resize
        self.train = train
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.original_data_frequency_hz = original_data_frequency_hz
        self.sampling_frequency_hz = sampling_frequency_hz

        self.temporal_step = max(1, round(self.original_data_frequency_hz / self.sampling_frequency_hz))
        self.sample_stride = sample_stride

        root_path = Path(root)
        # Caches full trajectories based on the frequency.
        cache_filename = f"cache_trajectories_freq{self.sampling_frequency_hz}.pkl"
        cache_path = root_path / cache_filename

        self.trajectories = []

        if cache_path.exists():
            print(f"Loading continuous trajectory data from cache: {cache_path}")
            with cache_path.open("rb") as f:
                self.trajectories = pickle.load(f)
        else:
            print(f"Cache not found. Processing trajectories and creating cache at: {cache_path}")

            trajectory_dirs = [p for p in root_path.glob("*/*") if p.is_dir()]
            if not trajectory_dirs:
                trajectory_dirs = [p for p in root_path.glob("*") if p.is_dir()]

            for traj_dir in trajectory_dirs:
                try:
                    metadata_path = next(traj_dir.glob("*.pkl"))
                except StopIteration:
                    continue

                with metadata_path.open("rb") as f:
                    traj_data = pickle.load(f)

                rgb_path = traj_dir / "rgb"
                if not rgb_path.exists():
                    continue

                all_frames_original = sorted(
                    list(rgb_path.iterdir()),
                    key=lambda x: int(x.name.split(".")[0]),
                )

                has_vw = traj_data.get("vw") is not None
                lengths = [len(traj_data["position"]), len(traj_data["yaw"]), len(all_frames_original)]
                if has_vw:
                    lengths.append(len(traj_data["vw"]))
                traj_len = min(lengths)

                if traj_len < (self.obs_len + self.pred_len + 1) * self.temporal_step:
                    continue  # Trajectory too short

                # Downsample the full trajectory and store it as a contiguous block
                world_pos = traj_data["position"][:traj_len:self.temporal_step].astype(np.float32)
                world_yaw = traj_data["yaw"][:traj_len:self.temporal_step].astype(np.float32)
                all_frames = all_frames_original[:traj_len:self.temporal_step]

                traj_dict = {
                    "position": world_pos,
                    "yaw": world_yaw,
                    "frames": all_frames,
                    "vw": traj_data["vw"][:traj_len:self.temporal_step].astype(np.float32) if has_vw else None
                }
                self.trajectories.append(traj_dict)

            with cache_path.open("wb") as f:
                pickle.dump(self.trajectories, f)

        # Map valid indexing windows
        self.valid_indices = []
        self._build_index_map()

        # define transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize, antialias=True),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _build_index_map(self):
        """
        Calculates all valid (trajectory_id, start_frame) tuples.
        """
        seq_len = self.obs_len + self.pred_len
        self.valid_indices = []

        for traj_idx, traj in enumerate(self.trajectories):
            M = len(traj["position"])
            # Start index begins at 1 because calculating the first delta requires start_idx - 1
            for start_idx in range(1, M - seq_len + 1, self.sample_stride):
                self.valid_indices.append((traj_idx, start_idx))

        print(f"Dataset mapped: {len(self.valid_indices)} valid temporal windows extracted from {len(self.trajectories)} trajectories.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        traj_idx, start_idx = self.valid_indices[idx]
        traj = self.trajectories[traj_idx]
        seq_len = self.obs_len + self.pred_len

        # 1. Slice Continuous State Windows
        pos_window = traj["position"][start_idx - 1 : start_idx + seq_len].copy()
        yaw_window = traj["yaw"][start_idx - 1 : start_idx + seq_len].copy()

        frame_paths = traj["frames"][start_idx : start_idx + seq_len]

        if traj["vw"] is not None:
            vw_window = traj["vw"][start_idx : start_idx + seq_len].copy()
            past_vw = torch.from_numpy(vw_window[:self.obs_len]).float()
            fut_vw = torch.from_numpy(vw_window[self.obs_len:]).float()
        else:
            past_vw, fut_vw = None, None

        # 2. Compute Deltas
        # Shapes: [seq_len, 2] and [seq_len]
        pos_deltas = pos_window[1:] - pos_window[:-1]
        yaw_deltas = yaw_window[1:] - yaw_window[:-1]
        yaw_deltas = (yaw_deltas + np.pi) % (2 * np.pi) - np.pi

        # Split into Past and Future
        past_pos_deltas = torch.from_numpy(pos_deltas[:self.obs_len]).float()
        fut_pos_deltas = torch.from_numpy(pos_deltas[self.obs_len:]).float()

        past_yaw_deltas = torch.from_numpy(yaw_deltas[:self.obs_len]).float()
        fut_yaw_deltas = torch.from_numpy(yaw_deltas[self.obs_len:]).float()

        past_yaw = torch.from_numpy(yaw_window[1 : 1 + self.obs_len]).float()
        fut_yaw = torch.from_numpy(yaw_window[1 + self.obs_len :]).float()

        # 3. Ego Rotation
        # Ego state is anchored exactly at the final observation index
        ego_yaw = yaw_window[self.obs_len]

        cos_y, sin_y = np.cos(-ego_yaw), np.sin(-ego_yaw)
        R = torch.tensor([[cos_y, -sin_y],
                          [sin_y,  cos_y]], dtype=torch.float32)

        past_pos_deltas = past_pos_deltas @ R.T
        fut_pos_deltas = fut_pos_deltas @ R.T

        # 4. Load & Transform Images
        past_frames = torch.stack([
            self.transform(imread(str(p))) for p in frame_paths[:self.obs_len]
        ], dim=0)

        future_frames = torch.stack([
            self.transform(imread(str(p))) for p in frame_paths[self.obs_len:]
        ], dim=0)

        return {
            "past_positions": torch.cat([past_pos_deltas, past_yaw_deltas.unsqueeze(-1)], dim=-1),
            "future_positions": torch.cat([fut_pos_deltas, fut_yaw_deltas.unsqueeze(-1)], dim=-1),
            "past_yaw": past_yaw,
            "future_yaw": fut_yaw,
            "past_vw": past_vw,
            "future_vw": fut_vw,
            "past_frames": past_frames,
            "future_frames": future_frames,
        }

if __name__ == "__main__":
    # This is for testing
    try:
        dataset = NavigationDataset(
            root="/path/to/your/dataset",
            obs_len=6,
            pred_len=20,
            train=False,
        )
        if len(dataset) > 0:
            data = dataset[0]
            print(f"Sample future positions:\n{data['future_positions']}")
            print(f"Past positions shape: {data['past_positions'].shape}")
        else:
            print("Dataset is empty. Check your root path and data structure.")
    except FileNotFoundError:
        print("Please update the root path in the `if __name__ == '__main__'` block to test the dataloader.")