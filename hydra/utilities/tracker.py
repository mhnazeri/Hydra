import torch
import numpy as np
from collections import defaultdict

class MetricTracker:
    """
    Helper class to track and compute running metrics automatically during training loops.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(list)

    def update(self, metrics_dict):
        for k, v in metrics_dict.items():
            # detach and unwrap tensors to standard scalar values
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().mean().item()
            elif isinstance(v, dict):
                continue  # ignore nested dicts, handle them separately if needed
            self.metrics[k].append(v)

    def get_mean(self, key):
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0.0

    def get_means(self):
        return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in self.metrics.items()}