"""File-backed dataset shared by Step0 and incremental training."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyDeviceDataset(Dataset):
    def __init__(self, root, allowed_labels=None, max_per_class=0):
        roots = [Path(p) for p in root] if isinstance(root, (list, tuple)) else [Path(root)]
        allowed = None if allowed_labels is None else {int(v) for v in allowed_labels}
        pattern = re.compile(r"device_(\d+)_\d+(?:_[\w\d]+)?\.npy$")
        by_label = defaultdict(list)
        for folder in roots:
            if not folder.exists():
                raise FileNotFoundError(folder)
            for path in folder.rglob("*.npy"):
                match = pattern.match(path.name)
                if match and (allowed is None or int(match.group(1)) in allowed):
                    by_label[int(match.group(1))].append(path)
        self.samples = []
        for label in sorted(by_label):
            paths = sorted(by_label[label])
            if max_per_class:
                paths = paths[:max_per_class]
            self.samples.extend((path, label) for path in paths)
        if not self.samples:
            raise RuntimeError(f"No samples found under {roots}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return torch.from_numpy(np.load(path).astype(np.float32, copy=False)), label

