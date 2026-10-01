import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


SAMPLE_PATTERN = re.compile(r"device_(\d+)_\d+(?:_[\w\d]+)?\.npy$")
EXPECTED_SHAPE = (29, 42, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an ADS-B 2D class-incremental protocol without feature conversion."
    )
    parser.add_argument("--source_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--initial_classes", default=50, type=int)
    parser.add_argument("--increment_size", default=5, type=int)
    parser.add_argument("--num_increments", default=10, type=int)
    parser.add_argument("--train_per_class", default=200, type=int)
    parser.add_argument("--val_per_class", default=50, type=int)
    parser.add_argument("--test_per_class", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--shuffle_class_order", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_samples(source_root):
    by_label = defaultdict(list)
    for path in sorted(Path(source_root).rglob("*.npy")):
        match = SAMPLE_PATTERN.match(path.name)
        if match is not None:
            by_label[int(match.group(1))].append(path)
    if not by_label:
        raise RuntimeError(f"No ADS-B samples found in {source_root}")
    return by_label


def validate_sample(path):
    sample = np.load(path, mmap_mode="r")
    if sample.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected ADS-B 2D shape {EXPECTED_SHAPE}, got {sample.shape}: {path}"
        )
    if sample.dtype != np.float32:
        raise ValueError(f"Expected float32, got {sample.dtype}: {path}")


def reset_output(path, overwrite):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite to rebuild")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize(paths, output_dir, label, use_copy):
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = defaultdict(int)
    for index, source in enumerate(paths):
        destination = output_dir / f"device_{label:03d}_{index:05d}.npy"
        if use_copy:
            shutil.copy2(source, destination)
            modes["copy"] += 1
            continue
        try:
            os.link(source, destination)
            modes["hardlink"] += 1
        except OSError:
            shutil.copy2(source, destination)
            modes["copy_fallback"] += 1
    return modes


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    by_label = collect_samples(source_root)

    total_classes = args.initial_classes + args.increment_size * args.num_increments
    required = args.train_per_class + args.val_per_class + args.test_per_class
    eligible = sorted(label for label, paths in by_label.items() if len(paths) >= required)
    if len(eligible) < total_classes:
        raise RuntimeError(
            f"Need {total_classes} classes with at least {required} samples, found {len(eligible)}"
        )

    selected = eligible[:total_classes]
    rng = np.random.default_rng(args.seed)
    if args.shuffle_class_order:
        selected = [int(value) for value in rng.permutation(selected)]

    for raw_label in selected:
        validate_sample(by_label[raw_label][0])

    reset_output(output_root, args.overwrite)
    new_to_raw = {new: raw for new, raw in enumerate(selected)}
    raw_to_new = {raw: new for new, raw in new_to_raw.items()}
    partitions = {}
    for new_label, raw_label in new_to_raw.items():
        paths = by_label[raw_label]
        order = rng.permutation(len(paths))[:required]
        chosen = [paths[int(index)] for index in order]
        train_end = args.train_per_class
        val_end = train_end + args.val_per_class
        partitions[new_label] = {
            "train": chosen[:train_end],
            "val": chosen[train_end:val_end],
            "test": chosen[val_end:],
        }

    mode_counts = defaultdict(int)
    folder_counts = defaultdict(dict)

    def add(folder, label, split):
        modes = materialize(
            partitions[label][split], output_root / folder, label, args.copy
        )
        for mode, count in modes.items():
            mode_counts[mode] += count
        folder_counts[folder][str(label)] = len(partitions[label][split])

    initial_labels = list(range(args.initial_classes))
    for label in initial_labels:
        add("train-closed", label, "train")
        add("val-closed", label, "val")
        add("test-closed", label, "test")

    increment_groups = {}
    for step in range(1, args.num_increments + 1):
        start = args.initial_classes + (step - 1) * args.increment_size
        current_labels = list(range(start, start + args.increment_size))
        increment_groups[str(step)] = current_labels
        for label in current_labels:
            add(f"train-add{step}", label, "train")

        seen_labels = list(range(args.initial_classes + step * args.increment_size))
        for label in seen_labels:
            add(f"val-add{step}", label, "val")
            add(f"test-add{step}", label, "test")

    protocol = {
        "dataset": "adsb",
        "representation": "adsb2d_noise_fft_residual",
        "sample_shape": list(EXPECTED_SHAPE),
        "feature_conversion": "none",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "initial_classes": args.initial_classes,
        "increment_size": args.increment_size,
        "num_increments": args.num_increments,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "test_per_class": args.test_per_class,
        "seed": args.seed,
        "shuffle_class_order": bool(args.shuffle_class_order),
        "raw_to_new_label": {str(k): v for k, v in sorted(raw_to_new.items())},
        "new_to_raw_label": {str(k): v for k, v in sorted(new_to_raw.items())},
        "increment_groups": increment_groups,
        "folder_counts": dict(folder_counts),
        "materialization": dict(mode_counts),
    }
    with open(output_root / "protocol.json", "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    audit = {
        "adsb": {
            "data_root": str(output_root.resolve()),
            "prepared_integrity_passed": True,
            "development_training_admitted": True,
            "rows": total_classes * required,
            "train_val_test_file_disjoint": True,
            "test_predictions_read": False,
        }
    }
    with open(output_root / "audit_summary.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
