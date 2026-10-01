import argparse
import copy
import json
import math
import os
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from loss import SupConLoss
from model_mid import ContrastiveLayer
from model_mix import ClassifierHead
from util.utils import adjust_lr_add, set_seed


DATASET_DEFAULTS = {
    "oracle": {
        "init_dir": "model-5woaux",
        "data_root": "data/oracle",
        "initial_classes": 50,
        "increment_size": 5,
        "num_increments": 10,
        "new_class_start": 50,
        "train_add_template": "train-add{step}",
        "test_add_template": "test-add{step}",
        "test_closed": "test-closed",
        "support_k": 4,
        "support_radius_stat": "p80",
    },
    "adsb": {
        "init_dir": "model-adsb3",
        "data_root": "data/adsb",
        "initial_classes": 20,
        "increment_size": 5,
        "num_increments": 4,
        "new_class_start": 100,
        "train_add_template": "train-openset",
        "test_add_template": "val-mixed",
        "test_closed": "test-closed",
        "support_k": 6,
        "support_radius_stat": "p99",
    },
}


class IdentityProjector(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=1)


class NpyDeviceDataset(Dataset):
    def __init__(self, root, allowed_labels=None, max_per_class=0):
        super().__init__()
        roots = [Path(item) for item in root] if isinstance(root, (list, tuple)) else [Path(root)]
        self.root = roots[0] if len(roots) == 1 else roots
        self.allowed_labels = None if allowed_labels is None else set(int(x) for x in allowed_labels)
        missing = [str(item) for item in roots if not item.exists()]
        if missing:
            raise FileNotFoundError(f"Data folder not found: {missing}")

        pattern = re.compile(r"device_(\d+)_\d+(?:_[\w\d]+)?\.npy$")
        by_label = defaultdict(list)
        for folder in roots:
            for dirpath, _, filenames in os.walk(folder):
                for fname in filenames:
                    if not fname.endswith(".npy"):
                        continue
                    match = pattern.match(fname)
                    if match is None:
                        continue
                    label = int(match.group(1))
                    if self.allowed_labels is not None and label not in self.allowed_labels:
                        continue
                    by_label[label].append(str(Path(dirpath) / fname))

        self.samples = []
        for label in sorted(by_label.keys()):
            paths = sorted(by_label[label])
            if max_per_class and max_per_class > 0:
                paths = paths[:max_per_class]
            self.samples.extend((path, label) for path in paths)

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        data = np.load(path)
        return torch.tensor(data, dtype=torch.float32), int(label)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Class-incremental training with multi-local support-domain memory."
    )
    parser.add_argument("--dataset", choices=["oracle", "adsb", "custom"], default="oracle")
    parser.add_argument("--init_dir", default=None, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--data_root", default=None, type=str)
    parser.add_argument("--initial_classes", default=None, type=int)
    parser.add_argument("--increment_size", default=None, type=int)
    parser.add_argument("--num_increments", default=None, type=int)
    parser.add_argument("--new_class_start", default=None, type=int)
    parser.add_argument("--train_add_template", default=None, type=str)
    parser.add_argument("--test_add_template", default=None, type=str)
    parser.add_argument("--test_closed", default=None, type=str)
    parser.add_argument("--val_add_template", default="", type=str)
    parser.add_argument("--val_closed", default="", type=str)
    parser.add_argument("--select_increment_by_val", action="store_true")
    parser.add_argument(
        "--val_selection_metric",
        choices=["total_acc", "old_acc", "new_acc", "balanced_acc", "harmonic_acc"],
        default="total_acc",
    )

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--embedding_dim", default=1024, type=int)
    parser.add_argument("--epochs", default=36, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--incr_batch_size", default=150, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--ratio", default=0.5, type=float)
    parser.add_argument("--max_feature_per_class", default=100, type=int)
    parser.add_argument("--max_train_per_class", default=0, type=int)
    parser.add_argument("--max_test_per_class", default=0, type=int)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--lr_decay_epochs", default="30,50,140", type=str)

    parser.add_argument("--ce_weight", default=2.0, type=float)
    parser.add_argument("--contrastive_weight", default=1.0, type=float)
    parser.add_argument("--distill_weight", default=0.5, type=float)
    parser.add_argument("--new_ce_weight", default=2.0, type=float)
    parser.add_argument("--old_ce_weight", default=1.0, type=float)
    parser.add_argument("--temperature", default=0.08, type=float)
    parser.add_argument("--projector_alpha", default=0.5, type=float)

    parser.add_argument("--support_keep_weight", default=0.0, type=float)
    parser.add_argument("--support_sep_weight", default=0.0, type=float)
    parser.add_argument("--support_replay_per_class", default=0, type=int)
    parser.add_argument(
        "--disable_feature_cache_replay",
        action="store_true",
        help="Disable old encoder-feature cache replay during incremental training.",
    )
    parser.add_argument("--support_k", default=None, type=int)
    parser.add_argument(
        "--support_center_mode",
        choices=["fps", "density_farthest", "kmeans", "random_features", "herding", "covariance"],
        default="fps",
        help=(
            "How to construct old-class support replay centers. "
            "fps is the proposed farthest-first multi-local construction; "
            "density_farthest multiplies farthest-first coverage by local density; "
            "kmeans uses Lloyd iterations; random_features replays random cached "
            "features; herding greedily approximates the class mean; covariance "
            "places anchors along principal covariance directions."
        ),
    )
    parser.add_argument("--support_radius_stat", default=None, type=str)
    parser.add_argument("--support_tau", default=1.0, type=float)
    parser.add_argument("--support_margin", default=0.15, type=float)
    parser.add_argument("--support_min_cluster_size", default=5, type=int)
    parser.add_argument("--support_eps", default=1e-6, type=float)
    parser.add_argument(
        "--support_loss_space",
        choices=["teacher_projector", "current_projector"],
        default="teacher_projector",
        help=(
            "Coordinate space for support-domain training losses. "
            "teacher_projector compares current encoder features through the frozen "
            "previous projector against the previous support domain."
        ),
    )
    parser.add_argument(
        "--cache_stage",
        choices=["feature", "bottleneck_layer4", "bottleneck_layer3"],
        default="feature",
    )
    parser.add_argument("--train_closed", default="train-closed", type=str)
    parser.add_argument(
        "--encoder_update_mode",
        choices=["frozen", "all", "fc", "layer4", "layer3_layer4"],
        default="frozen",
    )
    parser.add_argument("--encoder_lr", default=1e-5, type=float)
    parser.add_argument("--encoder_lr_after_warmup", default=0.0, type=float)
    parser.add_argument("--encoder_lr_warmup_epochs", default=0, type=int)
    parser.add_argument("--old_raw_source_template", default=None, type=str)
    parser.add_argument("--old_raw_conf", default=None, type=float)
    parser.add_argument("--old_raw_support_tau", default=None, type=float)
    parser.add_argument("--old_raw_max_per_class", default=None, type=int)
    parser.add_argument("--old_raw_ratio", default=None, type=float)
    parser.add_argument("--old_raw_require_label_match", action="store_true")
    parser.add_argument("--replace_old_cache_with_old_raw", action="store_true")
    parser.add_argument("--pseudo_old_source_template", default="", type=str)
    parser.add_argument("--pseudo_old_conf", default=0.90, type=float)
    parser.add_argument("--pseudo_old_support_tau", default=1.0, type=float)
    parser.add_argument("--pseudo_old_max_per_class", default=100, type=int)
    parser.add_argument("--pseudo_old_ratio", default=0.0, type=float)
    parser.add_argument(
        "--old_filter_mode",
        choices=[
            "conf_support",
            "confidence",
            "support",
            "random",
            "raw_random",
            "label_conf_support",
            "raw_farthest",
            "raw_density_farthest",
            "raw_herding",
            "raw_adaptive_mix",
        ],
        default="conf_support",
        help=(
            "Old training memory selection rule: confidence and support distance, "
            "confidence only, support distance only, random labeled old replay, "
            "confidence/support filtering with known old training labels, "
            "or feature-guided true-label raw memory selection."
        ),
    )
    parser.add_argument("--pseudo_old_require_label_match", action="store_true")
    parser.add_argument("--replace_old_cache_with_pseudo", action="store_true")
    parser.add_argument(
        "--old_replay_sampling",
        choices=["uniform", "similarity"],
        default="uniform",
        help="Sample old raw memory uniformly by class or with new-old feature similarity weights.",
    )
    parser.add_argument(
        "--old_replay_similarity_tau",
        default=1.0,
        type=float,
        help="Temperature for converting new-to-old support distances into replay weights.",
    )
    parser.add_argument(
        "--old_replay_similarity_gamma",
        default=0.0,
        type=float,
        help="Strength of similarity-aware old-class replay weighting.",
    )
    parser.add_argument(
        "--anchor_replay_mode",
        choices=["none", "topk", "topk_plus_global", "all_plus_anchor"],
        default="none",
        help=(
            "Use new-class anchor neighborhoods in the old support domain for "
            "feature-cache replay. none keeps the original all-old balanced replay."
        ),
    )
    parser.add_argument(
        "--anchor_replay_topk",
        default=10,
        type=int,
        help="Number of nearest old classes retained as anchors for each new class.",
    )
    parser.add_argument(
        "--anchor_replay_global_fraction",
        default=0.25,
        type=float,
        help=(
            "Fraction of the feature-cache replay budget still sampled from all old "
            "classes when anchor_replay_mode=topk_plus_global."
        ),
    )
    parser.add_argument(
        "--anchor_replay_extra_ratio",
        default=0.25,
        type=float,
        help=(
            "Extra anchor replay budget relative to the original all-old replay "
            "budget when anchor_replay_mode=all_plus_anchor."
        ),
    )
    parser.add_argument("--eval_support_alphas", default="0,0.1,0.25,0.5,1.0", type=str)
    parser.add_argument("--primary_eval", default="standard", type=str)
    parser.add_argument("--eval_stage0", action="store_true")
    parser.add_argument(
        "--early_stop_monitor",
        choices=["none", "val_metric", "train_loss"],
        default="none",
        help="Stop the current incremental stage early when the monitored quantity stops improving.",
    )
    parser.add_argument("--early_stop_patience", default=0, type=int)
    parser.add_argument("--early_stop_min_epochs", default=0, type=int)
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float)
    parser.add_argument("--protocol_note", default="", type=str)
    return resolve_defaults(parser.parse_args())


def resolve_defaults(args):
    if args.old_raw_source_template is not None:
        args.pseudo_old_source_template = args.old_raw_source_template
    if args.old_raw_conf is not None:
        args.pseudo_old_conf = args.old_raw_conf
    if args.old_raw_support_tau is not None:
        args.pseudo_old_support_tau = args.old_raw_support_tau
    if args.old_raw_max_per_class is not None:
        args.pseudo_old_max_per_class = args.old_raw_max_per_class
    if args.old_raw_ratio is not None:
        args.pseudo_old_ratio = args.old_raw_ratio
    if args.old_raw_require_label_match:
        args.pseudo_old_require_label_match = True
    if args.replace_old_cache_with_old_raw:
        args.replace_old_cache_with_pseudo = True

    if args.dataset != "custom":
        defaults = DATASET_DEFAULTS[args.dataset]
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    missing = [
        name
        for name in [
            "init_dir",
            "data_root",
            "initial_classes",
            "increment_size",
            "num_increments",
            "new_class_start",
            "train_add_template",
            "test_add_template",
            "test_closed",
            "support_k",
            "support_radius_stat",
        ]
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")
    args.lr_decay_epochs = [int(x) for x in parse_float_list(args.lr_decay_epochs)]
    args.eval_support_alphas = parse_float_list(args.eval_support_alphas)
    args.incre_lr = args.lr
    args.incre_lr_decay_rate = args.lr_decay_rate
    args.incre_lr_decay_epochs = args.lr_decay_epochs
    return args


def bottleneck_stage_name(args):
    if args.cache_stage == "bottleneck_layer4":
        return "layer4"
    if args.cache_stage == "bottleneck_layer3":
        return "layer3"
    return None


def cache_values_from_raw(encoder, x, args):
    stage = bottleneck_stage_name(args)
    if stage is None:
        return encoder(x)
    if not hasattr(encoder, "forward_to_bottleneck"):
        raise AttributeError("Encoder does not expose forward_to_bottleneck for bottleneck cache.")
    return encoder.forward_to_bottleneck(x, stage=stage)


def features_from_cache(encoder, cache_tensor, args):
    if cache_tensor.numel() == 0:
        return cache_tensor
    stage = bottleneck_stage_name(args)
    if stage is None:
        return cache_tensor
    if not hasattr(encoder, "forward_from_bottleneck"):
        raise AttributeError("Encoder does not expose forward_from_bottleneck for bottleneck cache.")
    return encoder.forward_from_bottleneck(cache_tensor, stage=stage)


def encoder_trainable_prefixes(update_mode):
    if update_mode == "frozen":
        return ()
    if update_mode == "all":
        return None
    if update_mode == "fc":
        return ("fc1",)
    if update_mode == "layer4":
        return ("layer4", "fc1")
    if update_mode == "layer3_layer4":
        return ("layer3", "layer4", "fc1")
    raise ValueError(f"Unsupported encoder_update_mode: {update_mode}")


def configure_encoder_update(encoder, update_mode):
    prefixes = encoder_trainable_prefixes(update_mode)
    if prefixes is None:
        for param in encoder.parameters():
            param.requires_grad = True
        encoder.train()
        return

    for name, param in encoder.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)

    encoder.eval()
    for child_name, child in encoder.named_children():
        if any(child_name == prefix for prefix in prefixes):
            child.train()


def trainable_encoder_parameters(encoder):
    return [param for param in encoder.parameters() if param.requires_grad]


def stage_seen_labels(args, stage):
    labels = list(range(args.initial_classes))
    labels.extend(
        range(
            args.new_class_start,
            args.new_class_start + stage * args.increment_size,
        )
    )
    return labels


def stage_new_labels(args, stage):
    start = args.new_class_start + (stage - 1) * args.increment_size
    return list(range(start, start + args.increment_size))


def stage_group_labels(args, max_stage):
    groups = {0: list(range(args.initial_classes))}
    for stage in range(1, max_stage + 1):
        groups[stage] = stage_new_labels(args, stage)
    return groups


def resolve_stage_folder(args, template, stage):
    name = template.format(step=stage)
    return str(Path(args.data_root) / name)


def cumulative_stage_folders(args, closed_folder, add_template, stage):
    folders = []
    if closed_folder:
        folders.append(str(Path(args.data_root) / closed_folder))
    folders.extend(resolve_stage_folder(args, add_template, idx) for idx in range(1, stage + 1))
    return folders


def get_num_classes(classifier):
    if hasattr(classifier, "classifier"):
        return classifier.classifier[-1].out_features
    if hasattr(classifier, "fc"):
        return classifier.fc.out_features
    raise AttributeError("Unsupported classifier structure.")


def copy_classifier_weights(old_classifier, new_classifier, rows):
    if not hasattr(old_classifier, "classifier") or not hasattr(new_classifier, "classifier"):
        raise AttributeError("Only ClassifierHead-style classifiers are supported.")
    new_classifier.classifier[0].load_state_dict(old_classifier.classifier[0].state_dict())
    old_out = old_classifier.classifier[-1]
    new_out = new_classifier.classifier[-1]
    with torch.no_grad():
        n_rows = min(rows, old_out.out_features, new_out.out_features)
        new_out.weight[:n_rows].copy_(old_out.weight[:n_rows])
        new_out.bias[:n_rows].copy_(old_out.bias[:n_rows])


def make_expanded_classifier(old_classifier, total_classes, embedding_dim, device):
    try:
        classifier = old_classifier.__class__(embedding_dim, total_classes).to(device)
    except TypeError:
        classifier = ClassifierHead(embedding_dim, total_classes).to(device)
    copy_classifier_weights(old_classifier, classifier, get_num_classes(old_classifier))
    return classifier


def cache_candidates(init_dir):
    return [
        Path(init_dir) / "feature_cache_step0.pkl",
        Path(init_dir) / "feature_cache.pkl",
    ]


def load_cache(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "features" in obj:
        return obj["features"]
    return obj


def load_stage_cache(args, stage):
    if stage == 0:
        path = Path(args.save_dir) / "feature_cache_step0.pkl"
        if path.exists():
            return load_cache(path)
        for candidate in cache_candidates(args.init_dir):
            if candidate.exists():
                return load_cache(candidate)
        raise FileNotFoundError(f"Cannot find initial feature cache in {args.init_dir}")

    path = Path(args.save_dir) / f"feature_cache_step{stage}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find feature cache for stage {stage}: {path}")
    return load_cache(path)


def save_cache(save_dir, stage, cache):
    with open(Path(save_dir) / f"feature_cache_step{stage}.pkl", "wb") as f:
        pickle.dump({"version": stage, "features": cache}, f)


@torch.no_grad()
def build_initial_cache_from_train(encoder, classifier, args, device):
    labels = stage_seen_labels(args, 0)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    train_folder = str(Path(args.data_root) / args.train_closed)
    dataset = NpyDeviceDataset(
        train_folder,
        allowed_labels=labels,
        max_per_class=args.max_train_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    cache = {int(label): [] for label in labels}
    encoder.eval()
    classifier.eval()

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y_raw = y_raw.to(device)
        cache_value = cache_values_from_raw(encoder, x, args)
        feat = features_from_cache(encoder, cache_value, args)
        logits = classifier(feat)
        pred_idx = logits.argmax(dim=1).detach().cpu().tolist()
        pred_raw = torch.tensor(
            [labels[int(idx)] for idx in pred_idx],
            dtype=torch.long,
            device=device,
        )
        correct = pred_raw == y_raw

        for raw in sorted(set(int(v) for v in y_raw.detach().cpu().tolist())):
            mask = (y_raw == raw) & correct
            if mask.any():
                cache[raw].extend(cache_value[mask].detach().cpu().tolist())
                max_keep = int(args.max_feature_per_class)
                if len(cache[raw]) > max_keep:
                    cache[raw] = cache[raw][-max_keep:]

    return cache


def prepare_initial_state(args, device):
    os.makedirs(args.save_dir, exist_ok=True)
    init_encoder = Path(args.init_dir) / "encoder.pth"
    init_classifier = Path(args.init_dir) / "classifier.pth"
    if not init_encoder.exists() or not init_classifier.exists():
        raise FileNotFoundError(f"Initial encoder/classifier not found in {args.init_dir}")

    dst_encoder = Path(args.save_dir) / "encoder.pth"
    dst_classifier = Path(args.save_dir) / "classifier.pth"
    if not dst_encoder.exists():
        shutil.copy2(init_encoder, dst_encoder)
    if not dst_classifier.exists():
        shutil.copy2(init_classifier, dst_classifier)

    encoder = torch.load(dst_encoder, map_location=device).to(device).eval()
    classifier = torch.load(dst_classifier, map_location=device).to(device).eval()

    dst_cache = Path(args.save_dir) / "feature_cache_step0.pkl"
    if not dst_cache.exists():
        if args.cache_stage == "feature":
            for candidate in cache_candidates(args.init_dir):
                if candidate.exists():
                    cache = load_cache(candidate)
                    save_cache(args.save_dir, 0, cache)
                    break
        else:
            cache = build_initial_cache_from_train(encoder, classifier, args, device)
            save_cache(args.save_dir, 0, cache)
        if not dst_cache.exists():
            raise FileNotFoundError(f"Initial feature cache not found in {args.init_dir}")

    return encoder, classifier


def load_previous_modules(args, stage, device):
    if stage == 1:
        encoder = torch.load(Path(args.save_dir) / "encoder.pth", map_location=device).to(device).eval()
        classifier = torch.load(Path(args.save_dir) / "classifier.pth", map_location=device).to(device).eval()
        projector = IdentityProjector().to(device).eval()
        return encoder, classifier, projector

    encoder = torch.load(
        Path(args.save_dir) / f"encoder_step{stage - 1}.pth",
        map_location=device,
    ).to(device).eval()
    classifier = torch.load(
        Path(args.save_dir) / f"classifier_step{stage - 1}.pth",
        map_location=device,
    ).to(device).eval()
    projector = torch.load(
        Path(args.save_dir) / f"contrastive_step{stage - 1}.pth",
        map_location=device,
    ).to(device).eval()
    return encoder, classifier, projector


def make_current_projector(prev_projector, args, stage, device):
    if stage == 1 or isinstance(prev_projector, IdentityProjector):
        return ContrastiveLayer(
            in_dim=args.embedding_dim,
            hidden_dim=512,
            out_dim=args.embedding_dim,
            use_norm=True,
            alpha=args.projector_alpha,
        ).to(device)
    return copy.deepcopy(prev_projector).to(device)


def sample_cache_balanced(feature_cache, labels, sample_per_class, device):
    feats, raw_labels = [], []
    for label in labels:
        flist = feature_cache.get(int(label), [])
        if len(flist) == 0:
            continue
        farr = torch.tensor(flist, dtype=torch.float32, device=device)
        if farr.size(0) >= sample_per_class:
            idx = torch.randperm(farr.size(0), device=device)[:sample_per_class]
            sel = farr[idx]
        else:
            idx = torch.randint(0, farr.size(0), (sample_per_class,), device=device)
            sel = farr[idx]
        feats.append(sel)
        raw_labels.extend([int(label)] * sel.size(0))

    if not feats:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(feats, dim=0), torch.tensor(raw_labels, dtype=torch.long, device=device)


def sample_cache_total(feature_cache, labels, total_samples, device):
    labels = [int(label) for label in labels if len(feature_cache.get(int(label), [])) > 0]
    total_samples = int(total_samples)
    if not labels or total_samples <= 0:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    counts = {label: total_samples // len(labels) for label in labels}
    for label in labels[: total_samples % len(labels)]:
        counts[label] += 1

    feats, raw_labels = [], []
    for label in labels:
        count = int(counts.get(label, 0))
        if count <= 0:
            continue
        farr = torch.tensor(feature_cache[int(label)], dtype=torch.float32, device=device)
        if farr.size(0) >= count:
            idx = torch.randperm(farr.size(0), device=device)[:count]
        else:
            idx = torch.randint(0, farr.size(0), (count,), device=device)
        sel = farr[idx]
        feats.append(sel)
        raw_labels.extend([int(label)] * sel.size(0))

    if not feats:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(feats, dim=0), torch.tensor(raw_labels, dtype=torch.long, device=device)


def anchor_labels_for_batch(y_new_raw, anchor_label_map, old_label_order):
    labels = []
    seen = set()
    new_batch_labels = sorted(set(int(label) for label in y_new_raw.detach().cpu().tolist()))
    for new_label in new_batch_labels:
        for old_label in anchor_label_map.get(int(new_label), []):
            old_label = int(old_label)
            if old_label not in seen:
                labels.append(old_label)
                seen.add(old_label)
    return labels if labels else [int(label) for label in old_label_order]


def sample_anchor_cache(
    feature_cache,
    old_label_order,
    anchor_labels,
    total_samples,
    global_fraction,
    device,
):
    total_samples = int(total_samples)
    if total_samples <= 0:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    global_fraction = min(max(float(global_fraction), 0.0), 1.0)
    n_global = int(round(total_samples * global_fraction))
    n_anchor = max(0, total_samples - n_global)

    cache_parts, label_parts = [], []
    anchor_x, anchor_y = sample_cache_total(feature_cache, anchor_labels, n_anchor, device)
    if anchor_x.numel() > 0:
        cache_parts.append(anchor_x)
        label_parts.append(anchor_y)

    if n_global > 0:
        global_x, global_y = sample_cache_total(feature_cache, old_label_order, n_global, device)
        if global_x.numel() > 0:
            cache_parts.append(global_x)
            label_parts.append(global_y)

    if not cache_parts:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(cache_parts, dim=0), torch.cat(label_parts, dim=0)


def sample_pseudo_old_pool(pseudo_pool, labels, sample_per_class, device):
    if not pseudo_pool or sample_per_class <= 0:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    xs, raw_labels = [], []
    for label in labels:
        data = pseudo_pool.get(int(label), None)
        if data is None or data.size(0) == 0:
            continue
        if data.size(0) >= sample_per_class:
            idx = torch.randperm(data.size(0))[:sample_per_class]
        else:
            idx = torch.randint(0, data.size(0), (sample_per_class,))
        sel = data[idx].to(device, non_blocking=True)
        xs.append(sel)
        raw_labels.extend([int(label)] * sel.size(0))
    if not xs:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(xs, dim=0), torch.tensor(raw_labels, dtype=torch.long, device=device)


def sample_pseudo_old_pool_weighted(pseudo_pool, labels, total_samples, class_weights, device):
    if not pseudo_pool or total_samples <= 0:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    available_labels = [int(label) for label in labels if pseudo_pool.get(int(label), None) is not None]
    available_labels = [label for label in available_labels if pseudo_pool[label].size(0) > 0]
    if not available_labels:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    probs = torch.tensor(
        [float(class_weights.get(label, 0.0)) for label in available_labels],
        dtype=torch.float32,
    )
    if probs.sum().item() <= 0:
        probs = torch.ones_like(probs)
    probs = probs / probs.sum()
    chosen = torch.multinomial(probs, int(total_samples), replacement=True).cpu().tolist()
    count_by_label = defaultdict(int)
    for idx in chosen:
        count_by_label[available_labels[int(idx)]] += 1

    xs, raw_labels = [], []
    for label, count in count_by_label.items():
        data = pseudo_pool.get(int(label), None)
        if data is None or data.size(0) == 0 or count <= 0:
            continue
        if data.size(0) >= count:
            idx = torch.randperm(data.size(0))[:count]
        else:
            idx = torch.randint(0, data.size(0), (count,))
        sel = data[idx].to(device, non_blocking=True)
        xs.append(sel)
        raw_labels.extend([int(label)] * sel.size(0))
    if not xs:
        return (
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(xs, dim=0), torch.tensor(raw_labels, dtype=torch.long, device=device)


def squared_distances_np(x, centers):
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    return np.maximum(x2 + c2 - 2.0 * np.matmul(x, centers.T), 0.0)


def farthest_point_clusters(feats, k_eff):
    class_mean = feats.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(squared_distances_np(feats, class_mean)[:, 0]))
    selected = [first_idx]
    min_dist2 = squared_distances_np(feats, feats[[first_idx]])[:, 0]

    while len(selected) < k_eff:
        next_idx = int(np.argmax(min_dist2))
        selected.append(next_idx)
        next_dist2 = squared_distances_np(feats, feats[[next_idx]])[:, 0]
        min_dist2 = np.minimum(min_dist2, next_dist2)

    centers = feats[selected].copy()
    labels = np.zeros(feats.shape[0], dtype=np.int64)
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    for cluster_id in range(k_eff):
        mask = labels == cluster_id
        if np.any(mask):
            centers[cluster_id] = feats[mask].mean(axis=0)
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    return labels, centers.astype(np.float32)


def density_farthest_clusters(feats, k_eff, density_k=10, eps=1e-12):
    if k_eff <= 1:
        return np.zeros(feats.shape[0], dtype=np.int64), feats.mean(axis=0, keepdims=True).astype(np.float32)
    if feats.shape[0] <= 2:
        return farthest_point_clusters(feats, k_eff)

    pair_dist2 = squared_distances_np(feats, feats)
    np.fill_diagonal(pair_dist2, np.inf)
    nn_k = max(1, min(density_k, feats.shape[0] - 1))
    nn_dist2 = np.partition(pair_dist2, nn_k - 1, axis=1)[:, :nn_k]
    finite = nn_dist2[np.isfinite(nn_dist2)]
    sigma2 = max(float(np.median(finite)) if finite.size else eps, eps)
    density = np.exp(-nn_dist2 / sigma2).mean(axis=1)
    density = density / max(float(np.max(density)), eps)
    density = np.maximum(density, 1e-3)

    class_mean = feats.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(squared_distances_np(feats, class_mean)[:, 0]))
    selected = [first_idx]
    min_dist2 = squared_distances_np(feats, feats[[first_idx]])[:, 0]

    while len(selected) < k_eff:
        score = np.sqrt(np.maximum(min_dist2, 0.0)) * density
        score[selected] = -np.inf
        next_idx = int(np.argmax(score))
        if not np.isfinite(score[next_idx]):
            next_idx = int(np.argmax(min_dist2))
        selected.append(next_idx)
        next_dist2 = squared_distances_np(feats, feats[[next_idx]])[:, 0]
        min_dist2 = np.minimum(min_dist2, next_dist2)

    centers = feats[selected].copy()
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    for cluster_id in range(k_eff):
        mask = labels == cluster_id
        if np.any(mask):
            centers[cluster_id] = feats[mask].mean(axis=0)
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    return labels, centers.astype(np.float32)


def kmeans_clusters(feats, k_eff, max_iter=25):
    if k_eff <= 1:
        return np.zeros(feats.shape[0], dtype=np.int64), feats.mean(axis=0, keepdims=True).astype(np.float32)
    _, centers = farthest_point_clusters(feats, k_eff)
    labels = np.zeros(feats.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        new_labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
        new_centers = centers.copy()
        for cluster_id in range(k_eff):
            mask = new_labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = feats[mask].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers, atol=1e-6):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    return labels, centers.astype(np.float32)


def random_feature_centers(feats, k_eff):
    if k_eff <= 1:
        centers = feats.mean(axis=0, keepdims=True)
    else:
        replace = feats.shape[0] < k_eff
        idx = np.random.choice(feats.shape[0], size=k_eff, replace=replace)
        centers = feats[idx].copy()
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    return labels, centers.astype(np.float32)


def farthest_point_indices(feats, budget):
    n_items = feats.shape[0]
    budget = max(0, min(int(budget), n_items))
    if budget == 0:
        return np.asarray([], dtype=np.int64)
    class_mean = feats.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(squared_distances_np(feats, class_mean)[:, 0]))
    selected = [first_idx]
    min_dist2 = squared_distances_np(feats, feats[[first_idx]])[:, 0]
    while len(selected) < budget:
        min_dist2[selected] = -np.inf
        next_idx = int(np.argmax(min_dist2))
        selected.append(next_idx)
        next_dist2 = squared_distances_np(feats, feats[[next_idx]])[:, 0]
        min_dist2 = np.minimum(min_dist2, next_dist2)
    return np.asarray(selected, dtype=np.int64)


def density_farthest_indices(feats, budget, density_k=10, eps=1e-12):
    n_items = feats.shape[0]
    budget = max(0, min(int(budget), n_items))
    if budget == 0:
        return np.asarray([], dtype=np.int64)
    if budget == 1 or n_items <= 2:
        return farthest_point_indices(feats, budget)

    pair_dist2 = squared_distances_np(feats, feats)
    np.fill_diagonal(pair_dist2, np.inf)
    nn_k = max(1, min(density_k, n_items - 1))
    nn_dist2 = np.partition(pair_dist2, nn_k - 1, axis=1)[:, :nn_k]
    finite = nn_dist2[np.isfinite(nn_dist2)]
    sigma2 = max(float(np.median(finite)) if finite.size else eps, eps)
    density = np.exp(-nn_dist2 / sigma2).mean(axis=1)
    density = density / max(float(np.max(density)), eps)
    density = np.maximum(density, 1e-3)

    class_mean = feats.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(squared_distances_np(feats, class_mean)[:, 0]))
    selected = [first_idx]
    min_dist2 = squared_distances_np(feats, feats[[first_idx]])[:, 0]
    while len(selected) < budget:
        score = np.sqrt(np.maximum(min_dist2, 0.0)) * density
        score[selected] = -np.inf
        next_idx = int(np.argmax(score))
        if not np.isfinite(score[next_idx]):
            next_idx = int(np.argmax(min_dist2))
        selected.append(next_idx)
        next_dist2 = squared_distances_np(feats, feats[[next_idx]])[:, 0]
        min_dist2 = np.minimum(min_dist2, next_dist2)
    return np.asarray(selected, dtype=np.int64)


def herding_indices(feats, budget):
    n_items = feats.shape[0]
    budget = max(0, min(int(budget), n_items))
    if budget == 0:
        return np.asarray([], dtype=np.int64)
    mu = feats.mean(axis=0)
    selected = []
    running = np.zeros_like(mu)
    available = np.ones(n_items, dtype=bool)
    for step in range(budget):
        target = (step + 1) * mu - running
        dist = np.sum((feats - target[None, :]) ** 2, axis=1)
        dist[~available] = np.inf
        idx = int(np.argmin(dist))
        selected.append(idx)
        running += feats[idx]
        available[idx] = False
    return np.asarray(selected, dtype=np.int64)


def adaptive_mix_indices(feats, budget):
    n_items = feats.shape[0]
    budget = max(0, min(int(budget), n_items))
    if budget == 0:
        return np.asarray([], dtype=np.int64)
    if budget <= 5:
        rep_ratio = 1.0
    elif budget >= 100:
        rep_ratio = 0.0
    else:
        rep_ratio = math.log(100.0 / float(budget)) / math.log(100.0 / 5.0)
        rep_ratio = min(1.0, max(0.0, rep_ratio))

    n_rep = min(budget, int(round(budget * rep_ratio)))
    selected = []
    if n_rep > 0:
        rep_idx = herding_indices(feats, n_rep)
        selected.extend(int(idx) for idx in rep_idx.tolist())
    remaining = [idx for idx in range(n_items) if idx not in set(selected)]
    n_div = budget - len(selected)
    if n_div > 0 and remaining:
        div_idx = np.random.choice(
            np.asarray(remaining, dtype=np.int64),
            size=min(n_div, len(remaining)),
            replace=False,
        )
        selected.extend(int(idx) for idx in div_idx.tolist())
    return np.asarray(selected[:budget], dtype=np.int64)


def select_raw_memory_indices(feats, budget, mode):
    if mode == "raw_random":
        budget = max(0, min(int(budget), feats.shape[0]))
        if budget == 0:
            return np.asarray([], dtype=np.int64)
        return np.random.choice(feats.shape[0], size=budget, replace=False).astype(np.int64)
    if mode == "raw_farthest":
        return farthest_point_indices(feats, budget)
    if mode == "raw_density_farthest":
        return density_farthest_indices(feats, budget)
    if mode == "raw_herding":
        return herding_indices(feats, budget)
    if mode == "raw_adaptive_mix":
        return adaptive_mix_indices(feats, budget)
    raise ValueError(f"Unsupported feature-guided old_filter_mode: {mode}")


def herding_centers(feats, k_eff):
    if k_eff <= 1:
        centers = feats.mean(axis=0, keepdims=True)
        return np.zeros(feats.shape[0], dtype=np.int64), centers.astype(np.float32)
    mu = feats.mean(axis=0)
    selected = []
    running = np.zeros_like(mu)
    available = np.ones(feats.shape[0], dtype=bool)
    for step in range(k_eff):
        target = (step + 1) * mu - running
        dist = np.sum((feats - target[None, :]) ** 2, axis=1)
        dist[~available] = np.inf
        idx = int(np.argmin(dist))
        selected.append(idx)
        running += feats[idx]
        available[idx] = False
        if not np.any(available):
            available[:] = True
    centers = feats[selected].copy()
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    return labels, centers.astype(np.float32)


def covariance_centers(feats, k_eff):
    mean = feats.mean(axis=0, keepdims=True)
    if k_eff <= 1 or feats.shape[0] < 2:
        return np.zeros(feats.shape[0], dtype=np.int64), mean.astype(np.float32)
    centered = feats - mean
    # SVD avoids explicitly forming a large covariance matrix when feature_dim is high.
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    dirs = vt[: max(1, min(vt.shape[0], (k_eff - 1 + 1) // 2))]
    scales = singular_values[: dirs.shape[0]] / max(1.0, np.sqrt(feats.shape[0] - 1))
    centers = [mean[0]]
    for direction, scale in zip(dirs, scales):
        if len(centers) < k_eff:
            centers.append(mean[0] + scale * direction)
        if len(centers) < k_eff:
            centers.append(mean[0] - scale * direction)
    centers = np.stack(centers[:k_eff], axis=0)
    labels = np.argmin(squared_distances_np(feats, centers), axis=1).astype(np.int64)
    return labels, centers.astype(np.float32)


def local_center_clusters(feats, k_eff, mode):
    if mode == "fps":
        return farthest_point_clusters(feats, k_eff)
    if mode == "density_farthest":
        return density_farthest_clusters(feats, k_eff)
    if mode == "kmeans":
        return kmeans_clusters(feats, k_eff)
    if mode == "random_features":
        return random_feature_centers(feats, k_eff)
    if mode == "herding":
        return herding_centers(feats, k_eff)
    if mode == "covariance":
        return covariance_centers(feats, k_eff)
    raise ValueError(f"Unsupported support_center_mode: {mode}")


def radius_value(distances, stat_name, eps):
    if distances.size == 0:
        return eps
    mean = float(np.mean(distances))
    std = float(np.std(distances))
    values = {
        "mean": mean,
        "mean_plus_std": mean + std,
        "mean_plus_2std": mean + 2.0 * std,
        "p80": float(np.quantile(distances, 0.80)),
        "p90": float(np.quantile(distances, 0.90)),
        "p95": float(np.quantile(distances, 0.95)),
        "p99": float(np.quantile(distances, 0.99)),
        "max": float(np.max(distances)),
    }
    if stat_name not in values:
        raise ValueError(f"Unsupported radius stat: {stat_name}")
    return max(values[stat_name], eps)


@torch.no_grad()
def project_cache_by_class(feature_cache, class_labels, encoder, projector, args, device, batch_size):
    projected = {}
    if encoder is not None:
        encoder.eval()
    projector.eval()
    for label in class_labels:
        feats = feature_cache.get(int(label), [])
        if len(feats) == 0:
            continue
        feats_t = torch.tensor(feats, dtype=torch.float32, device=device)
        outs = []
        for start in range(0, feats_t.size(0), batch_size):
            cache_batch = feats_t[start : start + batch_size]
            enc_feat = features_from_cache(encoder, cache_batch, args)
            outs.append(projector(enc_feat).detach().cpu())
        projected[int(label)] = torch.cat(outs, dim=0).numpy().astype(np.float32)
    return projected


def build_support_domain_from_projected(projected_cache, args):
    centers, radii, labels = [], [], []
    summary = {}
    for label in sorted(projected_cache.keys()):
        feats = projected_cache[label]
        if feats.shape[0] == 0:
            continue
        k_eff = max(1, min(args.support_k, feats.shape[0] // max(1, args.support_min_cluster_size)))
        if k_eff <= 1:
            cluster_labels = np.zeros(feats.shape[0], dtype=np.int64)
            cls_centers = feats.mean(axis=0, keepdims=True)
        else:
            cluster_labels, cls_centers = local_center_clusters(
                feats,
                k_eff,
                args.support_center_mode,
            )

        cls_summary = []
        for cluster_id in range(cls_centers.shape[0]):
            mask = cluster_labels == cluster_id
            cluster_feats = feats[mask]
            center = cls_centers[cluster_id]
            dist = np.sqrt(np.maximum(((cluster_feats - center) ** 2).sum(axis=1), 0.0))
            radius = radius_value(dist, args.support_radius_stat, args.support_eps)
            centers.append(center.astype(np.float32))
            radii.append(radius)
            labels.append(int(label))
            cls_summary.append(
                {
                    "cluster_id": int(cluster_id),
                    "num_features": int(cluster_feats.shape[0]),
                    "radius": float(radius),
                }
            )
        summary[int(label)] = {
            "num_features": int(feats.shape[0]),
            "num_local_domains": int(len(cls_summary)),
            "clusters": cls_summary,
        }

    if not centers:
        return None

    center_labels = np.asarray(labels, dtype=np.int64)
    return {
        "centers": torch.tensor(np.stack(centers, axis=0), dtype=torch.float32),
        "radii": torch.tensor(np.asarray(radii, dtype=np.float32), dtype=torch.float32),
        "labels": center_labels,
        "summary": summary,
    }


def build_support_domain(feature_cache, class_labels, encoder, projector, args, device):
    projected = project_cache_by_class(
        feature_cache,
        class_labels,
        encoder,
        projector,
        args,
        device,
        args.batch_size,
    )
    return build_support_domain_from_projected(projected, args)


@torch.no_grad()
def build_pseudo_old_pool(
    args,
    stage,
    teacher_encoder,
    teacher_projector,
    teacher_classifier,
    old_domain,
    old_label_order,
    device,
):
    if not args.pseudo_old_source_template or args.pseudo_old_ratio <= 0:
        return {}, {}
    if old_domain is None:
        return {}, {}

    folder = resolve_stage_folder(args, args.pseudo_old_source_template, stage)
    dataset = NpyDeviceDataset(
        folder,
        allowed_labels=stage_seen_labels(args, stage),
        max_per_class=args.max_test_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    teacher_encoder.eval()
    teacher_projector.eval()
    teacher_classifier.eval()

    pool_chunks = defaultdict(list)
    guided_raw_chunks = defaultdict(list)
    guided_raw_feats = defaultdict(list)
    raw_true_counts = defaultdict(lambda: defaultdict(int))
    accepted_counts = defaultdict(int)
    candidate_counts = defaultdict(int)
    index_to_old_label = {idx: int(label) for idx, label in enumerate(old_label_order)}
    old_label_to_index = {int(label): idx for idx, label in enumerate(old_label_order)}
    old_label_set = set(int(label) for label in old_label_order)
    guided_raw_modes = {
        "raw_random",
        "raw_farthest",
        "raw_density_farthest",
        "raw_herding",
        "raw_adaptive_mix",
    }

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        h = teacher_projector(teacher_encoder(x))
        logits = teacher_classifier(h)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
        support_dist = support_distance_matrix(h, old_domain, old_label_order, device)
        own_dist = support_dist.gather(1, pred_idx.view(-1, 1)).squeeze(1)

        pred_raw_batch = torch.tensor(
            [index_to_old_label[int(idx)] for idx in pred_idx.detach().cpu().tolist()],
            dtype=y_raw.dtype,
            device=conf.device,
        )
        y_raw_device = y_raw.to(conf.device)

        if args.old_filter_mode in guided_raw_modes:
            keep = torch.tensor(
                [int(label) in old_label_set for label in y_raw.detach().cpu().tolist()],
                dtype=torch.bool,
                device=conf.device,
            )
            for idx in torch.nonzero(keep, as_tuple=False).view(-1).detach().cpu().tolist():
                raw_label = int(y_raw[idx].item())
                guided_raw_chunks[raw_label].append(x[idx : idx + 1].detach().cpu())
                guided_raw_feats[raw_label].append(h[idx : idx + 1].detach().cpu())
                candidate_counts[raw_label] += 1
            continue
        elif args.old_filter_mode == "random":
            keep = torch.tensor(
                [int(label) in old_label_set for label in y_raw.detach().cpu().tolist()],
                dtype=torch.bool,
                device=conf.device,
            )
            selected_raw_batch = y_raw_device
        elif args.old_filter_mode == "label_conf_support":
            label_known = torch.tensor(
                [int(label) in old_label_set for label in y_raw.detach().cpu().tolist()],
                dtype=torch.bool,
                device=conf.device,
            )
            true_cols = torch.tensor(
                [old_label_to_index.get(int(label), 0) for label in y_raw.detach().cpu().tolist()],
                dtype=torch.long,
                device=conf.device,
            )
            true_dist = support_dist.gather(1, true_cols.view(-1, 1)).squeeze(1)
            keep = (
                label_known
                & (pred_raw_batch == y_raw_device)
                & (conf >= args.pseudo_old_conf)
                & (true_dist <= args.pseudo_old_support_tau)
            )
            selected_raw_batch = y_raw_device
        else:
            if args.old_filter_mode == "confidence":
                keep = conf >= args.pseudo_old_conf
            elif args.old_filter_mode == "support":
                keep = own_dist <= args.pseudo_old_support_tau
            elif args.old_filter_mode == "conf_support":
                keep = (conf >= args.pseudo_old_conf) & (own_dist <= args.pseudo_old_support_tau)
            else:
                raise ValueError(f"Unsupported old_filter_mode: {args.old_filter_mode}")
            selected_raw_batch = pred_raw_batch

        if args.pseudo_old_require_label_match and args.old_filter_mode not in ("random", "label_conf_support"):
            y_raw_device = y_raw.to(keep.device)
            keep = keep & (pred_raw_batch == y_raw_device)
        for idx in torch.nonzero(keep, as_tuple=False).view(-1).detach().cpu().tolist():
            pred_raw = int(selected_raw_batch[idx].item())
            if pred_raw not in old_label_set:
                continue
            if accepted_counts[pred_raw] >= args.pseudo_old_max_per_class:
                continue
            pool_chunks[pred_raw].append(x[idx : idx + 1].detach().cpu())
            accepted_counts[pred_raw] += 1
            raw_true_counts[pred_raw][int(y_raw[idx].item())] += 1
        if args.old_filter_mode in ("random", "label_conf_support"):
            for label in y_raw.detach().cpu().tolist():
                if int(label) in old_label_set:
                    candidate_counts[int(label)] += 1
        else:
            for idx in pred_idx.detach().cpu().tolist():
                candidate_counts[index_to_old_label[int(idx)]] += 1

    pseudo_pool = {}
    if args.old_filter_mode in guided_raw_modes:
        for label in old_label_order:
            raw_label = int(label)
            chunks = guided_raw_chunks.get(raw_label, [])
            feats = guided_raw_feats.get(raw_label, [])
            if not chunks or not feats:
                continue
            data = torch.cat(chunks, dim=0).contiguous()
            feat_tensor = torch.cat(feats, dim=0).contiguous()
            feats_np = feat_tensor.numpy().astype(np.float32).reshape(feat_tensor.size(0), -1)
            selected_idx = select_raw_memory_indices(
                feats_np,
                args.pseudo_old_max_per_class,
                args.old_filter_mode,
            )
            if selected_idx.size == 0:
                continue
            selected_data = data[torch.tensor(selected_idx, dtype=torch.long)]
            pseudo_pool[raw_label] = selected_data.contiguous()
            accepted_counts[raw_label] = int(selected_data.size(0))
            raw_true_counts[raw_label][raw_label] = int(selected_data.size(0))

    summary = {}
    for label in old_label_order:
        chunks = pool_chunks.get(int(label), [])
        if chunks:
            pseudo_pool[int(label)] = torch.cat(chunks, dim=0).contiguous()
        true_counts = dict(sorted(raw_true_counts[int(label)].items()))
        correct = true_counts.get(int(label), 0)
        total = int(sum(true_counts.values()))
        summary[str(label)] = {
            "accepted": total,
            "candidate_predicted": int(candidate_counts[int(label)]),
            "true_label_counts": {str(k): int(v) for k, v in true_counts.items()},
            "known_label_purity": 0.0 if total == 0 else float(100.0 * correct / total),
        }
    summary["_filter_mode"] = args.old_filter_mode
    return pseudo_pool, summary


def build_encoder_support_replay_cache(feature_cache, class_labels, args):
    support_cache = {}
    summary = {}
    for label in class_labels:
        feats = feature_cache.get(int(label), [])
        if len(feats) == 0:
            continue
        feats_np = np.asarray(feats, dtype=np.float32)
        original_shape = feats_np.shape[1:]
        feats_flat = feats_np.reshape(feats_np.shape[0], -1)
        k_eff = max(
            1,
            min(
                args.support_k,
                feats_flat.shape[0] // max(1, args.support_min_cluster_size),
            ),
        )
        if k_eff <= 1:
            centers_flat = feats_flat.mean(axis=0, keepdims=True)
        else:
            _, centers_flat = local_center_clusters(
                feats_flat,
                k_eff,
                args.support_center_mode,
            )
        centers = centers_flat.reshape((centers_flat.shape[0],) + original_shape)
        support_cache[int(label)] = centers.astype(np.float32).tolist()
        summary[int(label)] = {
            "num_cache_features": int(feats_np.shape[0]),
            "num_support_replay_features": int(centers.shape[0]),
            "support_center_mode": args.support_center_mode,
        }
    return support_cache, summary


def pairwise_l2(x, y):
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1).view(1, -1)
    dist2 = torch.clamp(x2 + y2 - 2.0 * x @ y.t(), min=0.0)
    return torch.sqrt(dist2 + 1e-12)


def support_distance_matrix(features, domain, class_labels, device):
    centers = domain["centers"].to(device)
    radii = domain["radii"].to(device).view(1, -1)
    domain_labels = domain["labels"]
    dist = pairwise_l2(features, centers) / radii
    columns = []
    for label in class_labels:
        idx = np.where(domain_labels == int(label))[0]
        if idx.size == 0:
            columns.append(torch.full((features.size(0),), 1e6, device=device))
        else:
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            columns.append(dist.index_select(1, idx_t).min(dim=1).values)
    return torch.stack(columns, dim=1)


def support_distance_for_labels(features, labels_raw, domain, device):
    class_labels = sorted(set(int(x) for x in labels_raw.detach().cpu().tolist()))
    matrix = support_distance_matrix(features, domain, class_labels, device)
    label_to_col = {label: idx for idx, label in enumerate(class_labels)}
    cols = torch.tensor(
        [label_to_col[int(label)] for label in labels_raw.detach().cpu().tolist()],
        dtype=torch.long,
        device=device,
    )
    return matrix.gather(1, cols.view(-1, 1)).squeeze(1)


@torch.no_grad()
def estimate_similarity_replay_weights(
    trainset,
    teacher_encoder,
    teacher_projector,
    old_domain,
    old_label_order,
    args,
    device,
):
    weights = {int(label): 1.0 for label in old_label_order}
    if (
        args.old_replay_sampling != "similarity"
        or old_domain is None
        or not old_label_order
        or args.old_replay_similarity_gamma <= 0
    ):
        return weights

    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    teacher_encoder.eval()
    teacher_projector.eval()
    sims = []
    tau = max(float(args.old_replay_similarity_tau), 1e-6)
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        h = teacher_projector(teacher_encoder(x))
        dist = support_distance_matrix(h, old_domain, old_label_order, device)
        sims.append(torch.exp(-dist / tau).detach().cpu())
    if not sims:
        return weights

    sim = torch.cat(sims, dim=0)
    class_risk = sim.max(dim=0).values
    class_risk = class_risk / class_risk.mean().clamp_min(1e-6)
    gamma = float(args.old_replay_similarity_gamma)
    for idx, label in enumerate(old_label_order):
        weights[int(label)] = float(1.0 + gamma * class_risk[idx].item())
    return weights


@torch.no_grad()
def build_anchor_label_map(
    trainset,
    teacher_encoder,
    teacher_projector,
    old_domain,
    old_label_order,
    args,
    device,
):
    if args.anchor_replay_mode == "none" or old_domain is None or not old_label_order:
        return {}, {}

    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    teacher_encoder.eval()
    teacher_projector.eval()
    dist_sums = {}
    counts = defaultdict(int)

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        h = teacher_projector(teacher_encoder(x))
        dist = support_distance_matrix(h, old_domain, old_label_order, device).detach().cpu()
        y_cpu = [int(label) for label in y_raw.detach().cpu().tolist()]
        for new_label in sorted(set(y_cpu)):
            idx = [i for i, label in enumerate(y_cpu) if label == new_label]
            if not idx:
                continue
            if new_label not in dist_sums:
                dist_sums[new_label] = torch.zeros(dist.size(1), dtype=torch.float64)
            dist_sums[new_label] += dist[idx].double().sum(dim=0)
            counts[new_label] += len(idx)

    k = max(1, min(int(args.anchor_replay_topk), len(old_label_order)))
    anchor_label_map = {}
    summary = {
        "mode": args.anchor_replay_mode,
        "topk": k,
        "global_fraction": float(args.anchor_replay_global_fraction),
        "extra_ratio": float(args.anchor_replay_extra_ratio),
        "new_classes": {},
    }
    for new_label in sorted(dist_sums.keys()):
        avg_dist = dist_sums[new_label] / max(1, counts[new_label])
        top_idx = torch.argsort(avg_dist)[:k].tolist()
        anchors = [int(old_label_order[int(idx)]) for idx in top_idx]
        anchor_label_map[int(new_label)] = anchors
        summary["new_classes"][str(int(new_label))] = {
            "num_train_samples": int(counts[new_label]),
            "anchors": anchors,
            "avg_normalized_distance": [
                float(avg_dist[int(idx)].item()) for idx in top_idx
            ],
        }

    return anchor_label_map, summary


def support_losses(feat_new_support, feat_old_raw_support, y_old_raw, old_domain, args, device):
    """Support constraints on current student features from raw samples.

    `feat_old_raw_support` is produced by re-encoding old raw replay with the
    current student encoder and mapping it into a support-domain coordinate space,
    so keep loss can constrain encoder drift. Fixed feature-cache replay is
    handled outside this function and should not be used as an encoder loss.
    """
    zero = feat_new_support.new_tensor(0.0)
    if old_domain is None:
        return zero, zero

    keep_loss = zero
    if feat_old_raw_support.numel() > 0 and args.support_keep_weight > 0:
        own_distance = support_distance_for_labels(feat_old_raw_support, y_old_raw, old_domain, device)
        keep_loss = F.relu(own_distance - args.support_tau).pow(2).mean()

    sep_loss = zero
    if feat_new_support.numel() > 0 and args.support_sep_weight > 0:
        centers = old_domain["centers"].to(device)
        radii = old_domain["radii"].to(device).view(1, -1)
        norm_dist = pairwise_l2(feat_new_support, centers) / radii
        min_old_distance = norm_dist.min(dim=1).values
        sep_loss = F.relu(args.support_tau + args.support_margin - min_old_distance).pow(2).mean()

    return keep_loss, sep_loss


def labels_to_indices(labels_raw, label_to_index, device):
    return torch.tensor(
        [label_to_index[int(label)] for label in labels_raw.detach().cpu().tolist()],
        dtype=torch.long,
        device=device,
    )


def clone_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


@torch.no_grad()
def build_candidate_stage_cache(
    feature_cache,
    trainset,
    encoder,
    projector,
    classifier,
    label_order,
    args,
    device,
):
    candidate_cache = copy.deepcopy(feature_cache)
    index_to_label = {idx: int(label) for idx, label in enumerate(label_order)}
    encoder.eval()
    projector.eval()
    classifier.eval()

    for x, y_raw in DataLoader(trainset, batch_size=args.batch_size, shuffle=False):
        x = x.to(device, non_blocking=True)
        y_raw = y_raw.to(device)
        cache_value = cache_values_from_raw(encoder, x, args)
        feat_enc = features_from_cache(encoder, cache_value, args)
        feat_proj = projector(feat_enc)
        logits = classifier(feat_proj)
        pred_idx = logits.argmax(dim=1).detach().cpu().tolist()
        pred_raw = torch.tensor(
            [index_to_label[int(idx)] for idx in pred_idx],
            dtype=torch.long,
            device=device,
        )
        correct = pred_raw == y_raw
        for cls in y_raw.unique():
            mask = (y_raw == cls) & correct
            if mask.any():
                key = int(cls.item())
                candidate_cache.setdefault(key, []).extend(cache_value[mask].detach().cpu().tolist())
                max_keep = int(args.max_feature_per_class)
                if len(candidate_cache[key]) > max_keep:
                    candidate_cache[key] = candidate_cache[key][-max_keep:]

    return candidate_cache


@torch.no_grad()
def collect_eval_predictions(
    encoder,
    projector,
    classifier,
    loader,
    label_order,
    support_domain,
    support_alphas,
    device,
):
    encoder.eval()
    projector.eval()
    classifier.eval()
    index_to_label = {idx: int(label) for idx, label in enumerate(label_order)}

    variants = ["standard"]
    variants.extend(f"support_alpha_{alpha:g}" for alpha in support_alphas if alpha > 0)
    variants.append("support_nearest")
    preds = {name: [] for name in variants}
    labels = []

    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        h = projector(encoder(x))
        logits = classifier(h)
        labels.extend([int(y) for y in y_raw.tolist()])

        standard_idx = logits.argmax(dim=1).detach().cpu().tolist()
        preds["standard"].extend([index_to_label[int(idx)] for idx in standard_idx])

        if support_domain is None:
            for name in variants:
                if name != "standard":
                    preds[name].extend([index_to_label[int(idx)] for idx in standard_idx])
            continue

        support_dist = support_distance_matrix(h, support_domain, label_order, device)
        nearest_idx = support_dist.argmin(dim=1).detach().cpu().tolist()
        preds["support_nearest"].extend([index_to_label[int(idx)] for idx in nearest_idx])

        support_dist = torch.clamp(support_dist, max=20.0)
        for alpha in support_alphas:
            if alpha <= 0:
                continue
            adjusted = logits - float(alpha) * support_dist
            pred_idx = adjusted.argmax(dim=1).detach().cpu().tolist()
            preds[f"support_alpha_{alpha:g}"].extend([index_to_label[int(idx)] for idx in pred_idx])

    labels = np.asarray(labels, dtype=np.int64)
    return {name: np.asarray(value, dtype=np.int64) for name, value in preds.items()}, labels


def accuracy_for_labels(y_true, y_pred, label_set):
    label_set = set(int(x) for x in label_set)
    mask = np.asarray([int(y) in label_set for y in y_true], dtype=bool)
    if mask.sum() == 0:
        return None
    return float(100.0 * (y_pred[mask] == y_true[mask]).sum() / mask.sum())


def metrics_from_predictions(y_true, y_pred, args, stage):
    label_order = stage_seen_labels(args, stage)
    old_labels = stage_seen_labels(args, stage - 1) if stage > 0 else []
    new_labels = stage_new_labels(args, stage) if stage > 0 else label_order
    groups = stage_group_labels(args, stage)

    total_acc = float(100.0 * (y_pred == y_true).sum() / max(1, y_true.shape[0]))
    old_acc = accuracy_for_labels(y_true, y_pred, old_labels) if stage > 0 else None
    new_acc = accuracy_for_labels(y_true, y_pred, new_labels)
    old_acc_value = old_acc if old_acc is not None else 0.0
    new_acc_value = new_acc if new_acc is not None else 0.0
    if stage > 0 and old_acc is not None and new_acc is not None:
        balanced_acc = 0.5 * (old_acc_value + new_acc_value)
        harmonic_acc = (
            2.0 * old_acc_value * new_acc_value / (old_acc_value + new_acc_value)
            if old_acc_value + new_acc_value > 0
            else 0.0
        )
    else:
        balanced_acc = total_acc
        harmonic_acc = total_acc
    per_group = {}
    for group_id, labels in groups.items():
        acc = accuracy_for_labels(y_true, y_pred, labels)
        if acc is not None:
            per_group[str(group_id)] = acc

    per_class = {}
    for label in sorted(set(y_true.tolist())):
        acc = accuracy_for_labels(y_true, y_pred, [label])
        if acc is not None:
            per_class[str(label)] = acc

    return {
        "total_acc": total_acc,
        "old_acc": old_acc_value,
        "new_acc": new_acc_value,
        "balanced_acc": float(balanced_acc),
        "harmonic_acc": float(harmonic_acc),
        "per_group_acc_now": per_group,
        "acc_when_learned": per_group.get(str(stage), None),
        "per_class_acc_now": per_class,
    }


def update_forgetting(history, stage):
    current = history[str(stage)]
    forgetting = []
    for group_id in range(0, stage):
        learned = history.get(str(group_id), {}).get("acc_when_learned", None)
        now = current.get("per_group_acc_now", {}).get(str(group_id), None)
        if learned is None or now is None:
            continue
        forgetting.append(max(0.0, float(learned) - float(now)))
    forget_rate = float(np.mean(forgetting)) if forgetting else 0.0
    previous = [
        value.get("forget_rate", 0.0)
        for key, value in history.items()
        if int(key) < stage
    ]
    cumulative = float(np.mean(previous + [forget_rate])) if previous else forget_rate
    current["forget_rate"] = forget_rate
    current["cumulative_forget"] = cumulative


def evaluate_folder_variants(
    args,
    stage,
    encoder,
    projector,
    classifier,
    feature_cache,
    label_order,
    folder,
    device,
):
    support_domain = build_support_domain(feature_cache, label_order, encoder, projector, args, device)
    dataset = NpyDeviceDataset(
        folder,
        allowed_labels=label_order,
        max_per_class=args.max_test_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    preds_by_variant, y_true = collect_eval_predictions(
        encoder,
        projector,
        classifier,
        loader,
        label_order,
        support_domain,
        args.eval_support_alphas,
        device,
    )

    variant_metrics = {}
    for name, pred in preds_by_variant.items():
        variant_metrics[name] = metrics_from_predictions(y_true, pred, args, stage)
    return variant_metrics


def evaluate_stage(
    args,
    stage,
    encoder,
    projector,
    classifier,
    feature_cache,
    label_order,
    history,
    device,
):
    if stage == 0:
        folder = str(Path(args.data_root) / args.test_closed)
    else:
        folder = resolve_stage_folder(args, args.test_add_template, stage)

    variant_metrics = evaluate_folder_variants(
        args,
        stage,
        encoder,
        projector,
        classifier,
        feature_cache,
        label_order,
        folder,
        device,
    )

    primary = args.primary_eval
    if primary not in variant_metrics:
        raise ValueError(
            f"primary_eval={primary} not found. Available: {sorted(variant_metrics.keys())}"
        )

    history[str(stage)] = dict(variant_metrics[primary])
    history[str(stage)]["primary_eval"] = primary
    history[str(stage)]["eval_variants"] = {
        name: {
            key: value
            for key, value in metrics.items()
            if key in ["total_acc", "old_acc", "new_acc", "acc_when_learned"]
        }
        for name, metrics in variant_metrics.items()
    }
    update_forgetting(history, stage)

    metrics = history[str(stage)]
    print(
        f"[Eval Step {stage} | {primary}] "
        f"Total={metrics['total_acc']:.2f} Old={metrics['old_acc']:.2f} "
        f"New={metrics['new_acc']:.2f} Forget={metrics['forget_rate']:.2f}"
    )
    return history


def train_one_stage(args, stage, device):
    encoder, old_classifier, prev_projector = load_previous_modules(args, stage, device)
    train_encoder = args.encoder_update_mode != "frozen"
    teacher_encoder = copy.deepcopy(encoder).to(device).eval() if train_encoder else encoder
    for param in teacher_encoder.parameters():
        param.requires_grad = False
    configure_encoder_update(encoder, args.encoder_update_mode)
    for param in prev_projector.parameters():
        param.requires_grad = False

    old_label_order = stage_seen_labels(args, stage - 1)
    label_order = stage_seen_labels(args, stage)
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    old_num_outputs = len(old_label_order)

    feature_cache = load_stage_cache(args, stage - 1)
    old_domain = build_support_domain(feature_cache, old_label_order, teacher_encoder, prev_projector, args, device)
    pseudo_old_pool, pseudo_old_summary = build_pseudo_old_pool(
        args,
        stage,
        teacher_encoder,
        prev_projector,
        old_classifier,
        old_domain,
        old_label_order,
        device,
    )
    support_replay_cache, support_replay_summary = ({}, {})
    if args.support_replay_per_class > 0:
        support_replay_cache, support_replay_summary = build_encoder_support_replay_cache(
            feature_cache,
            old_label_order,
            args,
        )

    classifier = make_expanded_classifier(
        old_classifier,
        len(label_order),
        args.embedding_dim,
        device,
    )
    projector = make_current_projector(prev_projector, args, stage, device)
    classifier.train()
    projector.train()
    configure_encoder_update(encoder, args.encoder_update_mode)

    new_labels = stage_new_labels(args, stage)
    train_folder = resolve_stage_folder(args, args.train_add_template, stage)
    trainset = NpyDeviceDataset(
        train_folder,
        allowed_labels=new_labels,
        max_per_class=args.max_train_per_class,
    )
    train_loader = DataLoader(trainset, batch_size=args.incr_batch_size, shuffle=True)
    old_replay_weights = estimate_similarity_replay_weights(
        trainset,
        teacher_encoder,
        prev_projector,
        old_domain,
        old_label_order,
        args,
        device,
    )
    anchor_label_map, anchor_summary = build_anchor_label_map(
        trainset,
        teacher_encoder,
        prev_projector,
        old_domain,
        old_label_order,
        args,
        device,
    )
    if anchor_summary:
        with open(Path(args.save_dir) / f"anchor_replay_step{stage}.json", "w", encoding="utf-8") as f:
            json.dump(anchor_summary, f, indent=2)

    param_groups = [
        {
            "params": list(projector.parameters()) + list(classifier.parameters()),
            "lr": args.lr,
            "initial_lr": args.lr,
            "name": "head",
        }
    ]
    if train_encoder:
        param_groups.append(
            {
                "params": trainable_encoder_parameters(encoder),
                "lr": args.encoder_lr,
                "initial_lr": args.encoder_lr,
                "name": "encoder",
            }
        )
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    con_loss_fn = SupConLoss(temperature=args.temperature)

    print(
        f"[Train Step {stage}] old={len(old_label_order)} total={len(label_order)} "
        f"new={new_labels} train_samples={len(trainset)} "
        f"encoder_update={args.encoder_update_mode}"
    )
    if pseudo_old_pool:
        print(
            f"[Old Raw Replay Step {stage}] accepted="
            f"{sum(data.size(0) for data in pseudo_old_pool.values())} "
            f"classes={sum(1 for data in pseudo_old_pool.values() if data.size(0) > 0)}"
        )
        if args.old_replay_sampling == "similarity":
            top_weights = sorted(old_replay_weights.items(), key=lambda item: item[1], reverse=True)[:5]
            print(
                "[Old Raw Replay Sampling] mode=similarity "
                f"gamma={args.old_replay_similarity_gamma} "
                f"tau={args.old_replay_similarity_tau} "
                f"top={[(int(k), round(float(v), 4)) for k, v in top_weights]}"
            )
    if anchor_summary:
        compact_map = {
            int(new_label): values["anchors"]
            for new_label, values in anchor_summary["new_classes"].items()
        }
        print(
            f"[Anchor Replay Step {stage}] mode={args.anchor_replay_mode} "
            f"topk={args.anchor_replay_topk} "
            f"global_fraction={args.anchor_replay_global_fraction} "
            f"anchors={compact_map}"
        )

    best_selection = None
    val_folder = None
    if args.select_increment_by_val:
        if not args.val_add_template or not args.val_closed:
            raise ValueError("--select_increment_by_val requires --val_closed and --val_add_template")
        val_folder = cumulative_stage_folders(args, args.val_closed, args.val_add_template, stage)

    early_monitor = args.early_stop_monitor
    if early_monitor == "val_metric" and not args.select_increment_by_val:
        early_monitor = "train_loss"
    early_best = None
    early_bad_epochs = 0

    for epoch in range(args.epochs):
        if (
            train_encoder
            and args.encoder_lr_after_warmup > 0
            and args.encoder_lr_warmup_epochs > 0
        ):
            encoder_base_lr = (
                args.encoder_lr
                if epoch < args.encoder_lr_warmup_epochs
                else args.encoder_lr_after_warmup
            )
            for param_group in optimizer.param_groups:
                if param_group.get("name") == "encoder":
                    param_group["initial_lr"] = encoder_base_lr
                    param_group["lr"] = encoder_base_lr
        classifier.train()
        projector.train()
        configure_encoder_update(encoder, args.encoder_update_mode)
        total_loss, total_acc = 0.0, 0.0
        total_keep, total_sep = 0.0, 0.0
        for x_new, y_new_raw in train_loader:
            x_new = x_new.to(device, non_blocking=True)
            y_new_raw = y_new_raw.to(device)

            if pseudo_old_pool and args.pseudo_old_ratio > 0:
                n_pseudo = int(x_new.size(0) * args.pseudo_old_ratio)
                if args.old_replay_sampling == "similarity":
                    x_pseudo_old, y_pseudo_old_raw = sample_pseudo_old_pool_weighted(
                        pseudo_old_pool,
                        old_label_order,
                        n_pseudo,
                        old_replay_weights,
                        device,
                    )
                else:
                    per_pseudo_class = max(1, math.ceil(n_pseudo / max(1, len(old_label_order))))
                    x_pseudo_old, y_pseudo_old_raw = sample_pseudo_old_pool(
                        pseudo_old_pool,
                        old_label_order,
                        per_pseudo_class,
                        device,
                    )
            else:
                x_pseudo_old = torch.empty(0, device=device)
                y_pseudo_old_raw = torch.empty(0, dtype=torch.long, device=device)

            if x_pseudo_old.numel() > 0:
                x_raw = torch.cat([x_new, x_pseudo_old], dim=0)
                y_raw_labeled = torch.cat([y_new_raw, y_pseudo_old_raw], dim=0)
            else:
                x_raw = x_new
                y_raw_labeled = y_new_raw

            if train_encoder:
                feat_raw_enc = encoder(x_raw)
            else:
                with torch.no_grad():
                    feat_raw_enc = encoder(x_raw)
            feat_new_enc = feat_raw_enc[: x_new.size(0)]
            feat_pseudo_old_enc = feat_raw_enc[x_new.size(0) :]

            cache_parts, y_cache_parts = [], []
            if old_num_outputs > 0:
                if not args.disable_feature_cache_replay:
                    n_old = int(x_new.size(0) * args.ratio)
                    per_class = max(1, math.ceil(n_old / max(1, len(old_label_order))))
                    if args.anchor_replay_mode in {"none", "all_plus_anchor"} or not anchor_label_map:
                        cache_old_part, y_cache_raw_part = sample_cache_balanced(
                            feature_cache,
                            old_label_order,
                            per_class,
                            device,
                        )
                        if cache_old_part.numel() > 0:
                            cache_parts.append(cache_old_part)
                            y_cache_parts.append(y_cache_raw_part)
                        if args.anchor_replay_mode == "all_plus_anchor" and anchor_label_map:
                            target_old = per_class * max(1, len(old_label_order))
                            extra_old = int(round(target_old * max(0.0, args.anchor_replay_extra_ratio)))
                            if extra_old > 0:
                                anchor_labels = anchor_labels_for_batch(
                                    y_new_raw,
                                    anchor_label_map,
                                    old_label_order,
                                )
                                cache_anchor_part, y_anchor_raw_part = sample_anchor_cache(
                                    feature_cache,
                                    old_label_order,
                                    anchor_labels,
                                    extra_old,
                                    0.0,
                                    device,
                                )
                                if cache_anchor_part.numel() > 0:
                                    cache_parts.append(cache_anchor_part)
                                    y_cache_parts.append(y_anchor_raw_part)
                    else:
                        target_old = per_class * max(1, len(old_label_order))
                        anchor_labels = anchor_labels_for_batch(
                            y_new_raw,
                            anchor_label_map,
                            old_label_order,
                        )
                        global_fraction = (
                            args.anchor_replay_global_fraction
                            if args.anchor_replay_mode == "topk_plus_global"
                            else 0.0
                        )
                        cache_old_part, y_cache_raw_part = sample_anchor_cache(
                            feature_cache,
                            old_label_order,
                            anchor_labels,
                            target_old,
                            global_fraction,
                            device,
                        )
                        if cache_old_part.numel() > 0:
                            cache_parts.append(cache_old_part)
                            y_cache_parts.append(y_cache_raw_part)
                if args.support_replay_per_class > 0:
                    cache_support, y_support_raw = sample_cache_balanced(
                        support_replay_cache,
                        old_label_order,
                        args.support_replay_per_class,
                        device,
                    )
                    if cache_support.numel() > 0:
                        cache_parts.append(cache_support)
                        y_cache_parts.append(y_support_raw)

            if cache_parts:
                cache_old = torch.cat(cache_parts, dim=0)
                y_cache_raw = torch.cat(y_cache_parts, dim=0)
                feat_cache_enc = features_from_cache(encoder, cache_old, args)
            else:
                cache_old = torch.empty(0, device=device)
                feat_cache_enc = torch.empty(0, args.embedding_dim, device=device)
                y_cache_raw = torch.empty(0, dtype=torch.long, device=device)

            feat_all_enc = torch.cat([feat_raw_enc, feat_cache_enc], dim=0)
            feat_all_proj = projector(feat_all_enc)
            feat_new_proj = feat_all_proj[: x_new.size(0)]
            feat_pseudo_old_proj = feat_all_proj[x_new.size(0) : x_raw.size(0)]
            feat_cache_proj = feat_all_proj[x_raw.size(0) :]

            with torch.no_grad():
                prev_cache_proj = (
                    prev_projector(features_from_cache(teacher_encoder, cache_old, args))
                    if old_num_outputs > 0 and feat_cache_enc.numel() > 0
                    else feat_cache_enc
                )
                prev_pseudo_old_proj = (
                    prev_projector(teacher_encoder(x_pseudo_old))
                    if x_pseudo_old.numel() > 0
                    else feat_pseudo_old_proj
                )

            y_all_raw = torch.cat([y_raw_labeled, y_cache_raw], dim=0)
            labels_all = labels_to_indices(y_all_raw, label_to_index, device)
            logits_all = classifier(feat_all_proj)

            is_new = labels_all >= old_num_outputs
            ce_weights = torch.where(
                is_new,
                torch.tensor(args.new_ce_weight, device=device),
                torch.tensor(args.old_ce_weight, device=device),
            )
            ce_loss = (F.cross_entropy(logits_all, labels_all, reduction="none") * ce_weights).mean()
            con_loss = con_loss_fn(feat_all_proj, labels_all)
            distill_loss = (
                F.mse_loss(feat_cache_proj, prev_cache_proj.detach())
                if feat_cache_enc.numel() > 0 and args.distill_weight > 0
                else feat_all_proj.new_tensor(0.0)
            )
            if feat_pseudo_old_proj.numel() > 0 and args.distill_weight > 0:
                distill_loss = distill_loss + F.mse_loss(
                    feat_pseudo_old_proj,
                    prev_pseudo_old_proj.detach(),
                )

            if args.support_loss_space == "teacher_projector":
                feat_new_support = prev_projector(feat_new_enc)
                feat_pseudo_old_support = (
                    prev_projector(feat_pseudo_old_enc)
                    if feat_pseudo_old_enc.numel() > 0
                    else feat_pseudo_old_proj
                )
            else:
                feat_new_support = feat_new_proj
                feat_pseudo_old_support = feat_pseudo_old_proj
            keep_loss, sep_loss = support_losses(
                feat_new_support,
                feat_pseudo_old_support,
                y_pseudo_old_raw,
                old_domain,
                args,
                device,
            )

            loss = (
                args.ce_weight * ce_loss
                + args.contrastive_weight * con_loss
                + args.distill_weight * distill_loss
                + args.support_keep_weight * keep_loss
                + args.support_sep_weight * sep_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits_all.argmax(dim=1)
            acc = (preds == labels_all).float().mean().item() * 100.0
            total_loss += float(loss.item())
            total_acc += acc
            total_keep += float(keep_loss.item())
            total_sep += float(sep_loss.item())

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = total_acc / len(train_loader)
        epoch_keep = total_keep / len(train_loader)
        epoch_sep = total_sep / len(train_loader)
        val_score = None
        print(
            f"[Step {stage} | Epoch {epoch + 1}] "
            f"Loss={epoch_loss:.4f} "
            f"Acc={epoch_acc:.2f} "
            f"Keep={epoch_keep:.4f} "
            f"Sep={epoch_sep:.4f}"
        )
        if args.select_increment_by_val:
            val_cache = build_candidate_stage_cache(
                feature_cache,
                trainset,
                encoder,
                projector,
                classifier,
                label_order,
                args,
                device,
            )
            val_variants = evaluate_folder_variants(
                args,
                stage,
                encoder,
                projector,
                classifier,
                val_cache,
                label_order,
                val_folder,
                device,
            )
            primary = args.primary_eval
            if primary not in val_variants:
                raise ValueError(
                    f"primary_eval={primary} not found. Available: {sorted(val_variants.keys())}"
                )
            val_metrics = val_variants[primary]
            val_score = float(val_metrics[args.val_selection_metric])
            print(
                f"[Val Step {stage} | Epoch {epoch + 1} | {primary}] "
                f"{args.val_selection_metric}={val_score:.2f} "
                f"Total={val_metrics['total_acc']:.2f} "
                f"Old={val_metrics['old_acc']:.2f} "
                f"New={val_metrics['new_acc']:.2f}"
            )
            if best_selection is None or val_score > best_selection["score"]:
                best_selection = {
                    "epoch": epoch + 1,
                    "score": val_score,
                    "metrics": {
                        key: float(value)
                        for key, value in val_metrics.items()
                        if key in ["total_acc", "old_acc", "new_acc", "acc_when_learned"]
                    },
                    "encoder": clone_state_dict(encoder),
                    "projector": clone_state_dict(projector),
                    "classifier": clone_state_dict(classifier),
                }
        adjust_lr_add(optimizer, epoch, args)
        if early_monitor != "none" and args.early_stop_patience > 0:
            if early_monitor == "val_metric":
                current_value = val_score
                improved = (
                    current_value is not None
                    and (
                        early_best is None
                        or current_value > early_best + float(args.early_stop_min_delta)
                    )
                )
                monitor_text = f"{args.val_selection_metric}={current_value:.4f}" if current_value is not None else "val_metric=None"
            else:
                current_value = epoch_loss
                improved = (
                    early_best is None
                    or current_value < early_best - float(args.early_stop_min_delta)
                )
                monitor_text = f"train_loss={current_value:.4f}"

            if current_value is not None:
                if improved:
                    early_best = float(current_value)
                    early_bad_epochs = 0
                else:
                    early_bad_epochs += 1
                if (
                    epoch + 1 >= int(args.early_stop_min_epochs)
                    and early_bad_epochs >= int(args.early_stop_patience)
                ):
                    print(
                        f"[Early Stop Step {stage}] epoch={epoch + 1} "
                        f"monitor={early_monitor} {monitor_text} "
                        f"best={early_best:.4f} "
                        f"patience={args.early_stop_patience}"
                    )
                    break

    if best_selection is not None:
        encoder.load_state_dict(best_selection["encoder"])
        projector.load_state_dict(best_selection["projector"])
        classifier.load_state_dict(best_selection["classifier"])
        encoder.to(device)
        projector.to(device)
        classifier.to(device)
        print(
            f"[Val Select Step {stage}] epoch={best_selection['epoch']} "
            f"{args.val_selection_metric}={best_selection['score']:.2f}"
        )

    classifier.eval()
    projector.eval()
    encoder.eval()
    feature_cache = build_candidate_stage_cache(
        feature_cache,
        trainset,
        encoder,
        projector,
        classifier,
        label_order,
        args,
        device,
    )

    if pseudo_old_pool:
        refreshed = {}
        for raw_label, data_cpu in pseudo_old_pool.items():
            feats_current = []
            for start in range(0, data_cpu.size(0), args.batch_size):
                x = data_cpu[start : start + args.batch_size].to(device, non_blocking=True)
                with torch.no_grad():
                    cache_value = cache_values_from_raw(encoder, x, args)
                    feat_enc = features_from_cache(encoder, cache_value, args)
                    feat_proj = projector(feat_enc)
                    logits = classifier(feat_proj)
                    pred_idx = logits.argmax(dim=1).detach().cpu().tolist()
                pred_raw = torch.tensor(
                    [label_order[int(idx)] for idx in pred_idx],
                    dtype=torch.long,
                    device=device,
                )
                keep = pred_raw == int(raw_label)
                if keep.any():
                    feats_current.extend(cache_value[keep.to(device)].detach().cpu().tolist())
            if feats_current:
                max_keep = int(args.max_feature_per_class)
                refreshed[int(raw_label)] = feats_current[-max_keep:]

        for raw_label, feats_current in refreshed.items():
            if args.replace_old_cache_with_pseudo:
                feature_cache[int(raw_label)] = feats_current
            else:
                feature_cache.setdefault(int(raw_label), []).extend(feats_current)
                max_keep = int(args.max_feature_per_class)
                if len(feature_cache[int(raw_label)]) > max_keep:
                    feature_cache[int(raw_label)] = feature_cache[int(raw_label)][-max_keep:]

    save_cache(args.save_dir, stage, feature_cache)
    torch.save(encoder, Path(args.save_dir) / f"encoder_step{stage}.pth")
    torch.save(classifier, Path(args.save_dir) / f"classifier_step{stage}.pth")
    torch.save(projector, Path(args.save_dir) / f"contrastive_step{stage}.pth")

    with open(Path(args.save_dir) / f"support_domain_step{stage - 1}_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "projected_support_domain": {} if old_domain is None else old_domain["summary"],
                "encoder_support_replay": support_replay_summary,
                "old_raw_replay": pseudo_old_summary,
                "pseudo_old": pseudo_old_summary,
            },
            f,
            indent=2,
        )

    return encoder, projector, classifier, feature_cache, label_order


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder0, classifier0 = prepare_initial_state(args, device)
    with open(Path(args.save_dir) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(Path(args.save_dir) / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": (
                    "class-incremental learning with true-label old raw replay, "
                    f"encoder_update_mode={args.encoder_update_mode}, "
                    "and multi-local support-center replay"
                ),
                "protocol_note": args.protocol_note,
                "initial_train_source": args.init_dir,
                "increment_train_template": args.train_add_template,
                "test_template": args.test_add_template,
                "initial_classes": args.initial_classes,
                "increment_size": args.increment_size,
                "num_increments": args.num_increments,
                "new_class_start": args.new_class_start,
                "support_replay_per_class": args.support_replay_per_class,
                "disable_feature_cache_replay": args.disable_feature_cache_replay,
                "support_k": args.support_k,
                "support_center_mode": args.support_center_mode,
                "support_radius_stat": args.support_radius_stat,
                "support_loss_space": args.support_loss_space,
                "cache_stage": args.cache_stage,
                "encoder_update_mode": args.encoder_update_mode,
                "encoder_lr": args.encoder_lr,
                "encoder_lr_after_warmup": args.encoder_lr_after_warmup,
                "encoder_lr_warmup_epochs": args.encoder_lr_warmup_epochs,
                "old_raw_source_template": args.pseudo_old_source_template,
                "old_raw_ratio": args.pseudo_old_ratio,
                "old_raw_max_per_class": args.pseudo_old_max_per_class,
                "pseudo_old_source_template": args.pseudo_old_source_template,
                "pseudo_old_ratio": args.pseudo_old_ratio,
                "old_filter_mode": args.old_filter_mode,
                "pseudo_old_require_label_match": args.pseudo_old_require_label_match,
                "anchor_replay_mode": args.anchor_replay_mode,
                "anchor_replay_topk": args.anchor_replay_topk,
                "anchor_replay_global_fraction": args.anchor_replay_global_fraction,
                "anchor_replay_extra_ratio": args.anchor_replay_extra_ratio,
                "val_add_template": args.val_add_template,
                "select_increment_by_val": args.select_increment_by_val,
                "val_selection_metric": args.val_selection_metric,
                "primary_eval": args.primary_eval,
                "early_stop_monitor": args.early_stop_monitor,
                "early_stop_patience": args.early_stop_patience,
                "early_stop_min_epochs": args.early_stop_min_epochs,
                "early_stop_min_delta": args.early_stop_min_delta,
            },
            f,
            indent=2,
        )

    metrics_path = Path(args.save_dir) / "incremental_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {}

    if args.eval_stage0 and "0" not in history:
        cache0 = load_stage_cache(args, 0)
        history = evaluate_stage(
            args,
            0,
            encoder0,
            IdentityProjector().to(device),
            classifier0,
            cache0,
            stage_seen_labels(args, 0),
            history,
            device,
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    for stage in range(1, args.num_increments + 1):
        if (Path(args.save_dir) / f"classifier_step{stage}.pth").exists() and str(stage) in history:
            print(f"[Skip] Step {stage} already has checkpoint and metrics in {args.save_dir}")
            continue

        encoder, projector, classifier, feature_cache, label_order = train_one_stage(args, stage, device)
        history = evaluate_stage(
            args,
            stage,
            encoder,
            projector,
            classifier,
            feature_cache,
            label_order,
            history,
            device,
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_key = str(args.num_increments)
    if final_key in history:
        final = history[final_key]
        print(
            f"[Final] Total={final['total_acc']:.2f} Old={final['old_acc']:.2f} "
            f"New={final['new_acc']:.2f} Forget={final['forget_rate']:.2f} "
            f"CF={final['cumulative_forget']:.2f}"
        )


if __name__ == "__main__":
    main()
