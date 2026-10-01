import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import SupConLoss_DynamicMargin
from dataset import NpyDeviceDataset
from util.utils import adjust_lr, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an initial closed-set CIL model without auxiliary unknown samples."
    )
    parser.add_argument("--backbone", choices=["adsb2d", "oracle1d"], default="adsb2d")
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--aux_unknown_dir", default=None, type=str)
    parser.add_argument("--val_dir", default=None, type=str)
    parser.add_argument("--test_dir", default=None, type=str)
    parser.add_argument("--selection_metric", choices=["train", "val", "test"], default="val")
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--num_classes", required=True, type=int)
    parser.add_argument("--label_start", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--open_batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--lr_decay_epochs", default="20,30,40", type=str)
    parser.add_argument("--embedding_dim", default=1024, type=int)
    parser.add_argument("--max_feature_per_class", default=100, type=int)
    parser.add_argument("--ce_weight", default=2.4, type=float)
    parser.add_argument("--con_weight", default=1.0, type=float)
    parser.add_argument("--aux_mode", choices=["margin", "mixed_supcon"], default="margin")
    parser.add_argument("--aux_sep_weight", default=0.2, type=float)
    parser.add_argument("--aux_margin", default=0.2, type=float)
    parser.add_argument("--aux_unknown_grad", action="store_true")
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--base_margin", default=0.3, type=float)
    parser.add_argument("--beta", default=0.4, type=float)
    return parser.parse_args()


def build_models(args, device):
    if args.backbone == "adsb2d":
        from model_mix import ClassifierHead, FeatureExtractor
    elif args.backbone == "oracle1d":
        from model_mix_oracle import ClassifierHead, FeatureExtractor
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    encoder = FeatureExtractor(in_channels=3, feature_dim=args.embedding_dim).to(device)
    classifier = ClassifierHead(args.embedding_dim, args.num_classes).to(device)
    return encoder, classifier


def make_label_maps(args):
    raw_labels = list(range(args.label_start, args.label_start + args.num_classes))
    raw_to_index = {raw: idx for idx, raw in enumerate(raw_labels)}
    index_to_raw = {idx: raw for raw, idx in raw_to_index.items()}
    return raw_labels, raw_to_index, index_to_raw


def remap_labels(y_raw, raw_to_index, device):
    return torch.tensor(
        [raw_to_index[int(y.item())] for y in y_raw],
        dtype=torch.long,
        device=device,
    )


def auxiliary_unknown_margin_loss(feat_known, feat_unknown, margin):
    if feat_unknown is None or feat_unknown.numel() == 0:
        return feat_known.new_tensor(0.0)
    known_norm = F.normalize(feat_known, dim=1)
    unknown_norm = F.normalize(feat_unknown, dim=1)
    sim = known_norm @ unknown_norm.t()
    return F.relu(sim - margin).mean()


@torch.no_grad()
def evaluate(encoder, classifier, dataset, raw_to_index, args, device):
    if dataset is None:
        return None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    per_class = defaultdict(lambda: [0, 0])
    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y = remap_labels(y_raw, raw_to_index, device)
        logits = classifier(encoder(x))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        for raw, ok in zip(y_raw.tolist(), (pred == y).detach().cpu().tolist()):
            per_class[int(raw)][1] += 1
            per_class[int(raw)][0] += int(ok)
    return {
        "acc": 100.0 * correct / max(1, total),
        "total": total,
        "per_class_acc": {
            str(raw): 100.0 * good / max(1, count)
            for raw, (good, count) in sorted(per_class.items())
        },
    }


@torch.no_grad()
def build_feature_cache(encoder, classifier, dataset, raw_to_index, args, device):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()
    cache = {raw: [] for raw in raw_to_index.keys()}
    for x, y_raw in loader:
        x = x.to(device, non_blocking=True)
        y = remap_labels(y_raw, raw_to_index, device)
        feat = encoder(x)
        pred = classifier(feat).argmax(dim=1)
        for raw in sorted(set(int(v) for v in y_raw.tolist())):
            mask_raw = torch.tensor(
                [int(v) == raw for v in y_raw.tolist()],
                dtype=torch.bool,
                device=device,
            )
            mask = mask_raw & (pred == y)
            if mask.any():
                cache[raw].extend(feat[mask].detach().cpu().tolist())
                if len(cache[raw]) > args.max_feature_per_class:
                    cache[raw] = cache[raw][-args.max_feature_per_class :]
    return cache


def main():
    args = parse_args()
    args.lr_decay_epochs = [
        int(item.strip()) for item in str(args.lr_decay_epochs).split(",") if item.strip()
    ]
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_labels, raw_to_index, index_to_raw = make_label_maps(args)
    train_set = NpyDeviceDataset(args.train_dir, allowed_labels=raw_labels)
    aux_unknown_set = (
        NpyDeviceDataset(args.aux_unknown_dir)
        if args.aux_unknown_dir
        else None
    )
    val_set = (
        NpyDeviceDataset(args.val_dir, allowed_labels=raw_labels)
        if args.val_dir
        else None
    )
    test_set = (
        NpyDeviceDataset(args.test_dir, allowed_labels=raw_labels)
        if args.test_dir
        else None
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    aux_unknown_loader = (
        DataLoader(aux_unknown_set, batch_size=args.open_batch_size, shuffle=True)
        if aux_unknown_set is not None
        else None
    )

    encoder, classifier = build_models(args, device)
    with torch.no_grad():
        x_probe, _ = next(iter(train_loader))
        encoder(x_probe[: min(4, x_probe.size(0))].to(device))

    ce_loss = nn.CrossEntropyLoss()
    con_loss = SupConLoss_DynamicMargin(
        temperature=args.temperature,
        base_margin=args.base_margin,
        beta=args.beta,
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_score = -1.0
    train_log = []
    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_acc = 0.0
        total_aux = 0.0
        aux_iter = itertools.cycle(aux_unknown_loader) if aux_unknown_loader is not None else None
        for x, y_raw in train_loader:
            x = x.to(device, non_blocking=True)
            y = remap_labels(y_raw, raw_to_index, device)
            feat = encoder(x)
            logits = classifier(feat)
            loss_ce = ce_loss(logits, y)

            aux_loss = feat.new_tensor(0.0)
            if aux_iter is not None:
                x_aux, _ = next(aux_iter)
                x_aux = x_aux.to(device, non_blocking=True)
                if args.aux_unknown_grad:
                    feat_aux = encoder(x_aux)
                else:
                    with torch.no_grad():
                        feat_aux = encoder(x_aux)
                if args.aux_mode == "mixed_supcon":
                    labels_aux = torch.full(
                        (feat_aux.size(0),),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
                    feat_for_con = torch.cat([feat, feat_aux], dim=0)
                    labels_for_con = torch.cat([y, labels_aux], dim=0)
                    loss_con = con_loss(feat_for_con, labels_for_con)
                else:
                    loss_con = con_loss(feat, y)
                    aux_loss = auxiliary_unknown_margin_loss(
                        feat,
                        feat_aux,
                        args.aux_margin,
                    )
            else:
                loss_con = con_loss(feat, y)

            loss = (
                args.ce_weight * loss_ce
                + args.con_weight * loss_con
                + args.aux_sep_weight * aux_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_acc += float((logits.argmax(dim=1) == y).float().mean().item() * 100.0)
            total_aux += float(aux_loss.item())

        val_metrics = evaluate(encoder, classifier, val_set, raw_to_index, args, device)
        test_metrics = evaluate(encoder, classifier, test_set, raw_to_index, args, device)
        if args.selection_metric == "val" and val_metrics is not None:
            select_score = val_metrics["acc"]
        elif args.selection_metric == "test" and test_metrics is not None:
            select_score = test_metrics["acc"]
        else:
            select_score = total_acc / len(train_loader)
        if select_score > best_score:
            best_score = select_score
            torch.save(encoder, save_dir / "encoder.pth")
            torch.save(classifier, save_dir / "classifier.pth")

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / len(train_loader),
            "train_acc": total_acc / len(train_loader),
            "val_acc": None if val_metrics is None else val_metrics["acc"],
            "test_acc": None if test_metrics is None else test_metrics["acc"],
            "best_select_acc": best_score,
        }
        train_log.append(row)
        print(
            f"[Epoch {epoch + 1}] Loss={row['loss']:.4f} "
            f"TrainAcc={row['train_acc']:.2f} "
            f"ValAcc={row['val_acc'] if row['val_acc'] is not None else -1:.2f} "
            f"TestAcc={row['test_acc'] if row['test_acc'] is not None else -1:.2f} "
            f"Best={best_score:.2f} "
            f"Aux={total_aux / len(train_loader):.4f}"
        )
        adjust_lr(optimizer, epoch, args)

    encoder = torch.load(save_dir / "encoder.pth", map_location=device).to(device)
    classifier = torch.load(save_dir / "classifier.pth", map_location=device).to(device)
    final_val = evaluate(encoder, classifier, val_set, raw_to_index, args, device)
    final_test = evaluate(encoder, classifier, test_set, raw_to_index, args, device)
    summary = {
        "protocol": (
            "initial training with auxiliary unknown samples"
            if aux_unknown_set is not None
            else "closed-set initial training only; no auxiliary unknown samples"
        ),
        "args": vars(args),
        "raw_labels": raw_labels,
        "index_to_raw": {str(k): v for k, v in index_to_raw.items()},
        "train_samples": len(train_set),
        "aux_unknown_samples": 0 if aux_unknown_set is None else len(aux_unknown_set),
        "val": final_val,
        "test": final_test,
        "feature_cache_exported": False,
    }
    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(save_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved closed-set initial model to {save_dir}")


if __name__ == "__main__":
    main()
