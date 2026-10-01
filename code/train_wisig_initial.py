import argparse
import json
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import SupConLoss
from model_mix_oracle import ClassifierHead, FeatureExtractor
from train_incremental_support_domain import NpyDeviceDataset
from util.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train initial WiSig closed-set model.")
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--save_dir", default="runs/incremental_support/wisig_init_node18_2_id_iqdphi", type=str)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--embedding_dim", default=1024, type=int)
    parser.add_argument("--max_feature_per_class", default=100, type=int)
    parser.add_argument("--ce_weight", default=2.0, type=float)
    parser.add_argument("--con_weight", default=0.5, type=float)
    parser.add_argument("--temperature", default=0.08, type=float)
    return parser.parse_args()


@torch.no_grad()
def evaluate(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        logits = classifier(encoder(x))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def build_feature_cache(encoder, classifier, dataset, args, device):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder.eval()
    classifier.eval()
    cache = {cls: [] for cls in range(args.num_classes)}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        feat = encoder(x)
        pred = classifier(feat).argmax(dim=1)
        for cls in y.unique():
            cls_int = int(cls.item())
            mask = (y == cls) & (pred == y)
            if mask.any():
                cache[cls_int].extend(feat[mask].detach().cpu().tolist())
                if len(cache[cls_int]) > args.max_feature_per_class:
                    cache[cls_int] = cache[cls_int][-args.max_feature_per_class :]
    return cache


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_set = NpyDeviceDataset(
        Path(args.data_root) / "train-closed",
        allowed_labels=range(args.num_classes),
    )
    val_root = Path(args.data_root) / "val-closed"
    val_set = NpyDeviceDataset(
        val_root if val_root.exists() else Path(args.data_root) / "test-closed",
        allowed_labels=range(args.num_classes),
    )
    test_set = NpyDeviceDataset(
        Path(args.data_root) / "test-closed",
        allowed_labels=range(args.num_classes),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    encoder = FeatureExtractor(in_channels=args.in_channels, feature_dim=args.embedding_dim).to(device)
    with torch.no_grad():
        x_probe, _ = next(iter(train_loader))
        encoder(x_probe[: min(4, x_probe.size(0))].to(device))
    classifier = ClassifierHead(args.embedding_dim, args.num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss()
    con_loss = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_acc = -1.0
    for epoch in range(args.epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_acc = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device)
            feat = encoder(x)
            logits = classifier(feat)
            loss = args.ce_weight * ce_loss(logits, y) + args.con_weight * con_loss(feat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_acc += float((logits.argmax(dim=1) == y).float().mean().item() * 100.0)

        val_acc = evaluate(encoder, classifier, val_loader, device)
        test_acc = evaluate(encoder, classifier, test_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(encoder, out / "encoder.pth")
            torch.save(classifier, out / "classifier.pth")
        print(
            f"[Epoch {epoch + 1}] Loss={total_loss / len(train_loader):.4f} "
            f"TrainAcc={total_acc / len(train_loader):.2f} ValAcc={val_acc:.2f} "
            f"TestAcc={test_acc:.2f} BestVal={best_acc:.2f}"
        )

    encoder = torch.load(out / "encoder.pth", map_location=device).to(device)
    classifier = torch.load(out / "classifier.pth", map_location=device).to(device)
    cache = build_feature_cache(encoder, classifier, train_set, args, device)
    with open(out / "feature_cache_step0.pkl", "wb") as f:
        pickle.dump({"version": 0, "features": cache}, f)
    print(f"Saved initial WiSig model and cache to {out}")


if __name__ == "__main__":
    main()
