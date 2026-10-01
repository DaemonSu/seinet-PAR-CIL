import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_support_module():
    here = Path(__file__).resolve().parent
    support_path = here / "15_train_incremental_anchor_domain.py"
    sys.path.insert(0, str(here))
    os.chdir(str(here))
    spec = importlib.util.spec_from_file_location("anchor_support", support_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict no-replay anchor tree with frozen encoder and trainable E2/MLP."
    )
    parser.add_argument("--init_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--initial_classes", default=50, type=int)
    parser.add_argument("--increment_size", default=5, type=int)
    parser.add_argument("--num_increments", default=10, type=int)
    parser.add_argument("--new_class_start", default=50, type=int)
    parser.add_argument("--train_closed", default="train-closed", type=str)
    parser.add_argument("--test_closed", default="test-closed", type=str)
    parser.add_argument("--train_add_template", default="train-add{step}", type=str)
    parser.add_argument("--test_add_template", default="test-add{step}", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--incr_batch_size", default=150, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--lr_decay_epochs", default="30,50,140", type=str)
    parser.add_argument("--embedding_dim", default=1024, type=int)
    parser.add_argument("--projector_alpha", default=0.9, type=float)
    parser.add_argument(
        "--e2_mode",
        choices=["trainable", "identity"],
        default="trainable",
        help="Use trainable residual E2 or an identity E2 ablation.",
    )
    parser.add_argument("--max_train_per_class", default=0, type=int)
    parser.add_argument("--max_test_per_class", default=0, type=int)
    parser.add_argument("--anchor_topk", default=10, type=int)
    parser.add_argument("--route_topk", default=10, type=int)
    parser.add_argument("--centers_per_class", default=1, type=int)
    parser.add_argument(
        "--anchor_candidate_scope",
        choices=["all_seen", "initial_only"],
        default="all_seen",
        type=str,
    )
    parser.add_argument("--radius_percentile", default=95.0, type=float)
    parser.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--new_ce_weight", default=1.0, type=float)
    parser.add_argument("--prototype_ce_weight", default=1.0, type=float)
    parser.add_argument("--distill_weight", default=1.0, type=float)
    parser.add_argument("--new_data_distill_weight", default=0.0, type=float)
    parser.add_argument("--new_data_distill_temperature", default=2.0, type=float)
    parser.add_argument("--anchor_ce_weight", default=1.0, type=float)
    parser.add_argument("--anchor_margin_weight", default=0.5, type=float)
    parser.add_argument("--anchor_margin", default=0.10, type=float)
    parser.add_argument("--anchor_temperature", default=0.07, type=float)
    parser.add_argument("--freeze_old_output_rows", action="store_true")
    parser.add_argument("--early_stop_patience", default=6, type=int)
    parser.add_argument("--early_stop_min_epochs", default=10, type=int)
    parser.add_argument("--early_stop_min_delta", default=0.001, type=float)
    parser.add_argument("--protocol_note", default="", type=str)
    return parser.parse_args()


def parse_int_list(value):
    return [int(float(item.strip())) for item in value.split(",") if item.strip()]


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
    return str(Path(args.data_root) / template.format(step=stage))


@torch.no_grad()
def collect_encoder_features(module, encoder, folder, allowed_labels, args, device):
    dataset = module.NpyDeviceDataset(
        folder,
        allowed_labels=allowed_labels,
        max_per_class=args.max_train_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)
        feats.append(h.detach().cpu())
        labels.extend(int(item) for item in y.tolist())
    if not feats:
        raise RuntimeError(f"No features collected from {folder}")
    return torch.cat(feats, dim=0), np.asarray(labels, dtype=np.int64)


def pairwise_distance(features, prototypes, distance):
    if distance == "cosine":
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        return 1.0 - torch.matmul(features, prototypes.T)
    return torch.cdist(features, prototypes, p=2)


def summarize_class(feats, percentile, distance):
    proto = F.normalize(feats.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    dist = pairwise_distance(feats, proto.view(1, -1), distance).squeeze(1)
    radius = float(np.percentile(dist.detach().cpu().numpy(), percentile))
    return proto.detach().cpu(), radius


def support_centers(feats, centers_per_class):
    k = max(1, min(int(centers_per_class), int(feats.size(0))))
    if k == 1:
        centers = feats.mean(dim=0, keepdim=True)
    else:
        feats_np = feats.detach().cpu().numpy().astype(np.float32)
        labels = np.zeros(feats_np.shape[0], dtype=np.int64)
        class_mean = feats_np.mean(axis=0, keepdims=True)
        dist = np.sum((feats_np - class_mean) ** 2, axis=1)
        selected = [int(np.argmin(dist))]
        min_dist = np.sum((feats_np - feats_np[selected[0] : selected[0] + 1]) ** 2, axis=1)
        while len(selected) < k:
            min_dist[selected] = -np.inf
            selected.append(int(np.argmax(min_dist)))
            next_dist = np.sum((feats_np - feats_np[selected[-1] : selected[-1] + 1]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, next_dist)
        centers_np = feats_np[selected].copy()
        for _ in range(3):
            dist_mat = np.sum((feats_np[:, None, :] - centers_np[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist_mat, axis=1).astype(np.int64)
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centers_np[idx] = feats_np[mask].mean(axis=0)
        centers = torch.tensor(centers_np, dtype=torch.float32)
    return F.normalize(centers, dim=1).detach().cpu()


def build_step0_state(module, encoder, args, device):
    labels = stage_seen_labels(args, 0)
    folder = str(Path(args.data_root) / args.train_closed)
    feats, y = collect_encoder_features(module, encoder, folder, labels, args, device)
    prototypes, support_prototypes, radii, counts = {}, {}, {}, {}
    for label in labels:
        cls_feats = feats[y == int(label)]
        proto, radius = summarize_class(cls_feats, args.radius_percentile, args.distance)
        prototypes[int(label)] = proto
        support_prototypes[int(label)] = support_centers(cls_feats, args.centers_per_class)
        radii[int(label)] = radius
        counts[int(label)] = int(cls_feats.size(0))

    return {
        "stage": 0,
        "label_order": labels,
        "prototypes": prototypes,
        "support_prototypes": support_prototypes,
        "radii": radii,
        "counts": counts,
        "parents": {int(label): None for label in labels},
        "anchors": {},
        "config": {
            "distance": args.distance,
            "anchor_topk": int(args.anchor_topk),
            "route_topk": int(args.route_topk),
            "centers_per_class": int(args.centers_per_class),
            "anchor_candidate_scope": args.anchor_candidate_scope,
            "radius_percentile": float(args.radius_percentile),
            "stored_state": "one route prototype, compact support prototypes, one radius, count, parent, and top-k anchors per class",
        },
    }


def prototype_matrix(state, labels=None):
    labels = state["label_order"] if labels is None else [int(item) for item in labels]
    protos = torch.stack([state["prototypes"][int(label)] for label in labels], dim=0)
    return labels, protos


def nearest_labels(features, state, labels, distance, topk=1):
    labels, protos = prototype_matrix(state, labels)
    dist = pairwise_distance(features.cpu(), protos, distance)
    k = max(1, min(int(topk), len(labels)))
    values, indices = torch.topk(dist, k=k, dim=1, largest=False)
    out = []
    for row_values, row_indices in zip(values.tolist(), indices.tolist()):
        out.append([(int(labels[int(idx)]), float(value)) for value, idx in zip(row_values, row_indices)])
    return out


def anchor_candidate_labels(args, state):
    if args.anchor_candidate_scope == "initial_only":
        candidates = range(args.initial_classes)
        return [int(label) for label in candidates if int(label) in state["prototypes"]]
    return list(state["label_order"])


def add_increment_nodes(module, encoder, state, args, stage, device):
    old_labels = anchor_candidate_labels(args, state)
    new_labels = stage_new_labels(args, stage)
    folder = resolve_stage_folder(args, args.train_add_template, stage)
    feats, y = collect_encoder_features(module, encoder, folder, new_labels, args, device)

    summary = {
        "stage": int(stage),
        "anchor_candidate_scope": args.anchor_candidate_scope,
        "anchor_candidate_count": len(old_labels),
        "new_classes": {},
    }
    for label in new_labels:
        cls_feats = feats[y == int(label)]
        proto, radius = summarize_class(cls_feats, args.radius_percentile, args.distance)
        anchors = nearest_labels(
            proto.view(1, -1),
            state,
            old_labels,
            args.distance,
            topk=args.anchor_topk,
        )[0]
        parent = int(anchors[0][0]) if anchors else None
        state["label_order"].append(int(label))
        state["prototypes"][int(label)] = proto
        state.setdefault("support_prototypes", {})[int(label)] = support_centers(cls_feats, args.centers_per_class)
        state["radii"][int(label)] = radius
        state["counts"][int(label)] = int(cls_feats.size(0))
        state["parents"][int(label)] = parent
        state["anchors"][int(label)] = anchors
        summary["new_classes"][str(int(label))] = {
            "parent": parent,
            "anchors": [{"label": item[0], "distance": item[1]} for item in anchors],
            "count": int(cls_feats.size(0)),
            "radius": float(radius),
        }
    state["stage"] = int(stage)
    return summary


def build_children(state):
    children = defaultdict(list)
    roots = []
    for label in state["label_order"]:
        parent = state["parents"].get(int(label), None)
        if parent is None:
            roots.append(int(label))
        else:
            children[int(parent)].append(int(label))
    for values in children.values():
        values.sort()
    roots.sort()
    return roots, children


def descendants(children, roots):
    keep = set(int(item) for item in roots)
    queue = deque(int(item) for item in roots)
    while queue:
        node = queue.popleft()
        for child in children.get(int(node), []):
            if int(child) not in keep:
                keep.add(int(child))
                queue.append(int(child))
    return sorted(keep)


def labels_to_indices(labels, label_to_index, device):
    return torch.tensor(
        [label_to_index[int(item)] for item in labels],
        dtype=torch.long,
        device=device,
    )


def make_projector(module, prev_projector, args, stage, device):
    if getattr(args, "e2_mode", "trainable") == "identity":
        # Use PyTorch's importable identity module so the structural ablation
        # remains serializable across separately launched incremental stages.
        return torch.nn.Identity().to(device)
    if stage == 1 or isinstance(prev_projector, module.IdentityProjector):
        return module.ContrastiveLayer(
            in_dim=args.embedding_dim,
            hidden_dim=512,
            out_dim=args.embedding_dim,
            use_norm=True,
            alpha=args.projector_alpha,
        ).to(device)
    # The teacher is frozen before copying; the student must learn at every step.
    return copy.deepcopy(prev_projector).to(device).requires_grad_(True)


def make_expanded_classifier(module, old_classifier, total_classes, args, device):
    return module.make_expanded_classifier(
        old_classifier,
        total_classes,
        args.embedding_dim,
        device,
    )


def output_layer(classifier):
    if hasattr(classifier, "classifier"):
        return classifier.classifier[-1]
    if hasattr(classifier, "fc"):
        return classifier.fc
    raise AttributeError("Unsupported classifier structure.")


def freeze_old_output_gradients(classifier, old_num_outputs):
    layer = output_layer(classifier)

    def zero_old_rows(grad):
        grad = grad.clone()
        grad[:old_num_outputs].zero_()
        return grad

    layer.weight.register_hook(zero_old_rows)
    layer.bias.register_hook(zero_old_rows)


def load_previous_models(module, args, stage, device):
    if stage == 1:
        encoder = torch.load(Path(args.save_dir) / "encoder.pth", map_location=device).to(device).eval()
        classifier = torch.load(Path(args.save_dir) / "classifier.pth", map_location=device).to(device).eval()
        projector = module.IdentityProjector().to(device).eval()
        return encoder, classifier, projector

    encoder = torch.load(Path(args.save_dir) / f"encoder_step{stage - 1}.pth", map_location=device).to(device).eval()
    classifier = torch.load(Path(args.save_dir) / f"classifier_step{stage - 1}.pth", map_location=device).to(device).eval()
    projector = torch.load(Path(args.save_dir) / f"contrastive_step{stage - 1}.pth", map_location=device).to(device).eval()
    return encoder, classifier, projector


def clone_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def prototype_tensors(state, labels, device):
    labels = [int(item) for item in labels]
    proto_parts = []
    raw_labels = []
    support_map = state.get("support_prototypes", {})
    for label in labels:
        centers = support_map.get(label, None)
        if centers is None:
            centers = state["prototypes"][label].view(1, -1)
        if centers.dim() == 1:
            centers = centers.view(1, -1)
        proto_parts.append(centers)
        raw_labels.extend([label] * int(centers.size(0)))
    protos = torch.cat(proto_parts, dim=0).to(device)
    return protos, torch.tensor(raw_labels, dtype=torch.long, device=device)


def anchor_losses(z_new, h_new, y_new_raw, projector, state, new_labels, label_to_index, args, device):
    if args.anchor_ce_weight <= 0 and args.anchor_margin_weight <= 0:
        zero = z_new.new_tensor(0.0)
        return zero, zero

    anchor_ce_losses = []
    margin_losses = []
    for label in sorted(set(int(item) for item in y_new_raw.detach().cpu().tolist())):
        anchors = [int(item[0]) for item in state["anchors"].get(int(label), [])]
        if not anchors:
            continue
        mask = y_new_raw == int(label)
        if not mask.any():
            continue
        own_proto_h = state["prototypes"][int(label)].view(1, -1).to(device)
        anchor_proto_h = torch.stack([state["prototypes"][int(anchor)] for anchor in anchors], dim=0).to(device)
        proto_h = torch.cat([own_proto_h, anchor_proto_h], dim=0)
        proto_z = projector(proto_h)
        sample_z = z_new[mask]
        dist = pairwise_distance(sample_z, proto_z, args.distance)
        if args.anchor_ce_weight > 0:
            logits = -dist / max(float(args.anchor_temperature), 1e-6)
            target = torch.zeros(sample_z.size(0), dtype=torch.long, device=device)
            anchor_ce_losses.append(F.cross_entropy(logits, target))
        if args.anchor_margin_weight > 0 and dist.size(1) > 1:
            own_dist = dist[:, 0]
            min_anchor_dist = dist[:, 1:].min(dim=1).values
            margin_losses.append(F.relu(float(args.anchor_margin) + own_dist - min_anchor_dist).mean())

    zero = z_new.new_tensor(0.0)
    ce_loss = torch.stack(anchor_ce_losses).mean() if anchor_ce_losses else zero
    margin_loss = torch.stack(margin_losses).mean() if margin_losses else zero
    return ce_loss, margin_loss


def train_one_stage(module, args, stage, state, device):
    encoder, old_classifier, prev_projector = load_previous_models(module, args, stage, device)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    old_classifier.eval()
    prev_projector.eval()
    for param in old_classifier.parameters():
        param.requires_grad = False
    for param in prev_projector.parameters():
        param.requires_grad = False

    old_label_order = stage_seen_labels(args, stage - 1)
    label_order = stage_seen_labels(args, stage)
    label_to_index = {int(label): idx for idx, label in enumerate(label_order)}
    old_num_outputs = len(old_label_order)
    new_labels = stage_new_labels(args, stage)

    train_folder = resolve_stage_folder(args, args.train_add_template, stage)
    trainset = module.NpyDeviceDataset(
        train_folder,
        allowed_labels=new_labels,
        max_per_class=args.max_train_per_class,
    )
    train_loader = DataLoader(trainset, batch_size=args.incr_batch_size, shuffle=True)

    classifier = make_expanded_classifier(module, old_classifier, len(label_order), args, device)
    projector = make_projector(module, prev_projector, args, stage, device)
    if args.e2_mode == "trainable" and not all(
        param.requires_grad for param in projector.parameters()
    ):
        raise RuntimeError("Incremental E2 student contains frozen parameters")
    if args.freeze_old_output_rows:
        freeze_old_output_gradients(classifier, old_num_outputs)
    classifier.train()
    projector.train()

    trainable_parameters = [
        param
        for param in list(projector.parameters()) + list(classifier.parameters())
        if param.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.incre_lr = args.lr
    args.incre_lr_decay_rate = args.lr_decay_rate
    args.incre_lr_decay_epochs = parse_int_list(args.lr_decay_epochs)
    args.min_lr = args.min_lr

    old_proto_h, old_proto_raw = prototype_tensors(state, old_label_order, device)
    all_proto_h, all_proto_raw = prototype_tensors(state, label_order, device)
    old_proto_count = int(old_proto_h.size(0))
    old_proto_idx = labels_to_indices(old_proto_raw.detach().cpu().tolist(), label_to_index, device)
    all_proto_idx = labels_to_indices(all_proto_raw.detach().cpu().tolist(), label_to_index, device)

    with torch.no_grad():
        old_prev_z = prev_projector(old_proto_h)
        old_prev_logits = old_classifier(old_prev_z)

    print(
        f"[Train Step {stage}] frozen_encoder=True old={len(old_label_order)} "
        f"total={len(label_order)} new={new_labels} train_samples={len(trainset)} "
        f"e2_mode={args.e2_mode}"
    )

    best_loss = None
    bad_epochs = 0
    for epoch in range(args.epochs):
        classifier.train()
        projector.train()
        total_loss = 0.0
        total_acc = 0.0
        total_new_ce = 0.0
        total_proto_ce = 0.0
        total_distill = 0.0
        total_new_data_distill = 0.0
        total_anchor_ce = 0.0
        total_anchor_margin = 0.0

        for x_new, y_new_raw in train_loader:
            x_new = x_new.to(device, non_blocking=True)
            y_new_raw = y_new_raw.to(device)
            y_new_idx = labels_to_indices(y_new_raw.detach().cpu().tolist(), label_to_index, device)

            with torch.no_grad():
                h_new = F.normalize(encoder(x_new), dim=1)

            z_new = projector(h_new)
            logits_new = classifier(z_new)
            new_ce = F.cross_entropy(logits_new, y_new_idx)

            if args.new_data_distill_weight > 0:
                with torch.no_grad():
                    prev_z_new = prev_projector(h_new)
                    prev_logits_new = old_classifier(prev_z_new)
                temp = max(float(args.new_data_distill_temperature), 1e-6)
                new_data_distill = F.kl_div(
                    F.log_softmax(logits_new[:, :old_num_outputs] / temp, dim=1),
                    F.softmax(prev_logits_new.detach() / temp, dim=1),
                    reduction="batchmean",
                ) * (temp * temp)
            else:
                new_data_distill = logits_new.new_tensor(0.0)

            z_all_proto = projector(all_proto_h)
            logits_all_proto = classifier(z_all_proto)
            proto_ce = F.cross_entropy(logits_all_proto, all_proto_idx)

            z_old_proto = z_all_proto[:old_proto_count]
            logits_old_proto = classifier(z_old_proto)[:, :old_num_outputs]
            distill = F.kl_div(
                F.log_softmax(logits_old_proto, dim=1),
                F.softmax(old_prev_logits.detach(), dim=1),
                reduction="batchmean",
            )

            anchor_ce, anchor_margin = anchor_losses(
                z_new,
                h_new,
                y_new_raw,
                projector,
                state,
                new_labels,
                label_to_index,
                args,
                device,
            )

            loss = (
                float(args.new_ce_weight) * new_ce
                + float(args.prototype_ce_weight) * proto_ce
                + float(args.distill_weight) * distill
                + float(args.new_data_distill_weight) * new_data_distill
                + float(args.anchor_ce_weight) * anchor_ce
                + float(args.anchor_margin_weight) * anchor_margin
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits_new.argmax(dim=1)
            acc = (preds == y_new_idx).float().mean().item() * 100.0
            total_loss += float(loss.item())
            total_acc += acc
            total_new_ce += float(new_ce.item())
            total_proto_ce += float(proto_ce.item())
            total_distill += float(distill.item())
            total_new_data_distill += float(new_data_distill.item())
            total_anchor_ce += float(anchor_ce.item())
            total_anchor_margin += float(anchor_margin.item())

        denom = max(1, len(train_loader))
        epoch_loss = total_loss / denom
        print(
            f"[Step {stage} | Epoch {epoch + 1}] "
            f"Loss={epoch_loss:.4f} NewAcc={total_acc / denom:.2f} "
            f"NewCE={total_new_ce / denom:.4f} ProtoCE={total_proto_ce / denom:.4f} "
            f"Distill={total_distill / denom:.4f} NewDataDistill={total_new_data_distill / denom:.4f} "
            f"AnchorCE={total_anchor_ce / denom:.4f} AnchorMargin={total_anchor_margin / denom:.4f}"
        )

        if best_loss is None or epoch_loss < best_loss - float(args.early_stop_min_delta):
            best_loss = epoch_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
        if (
            args.early_stop_patience > 0
            and epoch + 1 >= int(args.early_stop_min_epochs)
            and bad_epochs >= int(args.early_stop_patience)
        ):
            print(
                f"[Early Stop Step {stage}] epoch={epoch + 1} "
                f"train_loss={epoch_loss:.4f} best={best_loss:.4f} "
                f"patience={args.early_stop_patience}"
            )
            break

        if hasattr(module, "adjust_lr_add"):
            module.adjust_lr_add(optimizer, epoch, args)

    classifier.eval()
    projector.eval()
    torch.save(encoder, Path(args.save_dir) / f"encoder_step{stage}.pth")
    torch.save(classifier, Path(args.save_dir) / f"classifier_step{stage}.pth")
    torch.save(projector, Path(args.save_dir) / f"contrastive_step{stage}.pth")
    return encoder, projector, classifier


@torch.no_grad()
def collect_test(module, encoder, projector, classifier, folder, label_order, args, device):
    dataset = module.NpyDeviceDataset(
        folder,
        allowed_labels=label_order,
        max_per_class=args.max_test_per_class,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder.eval()
    projector.eval()
    classifier.eval()
    h_parts, z_parts, logits_parts, labels = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)
        z = projector(h)
        logits = classifier(z)
        h_parts.append(h.detach().cpu())
        z_parts.append(z.detach().cpu())
        logits_parts.append(logits.detach().cpu())
        labels.extend(int(item) for item in y.tolist())
    return (
        torch.cat(h_parts, dim=0),
        torch.cat(z_parts, dim=0),
        torch.cat(logits_parts, dim=0),
        np.asarray(labels, dtype=np.int64),
    )


@torch.no_grad()
def projected_prototypes(projector, state, labels, device):
    labels, protos = prototype_matrix(state, labels)
    z = projector(protos.to(device)).detach().cpu()
    return labels, z


def predict_tree_nearest(features, state, args, space="e1", projector=None, device=None, topk=None):
    roots, children = build_children(state)
    topk = args.route_topk if topk is None else topk
    preds = []
    if space == "e1":
        root_choices = nearest_labels(features, state, roots, args.distance, topk=topk)
        for feat, choices in zip(features, root_choices):
            candidates = descendants(children, [label for label, _ in choices])
            pred = nearest_labels(feat.view(1, -1), state, candidates, args.distance, topk=1)[0][0][0]
            preds.append(int(pred))
        return np.asarray(preds, dtype=np.int64)

    if projector is None or device is None:
        raise ValueError("projector and device are required for e2 tree prediction")
    root_labels, root_proto_z = projected_prototypes(projector, state, roots, device)
    for feat in features:
        feat_z = feat.view(1, -1)
        dist_root = pairwise_distance(feat_z, root_proto_z, args.distance)
        k = max(1, min(int(topk), len(root_labels)))
        root_idx = torch.topk(dist_root, k=k, dim=1, largest=False).indices[0].tolist()
        candidates = descendants(children, [root_labels[int(idx)] for idx in root_idx])
        cand_labels, cand_proto_z = projected_prototypes(projector, state, candidates, device)
        pred_idx = int(pairwise_distance(feat_z, cand_proto_z, args.distance).argmin(dim=1).item())
        preds.append(int(cand_labels[pred_idx]))
    return np.asarray(preds, dtype=np.int64)


def predict_tree_mlp(h_features, logits, state, args):
    roots, children = build_children(state)
    root_choices = nearest_labels(h_features, state, roots, args.distance, topk=args.route_topk)
    label_order = list(state["label_order"])
    label_to_index = {int(label): idx for idx, label in enumerate(label_order)}
    preds = []
    for row, choices in zip(logits, root_choices):
        candidates = descendants(children, [label for label, _ in choices])
        cols = torch.tensor([label_to_index[int(label)] for label in candidates], dtype=torch.long)
        best = int(torch.argmax(row[cols]).item())
        preds.append(int(candidates[best]))
    return np.asarray(preds, dtype=np.int64)


def accuracy_for_labels(y_true, y_pred, label_set):
    label_set = set(int(item) for item in label_set)
    mask = np.asarray([int(item) in label_set for item in y_true], dtype=bool)
    if mask.sum() == 0:
        return None
    return float(100.0 * (y_pred[mask] == y_true[mask]).sum() / mask.sum())


def metrics_from_predictions(y_true, y_pred, args, stage):
    old_labels = stage_seen_labels(args, stage - 1) if stage > 0 else []
    new_labels = stage_new_labels(args, stage) if stage > 0 else stage_seen_labels(args, stage)
    groups = stage_group_labels(args, stage)
    total_acc = float(100.0 * (y_pred == y_true).sum() / max(1, y_true.shape[0]))
    old_acc = accuracy_for_labels(y_true, y_pred, old_labels) if stage > 0 else None
    new_acc = accuracy_for_labels(y_true, y_pred, new_labels)
    old_value = old_acc if old_acc is not None else 0.0
    new_value = new_acc if new_acc is not None else 0.0
    if stage > 0 and old_acc is not None and new_acc is not None:
        balanced = 0.5 * (old_value + new_value)
        harmonic = 2.0 * old_value * new_value / (old_value + new_value) if old_value + new_value > 0 else 0.0
    else:
        balanced = total_acc
        harmonic = total_acc

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
        "old_acc": old_value,
        "new_acc": new_value,
        "balanced_acc": float(balanced),
        "harmonic_acc": float(harmonic),
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
    current["forget_rate"] = float(np.mean(forgetting)) if forgetting else 0.0
    previous = [
        value.get("forget_rate", 0.0)
        for key, value in history.items()
        if int(key) < stage
    ]
    current["cumulative_forget"] = (
        float(np.mean(previous + [current["forget_rate"]]))
        if previous
        else current["forget_rate"]
    )


def evaluate_stage(module, encoder, projector, classifier, state, args, stage, history, device):
    folder = str(Path(args.data_root) / args.test_closed) if stage == 0 else resolve_stage_folder(args, args.test_add_template, stage)
    label_order = stage_seen_labels(args, stage)
    h, z, logits, y_true = collect_test(module, encoder, projector, classifier, folder, label_order, args, device)
    label_to_index = {int(label): idx for idx, label in enumerate(label_order)}
    index_to_label = {idx: int(label) for idx, label in enumerate(label_order)}

    mlp_pred_idx = logits.argmax(dim=1).numpy().tolist()
    pred_mlp = np.asarray([index_to_label[int(idx)] for idx in mlp_pred_idx], dtype=np.int64)
    pred_e1_global = np.asarray(
        [items[0][0] for items in nearest_labels(h, state, label_order, args.distance, topk=1)],
        dtype=np.int64,
    )
    labels_e2, proto_z = projected_prototypes(projector, state, label_order, device)
    e2_nearest_idx = pairwise_distance(z, proto_z, args.distance).argmin(dim=1).numpy().tolist()
    pred_e2_global = np.asarray([labels_e2[int(idx)] for idx in e2_nearest_idx], dtype=np.int64)
    pred_tree_e1 = predict_tree_nearest(h, state, args, space="e1")
    pred_tree_e2 = predict_tree_nearest(z, state, args, space="e2", projector=projector, device=device)
    pred_tree_mlp = predict_tree_mlp(h, logits, state, args)

    variants = {
        "mlp_standard": pred_mlp,
        "proto_e1_global": pred_e1_global,
        "proto_e2_global": pred_e2_global,
        f"tree_e1_topk_{args.route_topk}": pred_tree_e1,
        f"tree_e2_topk_{args.route_topk}": pred_tree_e2,
        f"tree_mlp_topk_{args.route_topk}": pred_tree_mlp,
    }
    variant_metrics = {
        name: metrics_from_predictions(y_true, pred, args, stage)
        for name, pred in variants.items()
    }
    primary = "mlp_standard"
    history[str(stage)] = dict(variant_metrics[primary])
    history[str(stage)]["primary_eval"] = primary
    history[str(stage)]["eval_variants"] = {
        name: {
            key: value
            for key, value in metrics.items()
            if key in ["total_acc", "old_acc", "new_acc", "balanced_acc", "harmonic_acc", "acc_when_learned"]
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
    best_variant = max(
        variant_metrics.items(),
        key=lambda item: item[1]["balanced_acc"],
    )
    print(
        f"[Eval Step {stage} | best_balanced={best_variant[0]}] "
        f"Total={best_variant[1]['total_acc']:.2f} Old={best_variant[1]['old_acc']:.2f} "
        f"New={best_variant[1]['new_acc']:.2f} Balanced={best_variant[1]['balanced_acc']:.2f}"
    )
    return history


def json_ready_state(state):
    children = defaultdict(list)
    for label, parent in state["parents"].items():
        if parent is not None:
            children[str(int(parent))].append(int(label))
    return {
        "stage": int(state["stage"]),
        "label_order": [int(item) for item in state["label_order"]],
        "parents": {str(int(k)): (None if v is None else int(v)) for k, v in state["parents"].items()},
        "children": {k: sorted(v) for k, v in sorted(children.items(), key=lambda item: int(item[0]))},
        "anchors": {
            str(int(k)): [{"label": int(label), "distance": float(dist)} for label, dist in values]
            for k, values in state["anchors"].items()
        },
        "counts": {str(int(k)): int(v) for k, v in state["counts"].items()},
        "radii": {str(int(k)): float(v) for k, v in state["radii"].items()},
        "support_prototype_counts": {
            str(int(k)): int(v.size(0))
            for k, v in state.get("support_prototypes", {}).items()
        },
        "config": state["config"],
    }


def save_state(save_dir, stage, state):
    path = Path(save_dir)
    torch.save(state, path / f"anchor_tree_state_step{stage}.pt")
    with open(path / f"anchor_tree_state_step{stage}.json", "w", encoding="utf-8") as f:
        json.dump(json_ready_state(state), f, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    module = load_support_module()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    init_encoder = torch.load(Path(args.init_dir) / "encoder.pth", map_location=device).to(device).eval()
    init_classifier = torch.load(Path(args.init_dir) / "classifier.pth", map_location=device).to(device).eval()
    for param in init_encoder.parameters():
        param.requires_grad = False
    torch.save(init_encoder, Path(args.save_dir) / "encoder.pth")
    torch.save(init_classifier, Path(args.save_dir) / "classifier.pth")

    protocol = {
        "protocol": "strict no-replay anchor tree with frozen E1 encoder and trainable E2/MLP",
        "protocol_note": args.protocol_note,
        "forbidden_incremental_state": ["old raw IQ samples", "old per-sample feature cache"],
        "allowed_incremental_state": [
            "frozen encoder parameters",
            "E2/MLP model parameters",
            "one route prototype per class",
            "a small number of support prototypes per class",
            "one radius per class",
            "tree parent/children",
            "top-k anchor labels",
        ],
        "training_rule": "Step1+ reads only current train-add{step}; old classes enter losses only through compact prototypes.",
        "initial_classes": args.initial_classes,
        "increment_size": args.increment_size,
        "num_increments": args.num_increments,
        "new_class_start": args.new_class_start,
        "anchor_topk": args.anchor_topk,
        "route_topk": args.route_topk,
        "centers_per_class": args.centers_per_class,
        "anchor_candidate_scope": args.anchor_candidate_scope,
        "distance": args.distance,
        "projector_alpha": args.projector_alpha,
        "e2_mode": args.e2_mode,
        "loss_weights": {
            "new_ce": args.new_ce_weight,
            "prototype_ce": args.prototype_ce_weight,
            "distill": args.distill_weight,
            "new_data_distill": args.new_data_distill_weight,
            "new_data_distill_temperature": args.new_data_distill_temperature,
            "anchor_ce": args.anchor_ce_weight,
            "anchor_margin": args.anchor_margin_weight,
        },
    }
    with open(Path(args.save_dir) / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    with open(Path(args.save_dir) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[Step 0] Build compact anchor state from initial training data.")
    state = build_step0_state(module, init_encoder, args, device)
    save_state(args.save_dir, 0, state)

    history = {}
    history = evaluate_stage(
        module,
        init_encoder,
        module.IdentityProjector().to(device),
        init_classifier,
        state,
        args,
        0,
        history,
        device,
    )
    with open(Path(args.save_dir) / "anchor_tree_mlp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    for stage in range(1, args.num_increments + 1):
        print(f"[Step {stage}] Add new classes {stage_new_labels(args, stage)} with current-increment data only.")
        summary = add_increment_nodes(module, init_encoder, state, args, stage, device)
        with open(Path(args.save_dir) / f"anchor_tree_add_step{stage}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        save_state(args.save_dir, stage, state)
        compact = {
            int(k): {
                "parent": int(v["parent"]) if v["parent"] is not None else None,
                "anchors": [int(item["label"]) for item in v["anchors"][: min(5, len(v["anchors"]))]],
            }
            for k, v in summary["new_classes"].items()
        }
        print(f"[Anchor Tree Step {stage}] {compact}")
        encoder, projector, classifier = train_one_stage(module, args, stage, state, device)
        history = evaluate_stage(module, encoder, projector, classifier, state, args, stage, history, device)
        with open(Path(args.save_dir) / "anchor_tree_mlp_metrics.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final = history[str(args.num_increments)]
    print(
        f"[Final] Total={final['total_acc']:.2f} Old={final['old_acc']:.2f} "
        f"New={final['new_acc']:.2f} Forget={final['forget_rate']:.2f} "
        f"CF={final['cumulative_forget']:.2f}"
    )


if __name__ == "__main__":
    main()
