"""Post-hoc test diagnostics. No fitting, optimizer, or feature-cache export."""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "mean": float(v.mean()),
            **{f"q{q}": float(np.percentile(v, q)) for q in (5, 25, 50, 75, 95)}}


def cpu(x):
    return x.detach().cpu().numpy()


def pct(x):
    a = np.asarray(x)
    return float(a.mean() * 100) if a.size else None


def class_rank_bins(distances, labels, errors):
    """Within-class bins reduce confounding by intrinsically harder devices."""
    bins = np.zeros(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        ids = np.flatnonzero(labels == c)
        ids = ids[np.argsort(distances[ids], kind="stable")]
        bins[ids] = np.arange(len(ids)) * 4 // len(ids)
    return [{"quartile": b + 1, "n": int((bins == b).sum()),
             "mlp_error_pct": pct(errors[bins == b]),
             "own_distance": summary(distances[bins == b])} for b in range(4)]


def error_groups(target, prediction, old_count):
    old = target < old_count
    new = ~old
    error = prediction != target
    return {"old_accuracy_pct": pct(~error[old]), "new_accuracy_pct": pct(~error[new]),
            "old_to_new_pct_of_old_samples": pct(prediction[old] >= old_count),
            "new_to_old_pct_of_new_samples": pct(prediction[new] < old_count),
            "old_to_old_wrong_pct_of_old_samples": pct(error[old] & (prediction[old] < old_count)),
            "old_to_new_pct_of_old_errors": pct(prediction[old & error] >= old_count),
            "counts": {"old": int(old.sum()), "new": int(new.sum()),
                       "old_errors": int((old & error).sum()), "new_errors": int((new & error).sum())}}


def positive_margin(scores, target):
    own = scores.gather(1, target[:, None]).squeeze(1)
    rivals = scores.clone()
    rivals.scatter_(1, target[:, None], -float("inf"))
    return own - rivals.max(dim=1).values


def flip_stats(before, after, target):
    good_before = before == target
    good_after = after == target
    return {"before_accuracy_pct": pct(good_before), "after_accuracy_pct": pct(good_after),
            "harmed_pct": pct(good_before & ~good_after),
            "rescued_pct": pct(~good_before & good_after),
            "prediction_changed_pct": pct(before != after)}


def apply(model, x, batch_size=512):
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def bank(state, device):
    order = [int(c) for c in state["label_order"]]
    support_map = state["support_prototypes"]
    route_map = state["prototypes"]
    parts = [support_map.get(c, support_map.get(str(c))).float() for c in order]
    counts = [len(p) for p in parts]
    if len(set(counts)) != 1:
        raise ValueError("This diagnostic expects equal support counts per class")
    supports = F.normalize(torch.cat(parts).to(device), dim=1)
    routes = F.normalize(torch.stack([route_map.get(c, route_map.get(str(c))) for c in order]).to(device), dim=1)
    radii = np.asarray([state["radii"].get(c, state["radii"].get(str(c))) for c in order])
    return order, supports, routes, counts[0], radii


def support_scores(features, supports, nclasses, centers, batch_size=512):
    results = []
    for start in range(0, len(features), batch_size):
        sims = features[start:start + batch_size] @ supports.T
        results.append(sims.reshape(-1, nclasses, centers).amax(dim=2))
    return torch.cat(results)


def sample_key(path, label):
    return (int(label), Path(path).name)


def validate_test_folder(dataset, reference, expected_classes):
    seen = set()
    for path, label in dataset.samples:
        key = sample_key(path, label)
        if key in seen or key not in reference or file_hash(path) != reference[key]:
            raise RuntimeError(f"Cumulative test identity/content mismatch: {path}")
        seen.add(key)
    expected = {key for key in reference if key[0] in expected_classes}
    if seen != expected:
        raise RuntimeError("Stage test is not exactly the enrolled-class subset of the final test")


def model_stats(previous, current, old_count):
    prev = [m for m in previous.modules() if isinstance(m, torch.nn.Linear)]
    curr = [m for m in current.modules() if isinstance(m, torch.nn.Linear)]
    old_w = curr[-1].weight[:old_count]
    new_w = curr[-1].weight[old_count:]
    result = {"old_output_weight_norm": summary(cpu(old_w.norm(dim=1))),
              "new_output_weight_norm": summary(cpu(new_w.norm(dim=1))),
              "old_output_rows_max_abs_change": float((old_w - prev[-1].weight).abs().max()),
              "shared_first_layer_relative_weight_change": float(
                  (curr[0].weight - prev[0].weight).norm() / prev[0].weight.norm().clamp_min(1e-12))}
    if curr[-1].bias is not None:
        result.update(old_output_bias=summary(cpu(curr[-1].bias[:old_count])),
                      new_output_bias=summary(cpu(curr[-1].bias[old_count:])),
                      old_output_bias_max_abs_change=float(
                          (curr[-1].bias[:old_count] - prev[-1].bias).abs().max()))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--code_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stages", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, code, output = map(Path, (args.run_dir, args.code_root, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code.resolve()))
    spec = importlib.util.spec_from_file_location("diagnostic_training_api", code / "20_train_anchor_tree_mlp_noreplay.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    api = training.load_support_module()
    config = json.loads((root / "args.json").read_text(encoding="utf-8-sig"))
    metrics = json.loads((root / "anchor_tree_mlp_metrics.json").read_text(encoding="utf-8-sig"))
    stages = sorted({int(s) for s in args.stages.split(",")})
    final = int(config["num_increments"])
    if not stages or min(stages) < 1 or max(stages) > final:
        raise ValueError("Invalid stage list")
    device = torch.device(args.device)
    checkpoint_paths = {root / "encoder.pth", root / "classifier.pth", root / f"anchor_tree_state_step{final}.pt"}
    for t in stages:
        checkpoint_paths.update(root / f"{name}_step{t}{ext}" for name, ext in
                                (("encoder", ".pth"), ("classifier", ".pth"), ("contrastive", ".pth"), ("anchor_tree_state", ".pt")))
        if t > 1:
            checkpoint_paths.update((root / f"classifier_step{t-1}.pth", root / f"contrastive_step{t-1}.pth"))
    hashes_before = {str(p): file_hash(p) for p in sorted(checkpoint_paths)}
    encoder = load(root / "encoder.pth", device).eval().requires_grad_(False)
    base_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    final_state = load(root / f"anchor_tree_state_step{final}.pt", device)
    final_order, final_supports, final_routes, centers, _ = bank(final_state, device)
    data = Path(config["data_root"])
    final_folder = data / config["test_add_template"].format(step=final)
    dataset = api.NpyDeviceDataset(str(final_folder), allowed_labels=final_order, max_per_class=config["max_test_per_class"])
    keys = [sample_key(path, label) for path, label in dataset.samples]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Nonunique final sample IDs")
    reference = {sample_key(path, label): file_hash(path) for path, label in dataset.samples}
    manifest_digest = hashlib.sha256(json.dumps(sorted((list(k), v) for k, v in reference.items())).encode()).hexdigest()
    features, labels = [], []
    for x, y in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
        features.append(F.normalize(encoder(x.to(device)), dim=1))
        labels.append(y.to(device))
    h_all, y_all = torch.cat(features), torch.cat(labels)
    records = []
    for t in stages:
        check_encoder = load(root / f"encoder_step{t}.pth", "cpu")
        if any(not torch.equal(base_encoder_state[k], v) for k, v in check_encoder.state_dict().items()):
            raise RuntimeError(f"Encoder parameters/buffers changed at Step{t}; cached-in-memory E1 pass is invalid")
        del check_encoder
        state = load(root / f"anchor_tree_state_step{t}.pt", device)
        order, supports, routes, count, radii = bank(state, device)
        nc = len(order)
        old_count = nc - int(config["increment_size"])
        if order != final_order[:nc] or count != centers or not torch.equal(supports, final_supports[:len(supports)]) or not torch.equal(routes, final_routes[:nc]):
            raise RuntimeError(f"Prototype state/order changed retrospectively at Step{t}")
        stage_folder = data / config["test_add_template"].format(step=t)
        stage_dataset = api.NpyDeviceDataset(str(stage_folder), allowed_labels=order, max_per_class=config["max_test_per_class"])
        validate_test_folder(stage_dataset, reference, set(order))
        mask = torch.isin(y_all, torch.tensor(order, device=device))
        h, raw_y = h_all[mask], y_all[mask]
        lookup = {c: i for i, c in enumerate(order)}
        target = torch.tensor([lookup[int(v)] for v in cpu(raw_y)], device=device)
        y = cpu(target)
        old_mask = target < old_count
        old_np = cpu(old_mask)
        e2 = load(root / f"contrastive_step{t}.pth", device).eval().requires_grad_(False)
        head = load(root / f"classifier_step{t}.pth", device).eval().requires_grad_(False)
        prev_e2 = torch.nn.Identity().to(device) if t == 1 else load(root / f"contrastive_step{t-1}.pth", device).eval().requires_grad_(False)
        prev_head = load(root / ("classifier.pth" if t == 1 else f"classifier_step{t-1}.pth"), device).eval().requires_grad_(False)
        z = F.normalize(apply(e2, h), dim=1)
        ps = F.normalize(apply(e2, supports), dim=1)
        pr = F.normalize(apply(e2, routes), dim=1)
        logits = apply(head, z)
        pred = cpu(logits.argmax(dim=1))
        error = pred != y
        observed = pct(~error)
        expected = float(metrics[str(t)]["total_acc"])
        if abs(observed - expected) > 0.05:
            raise RuntimeError(f"Evaluation mismatch Step{t}: {observed} vs archived {expected}")
        e1scores = support_scores(h, supports, nc, count)
        e2scores = support_scores(z, ps, nc, count)
        e1route = h @ routes.T
        e2route = z @ pr.T
        d_own = cpu(1 - e1scores.gather(1, target[:, None]).squeeze(1)).clip(0)
        d_route = cpu(1 - e1route.gather(1, target[:, None]).squeeze(1)).clip(0)
        margin1 = cpu(positive_margin(e1scores, target))
        margin2 = cpu(positive_margin(e2scores, target))
        e1pred, e2pred = cpu(e1scores.argmax(dim=1)), cpu(e2scores.argmax(dim=1))
        coverage = {"nearest_own_support_distance_correct": summary(d_own[~error]),
                    "nearest_own_support_distance_error": summary(d_own[error]),
                    "within_class_distance_quartiles": class_rank_bins(d_own, y, error),
                    "test_beyond_registration_route_q95_pct": pct(d_route > radii[y]),
                    "beyond_route_radius_error_pct": pct(error[d_route > radii[y]]),
                    "inside_route_radius_error_pct": pct(error[d_route <= radii[y]]),
                    "e1_support_accuracy_pct": pct(e1pred == y),
                    "e2_support_accuracy_pct": pct(e2pred == y),
                    "e1_route_accuracy_pct": pct(cpu(e1route.argmax(dim=1)) == y),
                    "e2_route_accuracy_pct": pct(cpu(e2route.argmax(dim=1)) == y),
                    "e1_support_correct_mlp_wrong_pct": pct((e1pred == y) & error),
                    "e2_support_correct_mlp_wrong_pct": pct((e2pred == y) & error),
                    "e2_support_wrong_mlp_correct_pct": pct((e2pred != y) & ~error),
                    "all_errors_e2_support_also_wrong_pct": pct(e2pred[error] != y[error]),
                    "e1_margin_correct": summary(margin1[~error]), "e1_margin_error": summary(margin1[error])}
        z_prev = F.normalize(apply(prev_e2, h[old_mask]), dim=1)
        ps_prev = F.normalize(apply(prev_e2, supports[:old_count * count]), dim=1)
        scores_prev = support_scores(z_prev, ps_prev, old_count, count)
        scores_old_now = e2scores[old_mask, :old_count]
        old_y = y[old_np]
        prev_margin = cpu(positive_margin(scores_prev, target[old_mask]))
        curr_margin = cpu(positive_margin(scores_old_now, target[old_mask]))
        boundary = {"old_only_previous_to_current_e2": flip_stats(cpu(scores_prev.argmax(1)), cpu(scores_old_now.argmax(1)), old_y),
                    "old_margin_change": summary(curr_margin - prev_margin),
                    "old_margin_before": summary(prev_margin), "old_margin_after": summary(curr_margin),
                    "e1_to_e2_all_seen": flip_stats(e1pred, e2pred, y),
                    "new_competitor_harms_old_pct": pct((cpu(scores_old_now.argmax(1)) == old_y) & (e2pred[old_np] != old_y)),
                    "old_sample_adapter_drift": summary(cpu(1 - (z_prev * z[old_mask]).sum(1)))}
        old_logits = logits[old_mask, :old_count]
        prev_logits = apply(prev_head, z_prev)
        grid = {"previous_e2_previous_head": pct(cpu(prev_logits.argmax(1)) == old_y),
                "current_e2_previous_head": pct(cpu(apply(prev_head, z[old_mask]).argmax(1)) == old_y),
                "previous_e2_current_head_old_only": pct(cpu(apply(head, z_prev)[:, :old_count].argmax(1)) == old_y),
                "current_e2_current_head_old_only": pct(cpu(old_logits.argmax(1)) == old_y)}
        group_score = cpu(torch.logsumexp(logits[:, old_count:], dim=1) - math.log(nc-old_count)
                          - torch.logsumexp(logits[:, :old_count], dim=1) + math.log(old_count))
        prev_pred = cpu(prev_logits.argmax(1))
        mlp = {**error_groups(y, pred, old_count), "counterfactual_old_accuracy_pct": grid,
               "old_only_teacher_to_student": flip_stats(prev_pred, cpu(old_logits.argmax(1)), old_y),
               "teacher_correct_student_wrong_pct_of_old": pct((prev_pred == old_y) & error[old_np]),
               "teacher_correct_student_wrong_to_new_pct_of_old": pct((prev_pred == old_y) & error[old_np] & (pred[old_np] >= old_count)),
               "new_minus_old_logmeanexp_old_samples": summary(group_score[old_np]),
               "new_minus_old_logmeanexp_new_samples": summary(group_score[~old_np]),
               "parameters": model_stats(prev_head, head, old_count)}
        per_class = [{"class": c, "group": "old" if ci < old_count else "new", "n": int((y == ci).sum()),
                      "mlp_accuracy_pct": pct(~error[y == ci]), "e1_support_accuracy_pct": pct(e1pred[y == ci] == ci),
                      "e2_support_accuracy_pct": pct(e2pred[y == ci] == ci),
                      "own_distance": summary(d_own[y == ci]), "e1_margin": summary(margin1[y == ci]),
                      "e2_margin": summary(margin2[y == ci])} for ci, c in enumerate(order)]
        support_target = torch.arange(nc, device=device).repeat_interleave(count)
        support_accuracy = pct(cpu(apply(head, ps).argmax(1) == support_target))
        row = {"stage": t, "samples": len(y), "classes": nc, "old_classes": old_count,
               "mlp_accuracy_pct": observed, "archived_mlp_accuracy_pct": expected,
               "retained_support_mlp_accuracy_pct": support_accuracy,
               "coverage": coverage, "boundary": boundary, "mlp_bias": mlp, "per_class": per_class}
        records.append(row)
        np.savez_compressed(output / f"samples_step{t}.npz", class_label=cpu(raw_y), target=y, mlp_prediction=pred,
                            nearest_support_e1=e1pred, nearest_support_e2=e2pred, own_e1_support_distance=d_own,
                            e1_support_margin=margin1, e2_support_margin=margin2,
                            new_minus_old_logmeanexp=group_score, old_mask=old_np,
                            previous_old_support_margin=prev_margin, current_old_support_margin=curr_margin)
        print(f"[Step{t}] MLP={observed:.2f} supports={support_accuracy:.2f} "
              f"old_to_new={mlp['old_to_new_pct_of_old_samples']:.2f} "
              f"E2_old_harmed={boundary['old_only_previous_to_current_e2']['harmed_pct']:.2f}", flush=True)
        (output / "diagnostics.partial.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        del e2, head, prev_e2, prev_head
    unchanged = all(file_hash(Path(p)) == value for p, value in hashes_before.items())
    if not unchanged:
        raise RuntimeError("Input checkpoint mutation detected")
    report = {"dataset": args.dataset, "seed": args.seed, "run_dir": str(root),
              "evaluation_only": True, "training_data_read": False, "feature_embeddings_exported": False,
              "checkpoint_hashes_unchanged": unchanged, "checkpoint_sha256": hashes_before,
              "cumulative_test_content_verified": True, "final_test_manifest_digest": manifest_digest,
              "sample_manifest": [{"label": c, "file": name} for c, name in keys], "records": records,
              "limitations": ["Post-hoc held-out test association is not causal evidence.",
                              "Support accuracy is measured on retained training summaries.",
                              "Counterfactual E2/head swaps can introduce representation mismatch.",
                              "No fitted bias correction, threshold selection, or model update was performed."]}
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"COMPLETE {args.dataset} seed={args.seed}; inputs unchanged", flush=True)


if __name__ == "__main__":
    main()
