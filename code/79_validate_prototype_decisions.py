"""Fixed-rule validation only. No test/training reads or feature exports."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_spec = importlib.util.spec_from_file_location("diagnostic_helpers", Path(__file__).with_name("75_diagnose_prototype_boundary_bias.py"))
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)


def decision_metrics(target, prediction, old_count, base_count):
    correct = prediction == target
    base = target < base_count
    added = ~base
    classes = sorted(set(target.tolist()))
    per_class = [{"class_index": int(c), "n": int((target == c).sum()),
                  "accuracy_pct": d.pct(correct[target == c])} for c in classes]
    result = {"accuracy_pct": d.pct(correct), "base_accuracy_pct": d.pct(correct[base]),
              "added_accuracy_pct": d.pct(correct[added]), "per_class": per_class,
              "worst_device_accuracy_pct": min(x["accuracy_pct"] for x in per_class)}
    if old_count:
        result.update(d.error_groups(target, prediction, old_count))
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--code_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stages", default="0,1,2,3,4,5,6,7,8,9,10")
    a = p.parse_args()
    root, code, output = map(Path, (a.run_dir, a.code_root, a.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "validation.json").exists():
        raise FileExistsError("Refusing to overwrite completed validation")
    config = json.loads((root / "args.json").read_text(encoding="utf-8-sig"))
    stages = sorted(set(map(int, a.stages.split(","))))
    final = int(config["num_increments"])
    if min(stages) < 0 or max(stages) > final:
        raise ValueError("Invalid stages")
    device = torch.device(a.device)
    sys.path.insert(0, str(code.resolve()))
    spec = importlib.util.spec_from_file_location("validation_training_api", code / "20_train_anchor_tree_mlp_noreplay.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    support_api = api.load_support_module()
    checkpoint_paths = {root / "encoder.pth", root / f"anchor_tree_state_step{final}.pt"}
    for t in stages:
        checkpoint_paths.add(root / ("classifier.pth" if t == 0 else f"classifier_step{t}.pth"))
        checkpoint_paths.add(root / f"anchor_tree_state_step{t}.pt")
        if t:
            checkpoint_paths.update((root / f"encoder_step{t}.pth", root / f"contrastive_step{t}.pth"))
    hashes = {str(f): d.file_hash(f) for f in sorted(checkpoint_paths)}
    encoder = d.load(root / "encoder.pth", device).eval().requires_grad_(False)
    encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    final_state = d.load(root / f"anchor_tree_state_step{final}.pt", device)
    final_order, final_supports, final_routes, centers, _ = d.bank(final_state, device)
    data_root = Path(config["data_root"])
    final_ds = support_api.NpyDeviceDataset(str(data_root / f"val-add{final}"), allowed_labels=final_order, max_per_class=0)
    manifest = {d.sample_key(f, c): d.file_hash(f) for f, c in final_ds.samples}
    if len(manifest) != len(final_ds.samples):
        raise RuntimeError("Duplicate validation sample identities")
    feats, labels = [], []
    for x, y in DataLoader(final_ds, batch_size=128, shuffle=False):
        feats.append(F.normalize(encoder(x.to(device)), dim=1))
        labels.append(y.to(device))
    all_h, all_y = torch.cat(feats), torch.cat(labels)
    base_count = int(config["initial_classes"])
    rows = []
    for t in stages:
        if t:
            stage_encoder = d.load(root / f"encoder_step{t}.pth", "cpu")
            if any(not torch.equal(encoder_state[k], v) for k, v in stage_encoder.state_dict().items()):
                raise RuntimeError(f"E1 changed at stage {t}")
            del stage_encoder
        state = d.load(root / f"anchor_tree_state_step{t}.pt", device)
        order, supports, routes, m, _ = d.bank(state, device)
        nc = len(order)
        if order != final_order[:nc] or m != centers or not torch.equal(supports, final_supports[:len(supports)]) or not torch.equal(routes, final_routes[:nc]):
            raise RuntimeError("Previously registered bank mutated")
        folder = data_root / ("val-closed" if t == 0 else f"val-add{t}")
        ds = support_api.NpyDeviceDataset(str(folder), allowed_labels=order, max_per_class=0)
        d.validate_test_folder(ds, manifest, set(order))
        mask = torch.isin(all_y, torch.tensor(order, device=device))
        h, raw_y = all_h[mask], all_y[mask]
        lookup = {c: i for i, c in enumerate(order)}
        target = np.array([lookup[int(v)] for v in d.cpu(raw_y)], dtype=np.int64)
        counts = np.bincount(target, minlength=nc)
        if not np.all(counts == 50):
            raise RuntimeError(f"Expected 50 validation samples/device, got {counts.tolist()}")
        e2 = torch.nn.Identity().to(device) if t == 0 else d.load(root / f"contrastive_step{t}.pth", device).eval().requires_grad_(False)
        head = d.load(root / ("classifier.pth" if t == 0 else f"classifier_step{t}.pth"), device).eval().requires_grad_(False)
        z = F.normalize(d.apply(e2, h), dim=1)
        ps = F.normalize(d.apply(e2, supports), dim=1)
        predictions = {"mlp": d.cpu(d.apply(head, z).argmax(1)),
                       "e1_support": d.cpu(d.support_scores(h, supports, nc, m).argmax(1)),
                       "e2_support": d.cpu(d.support_scores(z, ps, nc, m).argmax(1))}
        old_count = nc - int(config["increment_size"]) if t else 0
        row = {"stage": t, "classes": nc, "samples": len(target), "label_order": order,
               "methods": {name: decision_metrics(target, pred, old_count, base_count) for name, pred in predictions.items()}}
        rows.append(row)
        np.savez_compressed(output / f"predictions_step{t}.npz", target=target,
                            class_label=d.cpu(raw_y), **predictions)
        (output / "validation.partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Step{t}: " + " ".join(f"{k}={v['accuracy_pct']:.2f}" for k, v in row["methods"].items()), flush=True)
        del e2, head
    if not all(d.file_hash(f) == value for f, value in hashes.items()):
        raise RuntimeError("Input checkpoint changed")
    report = {"dataset": a.dataset, "seed": a.seed, "records": rows,
              "split": "validation", "training_data_read": False, "test_data_read": False,
              "model_updated": False, "feature_embeddings_exported": False,
              "checkpoint_sha256": hashes, "checkpoint_hashes_unchanged": True,
              "e1_parameters_buffers_unchanged": True, "old_bank_unchanged": True,
              "cumulative_validation_contents_verified": True,
              "validation_use": "offline development evaluation; not incremental training inputs",
              "source_record_leakage_audited": False}
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE validation; checkpoints unchanged", flush=True)


if __name__ == "__main__":
    main()
