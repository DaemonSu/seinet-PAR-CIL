"""Reuse the existing trainer; change only support-center aggregation."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Avoid duplicate Intel OpenMP runtimes when NumPy and torch share a process.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from importlib import util

JOINT_LOSS_OVERRIDES = {"anchor_ce_weight": 0.0, "anchor_margin_weight": 0.0, "distill_weight": 0.0}


def import_file(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def evaluate(module, encoder, e2, head, state, args, stage, helpers, validation):
    folder = Path(args.data_root) / ("val-closed" if stage == 0 else f"val-add{stage}")
    order, supports, _, m, _ = helpers.bank(state, torch.device(args.device))
    ds = module.NpyDeviceDataset(str(folder), allowed_labels=order, max_per_class=0)
    encoder.eval()
    e2.eval()
    head.eval()
    lookup = {c: i for i, c in enumerate(order)}
    ps = F.normalize(helpers.apply(e2, supports), dim=1)
    targets = []
    predictions = {k: [] for k in ("mlp", "e1_support", "e2_support")}
    for x, labels in DataLoader(ds, batch_size=args.batch_size, shuffle=False):
        h = F.normalize(encoder(x.to(args.device)), dim=1)
        z = F.normalize(e2(h), dim=1)
        targets.extend(lookup[int(c)] for c in labels)
        predictions["mlp"].append(helpers.cpu(head(z).argmax(1)))
        predictions["e1_support"].append(helpers.cpu(helpers.support_scores(h, supports, len(order), m).argmax(1)))
        predictions["e2_support"].append(helpers.cpu(helpers.support_scores(z, ps, len(order), m).argmax(1)))
    y = np.asarray(targets, dtype=np.int64)
    if not np.all(np.bincount(y, minlength=len(order)) == 50):
        raise RuntimeError("Expected 50 validation samples per device")
    preds = {k: np.concatenate(v) for k, v in predictions.items()}
    np.savez_compressed(Path(args.save_dir) / f"validation_predictions_step{stage}.npz", target=y, **preds)
    old_count = len(order) - args.increment_size if stage else 0
    return {"stage": stage, "classes": len(order), "methods": {
        k: validation.decision_metrics(y, pred, old_count, args.initial_classes) for k, pred in preds.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_run", required=True)
    p.add_argument("--code_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--decision_summary", required=True)
    p.add_argument("--strategy", required=True, choices=("coverage_means", "hybrid_tail_means", "original"))
    p.add_argument("--dataset", default="adsb", choices=("adsb", "wisig"))
    p.add_argument("--ablation", default="none", choices=("none", "single_neighbor", "no_residual", "no_anchor_no_kd"))
    p.add_argument("--result_prefix", default="coverage_validation")
    a = p.parse_args()
    decision = json.loads(Path(a.decision_summary).read_text(encoding="utf-8"))
    if not decision["phase1_passed"]:
        raise RuntimeError("Phase1 validation gate did not pass")
    code = Path(a.code_root).resolve()
    source = Path(a.source_run).resolve()
    output = Path(a.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Pilot requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code))
    trainer = import_file("coverage_original_training", code / "20_train_anchor_tree_mlp_noreplay.py")
    helpers = import_file("coverage_helpers", code / "75_diagnose_prototype_boundary_bias.py")
    validation = import_file("coverage_validation", code / "79_validate_prototype_decisions.py")
    aggregation = import_file("coverage_aggregation", code / "83_aggregate_coverage_centers.py")
    config = json.loads((source / "args.json").read_text(encoding="utf-8-sig"))
    if (config["initial_classes"], config["increment_size"], config["num_increments"], config["centers_per_class"]) != (50, 5, 10, 32) or config["seed"] not in (42, 43, 44):
        raise ValueError("Training must match the fixed protocol, seeds, and budget")
    if a.strategy != "original" and (config["seed"] != 42 or a.dataset != "adsb" or a.ablation != "none"):
        raise ValueError("Coverage variants remain ADS-B seed42-only pilots")
    if a.strategy == "original" and a.ablation == "none":
        raise ValueError("Original-center training requires an explicit ablation")
    if a.ablation == "single_neighbor":
        config["anchor_topk"] = 1
    elif a.ablation == "no_residual":
        config["projector_alpha"] = 0.0
    elif a.ablation == "no_anchor_no_kd":
        config.update(JOINT_LOSS_OVERRIDES)
    if Path(a.result_prefix).name != a.result_prefix or not a.result_prefix:
        raise ValueError("Result prefix must be a filename stem")
    config.update(init_dir=str(source), save_dir=str(output),
                  protocol_note=f"{a.dataset}; strategy={a.strategy}; ablation={a.ablation}; validation only; no old sample-feature replay")
    args = SimpleNamespace(**config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    module = trainer.load_support_module()
    original_dataset = module.NpyDeviceDataset
    context = {"stage": 0}
    read_ledger = []

    def guarded_dataset(folder, allowed_labels=None, **kwargs):
        if Path(folder).resolve().parent != Path(args.data_root).resolve():
            raise RuntimeError("Dataset outside the fixed protocol root")
        name = Path(folder).name
        stage = context["stage"]
        expected_train = "train-closed" if stage == 0 else f"train-add{stage}"
        expected_val = "val-closed" if stage == 0 else f"val-add{stage}"
        if name not in (expected_train, expected_val):
            raise RuntimeError(f"Forbidden dataset access: {folder}")
        allowed = set(map(int, allowed_labels))
        expected = set(trainer.stage_seen_labels(args, 0) if stage == 0 else trainer.stage_new_labels(args, stage))
        if name == expected_train and allowed != expected:
            raise RuntimeError("Training labels must be exactly the current device cohort")
        read_ledger.append({"stage": stage, "folder": str(folder), "labels": sorted(allowed), "use": "training" if name == expected_train else "offline_validation"})
        return original_dataset(folder, allowed_labels=allowed_labels, **kwargs)

    module.NpyDeviceDataset = guarded_dataset
    original_centers = trainer.support_centers
    group_sizes = []

    def centers(features, count):
        if a.strategy == "original":
            return original_centers(features, count)
        result, sizes = aggregation.make_centers(features, count, a.strategy, original_centers)
        group_sizes.append({"stage": context["stage"], "aggregate_group_sizes": sizes})
        return result

    trainer.support_centers = centers
    init_paths = [source / "encoder.pth", source / "classifier.pth"]
    hashes = {str(f): helpers.file_hash(f) for f in init_paths}
    encoder = helpers.load(init_paths[0], args.device).eval().requires_grad_(False)
    head = helpers.load(init_paths[1], args.device).eval().requires_grad_(False)
    initial_encoder = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    torch.save(encoder, output / "encoder.pth")
    torch.save(head, output / "classifier.pth")
    (output / "args.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps({
        "strategy": a.strategy, "ablation": a.ablation, "dataset": a.dataset,
        "source_run": str(source), "centers_per_class": 32,
        "initial_training_data": "train-closed", "incremental_training_data": "current train-add stage only",
        "evaluation_split": "validation", "test_evaluation": False,
        "offline_old_validation_used_for_development": True,
        "old_sample_feature_cache": False, "individual_fps_feature_storage": False,
        "early_stopping": "unchanged training-loss patience rule",
        "source_code_sha256": {f: helpers.file_hash(code / f) for f in (
            "20_train_anchor_tree_mlp_noreplay.py", "15_train_incremental_anchor_domain.py",
            "83_aggregate_coverage_centers.py", "84_train_coverage_pilot.py")}}, indent=2), encoding="utf-8")
    state = trainer.build_step0_state(module, encoder, args, torch.device(args.device))
    state["config"]["support_aggregation"] = a.strategy
    records = []
    for stage in range(args.num_increments + 1):
        context["stage"] = stage
        if stage:
            old = {c: state["support_prototypes"][c].clone() for c in state["label_order"]}
            old_routes = {c: state["prototypes"][c].clone() for c in state["label_order"]}
            trainer.add_increment_nodes(module, encoder, state, args, stage, torch.device(args.device))
            if any(not torch.equal(state["support_prototypes"][c], value) for c, value in old.items()) or any(not torch.equal(state["prototypes"][c], value) for c, value in old_routes.items()):
                raise RuntimeError("Old prototype mutation")
        if any(int(count) != 200 for count in state["counts"].values()):
            raise RuntimeError("Expected exactly 200 registration samples per device")
        trainer.save_state(output, stage, state)
        if stage:
            encoder, e2, head = trainer.train_one_stage(module, args, stage, state, torch.device(args.device))
        else:
            e2 = torch.nn.Identity().to(args.device)
        if any(not torch.equal(initial_encoder[k], v.cpu()) for k, v in encoder.state_dict().items()):
            raise RuntimeError("Frozen E1 parameters/buffers changed")
        row = evaluate(module, encoder, e2, head, state, args, stage, helpers, validation)
        records.append(row)
        (output / f"{a.result_prefix}.partial.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"VALIDATION Step{stage}: " + " ".join(f"{k}={v['accuracy_pct']:.2f}" for k, v in row["methods"].items()), flush=True)
        del e2
    if not all(helpers.file_hash(f) == value for f, value in hashes.items()):
        raise RuntimeError("Source Step0 mutation")
    result = {"strategy": a.strategy, "ablation": a.ablation, "seed": args.seed, "dataset": a.dataset, "records": records,
              "read_ledger": read_ledger, "group_sizes": group_sizes, "test_data_read": False,
              "old_training_data_read_during_increment": False, "feature_embeddings_exported": False,
              "centers_per_class": 32, "source_checkpoint_sha256": hashes,
              "e1_parameters_buffers_unchanged": True, "source_checkpoints_unchanged": True,
              "mean_incremental_e2_support_accuracy_pct": float(np.mean([r["methods"]["e2_support"]["accuracy_pct"] for r in records[1:]]))}
    (output / f"{a.result_prefix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE controlled training; no test evaluation", flush=True)


if __name__ == "__main__":
    main()
