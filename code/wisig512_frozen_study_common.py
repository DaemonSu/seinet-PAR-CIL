"""Shared audited protocol access; training features remain current-stage RAM only."""
import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Study:
    def __init__(self, code, source, output):
        self.code, self.source, self.out = map(lambda x: Path(x).resolve(), (code, source, output))
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError('Fresh output required')
        self.out.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(self.code))
        self.cfg = json.loads((self.source / 'args.json').read_text(encoding='utf-8-sig'))
        if (self.cfg['initial_classes'], self.cfg['increment_size'], self.cfg['num_increments'],
                self.cfg.get('input_segment_len')) != (50, 5, 10, 512):
            raise ValueError('Not the locked WiSig512 protocol')
        self.data = Path(self.cfg['data_root']).resolve()
        audit = json.loads((self.data / 'audit_summary.json').read_text())['wisig']
        if not audit['development_training_admitted'] or not audit['source_record_passed']:
            raise RuntimeError('Source audit failed')
        self.trainer = module('locked_study_trainer', self.code / '20_train_anchor_tree_mlp_noreplay.py')
        self.api = self.trainer.load_support_module()
        self.metrics = module('locked_study_metrics', self.code / '79_validate_prototype_decisions.py')
        self.args = SimpleNamespace(**self.cfg)
        self.device = torch.device(self.cfg['device'])
        self.encoder = torch.load(self.source / 'encoder.pth', map_location=self.device,
                                  weights_only=False).eval().requires_grad_(False)
        self.encoder_state = {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
        self.hashes = {f: digest(self.source / f) for f in ('encoder.pth', 'classifier.pth', 'args.json')}
        self.ledger = []
        torch.manual_seed(self.cfg['seed'])
        np.random.seed(self.cfg['seed'])

    def dataset(self, stage, use):
        if not 0 <= stage <= 10 or use not in ('train', 'val'):
            raise ValueError('Forbidden dataset access')
        name = f'{use}-closed' if stage == 0 else f'{use}-add{stage}'
        labels = list(range(50 + stage * 5)) if use == 'val' or stage == 0 else list(range(45 + stage * 5, 50 + stage * 5))
        ds = self.api.NpyDeviceDataset(str(self.data / name), allowed_labels=labels, max_per_class=0)
        counts = np.bincount([y for _, y in ds.samples], minlength=100)
        if any(counts[c] != (200 if use == 'train' else 50) for c in labels) or counts.sum() != len(labels) * (200 if use == 'train' else 50):
            raise RuntimeError('Quota mismatch')
        self.ledger.append({'stage': stage, 'use': use, 'folder': name, 'labels': labels})
        return ds

    @torch.no_grad()
    def features(self, stage, use):
        self.encoder.eval()
        xs, ys = [], []
        for x, y in DataLoader(self.dataset(stage, use), batch_size=256):
            xs.append(self.encoder(x.to(self.device)).cpu())
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    def unchanged(self):
        if any(digest(self.source / f) != h for f, h in self.hashes.items()):
            raise RuntimeError('Source checkpoints changed')
        if any(not torch.equal(self.encoder_state[k], v.cpu()) for k, v in self.encoder.state_dict().items()):
            raise RuntimeError('Frozen E1 or BN changed')

    def row(self, stage, y, pred):
        y, pred = np.asarray(y), np.asarray(pred)
        np.savez_compressed(self.out / f'validation_predictions_step{stage}.npz', target=y, prediction=pred)
        return {'stage': stage, 'classes': 50 + stage * 5, 'metrics':
                self.metrics.decision_metrics(y, pred, 45 + stage * 5 if stage else 0, 50)}

    def report(self, extra):
        self.unchanged()
        return dict(dataset='wisig', seed=self.cfg['seed'], protocol='50+5x10', split='validation',
                    development_only=True, independent_confirmation=False, test_data_read=False,
                    per_sample_training_features_exported=False, read_ledger=self.ledger,
                    initial_checkpoint_sha256=self.hashes, e1_parameters_buffers_unchanged=True, **extra)
