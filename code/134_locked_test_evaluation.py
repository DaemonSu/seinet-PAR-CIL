"""Read-only cumulative test evaluation authorized by a pre-evaluation hash lock."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from wisig512_frozen_study_common import module, digest, Study
from wisig512_official_mechanisms import acil, fecam, normalized_mahalanobis


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class ProtocolStudy(Study):
    """Generalize the existing mechanism runner to the audited ADS-B protocol."""
    def __init__(self, code, source, output):
        self.code, self.source, self.out = [Path(x).resolve() for x in (code, source, output)]
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError('Fresh output required')
        self.out.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(self.code))
        self.cfg = read(self.source/'args.json')
        if tuple(self.cfg[k] for k in ('initial_classes','increment_size','num_increments')) != (50,5,10):
            raise ValueError('Not the locked class schedule')
        self.data = Path(self.cfg['data_root']).resolve()
        self.dataset_name = 'wisig' if self.cfg.get('input_segment_len') == 512 else 'adsb'
        if self.dataset_name == 'wisig':
            audit = read(self.data/'audit_summary.json')['wisig']
        else:
            audit = read(self.code/'locked_adsb_source_audit.json')['adsb']
        if not audit['development_training_admitted'] or not audit['source_record_passed']:
            raise RuntimeError('Source audit failed')
        if Path(audit['data_root']).resolve() != self.data:
            raise RuntimeError('Audit/data root mismatch')
        self.trainer = module('protocol_study_trainer', self.code/'20_train_anchor_tree_mlp_noreplay.py')
        self.api = self.trainer.load_support_module()
        self.metrics = module('protocol_study_metrics', self.code/'79_validate_prototype_decisions.py')
        self.args = SimpleNamespace(**self.cfg)
        self.device = torch.device(self.cfg['device'])
        self.encoder = torch.load(self.source/'encoder.pth', map_location=self.device,
                                  weights_only=False).eval().requires_grad_(False)
        self.encoder_state = {k:v.cpu().clone() for k,v in self.encoder.state_dict().items()}
        self.hashes = {f:digest(self.source/f) for f in ('encoder.pth','classifier.pth','args.json')}
        self.ledger = []
        torch.manual_seed(self.cfg['seed']); np.random.seed(self.cfg['seed'])

    def report(self, extra):
        result = super().report(extra)
        result['dataset'] = self.dataset_name
        return result


def model_paths(run, kind):
    run = Path(run)
    paths = [run/'args.json']
    if kind == 'core':
        paths += [run/'encoder.pth']
        paths += [run/f'anchor_tree_state_step{t}.pt' for t in range(11)]
        paths += [run/f'contrastive_step{t}.pth' for t in range(1,11)]
    else:
        paths += [run/'baseline_validation.json']
        paths += [run/f'model_step{t}.pt' for t in range(11)]
    return paths


def make_lock(run, source, kind, output):
    path = Path(output)
    if path.exists():
        raise FileExistsError('Never overwrite a configuration lock')
    paths = model_paths(run, kind) + [Path(source)/f for f in ('encoder.pth','classifier.pth','args.json')]
    data = Path(read(Path(source)/'args.json')['data_root'])
    paths += [p for p in (data/'protocol.json',data/'source_records.json',data/'audit_summary.json') if p.exists()]
    lock = dict(run=str(Path(run).resolve()), source=str(Path(source).resolve()), kind=kind,
                created_before_test_evaluation=True, selection='prespecified core or validation-only baseline selection',
                checkpoint_sha256={str(p.resolve()):digest(p) for p in paths})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2))
    return lock


@torch.inference_mode()
def evaluate(run, source, kind, code, output, lock_path, independent=False):
    run, source, code, out = [Path(p).resolve() for p in (run, source, code, output)]
    lock = read(lock_path)
    if (lock['run'],lock['source'],lock['kind']) != (str(run),str(source),kind):
        raise RuntimeError('Evaluation does not match configuration lock')
    if any(digest(p) != h for p,h in lock['checkpoint_sha256'].items()):
        raise RuntimeError('Locked model changed before test access')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Test outputs cannot be overwritten')
    out.mkdir(parents=True)
    sys.path.insert(0,str(code))
    cfg = read(run/'args.json') if kind == 'core' else read(source/'args.json')
    data = Path(cfg['data_root'])
    encoder = torch.load(source/'encoder.pth', map_location=cfg['device'], weights_only=False).eval().requires_grad_(False)
    trainer = module('test_trainer',code/'20_train_anchor_tree_mlp_noreplay.py')
    api = trainer.load_support_module()
    helpers = module('test_helpers',code/'75_diagnose_prototype_boundary_bias.py')
    metrics = module('test_metrics',code/'79_validate_prototype_decisions.py')
    device = torch.device(cfg['device'])
    records, ledger = [], []
    report = read(run/'baseline_validation.json') if kind != 'core' else None
    for t in range(11):
        nc = 50+5*t
        folder = data/('test-closed' if t == 0 else f'test-add{t}')
        ds = api.NpyDeviceDataset(str(folder), allowed_labels=range(nc), max_per_class=0)
        ys, features = [], []
        for x,y in DataLoader(ds,batch_size=256,shuffle=False):
            features.append(encoder(x.to(device)));ys.append(y)
        h,y = torch.cat(features),torch.cat(ys).numpy()
        if not np.all(np.bincount(y,minlength=nc) == 50):
            raise RuntimeError('Test quota mismatch')
        predictions = {}
        if kind == 'core':
            state = torch.load(run/f'anchor_tree_state_step{t}.pt',map_location=device,weights_only=False)
            order,ps,_,m,_ = helpers.bank(state,device)
            if order != list(range(nc)):
                raise RuntimeError('Unexpected class ordering')
            e2 = torch.nn.Identity().to(device) if t == 0 else torch.load(
                run/f'contrastive_step{t}.pth',map_location=device,weights_only=False).eval().requires_grad_(False)
            h = F.normalize(h,dim=1)
            projected = F.normalize(helpers.apply(e2,ps),dim=1)
            query = F.normalize(helpers.apply(e2,h),dim=1)
            predictions['core'] = helpers.cpu(helpers.support_scores(query,projected,nc,m).argmax(1))
            predictions['same_bank_fixed'] = helpers.cpu(helpers.support_scores(h,ps,nc,m).argmax(1))
        else:
            state = torch.load(run/f'model_step{t}.pt',map_location=device,weights_only=False)
            if kind == 'fetril':
                q = F.normalize(h,dim=1).cpu().numpy().astype(np.float64)
                predictions[kind] = (q@np.asarray(state['weight']).T+np.asarray(state['bias'])).argmax(1)
            elif kind == 'fecam':
                means = state['means'].to(device)
                settings = state['settings']; maha = fecam(code/'official_frozen512')
                maha.args.update(settings.get('author_args',{}))
                query = F.normalize(h,dim=1)
                tukey = settings.get('tukey_enabled',False)
                if tukey and t:
                    query = maha._tukeys_transform(query).to(device)
                scores = []
                for c,mean in enumerate(means):
                    if tukey and t: mean = maha._tukeys_transform(mean).to(device)
                    if not t:
                        score = ((F.normalize(query,dim=1)-F.normalize(mean,dim=0))**2).sum(1)
                    else:
                        inverse = torch.linalg.pinv(state['covariances'][c].to(device)).float()
                        score = normalized_mahalanobis(query,mean,inverse)
                    scores.append(score)
                predictions[kind] = torch.stack(scores,dim=1).argmin(1).cpu().numpy()
            elif kind == 'acil':
                settings = report['settings']
                from wisig512_official_mechanisms import author_class
                root = code/'official_frozen512/ACIL'
                linear = module('test_acil_linear',root/'AnalyticLinear.py')
                buffer = module('test_acil_buffer',root/'Buffer.py')
                cls = author_class(root/'ACIL.py','ACIL',None,dict(torch=torch,RandomBuffer=buffer.RandomBuffer,
                    AnalyticLinear=linear.AnalyticLinear,RecursiveLinear=linear.RecursiveLinear))
                analytic = cls(1024,buffer_size=settings['buffer_size'],gamma=settings['gamma'],device=device,dtype=torch.double)
                # Author classifier grows dynamically; recreate its output dimension before loading.
                analytic.analytic_linear.weight = torch.zeros_like(state['analytic_linear.weight'])
                analytic.load_state_dict(state)
                predictions[kind] = analytic(h).argmax(1).cpu().numpy()
            else: raise ValueError(kind)
        values = {name:metrics.decision_metrics(y,pred, nc-5 if t else 0,50) for name,pred in predictions.items()}
        np.savez_compressed(out/f'test_predictions_step{t}.npz',target=y,**predictions)
        records.append(dict(stage=t,classes=nc,methods=values))
        ledger.append(dict(stage=t,folder=str(folder),use='test_only_no_training'))
        (out/'test.partial.json').write_text(json.dumps(records,indent=2))
        print(f'TEST {run.name} stage{t}: '+str({k:v['accuracy_pct'] for k,v in values.items()}),flush=True)
    if any(digest(p) != h for p,h in lock['checkpoint_sha256'].items()):
        raise RuntimeError('Locked models modified during evaluation')
    result = dict(kind=kind,seed=cfg['seed'],split='test',records=records,read_ledger=ledger,
                  models_updated=False,training_data_read=False,test_data_read=True,
                  independent_source_confirmation=independent,configuration_lock=str(lock_path),
                  lock_sha256=digest(lock_path),historical_development_exposure=not independent)
    (out/'test_results.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ('run','source','kind','code','output','lock'):p.add_argument('--'+k,required=True)
    p.add_argument('--independent',action='store_true')
    a=p.parse_args()
    evaluate(a.run,a.source,a.kind,a.code,a.output,a.lock,a.independent)


if __name__ == '__main__':
    from gpu_study_lock import gpu_study_lock
    with gpu_study_lock():main()
