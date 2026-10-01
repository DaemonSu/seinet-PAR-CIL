"""Final-core controls; unchanged current-only trainer and frozen Step0 encoder."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import torch


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source_run',required=True)
    p.add_argument('--code_root',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--audit_summary',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--variant',choices=['core','new_ce_only','plain_mlp','support_scan','combination','identity'],required=True)
    p.add_argument('--combination_profile')
    p.add_argument('--centers',type=int,default=32)
    a=p.parse_args()
    audit=json.loads(Path(a.audit_summary).read_text(encoding='utf-8-sig'))
    if not audit[a.dataset]['development_training_admitted']:
        raise RuntimeError('Source-record audit has not admitted this dataset')
    source,code,out=map(Path,(a.source_run,a.code_root,a.output_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(code.resolve()))
    trainer=module('core_control_training',code/'20_train_anchor_tree_mlp_noreplay.py')
    helpers=module('core_control_helpers',code/'75_diagnose_prototype_boundary_bias.py')
    evaluator=module('core_control_evaluate',code/'84_train_coverage_pilot.py')
    validation=module('core_control_metrics',code/'79_validate_prototype_decisions.py')
    cfg=json.loads((source/'args.json').read_text(encoding='utf-8-sig'))
    if 'data_root' in audit[a.dataset] and Path(audit[a.dataset]['data_root']).resolve()!=Path(cfg['data_root']).resolve():
        raise RuntimeError('Audit does not belong to the requested data root')
    expected={'initial_classes':50,'increment_size':5,'num_increments':10,'centers_per_class':32,
        'projector_alpha':.95,'prototype_ce_weight':4.,'new_ce_weight':1.,
        'anchor_ce_weight':0.,'anchor_margin_weight':0.,'distill_weight':0.,'new_data_distill_weight':0.}
    if any(cfg[k]!=v for k,v in expected.items()) or cfg['seed'] not in (42,43,44):
        raise ValueError('Not the final core source configuration')
    if a.centers not in (1,4,8,16,32):
        raise ValueError('Unregistered support budget')
    cfg.update(init_dir=str(source),save_dir=str(out),centers_per_class=a.centers)
    if a.variant=='combination':
        from wisig512_combination_profiles import get_profile
        if a.dataset!='wisig' or cfg.get('input_segment_len')!=512:
            raise ValueError('Combination profiles require the clean WiSig512 initialization')
        profile=get_profile(a.combination_profile)
        if a.centers!=profile['centers_per_class']:raise ValueError('Support count M does not match profile')
        cfg.update(profile)
    elif a.combination_profile:
        raise ValueError('Profile is only valid with combination variant')
    if a.variant=='new_ce_only':cfg['prototype_ce_weight']=0.
    if a.variant=='identity':cfg['e2_mode']='identity'
    if a.variant=='plain_mlp':cfg['projector_alpha']=0.
    cfg['protocol_note']=f'final-core control {a.variant}/{a.combination_profile}; M={a.centers}; development validation only'
    args=SimpleNamespace(**cfg)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    api=trainer.load_support_module()
    original=api.NpyDeviceDataset
    context={'stage':0};ledger=[]
    def guarded(folder,allowed_labels=None,**kw):
        path=Path(folder);t=context['stage']
        name=path.name
        train='train-closed' if t==0 else f'train-add{t}'
        val='val-closed' if t==0 else f'val-add{t}'
        if path.resolve().parent!=Path(args.data_root).resolve() or name not in (train,val):
            raise RuntimeError(f'Forbidden data access {path}')
        allowed=set(map(int,allowed_labels))
        if name==train:
            expected=set(trainer.stage_seen_labels(args,0) if t==0 else trainer.stage_new_labels(args,t))
            if allowed!=expected:raise RuntimeError('Old/future training class access')
        ledger.append({'stage':t,'folder':str(path),'use':'current_training' if name==train else 'offline_validation'})
        return original(folder,allowed_labels=allowed_labels,**kw)
    api.NpyDeviceDataset=guarded
    device=torch.device(args.device)
    encoder=helpers.load(source/'encoder.pth',device).eval().requires_grad_(False)
    head=helpers.load(source/'classifier.pth',device).eval().requires_grad_(False)
    e1_state={k:v.cpu().clone() for k,v in encoder.state_dict().items()}
    source_hashes={str(source/f):helpers.file_hash(source/f) for f in ('encoder.pth','classifier.pth','args.json')}
    torch.save(encoder,out/'encoder.pth');torch.save(head,out/'classifier.pth')
    (out/'args.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    state=trainer.build_step0_state(api,encoder,args,device)
    records=[]
    def memory_digest():
        digest=hashlib.sha256()
        for c in sorted(state['prototypes']):
            digest.update(str(c).encode())
            for item in (state['prototypes'][c],state['support_prototypes'][c]):
                x=item.detach().cpu().contiguous().numpy()
                digest.update(str(x.shape).encode());digest.update(x.tobytes())
        return digest.hexdigest()
    for t in range(11):
        context['stage']=t
        if t:
            old={c:p.clone() for c,p in state['support_prototypes'].items()}
            old_route={c:p.clone() for c,p in state['prototypes'].items()}
            trainer.add_increment_nodes(api,encoder,state,args,t,device)
            if any(not torch.equal(state['support_prototypes'][c],p) for c,p in old.items()):raise RuntimeError('Old memory moved')
            if any(not torch.equal(state['prototypes'][c],p) for c,p in old_route.items()):raise RuntimeError('Old route mean moved')
            if args.anchor_ce_weight>0 or args.anchor_margin_weight>0:
                candidate=set(old) if args.anchor_candidate_scope=='all_seen' else set(range(args.initial_classes))
                for c in trainer.stage_new_labels(args,t):
                    anchors=[int(item[0]) for item in state['anchors'][c]]
                    if len(anchors)!=min(args.anchor_topk,len(candidate)) or len(set(anchors))!=len(anchors) or not set(anchors)<=candidate:
                        raise RuntimeError('Anchor registration violated declared old-class neighborhood')
        if any(int(v)!=200 for v in state['counts'].values()):raise RuntimeError('Registration quota mismatch')
        trainer.save_state(out,t,state)
        if t:encoder,e2,head=trainer.train_one_stage(api,args,t,state,device)
        else:e2=torch.nn.Identity().to(device)
        if any(not torch.equal(e1_state[k],v.cpu()) for k,v in encoder.state_dict().items()):raise RuntimeError('E1/BN changed')
        row=evaluator.evaluate(api,encoder,e2,head,state,args,t,helpers,validation)
        row['memory_sha256']=memory_digest()
        records.append(row)
        (out/'control_validation.partial.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(f'CONTROL {a.dataset}/{args.seed}/{a.variant}/M{a.centers} Step{t} e2={row["methods"]["e2_support"]["accuracy_pct"]:.2f}',flush=True)
        del e2
    if any(helpers.file_hash(Path(f))!=h for f,h in source_hashes.items()):raise RuntimeError('Step0 source modified')
    report={'dataset':a.dataset,'seed':args.seed,'variant':a.variant,'combination_profile':a.combination_profile,'centers':a.centers,'records':records,
        'split':'validation','development_only':True,'read_ledger':ledger,'test_data_read':False,
        'old_training_data_read_during_increment':False,'source_checkpoint_hashes':source_hashes,
        'e1_parameters_buffers_unchanged':True,'old_supports_unchanged':True,
        'support_payload_bytes_per_class':a.centers*1024*4,'retained_vector_bytes_per_class':(a.centers+1)*1024*4,
        'final_core_configuration':expected,'actual_configuration':cfg,
        'source_code_sha256':{f:helpers.file_hash(code/f) for f in ('20_train_anchor_tree_mlp_noreplay.py','15_train_incremental_anchor_domain.py','100_train_fixed_memory_control.py')}}
    (out/'control_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    from gpu_study_lock import gpu_study_lock
    with gpu_study_lock():main()
