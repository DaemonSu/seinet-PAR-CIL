"""Split packet regions first; train stride1, validation/test stride4, 1024 inputs."""
import argparse
import hashlib
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
from wisig512_frozen_study_common import module,digest


def regions(n):
    a,b=int(n*.4),int(n*.7)
    return {'train':(0,a,1),'val':(a,b,4),'test':(b,n,4)}


def starts(n,use):
    left,right,stride=regions(n)[use]
    return list(range(left,right-3,stride))


def capacity(captures):
    return {use:sum(len(starts(n,use)) for n in captures) for use in ('train','val','test')}


def overlap_audit(records):
    groups=defaultdict(list);cross=0;train=0
    for r in records:groups[(r['raw_device'],r['source_id'])].append(r)
    for group in groups.values():
        active=[]
        for r in sorted(group,key=lambda x:x['start_iq']):
            active=[q for q in active if q['end_iq']>r['start_iq']]
            for q in active:
                if q['split']!=r['split']:cross+=1
                elif r['split']=='train':train+=1
                else:raise RuntimeError('Validation/test internal overlap')
            active.append(r)
    if cross:raise RuntimeError('Cross-split IQ overlap')
    return {'cross_split_shared_iq_pairs':cross,'train_internal_overlap_pairs':train,
            'val_internal_overlap_pairs':0,'test_internal_overlap_pairs':0}


def main():
    p=argparse.ArgumentParser()
    for k in ('output','code','reference','raw_source'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);code=Path(a.code);reference=Path(a.reference)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh data directory required')
    out.mkdir(parents=True)
    reference_protocol=json.loads((reference/'protocol.json').read_text())
    labels={int(k):v for k,v in reference_protocol['label_to_raw_device'].items()}
    if set(labels)!=set(range(100)):raise RuntimeError('Original100 labels required')
    days=reference_protocol['days'];rx=reference_protocol['receiver'];raw=Path(a.raw_source)
    chunks=defaultdict(list);source_hashes={}
    for day in days:
        path=raw/f'pkl_wifi_{day}'/f'dataset_{day}_node{rx}.pkl'
        source_hashes[str(path)]=digest(path)
        print('LOAD1024 '+str(path),flush=True)
        with path.open('rb') as f:obj=pickle.load(f)
        lookup={str(dev):arr for dev,arr in zip(obj['node_list'],obj['data'])}
        for c,dev in labels.items():
            arr=lookup.get(dev)
            if arr is not None and getattr(arr,'ndim',0)==3 and arr.shape[1:]==(256,2):chunks[c].append((day,arr))
        del obj,lookup
    eligibility=[dict(label=c,device=labels[c],**capacity([len(arr) for _,arr in chunks[c]])) for c in range(100)]
    summary=dict(classes=100,all_classes_admitted=all(r['train']>=200 and r['val']>=50 and r['test']>=50 for r in eligibility),
                 minimum_candidates={use:min(r[use] for r in eligibility) for use in ('train','val','test')},
                 segment_len=1024,source_file_sha256=source_hashes,classes_detail=eligibility)
    (out/'capacity.json').write_text(json.dumps(summary,indent=2))
    if not summary['all_classes_admitted']:raise RuntimeError('1024 quotas failed; no label/quota relaxation')
    prep=module('wisig1024_frontend',code/'fixed_memory_wisig_preprocessing_api.py')
    rows=[];hashes=set();packet_coverage=[]
    for c in range(100):
        coverage={use:set() for use in ('train','val','test')}
        for use,quota in [('train',200),('val',50),('test',50)]:
            pool=[(day,arr,i) for day,arr in chunks[c] for i in starts(len(arr),use)]
            rng=np.random.default_rng(20260914+c*3+['train','val','test'].index(use))
            chosen=rng.permutation(len(pool))[:quota]
            for index,k in enumerate(chosen):
                day,arr,start=pool[int(k)];sid=f'{day}_node{rx}'
                x=prep.make_wisig_segment(prep.normalize_wisig_packets(arr[start:start+4]),0,1024).T.copy()
                if x.shape!=(1024,3) or x.dtype!=np.float32 or not np.isfinite(x).all():raise RuntimeError('Bad1024 tensor')
                h=hashlib.sha256(x.tobytes()).hexdigest()
                if h in hashes:raise RuntimeError('Duplicate tensor content')
                hashes.add(h);folder=out/'records'/use;folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'device_{c:03d}_{index:05d}.npy';np.save(path,x)
                left,right,stride=regions(len(arr))[use]
                rows.append(dict(label=c,raw_device=labels[c],split=use,path=str(path),source_id=sid,
                    start_packet=start,packets_needed=4,start_iq=start*256,end_iq=(start+4)*256,sha256=h,
                    source_region_start_packet=left,source_region_end_packet=right,candidate_stride_packets=stride))
                coverage[use].update((sid,p) for p in range(start,start+4))
        if any(coverage[l]&coverage[r] for l,r in [('train','val'),('train','test'),('val','test')]):raise RuntimeError('Shared raw packet')
        packet_coverage.append(dict(label=c,unique_packets={use:len(s) for use,s in coverage.items()}))
        print(f'WRITE1024 class{c}/100',flush=True)
    audit_counts=overlap_audit(rows)
    for t in range(11):
        for use in ('train','val','test'):
            dst=out/(f'{use}-closed' if not t else f'{use}-add{t}');dst.mkdir()
            for r in rows:
                c=r['label'];selected=c<50 if not t else (45+5*t<=c<50+5*t if use=='train' else c<50+5*t)
                if r['split']==use and selected:os.link(r['path'],dst/Path(r['path']).name)
    protocol=dict(dataset='wisig',output_root=str(out),receiver=rx,days=days,label_to_raw_device=labels,
        initial_classes=50,increment_size=5,num_increments=10,train_per_class=200,val_per_class=50,test_per_class=50,
        segment_len=1024,packets_per_input=4,feature_mode='q_abs_fft',normalization='packet_rms',split_seed=20260914,
        raw_region_split='per device/date capture: first40% train, next30% val, final30% test; split before windowing',
        candidate_stride_packets={'train':1,'val':4,'test':4},train_overlap_allowed=True,cross_split_overlap_allowed=False,
        class_order_unchanged=True,previous_development_regions_allowed=True,independent_confirmation=False,
        comparison_to512='length and source partition/augmentation change together; not a pure length-effect estimate',
        source_file_sha256=source_hashes,preprocessor_sha256=digest(code/'fixed_memory_wisig_preprocessing_api.py'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'source_records.json').write_text(json.dumps(rows,indent=2))
    (out/'packet_coverage.json').write_text(json.dumps(packet_coverage,indent=2))
    audit={'wisig':dict(data_root=str(out),prepared_integrity_passed=True,source_record_passed=True,
        development_training_admitted=True,independent_confirmation_admitted=False,rows=len(rows),
        unique_tensor_hashes=len(hashes),test_predictions_read=False,source_regions_partitioned_before_windowing=True,
        source_code_sha256=digest(Path(__file__)),**audit_counts)}
    (out/'audit_summary.json').write_text(json.dumps(audit,indent=2))
    print('PREPARED1024 '+json.dumps(audit_counts),flush=True)


if __name__=='__main__':main()
