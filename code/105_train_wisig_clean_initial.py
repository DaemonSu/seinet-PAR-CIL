"""Original WiSig Step0 architecture/objective, but no test reads or feature cache."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import torch
from torch.utils.data import DataLoader


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code_root',required=True)
    p.add_argument('--data_root',required=True)
    p.add_argument('--save_dir',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--template',required=True)
    a=p.parse_args()
    code,out,data=Path(a.code_root),Path(a.save_dir),Path(a.data_root)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh Step0 output required')
    out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(code.resolve()))
    import train_wisig_initial as original
    original.set_seed(a.seed);device=torch.device('cuda:0')
    audit=json.loads((data/'audit_summary.json').read_text())
    if not audit['wisig']['source_record_passed']:raise RuntimeError('Clean data audit failed')
    train=original.NpyDeviceDataset(data/'train-closed',allowed_labels=range(50))
    val=original.NpyDeviceDataset(data/'val-closed',allowed_labels=range(50))
    loader=DataLoader(train,batch_size=256,shuffle=True);vl=DataLoader(val,batch_size=256,shuffle=False)
    encoder=original.FeatureExtractor(in_channels=3,feature_dim=1024).to(device)
    with torch.no_grad():encoder(next(iter(loader))[0][:4].to(device))
    head=original.ClassifierHead(1024,50).to(device)
    ce=torch.nn.CrossEntropyLoss();con=original.SupConLoss(temperature=.08)
    opt=torch.optim.Adam(list(encoder.parameters())+list(head.parameters()),lr=.001,weight_decay=1e-4)
    best=-1;history=[]
    for epoch in range(35):
        encoder.train();head.train();total=0.
        for x,y in loader:
            x,y=x.to(device),y.to(device);feat=encoder(x);loss=2*ce(head(feat),y)+.5*con(feat,y)
            opt.zero_grad();loss.backward();opt.step();total+=float(loss.item())
        accuracy=original.evaluate(encoder,head,vl,device)
        if accuracy>best:
            best=accuracy;torch.save(encoder,out/'encoder.pth');torch.save(head,out/'classifier.pth')
        history.append({'epoch':epoch+1,'training_loss':total/len(loader),'validation_accuracy_pct':accuracy})
        print(f'CLEAN STEP0 seed{a.seed} epoch{epoch+1}/35 val={accuracy:.2f} best={best:.2f}',flush=True)
    cfg=json.loads(Path(a.template).read_text(encoding='utf-8-sig'))
    protocol=json.loads((data/'protocol.json').read_text())
    cfg.update(seed=a.seed,data_root=str(data),init_dir=str(out),save_dir=str(out),
        input_segment_len=protocol['segment_len'],protocol_note='packet-disjoint WiSig Step0; development only; no test reads or feature-cache exports')
    (out/'args.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    report={'seed':a.seed,'best_validation_accuracy_pct':best,'history':history,'test_data_read':False,'feature_cache_exported':False,
        'epochs':35,'batch_size':256,'ce_weight':2.,'supcon_weight':.5,'temperature':.08,'lr':.001,'weight_decay':1e-4,
        'selection':'best Step0 validation; no test reads','protocol_path':str(data/'protocol.json')}
    (out/'initial_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    from gpu_study_lock import gpu_study_lock
    with gpu_study_lock():main()
