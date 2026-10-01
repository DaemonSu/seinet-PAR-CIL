"""Fresh 1024 Step0 using unchanged initial architecture, objective and batch size."""
import importlib
import json
import sys
from pathlib import Path


def main():
    argv=sys.argv[1:]
    data=Path(argv[argv.index('--data_root')+1])
    out=Path(argv[argv.index('--save_dir')+1])
    protocol=json.loads((data/'protocol.json').read_text(encoding='utf-8-sig'))
    if protocol['segment_len']!=1024 or protocol['candidate_stride_packets']!={'train':1,'val':4,'test':4}:
        raise RuntimeError('Not the prespecified 1024 block-split dataset')
    importlib.import_module('105_train_wisig_clean_initial').main()
    path=out/'args.json';cfg=json.loads(path.read_text(encoding='utf-8-sig'))
    cfg['protocol_note']='WiSig1024: packet-region-disjoint splits; train overlap allowed; validation/test nonoverlap; no test reads'
    cfg['window_protocol']=protocol
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')


if __name__=='__main__':
    from gpu_study_lock import gpu_study_lock
    with gpu_study_lock():main()
