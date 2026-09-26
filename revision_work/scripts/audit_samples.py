"""Audit the actual legacy dataset at sequence boundaries without modifying it."""
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Format
from ultralytics.utils.ops import scale_boxes,xywh2xyxy


class ProbeDataset(YOLODataset):
    def cache_labels(self,path):
        return super().cache_labels(ROOT/'revision_work/evidence/sample_labels_f5.cache')


def main():
    cfg=yaml.safe_load((ROOT/'ultralytics/cfg/datasets/DAUB.yaml').read_text())
    hyp=yaml.safe_load((ROOT/'ultralytics/cfg/default.yaml').read_text())
    hyp.update(frame_num=5,cube=True,gray=False,mod=False,rect=False)
    ds=ProbeDataset(img_path=str(Path(cfg['path'])/cfg['val']),imgsz=640,batch_size=8,
                    augment=False,hyp=SimpleNamespace(**hyp),rect=False,cache=False,
                    data=cfg,stride=32,pad=.5)
    rows=[json.loads(s) for s in (ROOT/'revision_work/evidence/data/DAUB_val_frames.jsonl').read_text().splitlines()]
    original={r['file']:r for r in rows}
    imfiles=ds.im_files
    seq=lambda p:Path(p).stem.rsplit('_',1)[0]
    boundaries=[i for i in range(1,len(imfiles)) if seq(imfiles[i])!=seq(imfiles[i-1])]
    indices=sorted(set([0,1,4,10,len(ds)-5,len(ds)-1]+[j for i in boundaries for j in [i-1,i,i+1,i+4]]))
    sample=[]
    for i in indices:
        item=ds[i];clip=list(ds.fs_im);p=item['im_file']
        box=xywh2xyxy(item['bboxes_t5'].clone())*640
        native=scale_boxes((640,640),box,item['ori_shape'],ratio_pad=item['ratio_pad'])
        truth=torch.tensor([b['xyxy'] for b in original[p]['boxes']])
        error=float((native-truth).abs().max()) if native.shape==truth.shape else None
        sample.append({'index':i,'nominal_file':imfiles[i],'label_file':p,'loaded_clip':clip,
                       'one_sequence':len(set(map(seq,clip)))==1,'keyframe_matches_label':p==clip[-1],
                       'coordinate_max_abs_error_px':error,'network_shape':list(item['img'].shape)})
    actual=[];cross=[]
    for requested in range(len(imfiles)):
        i=min(requested,len(imfiles)-5)
        fs=[imfiles[i+x-4] for x in range(5)]
        ids=[Path(f).name.split('_')[-2] for f in fs]
        change=5-next((k for k,v in enumerate(ids) if v!=ids[0]),-1)
        if change<5:fs=[imfiles[i+x-change-4] for x in range(5)]
        actual.append(fs[-1])
        if len(set(map(seq,fs)))!=1:cross.append({'index':requested,'clip':fs})
    counts=collections.Counter(actual)
    channel_input=np.tile(np.array([101,102,103,1,2,3,4,5],dtype=np.uint8),(2,3,1))
    formatted=Format()._format_img(channel_input)
    result={'num_requested':len(imfiles),'num_unique_loaded_keyframes':len(counts),
            'omitted_nominal_frames':sorted(set(original)-set(actual)),
            'duplicate_keyframes':{k:v for k,v in counts.items() if v>1},
            'cross_sequence_windows':cross,'sample_checks':sample,
            'channel_order_probe':formatted[:,0,0].tolist(),
            'channel_semantics':['gray_i','gray_i-1','gray_i-2','gray_i-3','gray_i-4','key_R','key_G','key_B'],
            'stf_reference_note':'TimeMambaStem uses its last temporal feature as alignment reference; under current Format this is the oldest source frame.',
            'formal_training_gate':'blocked: duplication/omission and reverse-order convention require an explicit protocol decision; no dataset code changed'}
    out=ROOT/'revision_work/evidence/sample_audit.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['sample_checks','duplicate_keyframes','omitted_nominal_frames']},indent=2))
    print('omitted',len(result['omitted_nominal_frames']),'duplicate_keys',len(result['duplicate_keyframes']))


if __name__=='__main__':main()
