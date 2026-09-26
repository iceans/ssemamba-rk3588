"""Re-evaluate complete serialized historical graphs with the current legacy loader.

These are explicitly diagnostic results. They cannot validate the historical
training protocol or the fresh-constructor incompatibilities recorded in the audit.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['YOLO_AUTOINSTALL']='false'
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class CapturingValidator(DetectionValidator):
    def build_dataset(self,img_path,mode='val',batch=None):
        # Preserve all loader semantics; redirect only a newly generated label cache.
        original=YOLODataset.cache_labels
        target=ROOT/'revision_work/evidence/diagnostic_labels_f5.cache'
        def cache_in_work(instance,path):return original(instance,target)
        from unittest.mock import patch
        with patch.object(YOLODataset,'cache_labels',cache_in_work):
            return super().build_dataset(img_path,mode,batch)

    def init_metrics(self,model):
        super().init_metrics(model)
        self.cache_file=(self.save_dir/'per_frame_predictions.jsonl').open('w')
        self.frame_keys=[]

    def update_metrics(self,preds,batch):
        for i,pred in enumerate(preds):
            info=self._prepare_batch(i,batch)
            native=self.scale_preds(pred,info)
            p=Path(info['im_file']);seq,frame=p.stem.rsplit('_',1)
            entry={'seq_id':seq,'frame_id':int(frame),'file':str(p),'original_hw':list(info['ori_shape']),
                   'boxes_xyxy':native['bboxes'].detach().cpu().tolist(),
                   'classes':native['cls'].detach().cpu().tolist(),
                   'confidences':native['conf'].detach().cpu().tolist(),
                   'model_source':str(self.args.model),'formal_result':False,
                   'sample_index':len(self.frame_keys)}
            self.cache_file.write(json.dumps(entry)+'\n');self.frame_keys.append(p.stem)
        self.cache_file.flush()
        return super().update_metrics(preds,batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        self.cache_file.close()
        counts=Counter(self.frame_keys)
        (self.save_dir/'sample_coverage.json').write_text(json.dumps({
            'num_samples':len(self.frame_keys),'num_unique_keys':len(counts),
            'duplicates':{k:v for k,v in counts.items() if v>1}},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['fullsize_model','conv3d-TIMSSM'],required=True)
    p.add_argument('--device',default='1')
    args=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in this execution context')
    path=ROOT/'runs/compare_result/ablation/Ablation/DAUB'/args.model/'weights/best.pt'
    out=ROOT/'revision_work/runs'/('REEVAL_DIAGNOSTIC_'+args.model)
    out.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'ultralytics/cfg/datasets/DAUB.yaml').read_text())
    cfg['channels']=8  # Explicit checkpoint graph input channels; current dataset YAML was changed to 23.
    cfgpath=out/'data.yaml';cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
    record={'id':'REEVAL_DIAGNOSTIC_'+args.model,'status':'running','pid':os.getpid(),
            'checkpoint':str(path),'checkpoint_sha256':sha(path),'formal_result':False,
            'purpose':'diagnostic re-evaluation; do not merge into formal paper statistics',
            'constructor_compatibility':'failed; see checkpoint_audit.json',
            'selection':'historical best.pt; trained with current test directory as val; not an independent test result',
            'config_override':{'channels':{'current_dataset_yaml':23,'checkpoint_graph':8}},
            'command':sys.argv,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [
                ROOT/'ultralytics/nn/modules/tgsmamba.py',ROOT/'ultralytics/nn/modules/timeguid.py',
                ROOT/'ultralytics/nn/modules/head.py',ROOT/'ultralytics/data/base.py',
                ROOT/'ultralytics/data/dataset.py',ROOT/'ultralytics/data/augment.py',
                ROOT/'ultralytics/models/yolo/detect/val.py',ROOT/'ultralytics/utils/metrics.py']}}
    def save(): (out/'run.json').write_text(json.dumps(record,indent=2,ensure_ascii=False,default=str))
    save();start=time.perf_counter()
    try:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        original=(ck.get('ema') or ck['model']).float()
        wrapper=YOLO(str(path))
        # Full serialized graph restore, with strict state validation. This is
        # deliberately distinguished from successful reconstruction using today's constructors.
        state=original.state_dict()
        wrapper.model.load_state_dict(state,strict=True)
        assert state.keys()==wrapper.model.state_dict().keys()
        for key,value in wrapper.model.state_dict().items():
            torch.testing.assert_close(value.cpu(),state[key],rtol=0,atol=0)
        record['serialized_state_restore']='all keys, shapes and values verified; strict=True'
        record['state_tensors']=len(state);save()
        del ck,original,state
        kwargs=dict(data=str(cfgpath),imgsz=640,batch=8,device=args.device,conf=.001,iou=.7,
                    rect=False,half=False,frame_num=5,cube=True,gray=False,workers=0,plots=False,
                    save_json=True,split='val',project=str(out),name='validation',exist_ok=False)
        record['resolved_evaluation_args']=kwargs;save()
        metrics=wrapper.val(validator=CapturingValidator,**kwargs)
        record.update(status='evaluated',metrics=metrics.results_dict,elapsed_s=time.perf_counter()-start)
    except Exception:
        record.update(status='failed',error=traceback.format_exc(),elapsed_s=time.perf_counter()-start)
        save();raise
    save();print(json.dumps(record,indent=2,default=str))


if __name__=='__main__':main()
