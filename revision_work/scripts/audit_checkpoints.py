"""Read local historical artifacts without transferring or discarding parameters."""
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from ultralytics.nn.tasks import parse_model


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    out = ROOT / 'revision_work/evidence/checkpoints'
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'runs/compare_result/ablation/Ablation/DAUB'
    paths = sorted(base.glob('*/weights/best.pt'))
    paths += [ROOT / 'runs/compare_result/DAUB' / n / 'weights/best.pt'
              for n in ['my_2timMix_945', 'fullsize_model', 'yolov8n_813']]
    paths += sorted((ROOT / 'runs/compare_result/ablation/Ablation/IRDST').glob('*/weights/best.pt'))
    records = []
    for path in paths:
        rid = str(path.relative_to(ROOT)).replace('/', '__').removesuffix('.pt')
        rec = {'checkpoint': str(path), 'sha256': sha(path)}
        try:
            ck = torch.load(path, map_location='cpu', weights_only=False)
            model = ck.get('ema') or ck.get('model')
            rec.update({k: ck.get(k) for k in ['epoch', 'date', 'version', 'git', 'train_args', 'train_metrics']})
            rec['optimizer_present'] = ck.get('optimizer') is not None
            rec['serialized_yaml'] = copy.deepcopy(model.yaml)
            rec['params'] = sum(p.numel() for p in model.parameters())
            rec['state_tensors'] = len(model.state_dict())
            rec['nonfinite_keys'] = [k for k,v in model.state_dict().items()
                                      if v.is_floating_point() and not torch.isfinite(v).all()]
            rec['graph'] = [{'i': i, 'from': getattr(m, 'f', None),
                             'class': type(m).__module__ + '.' + type(m).__name__,
                             'params': sum(p.numel() for p in m.parameters())}
                            for i,m in enumerate(model.model)]
            rec['modules'] = [{'path': n, 'class': type(m).__module__+'.'+type(m).__name__,
                               'kernel': getattr(m, 'kernel_size', None),
                               'stride': getattr(m, 'stride', None) if isinstance(m, (torch.nn.Conv2d,torch.nn.Conv3d)) else None,
                               'groups': getattr(m, 'groups', None)} for n,m in model.named_modules()]
            rec['state_shapes'] = {k: list(v.shape) for k,v in model.state_dict().items()}
            # Build the graph from the checkpoint YAML, bypassing DetectionModel's
            # CUDA stride probe only. No parameters or modules are omitted.
            try:
                graph,_ = parse_model(copy.deepcopy(model.yaml), ch=model.yaml.get('channels',8), verbose=False)
                state = model.model.state_dict()
                expected = graph.state_dict()
                rec['strict_missing_keys'] = sorted(expected.keys()-state.keys())
                rec['strict_unexpected_keys'] = sorted(state.keys()-expected.keys())
                rec['strict_shape_mismatches'] = {k: [list(expected[k].shape),list(state[k].shape)]
                         for k in expected.keys() & state.keys() if expected[k].shape != state[k].shape}
                graph.load_state_dict(state, strict=True)
                rec['strict_reconstruction'] = 'passed'
                del graph
            except Exception as exc:
                rec['strict_reconstruction'] = 'failed'
                rec['strict_error'] = str(exc)
            (out/(rid+'.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2,default=str))
            yaml_path = out / (rid+'.yaml')
            yaml_path.write_text(yaml.safe_dump(rec['serialized_yaml'],sort_keys=False))
            print(path.parent.parent.name, 'strict='+rec['strict_reconstruction'],
                  'params='+str(rec['params']), 'pretrained='+str(rec['train_args'].get('pretrained')),flush=True)
        except Exception:
            rec['error'] = traceback.format_exc()
            print(path, rec['error'],flush=True)
        records.append(rec)
    (out.parent/'checkpoint_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str))


if __name__ == '__main__':
    main()
