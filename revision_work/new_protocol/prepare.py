"""Materialize the approved fifteen-run series and freeze it after preflight."""
import argparse
import shutil
import tarfile
from pathlib import Path

import yaml

from .common import ROOT, WORK, SERIES, PYTHON, now, sha, digest, atomic_json, read_json


def prepare():
    if (SERIES / 'protocol.json').exists():
        print('Existing series retained; no runs or protocol overwritten.')
        return
    history = WORK / 'history/preapproval'
    history.mkdir(parents=True, exist_ok=False)
    for name in ['protocol_lock.json', 'resolved_experiments.json', 'run_registry.json', 'STATUS.md',
                 'FINAL_REPORT.md', 'commands.md', 'baseline_registry.json', 'remediation_options.md']:
        shutil.copy2(WORK / name, history / name)
    shutil.copytree(WORK / 'results', history / 'results')
    (SERIES / 'configs').mkdir(parents=True)
    (SERIES / 'runs').mkdir()
    (SERIES / 'preflight').mkdir()
    approval = {'approved_at': now(), 'user_message': '批准新批准新的协议与增加训练预算',
                'scope': 'Previous proposal B: new common protocol and fifteen complete DAUB trainings',
                'full_training_budget': 15, 'additional_over_default': 2,
                'not_included': ['two extra component baselines required for a 17-run series',
                                 'IRDST training', 'new scan-order/placement/baseline training', 'target-domain training']}
    atomic_json(SERIES / 'approval.json', approval)
    evidence = WORK / 'evidence/checkpoints'
    paths = {}
    for name, historic in [('full', 'frame5'), ('stf_3dconv', 'conv3d-TIMSSM'), ('sste_saam', 'sste_stage')]:
        source = next(evidence.glob('*DAUB__' + historic + '__weights__best.yaml'))
        model = yaml.safe_load(source.read_text())
        model.pop('yaml_file', None)
        path = SERIES / 'configs' / (name + '.yaml')
        path.write_text(yaml.safe_dump(model, sort_keys=False))
        paths[name] = str(path)
    data = read_json(WORK / 'evidence/data/summary.json')['DAUB']
    lock = {
        'protocol_id': 'SSE_DAUB_SERIES_B_20260915', 'approved': True, 'formal_training_locked': False,
        'approval': approval, 'created_at': now(), 'initialization': 'fresh random; no trained checkpoint loaded',
        'BASE_SEED': 0, 'COMMON_SEEDS': [0, 2027, 3407], 'historical_seeds_reused': [],
        'train_manifest': str(WORK / 'evidence/data/DAUB_train_frames.jsonl'),
        'test_manifest': str(WORK / 'evidence/data/DAUB_val_frames.jsonl'),
        'train_manifest_sha256': data['train']['manifest_sha256'],
        'test_manifest_sha256': data['val']['manifest_sha256'],
        'split_note': 'Same physical train/test sequence sets; previous val=test selection removed for this new series',
        'train_frames': 8982, 'test_frames': 4795, 'imgsz': 640, 'frames': 5, 'channels': 8,
        'temporal_stride_train': 1, 'channel_order': ['g_i-4', 'g_i-3', 'g_i-2', 'g_i-1', 'g_i', 'R_i', 'G_i', 'B_i'],
        'boundary': 'replicate first sequence frame; one clip per unique keyframe',
        'epochs': 100, 'batch': 8, 'nbs': 64, 'steady_accumulation': 8, 'amp': False,
        'optimizer': 'Adam', 'lr0': .001, 'lrf': .01, 'betas': [.937, .999], 'weight_decay': .0005,
        'scheduler': 'lr(epoch)=lr0*(lrf+(1-lrf)*(1+cos(pi*epoch/99))/2), epoch=0..99',
        'warmup_epochs': 3, 'warmup_bias_lr': 0., 'gradient_clip_norm': 10.,
        'accumulation': 'ramp 1->8 over warmup; carry pending gradients across epochs and save them for resume',
        'loss_default': [7.5, .5, 1.5], 'lfss_depth': 4, 'kernel_default': 3,
        'checkpoint_selection': 'EMA of fixed final epoch 100; no test during training; no best-by-AP file',
        'ema': {'decay': .9999, 'tau': 2000, 'implementation': 'local ultralytics.utils.torch_utils.ModelEMA'},
        'save_resume': 'FP32 model, EMA, optimizer, gradients, RNG and progress at every completed epoch',
        'augmentation_active': {'horizontal_flip_probability': .5, 'vertical_flip_probability': 0.,
                                'mosaic': False, 'mixup': False, 'HSV': False, 'albumentations': False},
        'augmentation_evidence': 'Current v8_transforms comments out pre_transform/HSV, mixup=0. Albumentations import raises ImportError for albucore.preserve_channel_dim and is skipped by the current wrapper. Geometric operations apply once to the whole clip and keyframe boxes.',
        'augmentation_rng': 'stateless by seed, epoch, unique keyframe index; independent of worker scheduling',
        'evaluation': {'conf': .001, 'nms_iou': .7, 'max_det': 300, 'half': False, 'batch': 8,
                       'fuse': False, 'working_threshold': .25, 'working_iou': .5,
                       'working_threshold_source': 'a priori source protocol, fixed before new model scores',
                       'metrics': 'local DetectionValidator and DetMetrics, AP50 and AP50-95'},
        'architecture_notes': 'Matched frame5/3D YAML, LFSS n=4; unused constructor-only TMixSSM.saam omitted. Explicit STF/DCN and Detect/SAAM forwards surface exceptions.',
        'E1': 'Historical SSTE YAML without CCCBlock, leaving the directly following C2f as shape-compatible baseline; SAAM explicitly active in Detect',
        'source_files': {}, 'source_sha256': None,
    }
    atomic_json(SERIES / 'protocol.json', lock)
    definitions = [('B_FULL_0', 'full', 0), ('B_3DCONV_0', 'stf_3dconv', 0),
                   ('R1', 'full', 2027), ('R2', 'stf_3dconv', 2027),
                   ('R3', 'full', 3407), ('R4', 'stf_3dconv', 3407)]
    definitions += [(f'E{i}', 'sste_saam' if i == 1 else 'full', 0) for i in range(1, 10)]
    tasks = []
    for position, (task_id, model, seed) in enumerate(definitions):
        loss = lock['loss_default'].copy()
        if task_id in ['E4', 'E5', 'E6', 'E7', 'E8', 'E9']:
            number = int(task_id[1:]) - 4
            loss[number // 2] *= .5 if number % 2 == 0 else 2.
        task = {'id': task_id, 'series': 'B', 'protocol_id': lock['protocol_id'], 'model': model, 'seed': seed,
                'model_yaml': paths[model], 'model_yaml_sha256': sha(paths[model]),
                'lfss_depth': 4, 'lfss_kernel': 1 if task_id == 'E2' else 5 if task_id == 'E3' else 3,
                'loss': loss, 'epochs': 100, 'batch': 8, 'workers': 8,
                'output_directory': str(SERIES / 'runs' / task_id), 'priority': position,
                'command': [PYTHON, '-m', 'revision_work.new_protocol.train', '--task', task_id]}
        task['config_sha256'] = digest(task)
        atomic_json(SERIES / 'configs' / (task_id + '.json'), task)
        tasks.append({'id': task_id, 'kind': 'training', 'status': 'pending', 'seed': seed,
                      'config_sha256': task['config_sha256'], 'output_directory': task['output_directory'],
                      'dependencies': ['PREFLIGHT_SERIES_B'], 'formal_result': False})
        tasks.append({'id': 'TEST_' + task_id, 'kind': 'formal_test', 'status': 'pending',
                      'dependencies': [task_id], 'formal_result': False})
    atomic_json(SERIES / 'registry.json', {'status': 'preparing', 'approved_budget': 15, 'updated_at': now(), 'tasks': tasks})
    atomic_json(WORK / 'protocol_lock.json', lock)
    atomic_json(WORK / 'resolved_experiments.json', {'resolution_complete': True, 'preflight_pending': True,
                 'tasks': [read_json(SERIES / 'configs' / (t[0] + '.json')) for t in definitions]})
    print('Prepared 15 independent run configurations. Formal training awaits necessary preflight only.')


def freeze():
    result = read_json(SERIES / 'preflight/result.json')
    assert result['passed'] and result['formal_training'] is False
    lock = read_json(SERIES / 'protocol.json')
    assert not lock['formal_training_locked'], 'Already frozen; retain the existing experiment version'
    original = read_json(WORK / 'evidence/source_manifest.json')['files']
    for file, expected in original.items():
        assert sha(ROOT / file) == expected, file
    runtime_names = {'__init__.py', 'common.py', 'model.py', 'data.py', 'train.py', 'evaluate.py'}
    added = {str(p.relative_to(ROOT)): sha(p) for p in sorted((WORK / 'new_protocol').glob('*.py'))}
    files = dict(original)
    files.update({name: value for name, value in added.items() if Path(name).name in runtime_names})
    workflow = {name: value for name, value in added.items() if Path(name).name not in runtime_names}
    lock.update(source_files=files, source_sha256=digest(files), workflow_files_at_freeze=workflow,
                hash_scope='original model/loss/metric code plus new data/model/train/evaluate/runtime utilities',
                formal_training_locked=True, locked_at=now())
    atomic_json(SERIES / 'protocol.json', lock)
    atomic_json(WORK / 'protocol_lock.json', lock)
    with tarfile.open(SERIES / 'source_snapshot.tar.gz', 'w:gz') as archive:
        for name in sorted({**files, **workflow}):
            archive.add(ROOT / name, arcname=name)
    resolved = read_json(WORK / 'resolved_experiments.json')
    resolved.update(preflight_pending=False, source_sha256=lock['source_sha256'])
    atomic_json(WORK / 'resolved_experiments.json', resolved)
    registry = read_json(SERIES / 'registry.json')
    for task in registry['tasks']:
        if task['kind'] == 'training':
            task['status'] = 'ready'
            task['dependencies'] = []
    registry.update(status='ready', updated_at=now(), source_sha256=lock['source_sha256'])
    atomic_json(SERIES / 'registry.json', registry)
    print('Protocol frozen:', lock['source_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else prepare()
