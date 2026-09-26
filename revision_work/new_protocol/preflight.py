"""Necessary checks for the newly approved model, loader, optimizer and resume path."""
import gc
import os
import time
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch

from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.torch_utils import init_seeds, ModelEMA
from .common import WORK, SERIES, now, read_json, atomic_json, load_task
from .data import ClipDataset, collate, to_device, letterbox
from .model import SSEModel, architecture_summary
from .train import build_optimizer, make_checkpoint, restore_checkpoint, atomic_torch
from .evaluate import evaluate_model


def data_checks(lock):
    test = ClipDataset(lock['test_manifest'])
    train = ClipDataset(lock['train_manifest'], train=True, seed=0)
    no_aug = ClipDataset(lock['train_manifest'], train=False, seed=0)
    assert len(test) == 4795 and len(train) == 8982
    boundary = [test.sequences[s][0] for s in test.sequences]
    for index in boundary:
        assert len(set(test.clip_indices(index))) == 1
        item = test[index]
        assert all(torch.equal(item['img'][0], item['img'][j]) for j in range(1, 5))
        rgb = item['img'][5:].permute(1, 2, 0).numpy()
        # Native grayscale conversion precedes resize; allow interpolation rounding.
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        assert np.abs(grey.astype(float) - item['img'][4].numpy()).max() <= 1
    for index in range(16):
        a, b = train[index], no_aug[index]
        if a['flipped']:
            assert torch.equal(a['img'], b['img'].flip(-1))
            expected = b['bboxes_t5'].clone()
            expected[:, 0] = 1 - expected[:, 0]
        else:
            assert torch.equal(a['img'], b['img'])
            expected = b['bboxes_t5']
        torch.testing.assert_close(a['bboxes_t5'], expected)
    common = read_json(WORK / 'evidence/data/DAUB_val_temporal_common.json')
    for stride in [1, 2, 4]:
        subset = ClipDataset(lock['test_manifest'], temporal_stride=stride, common_only=True)
        assert len(subset) == 4683
        for i, expected in enumerate(common):
            actual = [subset.rows[j]['file'] for j in subset.clip_indices(i)]
            assert actual == expected['clips'][str(stride)]
    image = np.zeros((80, 120, 8), dtype=np.uint8)
    box = np.array([[7., 9., 17., 22.]], dtype=np.float32)
    resized, transformed, ratio = letterbox(image, box, 640)
    restored = scale_boxes((640, 640), torch.from_numpy(transformed), (80, 120), ratio_pad=ratio)
    torch.testing.assert_close(restored, torch.from_numpy(box), rtol=0, atol=1e-5)
    return {'passed': True, 'train_frames': len(train), 'test_unique_frames': len(test),
            'sequence_start_replication_checks': len(boundary), 'clip_flip_and_label_checks': 16,
            'P1_common_frames_per_stride': 4683, 'non_square_coordinate_roundtrip': True}


def step(model, optimizer, ema, batch):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss, parts = model(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(parts).all()
    loss.sum().backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
    optimizer.step()
    ema.update(model)
    return parts.detach().cpu().tolist(), float(norm)


def run():
    started = time.monotonic()
    lock = read_json(SERIES / 'protocol.json')
    report = {'formal_training': False, 'formal_metrics': False, 'started_at': now(), 'passed': False}
    report['data'] = data_checks(lock)
    atomic_json(SERIES / 'preflight/result.json', report)
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dataset = ClipDataset(lock['train_manifest'], train=True, seed=0)
    batch = to_device(collate([dataset[i] for i in range(8)]), 'cuda:0')
    results = []
    for task_id, historical in [('B_FULL_0', 'frame5'), ('B_3DCONV_0', 'conv3d-TIMSSM'),
                                ('E1', 'sste_stage'), ('E2', None), ('E3', None)]:
        init_seeds(0, deterministic=True)
        task = load_task(task_id)
        model = SSEModel(task, device='cuda:0', verify_stride=True)
        architecture = architecture_summary(model)
        if historical:
            expected = read_json(next((WORK / 'evidence/checkpoints').glob('*DAUB__' + historical + '__weights__best.json')))
            assert architecture['state_shapes'] == expected['state_shapes'], task_id
        optimizer = build_optimizer(model, lock)
        ema = ModelEMA(model)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.monotonic()
        losses = [step(model, optimizer, ema, batch)[0] for _ in range(2)]
        torch.cuda.synchronize()
        result = {'task': task_id, 'passed': True, 'batch': 8, 'imgsz': 640, 'loss_components': losses,
                  'two_training_steps_elapsed_s': time.monotonic() - t,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'architecture': architecture}
        if task_id == 'B_FULL_0':
            # Verify an accumulation boundary, including genuinely pending gradients.
            optimizer.zero_grad(set_to_none=True)
            pending_loss, _ = model(batch)
            pending_loss.sum().backward()
            def finish_accumulation():
                loss, parts = model(batch)
                loss.sum().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                return parts.detach().cpu().tolist()
            checkpoint = make_checkpoint(task, lock, model, ema, optimizer, 0, 1, 0.)
            checkpoint_path = SERIES / 'preflight/resume_smoke.pt'
            atomic_torch(checkpoint_path, checkpoint)
            del checkpoint
            loss_expected = finish_accumulation()
            expected_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            restore_checkpoint(checkpoint, task, lock, model, ema, optimizer)
            loss_actual = finish_accumulation()
            np.testing.assert_allclose(loss_actual, loss_expected, rtol=1e-5, atol=1e-6)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), expected_state[key], rtol=1e-5, atol=1e-6)
            result['resume_continuation'] = 'passed including pending gradients; rtol=1e-5, atol=1e-6, custom CUDA deterministic limitations retained'
            smoke_set = ClipDataset(lock['train_manifest'], train=False, limit=8)
            result['evaluation_smoke'] = evaluate_model(ema.ema, smoke_set, SERIES / 'preflight/evaluation',
                                                        'SMOKE_ONLY_UNTRAINED', lock['evaluation'], formal=False, workers=0)
            del checkpoint, expected_state
        results.append(result)
        print('PREFLIGHT PASS', task_id, result['peak_allocated_bytes'], flush=True)
        atomic_json(SERIES / 'preflight/' / (task_id + '.json'), result)
        del model, optimizer, ema
        gc.collect()
        torch.cuda.empty_cache()
    report.update(passed=True, models=results, gpu=torch.cuda.get_device_name(0),
                  physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), elapsed_s=time.monotonic() - started,
                  completed_at=now(), loss_checks='All six coefficient changes previously passed actual v8DetectionLoss checks; new model smoke uses the same loss')
    atomic_json(SERIES / 'preflight/result.json', report)
    print('All new-protocol preflight checks passed', flush=True)


if __name__ == '__main__':
    run()
