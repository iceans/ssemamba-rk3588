"""Formal final-epoch evaluation and fixed-seed temporal mismatch tests."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
import numpy as np
import torch

from ultralytics.models.yolo.detect.val import DetectionValidator
from .common import SERIES, now, atomic_json, read_json, sha, digest, load_task, assert_frozen
from .data import ClipDataset, loader, to_device
from .model import SSEModel


@torch.inference_mode()
def evaluate_model(model, dataset, out, checkpoint_source, config, formal=True, workers=8):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validator = DetectionValidator(save_dir=out, args={
        'task': 'detect', 'mode': 'val', 'imgsz': 640, 'batch': 8,
        'conf': config['conf'], 'iou': config['nms_iou'], 'max_det': config['max_det'],
        'half': False, 'plots': False, 'save_json': False, 'split': 'test',
        'cube': True, 'frame_num': 5, 'model': checkpoint_source, 'verbose': False})
    validator.device = next(model.parameters()).device
    validator.training = False
    validator.data = {'test': 'DAUB/test/images', 'nc': 1, 'names': {0: 'target'}, 'channels': 8}
    validator.init_metrics(model)
    cache = out / 'per_frame_predictions.jsonl'
    temporary = out / 'per_frame_predictions.partial.jsonl'
    keys, sample_info = [], []
    false_positives, empty = 0, 0
    sequence_stats = {}
    model.eval()
    start = time.monotonic()
    with temporary.open('w') as handle:
        for step, batch in enumerate(loader(dataset, batch=config['batch'], workers=workers)):
            batch = to_device(batch, validator.device)
            output = model(batch['img'])
            predictions = validator.postprocess(output)
            validator.update_metrics(predictions, batch)
            for i, pred in enumerate(predictions):
                prepared = validator._prepare_batch(i, batch)
                native = validator.scale_preds(pred, prepared)
                classes = native['cls'].cpu().tolist()
                entry = {'seq_id': batch['seq_id'][i], 'frame_id': batch['frame_id'][i],
                         'file': batch['im_file'][i], 'original_hw': list(batch['ori_shape'][i]),
                         'clip_files': batch['clip_files'][i],
                         'boxes_xyxy': native['bboxes'].cpu().tolist(), 'classes': classes,
                         'confidences': native['conf'].cpu().tolist(), 'sample_index': len(keys),
                         'model_source': checkpoint_source, 'formal_result': formal}
                handle.write(json.dumps(entry) + '\n')
                keys.append((entry['seq_id'], entry['frame_id']))
                sample_info.append([entry['file'], entry['clip_files'], entry['original_hw']])
                empty += not classes
                keep = pred['conf'] >= config['working_threshold']
                threshold_pred = {k: v[keep] for k, v in pred.items()}
                tp = validator._process_batch(threshold_pred, prepared)['tp'][:, 0]
                fp = len(tp) - int(tp.sum())
                false_positives += fp
                seq = sequence_stats.setdefault(entry['seq_id'], {'frames': 0, 'false_positives': 0})
                seq['frames'] += 1
                seq['false_positives'] += fp
            if step % 50 == 0:
                print(f'EVAL {out.name}: {len(keys)}/{len(dataset)} frames', flush=True)
    assert len(keys) == len(set(keys)) == len(dataset), 'Evaluation keyframes duplicated or omitted'
    os.replace(temporary, cache)
    metrics = validator.get_stats()
    validator.finalize_metrics()
    assert all(np.isfinite(float(value)) for value in metrics.values())
    for row in sequence_stats.values():
        row['false_positives_per_frame'] = row['false_positives'] / row['frames']
    result = {'status': 'evaluated', 'formal_result': formal, 'metrics': metrics,
              'samples': len(keys), 'unique_keyframes': len(set(keys)), 'empty_prediction_frames': empty,
              'elapsed_s': time.monotonic() - start, 'prediction_cache': str(cache), 'prediction_cache_sha256': sha(cache),
              'evaluated_sample_hash': digest(sample_info), 'common_keyframe_hash': digest(keys), 'false_positives': false_positives,
              'false_positives_per_frame': false_positives / len(keys),
              'working_threshold': config['working_threshold'], 'working_iou': config['working_iou'],
              'sequence_false_alarms': sequence_stats}
    atomic_json(out / 'evaluation.json', result)
    return result


def selected_model(task, lock):
    record = read_json(Path(task['output_directory']) / 'run.json')
    assert record['status'] in ['trained', 'evaluated', 'verified', 'complete']
    path = record['selected_checkpoint']
    assert sha(path) == record['selected_checkpoint_sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch_completed'] == 100 and not checkpoint['test_used_during_training']
    assert checkpoint['task']['config_sha256'] == task['config_sha256']
    assert checkpoint['source_sha256'] == lock['source_sha256']
    model = SSEModel(task, device='cuda:0')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), checkpoint['state_dict'][key], rtol=0, atol=0)
    return model.eval(), path


def run(task_id, temporal_stride=None, benchmark=False):
    task = load_task(task_id)
    lock = assert_frozen(task)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    suffix = 'P4' if benchmark else f'P1_s{temporal_stride}' if temporal_stride else 'TEST'
    out = Path(task['output_directory']) / suffix
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'run.json'
    if result_path.exists() and read_json(result_path).get('status') == 'verified':
        old = read_json(result_path)
        assert old['config_sha256'] == task['config_sha256']
        assert old['source_sha256'] == lock['source_sha256']
        if old.get('prediction_cache'):
            assert sha(old['prediction_cache']) == old['prediction_cache_sha256']
        print('Verified evaluation already exists:', out, flush=True)
        return
    record = {'id': f'{suffix}_{task_id}', 'training_id': task_id, 'status': 'running', 'started_at': now(),
              'pid': os.getpid(), 'protocol_id': lock['protocol_id'], 'seed': task['seed'],
              'config_sha256': task['config_sha256'], 'source_sha256': lock['source_sha256'],
              'formal_result': True, 'selection': lock['checkpoint_selection'],
              'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'temporal_stride': temporal_stride or 1}
    atomic_json(result_path, record)
    try:
        model, path = selected_model(task, lock)
        record.update(checkpoint=path, checkpoint_sha256=sha(path), strict_load='all keys, shapes and values verified')
        if benchmark:
            import subprocess
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
            other = [int(p.strip()) for p in active.splitlines() if p.strip().isdigit() and int(p.strip()) != os.getpid()]
            if other:
                raise RuntimeError(f'Competing compute processes; benchmark must wait: {other}')
            x = torch.zeros(1, 8, 640, 640, device='cuda:0')
            with torch.inference_mode():
                for _ in range(50):
                    model(x)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for _ in range(950):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
            record.update(status='verified', gpu=torch.cuda.get_device_name(0), precision='FP32',
                          batch=1, frames=5, channels=8, imgsz=640, warmup=50, iterations=950,
                          preprocess_and_nms_included=False, fuse=False, latency_ms=samples,
                          mean_latency_ms=float(np.mean(samples)), throughput_clips_s=1000 / float(np.mean(samples)),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          parameters=sum(p.numel() for p in model.parameters()), FLOPs=None,
                          FLOPs_note='Custom SSM/DCN counting not audited')
        else:
            dataset = ClipDataset(lock['test_manifest'], temporal_stride=temporal_stride or 1,
                                  common_only=temporal_stride is not None)
            expected = 4683 if temporal_stride else lock['test_frames']
            assert len(dataset) == expected
            result = evaluate_model(model, dataset, out, path, lock['evaluation'], formal=True)
            record.update(result, status='verified', split_hash=lock['test_manifest_sha256'])
            if temporal_stride is None:
                # All DAUB GT boxes fall in this one bin; no out-of-bin GT exist to ignore.
                assert all(b['max_side_px'] > 7 for row in dataset.rows for b in row['boxes'])
                record['size_groups'] = [
                    {'group': 'd_le_3', 'boxes': 0, 'AP50': None, 'AP50-95': None},
                    {'group': '3_lt_d_le_7', 'boxes': 0, 'AP50': None, 'AP50-95': None},
                    {'group': 'd_gt_7', 'boxes': 4795, 'AP50': result['metrics']['metrics/mAP50(B)'],
                     'AP50-95': result['metrics']['metrics/mAP50-95(B)']}]
                record['group_ignore_rule'] = 'Other GT bins are empty in DAUB; >7 is exactly full-set evaluation, empty bins have undefined AP'
        record['completed_at'] = now()
        atomic_json(result_path, record)
        print('VERIFIED', record['id'], flush=True)
    except Exception:
        record.update(status='failed', error=traceback.format_exc(), failed_at=now())
        atomic_json(result_path, record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--temporal-stride', type=int, choices=[1, 2, 4])
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    run(args.task, args.temporal_stride, args.benchmark)
