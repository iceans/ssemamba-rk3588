"""Run one approved independent experiment, with exact-state epoch-boundary resume."""
import argparse
import fcntl
import math
import os
import random
import time
import traceback
from pathlib import Path

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from ultralytics.utils.torch_utils import init_seeds, ModelEMA
from .common import SERIES, now, atomic_json, read_json, sha, load_task, assert_frozen, run_identity
from .data import ClipDataset, loader, to_device
from .model import SSEModel, architecture_summary


def build_optimizer(model, lock):
    # Match the local trainer's parameter grouping and Adam defaults.
    groups = [[], [], []]
    norm_types = tuple(v for name, v in torch.nn.__dict__.items() if 'Norm' in name and isinstance(v, type))
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f'{module_name}.{param_name}'
            if 'bias' in fullname:
                groups[2].append(param)
            elif isinstance(module, norm_types) or 'logit_scale' in fullname:
                groups[1].append(param)
            else:
                groups[0].append(param)
    optimizer = torch.optim.Adam(groups[2], lr=lock['lr0'], betas=tuple(lock['betas']), weight_decay=0.)
    optimizer.add_param_group({'params': groups[0], 'weight_decay': lock['weight_decay']})
    optimizer.add_param_group({'params': groups[1], 'weight_decay': 0.})
    return optimizer


def learning_rate(lock, epoch, microstep, batches):
    lr = lock['lr0'] * (lock['lrf'] + (1 - lock['lrf']) * (1 + math.cos(math.pi * epoch / (lock['epochs'] - 1))) / 2)
    warmup = max(round(lock['warmup_epochs'] * batches), 100)
    accumulation = lock['steady_accumulation']
    if microstep <= warmup:
        fraction = microstep / warmup
        lr *= fraction
        accumulation = max(1, round(1 + (lock['steady_accumulation'] - 1) * fraction))
    return lr, accumulation


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def random_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_random(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def make_checkpoint(task, lock, model, ema, optimizer, epoch, last_opt_step, elapsed):
    return {'format': 'sse_series_b_resume_v1', 'epoch_completed': epoch + 1,
            'task': task, 'source_sha256': lock['source_sha256'],
            'model_state': cpu_state(model), 'ema_state': cpu_state(ema.ema), 'ema_updates': ema.updates,
            'optimizer': optimizer.state_dict(),
            'gradients': {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None},
            'last_opt_step': last_opt_step, 'rng': random_state(), 'elapsed_training_s': elapsed,
            'precision': 'float32', 'checkpoint_selection': lock['checkpoint_selection'],
            'training_test_evaluations': 0}


def restore_checkpoint(checkpoint, task, lock, model, ema, optimizer):
    assert checkpoint['format'] == 'sse_series_b_resume_v1'
    assert checkpoint['task']['config_sha256'] == task['config_sha256']
    assert checkpoint['source_sha256'] == lock['source_sha256']
    model.load_state_dict(checkpoint['model_state'], strict=True)
    ema.ema.load_state_dict(checkpoint['ema_state'], strict=True)
    ema.updates = checkpoint['ema_updates']
    optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        if name in checkpoint['gradients']:
            param.grad = checkpoint['gradients'][name].to(param.device)
    restore_random(checkpoint['rng'])


def run(task_id, resume=False):
    task = load_task(task_id)
    lock = assert_frozen(task)
    out = Path(task['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(task, lock, out, resume)


def run_locked(task, lock, out, resume):
    record_path = out / 'run.json'
    previous = read_json(record_path) if record_path.exists() else {}
    if previous.get('status') in ['trained', 'evaluated', 'verified', 'complete']:
        print('Already trained:', task['id'], flush=True)
        return
    if previous and not resume:
        raise RuntimeError('Existing attempt retained. Pass --resume to restore a compatible epoch checkpoint.')
    assert torch.cuda.is_available(), 'CUDA access is required'
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    init_seeds(task['seed'], deterministic=True)
    start = time.monotonic()
    record = {'id': task['id'], 'status': 'running', 'started_at': now(), 'pid': os.getpid(),
              'config_sha256': task['config_sha256'], 'source_sha256': lock['source_sha256'],
              'run_identity_sha256': run_identity(task, lock),
              'protocol_id': lock['protocol_id'], 'seed': task['seed'], 'model': task['model'],
              'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': torch.cuda.get_device_name(0),
              'formal_training': True, 'formal_test_verified': False, 'epochs_planned': task['epochs'],
              'attempts': previous.get('attempts', []) + [{'started_at': now(), 'pid': os.getpid()}],
              'test_used_during_training': False, 'checkpoint_selection': lock['checkpoint_selection']}
    atomic_json(record_path, record)
    offset_elapsed = 0.
    try:
        model = SSEModel(task, device='cuda:0', verify_stride=True)
        for name, param in model.named_parameters():
            if '.dfl.' in name:
                param.requires_grad_(False)
        atomic_json(out / 'architecture.json', architecture_summary(model))
        atomic_json(out / 'resolved_config.json', {'task': task, 'protocol': lock})
        optimizer = build_optimizer(model, lock)
        ema = ModelEMA(model, decay=lock['ema']['decay'], tau=lock['ema']['tau'])
        weights = out / 'weights'
        weights.mkdir(exist_ok=True)
        last_path = weights / 'last.pt'
        start_epoch, last_opt_step = 0, -1
        optimizer.zero_grad(set_to_none=True)
        if resume and last_path.exists():
            checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
            restore_checkpoint(checkpoint, task, lock, model, ema, optimizer)
            start_epoch = checkpoint['epoch_completed']
            last_opt_step = checkpoint['last_opt_step']
            offset_elapsed = checkpoint['elapsed_training_s']
            record['resume'] = {'checkpoint_sha256': sha(last_path), 'epoch_completed': start_epoch,
                                'restored': ['model', 'EMA', 'optimizer', 'gradients', 'RNG', 'step counters']}
            del checkpoint
        elif resume and previous:
            record['resume'] = {'checkpoint': None, 'action': 'restart identical configuration and seed; no completed epoch checkpoint'}
        atomic_json(record_path, record)
        dataset = ClipDataset(lock['train_manifest'], train=True, seed=task['seed'])
        batches = math.ceil(len(dataset) / task['batch'])
        assert len(dataset) == lock['train_frames'] and batches == 1123
        torch.cuda.reset_peak_memory_stats()
        epoch_history_path = out / 'epochs.jsonl'
        for epoch in range(start_epoch, task['epochs']):
            dataset.epoch = epoch
            data_loader = loader(dataset, batch=task['batch'], workers=task['workers'])
            model.train()
            totals = torch.zeros(3, device='cuda:0')
            epoch_start = time.monotonic()
            seen = 0
            for batch_index, batch in enumerate(data_loader):
                microstep = epoch * batches + batch_index
                lr, accumulation = learning_rate(lock, epoch, microstep, batches)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                batch = to_device(batch, 'cuda:0')
                loss, parts = model(batch)
                loss = loss.sum()
                if not torch.isfinite(loss).item() or not torch.isfinite(parts).all().item():
                    raise FloatingPointError(f'Nonfinite loss at epoch={epoch + 1}, batch={batch_index}')
                loss.backward()
                if microstep - last_opt_step >= accumulation:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), lock['gradient_clip_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                    last_opt_step = microstep
                totals += parts.detach()
                seen += batch['img'].shape[0]
                if batch_index % 20 == 0 or batch_index == batches - 1:
                    progress = {'id': task['id'], 'status': 'running', 'updated_at': now(),
                                'epoch': epoch + 1, 'epochs': task['epochs'], 'batch': batch_index + 1, 'batches': batches,
                                'mean_loss_components': (totals / (batch_index + 1)).tolist(), 'lr': lr,
                                'accumulation': accumulation, 'elapsed_s': offset_elapsed + time.monotonic() - start,
                                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'pid': os.getpid()}
                    atomic_json(out / 'progress.json', progress)
                    print(f"{task['id']} epoch {epoch + 1}/100 batch {batch_index + 1}/{batches} "
                          f"loss={progress['mean_loss_components']} lr={lr:.8g}", flush=True)
            assert seen == lock['train_frames']
            elapsed = offset_elapsed + time.monotonic() - start
            epoch_result = {'epoch': epoch + 1, 'mean_loss_components': (totals / batches).tolist(),
                            'samples': seen, 'batches': batches, 'lr': lr, 'ema_updates': ema.updates,
                            'epoch_elapsed_s': time.monotonic() - epoch_start, 'total_elapsed_s': elapsed,
                            'completed_at': now(), 'attempt': len(record['attempts'])}
            checkpoint = make_checkpoint(task, lock, model, ema, optimizer, epoch, last_opt_step, elapsed)
            atomic_torch(last_path, checkpoint)
            if (epoch + 1) % 25 == 0:
                atomic_torch(weights / f'epoch{epoch + 1:03d}.pt', checkpoint)
            del checkpoint
            with epoch_history_path.open('a') as f:
                import json
                f.write(json.dumps(epoch_result) + '\n')
            record.update(epoch_completed=epoch + 1, elapsed_training_s=elapsed, updated_at=now(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
            atomic_json(record_path, record)
            del data_loader
        selected = weights / 'selected_final.pt'
        atomic_torch(selected, {'format': 'sse_series_b_selected_v1', 'task': task,
                     'source_sha256': lock['source_sha256'], 'epoch_completed': 100,
                     'state_dict': cpu_state(ema.ema), 'ema_updates': ema.updates,
                     'selection': lock['checkpoint_selection'], 'test_used_during_training': False})
        record.update(status='trained', completed_at=now(), epoch_completed=100,
                      selected_checkpoint=str(selected), selected_checkpoint_sha256=sha(selected),
                      elapsed_training_s=offset_elapsed + time.monotonic() - start,
                      optimizer_steps=ema.updates)
        record['attempts'][-1].update(status='trained', elapsed_s=time.monotonic() - start)
        atomic_json(record_path, record)
        print(f"TRAINED {task['id']}: fixed epoch 100, awaiting independent TEST task", flush=True)
    except Exception:
        record.update(status='failed', error=traceback.format_exc(), failed_at=now(),
                      attempt_elapsed_s=time.monotonic() - start)
        record['attempts'][-1].update(status='failed', elapsed_s=time.monotonic() - start)
        atomic_json(record_path, record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.task, resume=args.resume)
