"""Persistent two-GPU queue for the approved fifteen runs and their test tasks."""
import argparse
import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

from .common import ROOT, WORK, SERIES, PYTHON, now, read_json, atomic_json, assert_frozen
from .refresh import refresh, optional


class ExistingProcess:
    """Track a still-running child across a manager restart without restarting training."""
    def __init__(self, pid, record_path):
        self.pid, self.record_path = pid, record_path
        self.start_ticks = self._identity()[1]

    def _identity(self):
        try:
            fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')', 1)[1].split()
            return fields[0], fields[19]
        except FileNotFoundError:
            return None, None

    def poll(self):
        state, ticks = self._identity()
        if ticks == self.start_ticks and state not in [None, 'Z']:
            return None
        return 0 if optional(self.record_path).get('status') in ['trained', 'evaluated', 'verified', 'complete'] else 1


def adopt_running(tasks, gpus):
    active = {}
    for task in tasks:
        out = Path(task['output_directory'])
        for phase in ['train', 'TEST', 'P1_s1', 'P1_s2', 'P1_s4', 'P3']:
            record_path = out / 'run.json' if phase == 'train' else out / phase / 'run.json'
            record = optional(record_path)
            pid = record.get('pid')
            if record.get('status') != 'running' or not pid:
                continue
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            except FileNotFoundError:
                continue
            module = 'train' if phase == 'train' else 'external' if phase == 'P3' else 'evaluate'
            if f'revision_work.new_protocol.{module}' not in command or '--task' not in command:
                continue
            if command[command.index('--task') + 1] != task['id']:
                continue
            gpu = int(record['physical_gpu'])
            assert gpu in gpus and gpu not in active, 'Conflicting running tasks for this GPU'
            process = ExistingProcess(pid, record_path)
            if process.poll() is not None:
                continue
            stdout = (out / (phase + '.stdout.log')).open('a', buffering=1)
            active[gpu] = {'task': task['id'], 'phase': phase, 'gpu': gpu, 'pid': pid,
                           'process': process, 'stdout': stdout, 'record_path': record_path,
                           'started_at': record['started_at'], 'adopted_existing_process': True}
            print('Adopted existing', task['id'], phase, 'PID', pid, 'GPU', gpu, flush=True)
    return active


def gpu_state():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for row in rows.splitlines():
        index, uuid, free = [v.strip() for v in row.split(',')]
        devices[int(index)] = {'uuid': uuid, 'free_MiB': float(free), 'compute_pids': []}
    for row in active.splitlines():
        fields = [v.strip() for v in row.split(',')]
        if len(fields) == 2 and fields[1].isdigit():
            for device in devices.values():
                if device['uuid'] == fields[0]:
                    device['compute_pids'].append(int(fields[1]))
    return devices


def next_job(task):
    out = Path(task['output_directory'])
    record = optional(out / 'run.json')
    state = record.get('status')
    if state == 'failed':
        return None
    if state not in ['trained', 'evaluated', 'verified', 'complete']:
        if state == 'running':
            pid = record.get('pid')
            if pid and Path(f'/proc/{pid}').exists():
                return None
        return ('train', [PYTHON, '-m', 'revision_work.new_protocol.train', '--task', task['id'], '--resume'], out / 'run.json')
    test = optional(out / 'TEST/run.json')
    if test.get('status') == 'failed':
        return None
    if test.get('status') != 'verified':
        return ('TEST', [PYTHON, '-m', 'revision_work.new_protocol.evaluate', '--task', task['id']], out / 'TEST/run.json')
    if task['id'] in ['B_FULL_0', 'B_3DCONV_0']:
        for stride in [1, 2, 4]:
            record_path = out / f'P1_s{stride}/run.json'
            state = optional(record_path).get('status')
            if state not in ['verified', 'failed']:
                return (f'P1_s{stride}', [PYTHON, '-m', 'revision_work.new_protocol.evaluate', '--task', task['id'],
                                        '--temporal-stride', str(stride)], record_path)
        if optional(SERIES / 'external/protocol.json').get('status') == 'verified':
            record_path = out / 'P3/run.json'
            if optional(record_path).get('status') not in ['verified', 'failed']:
                return ('P3', [PYTHON, '-m', 'revision_work.new_protocol.external', '--task', task['id']], record_path)
    return None


def start_job(task, job, gpu):
    phase, command, record_path = job
    out = Path(task['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    stdout = (out / (phase + '.stdout.log')).open('a', buffering=1)
    stdout.write(f'\nAttempt started {now()} physical GPU={gpu}\n')
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(task['seed']),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       YOLO_AUTOINSTALL='false', MPLCONFIGDIR=str(SERIES / 'matplotlib_config'))
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=subprocess.STDOUT)
    return {'task': task['id'], 'phase': phase, 'process': process, 'stdout': stdout,
            'record_path': record_path, 'gpu': gpu, 'started_at': now(), 'pid': process.pid}


def main(gpus):
    guard = (SERIES / '.queue.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = sorted([read_json(p) for p in (SERIES / 'configs').glob('*.json')], key=lambda t: t['priority'])
    assert len(tasks) == 15
    for task in tasks:
        assert_frozen(task)
    atomic_json(SERIES / 'queue.json', {'status': 'running', 'pid': os.getpid(), 'started_at': now(), 'gpus': gpus})
    active = adopt_running(tasks, gpus)
    while True:
        for gpu, item in list(active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['stdout'].close()
            record = optional(item['record_path'])
            if code != 0:
                record.update(status='failed', exit_code=code, failed_at=now(),
                              queue_note='Other independent tasks remain scheduled; failure is not a completed result')
                atomic_json(item['record_path'], record)
            print('Finished', item['task'], item['phase'], 'exit', code, flush=True)
            del active[gpu]
        counts = refresh()
        state = gpu_state()
        free_bytes = shutil.disk_usage(WORK).free
        if free_bytes < 12 * 2**30:
            reason = 'Less than 12 GiB disk free; no additional runs launched'
        else:
            reason = None
            in_flight = {item['task'] for item in active.values()}
            for gpu in gpus:
                if gpu in active or state[gpu]['compute_pids'] or state[gpu]['free_MiB'] < 10000:
                    continue
                for task in tasks:
                    if task['id'] in in_flight:
                        continue
                    job = next_job(task)
                    if job is not None:
                        active[gpu] = start_job(task, job, gpu)
                        in_flight.add(task['id'])
                        print('Started', task['id'], job[0], 'on GPU', gpu, flush=True)
                        break
        pending = any(next_job(task) is not None for task in tasks if task['id'] not in {v['task'] for v in active.values()})
        snapshot = {'status': 'running', 'pid': os.getpid(), 'updated_at': now(), 'gpus': gpus,
                    'active': [{k: v for k, v in item.items() if k not in ['process', 'stdout', 'record_path']} for item in active.values()],
                    'device_snapshot': state, 'disk_free_bytes': free_bytes, 'waiting_reason': reason, 'counts': counts}
        atomic_json(SERIES / 'queue.json', snapshot)
        if not active and not pending:
            break
        time.sleep(15)
    # Speed measurements only after all our training/evaluation processes have ended.
    for task_id in ['B_FULL_0', 'B_3DCONV_0']:
        task = next(t for t in tasks if t['id'] == task_id)
        out = Path(task['output_directory'])
        if optional(out / 'run.json').get('status') not in ['trained', 'evaluated', 'verified', 'complete']:
            continue
        if optional(out / 'P4/run.json').get('status') == 'verified':
            continue
        state = gpu_state()
        if any(device['compute_pids'] for device in state.values()):
            atomic_json(out / 'P4/run.json', {'status': 'blocked', 'reason': 'Other compute workloads present; rerun queue after they finish'})
            continue
        job = ('P4', [PYTHON, '-m', 'revision_work.new_protocol.evaluate', '--task', task_id, '--benchmark'], out / 'P4/run.json')
        item = start_job(task, job, gpus[0])
        code = item['process'].wait()
        item['stdout'].close()
        print('Benchmark', task_id, 'exit', code, flush=True)
    counts = refresh()
    atomic_json(SERIES / 'queue.json', {'status': 'finished', 'pid': os.getpid(), 'completed_at': now(), 'counts': counts,
                'scope': 'approved training, corresponding TEST, available-model P1/P3 and isolated P4',
                'remaining_gaps': 'unapproved component repair runs, noninterleaved checkpoint, DFAR provenance, scene/manuscript metadata; consult registry for any failed or blocked inference rows'})
    print('Queue finished', counts, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, default=[1, 0])
    args = parser.parse_args()
    main(args.gpus)
