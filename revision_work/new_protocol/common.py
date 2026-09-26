import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'revision_work'
SERIES = WORK / 'series_b'
PYTHON = '/opt/anaconda3/envs/mambayolo/bin/python'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def load_task(task_id):
    return read_json(SERIES / 'configs' / (task_id + '.json'))


def run_identity(task, lock):
    return digest({'id': task['id'], 'seed': task['seed'], 'config_sha256': task['config_sha256'],
                   'source_sha256': lock['source_sha256'], 'train_split_sha256': lock['train_manifest_sha256'],
                   'test_split_sha256': lock['test_manifest_sha256']})


def assert_frozen(task):
    lock = read_json(SERIES / 'protocol.json')
    assert lock['approved'] and lock['formal_training_locked']
    assert task['protocol_id'] == lock['protocol_id']
    clean = {k: v for k, v in task.items() if k != 'config_sha256'}
    assert digest(clean) == task['config_sha256'], 'Run configuration changed'
    assert sha(task['model_yaml']) == task['model_yaml_sha256']
    for file, expected in lock['source_files'].items():
        assert sha(ROOT / file) == expected, f'Frozen source changed: {file}'
    for name in ['train_manifest', 'test_manifest']:
        assert sha(lock[name]) == lock[name + '_sha256']
    return lock
