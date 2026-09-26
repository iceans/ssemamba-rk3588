"""Freeze the official NUDT-MIRSDT test split and evaluate DAUB final weights."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import traceback
import zipfile

import cv2
import numpy as np

from .common import ROOT, WORK, SERIES, atomic_json, assert_frozen, digest, load_task, now, read_json, sha

EVIDENCE = WORK / 'evidence/external/NUDT-MIRSDT'
EXTERNAL = SERIES / 'external'
LOCK = EXTERNAL / 'protocol.json'
SELF = Path(__file__).resolve()


def pixel_hash(image):
    if image.ndim == 3 and image.shape[2] == 3:
        if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2]):
            image = image[:, :, 0]
    h = hashlib.sha256()
    h.update(str((image.shape, str(image.dtype))).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def mask_boxes(mask):
    assert mask.ndim == 2 and mask.dtype == np.uint8
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        assert w > 0 and h > 0 and area > 0
        boxes.append({'class': 0, 'xyxy': [int(x), int(y), int(x + w), int(y + h)],
                      'max_side_px': int(max(w, h)), 'mask_pixels': int(area)})
    return boxes


def assert_external():
    lock = read_json(LOCK)
    assert lock['status'] == 'verified' and lock['frozen_before_target_model_scores']
    assert sha(SELF) == lock['adapter_sha256'], 'Frozen external adapter changed'
    for key in ['official_test_list', 'official_train_list', 'manifest']:
        assert sha(lock[key]) == lock[key + '_sha256'], key
    rows = [json.loads(line) for line in Path(lock['manifest']).read_text().splitlines()]
    assert len(rows) == lock['frames']
    for row in rows:
        assert sha(row['file']) == row['image_sha256'], row['file']
        assert sha(row['mask_file']) == row['mask_sha256'], row['mask_file']
    return lock


def prepare():
    cv2.setNumThreads(0)
    if LOCK.exists():
        print('Already frozen:', assert_external()['dataset'], flush=True)
        return
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    source_lock = assert_frozen(load_task('B_FULL_0'))
    archive = EVIDENCE / 'NUDT-MIRSDT.zip'
    download = read_json(EVIDENCE / 'download.json')
    assert sha(archive) == download['sha256']
    official = [line.strip() for line in (EVIDENCE / 'test.txt').read_text().splitlines() if line.strip()]
    train = [line.strip() for line in (EVIDENCE / 'train.txt').read_text().splitlines() if line.strip()]
    assert len(official) == len(set(official)) == 2000
    assert len(train) == len(set(train)) == 8000
    test_sequences = {s.split('/')[0] for s in official}
    train_sequences = {s.split('/')[0] for s in train}
    assert len(test_sequences) == 20 and len(train_sequences) == 80
    assert not test_sequences & train_sequences
    tiny = np.zeros((5, 5), np.uint8)
    tiny[0, 0] = 255
    assert mask_boxes(tiny)[0]['xyxy'] == [0, 0, 1, 1]
    tiny[1, 1] = 255
    assert mask_boxes(tiny)[0]['xyxy'] == [0, 0, 2, 2]
    tiny[4, 4] = 255
    assert len(mask_boxes(tiny)) == 2
    frames, image_hash_to_rows = [], defaultdict(list)
    directory = EVIDENCE / 'official_test'
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP members'
        available = set(names)
        for item in official:
            parts = PurePosixPath(item).parts
            assert len(parts) == 3 and parts[1] == 'Mix' and parts[2].endswith('.mat')
            seq, _, basename = parts
            assert seq.startswith('Sequence') and seq[len('Sequence'):].isdigit()
            stem = Path(basename).stem
            assert stem.isdigit()
            files, payloads = {}, {}
            for kind in ['images', 'masks']:
                relative = PurePosixPath('NUDT-MIRSDT') / seq / kind / (stem + '.png')
                assert str(relative) in available, str(relative)
                payload = z.read(str(relative))  # ZipFile validates each extracted member's CRC.
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert target.read_bytes() == payload
                else:
                    target.write_bytes(payload)
                files[kind], payloads[kind] = str(target), payload
            image = cv2.imdecode(np.frombuffer(payloads['images'], np.uint8), cv2.IMREAD_UNCHANGED)
            mask = cv2.imdecode(np.frombuffer(payloads['masks'], np.uint8), cv2.IMREAD_UNCHANGED)
            assert image is not None and image.dtype == np.uint8 and image.ndim == 2
            assert mask is not None and mask.shape == image.shape
            row = {'file': files['images'], 'mask_file': files['masks'], 'seq_id': seq,
                   'frame_id': int(stem), 'height': image.shape[0], 'width': image.shape[1],
                   'boxes': mask_boxes(mask), 'official_split_entry': item,
                   'image_sha256': hashlib.sha256(payloads['images']).hexdigest(),
                   'mask_sha256': hashlib.sha256(payloads['masks']).hexdigest(),
                   'decoded_pixel_sha256': pixel_hash(image)}
            image_hash_to_rows[row['decoded_pixel_sha256']].append(row)
            frames.append(row)
    by_sequence = defaultdict(list)
    for row in frames:
        by_sequence[row['seq_id']].append(row)
    for seq, rows in by_sequence.items():
        assert sorted(r['frame_id'] for r in rows) == list(range(1, 101)), seq
        assert len({(r['height'], r['width']) for r in rows}) == 1, seq
    overlap, source_count, source_sequences = [], 0, set()
    for split in ['train_manifest', 'test_manifest']:
        for line in Path(source_lock[split]).read_text().splitlines():
            row = json.loads(line)
            source_sequences.add(str(row['seq_id']))
            image = cv2.imread(row['file'], cv2.IMREAD_UNCHANGED)
            assert image is not None
            for target in image_hash_to_rows.get(pixel_hash(image), []):
                overlap.append({'source_split': split, 'source_file': row['file'],
                                'target_file': target['file'], 'target_sequence': target['seq_id']})
            source_count += 1
    excluded = sorted({r['target_sequence'] for r in overlap})
    included = [r for r in frames if r['seq_id'] not in excluded]
    assert included, 'All external test sequences overlap the source'
    manifest = EXTERNAL / 'NUDT-MIRSDT_test_frames.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in included))
    atomic_json(EXTERNAL / 'overlap_audit.json', {
        'checked_at': now(), 'source_frames_checked': source_count,
        'source_manifest_sha256': {k: source_lock[k + '_sha256'] for k in ['train_manifest', 'test_manifest']},
        'target_frames_checked': len(frames), 'exact_decoded_pixel_matches': overlap,
        'excluded_entire_sequences': excluded,
        'literal_sequence_identifier_intersection': sorted(source_sequences & test_sequences),
        'limitation': 'Exact decoded-image comparison cannot rule out transformed, cropped or reused background素材. The author README states synthetic data but does not identify every background source.'})
    from .data import ClipDataset
    dataset = ClipDataset(manifest)
    checked = []
    for seq in sorted(by_sequence):
        if seq in excluded:
            continue
        indices = [i for i in range(len(dataset)) if dataset.rows[dataset.indices[i]]['seq_id'] == seq]
        for i in [indices[0], indices[-1]]:
            sample = dataset[i]
            assert sample['img'].shape == (8, 640, 640)
            assert sample['clip_files'][-1] == sample['im_file']
            if i == indices[0]:
                assert len(set(sample['clip_files'])) == 1
            checked.append([seq, sample['frame_id']])
    sizes = Counter('d_le_3' if b['max_side_px'] <= 3 else '3_lt_d_le_7' if b['max_side_px'] <= 7 else 'd_gt_7'
                    for row in included for b in row['boxes'])
    lock = {
        'status': 'verified', 'frozen_at': now(), 'dataset': 'NUDT-MIRSDT',
        'protocol_id': 'P3_NUDT_MIRSDT_OFFICIAL_TEST_20260915',
        'selection_basis': 'Official split, mask availability and accessible author release; no candidate target-model scores used',
        'frozen_before_target_model_scores': True, 'target_model_evaluations_before_freeze': 0,
        'source_protocol_id': source_lock['protocol_id'], 'source_sha256': source_lock['source_sha256'],
        'adapter': str(SELF), 'adapter_sha256': sha(SELF),
        'official_repository': download['official_repository'], 'archive_source': download['source_url'],
        'archive_sha256': download['sha256'], 'archive_bytes': download['bytes'],
        'official_test_list': str(EVIDENCE / 'test.txt'), 'official_test_list_sha256': sha(EVIDENCE / 'test.txt'),
        'official_train_list': str(EVIDENCE / 'train.txt'), 'official_train_list_sha256': sha(EVIDENCE / 'train.txt'),
        'official_test_frames': len(frames), 'official_test_sequences': len(test_sequences),
        'split': 'official test' if not excluded else 'official test excluding source-overlap sequences',
        'frames': len(included), 'sequences': len({r['seq_id'] for r in included}),
        'manifest': str(manifest), 'manifest_sha256': sha(manifest), 'excluded_sequences': excluded,
        'annotation_source': 'Official masks; Mix/NNNNN.mat test-list entries map to images/NNNNN.png and masks/NNNNN.png',
        'box_conversion': 'mask > 0; 8-connected components; half-open [xmin,ymin,xmax+1,ymax+1]; each component one class-0 GT; no GT confidence',
        'GT_boxes': sum(len(r['boxes']) for r in included), 'GT_size_counts': dict(sizes),
        'empty_GT_frames': sum(not r['boxes'] for r in included),
        'input': 'Source ClipDataset unchanged: five chronological grayscale frames, keyframe RGB replicated from grayscale, first-frame replication, letterbox640, uint8/255',
        'sampling': 'One unique clip per official test keyframe; stride1; frames1..100 verified for every included sequence',
        'boundary_shape_checks': checked, 'evaluation': source_lock['evaluation'],
        'source_models': {'full': 'B_FULL_0', 'STF-3DConv': 'B_3DCONV_0'},
        'checkpoint_rule': 'DAUB seed0 fixed epoch100 EMA, chosen before target evaluation; no target training, tuning or seed/epoch selection',
        'DFAR': 'blocked: source-training provenance not verified',
        'independence_limitation': 'No claim of complete background-source independence; exact duplicate audit and unresolved transformed/shared background provenance are reported separately',
        'release_note': 'Author README mentions120 sequences; published lists define80 train and20 test. Evaluation follows the20 listed sequences only.'}
    atomic_json(LOCK, lock)
    print(json.dumps({k: lock[k] for k in ['status', 'dataset', 'frames', 'sequences', 'GT_boxes', 'GT_size_counts', 'excluded_sequences']}), flush=True)


def run(task_id):
    assert task_id in ['B_FULL_0', 'B_3DCONV_0']
    task = load_task(task_id)
    source_lock = assert_frozen(task)
    lock = assert_external()
    assert lock['source_sha256'] == source_lock['source_sha256']
    out = Path(task['output_directory']) / 'P3'
    out.mkdir(parents=True, exist_ok=True)
    guard = (out / '.run.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result_path = out / 'run.json'
    if result_path.exists():
        old = read_json(result_path)
        if old.get('status') == 'verified':
            assert old['external_protocol_sha256'] == sha(LOCK)
            assert old['config_sha256'] == task['config_sha256']
            assert sha(old['checkpoint']) == old['checkpoint_sha256']
            assert sha(old['prediction_cache']) == old['prediction_cache_sha256']
            print('Verified external evaluation already exists:', task_id, flush=True)
            return
    record = {'id': 'P3_' + task_id, 'training_id': task_id, 'status': 'running', 'pid': os.getpid(),
              'started_at': now(), 'dataset': lock['dataset'], 'split': lock['split'],
              'seed': task['seed'], 'source_dataset': 'DAUB', 'source_protocol_id': source_lock['protocol_id'],
              'source_sha256': source_lock['source_sha256'], 'config_sha256': task['config_sha256'],
              'protocol_id': lock['protocol_id'], 'external_protocol_sha256': sha(LOCK),
              'external_adapter_sha256': lock['adapter_sha256'], 'split_hash': lock['manifest_sha256'],
              'selection': lock['checkpoint_rule'], 'target_training': False, 'target_tuning': False,
              'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'formal_result': True,
              'independence_limitation': lock['independence_limitation']}
    atomic_json(result_path, record)
    try:
        import torch
        from .data import ClipDataset
        from .evaluate import evaluate_model, selected_model
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, checkpoint = selected_model(task, source_lock)
        record.update(checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint), strict_load='all keys, shapes and values verified')
        dataset = ClipDataset(lock['manifest'])
        result = evaluate_model(model, dataset, out, checkpoint, lock['evaluation'], formal=True)
        assert result['samples'] == lock['frames']
        record.update(result, status='verified', completed_at=now())
        atomic_json(result_path, record)
        print('VERIFIED', record['id'], flush=True)
    except Exception:
        record.update(status='failed', failed_at=now(), error=traceback.format_exc())
        atomic_json(result_path, record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--task', choices=['B_FULL_0', 'B_3DCONV_0'])
    args = parser.parse_args()
    prepare() if args.prepare else run(args.task)
