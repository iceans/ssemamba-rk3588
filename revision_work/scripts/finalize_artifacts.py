"""Validate saved evidence and add provenance sidecars without rerunning models.

This certifies file consistency, not historical training validity or formal AP.
"""
import csv
import hashlib
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'revision_work'
STATES = set('pending blocked ready running trained evaluated verified failed invalid superseded reused complete'.split())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def check_predictions(folder, gt, raw_split_sha):
    record = read(folder / 'run.json')
    assert record['status'] == 'evaluated' and record['formal_result'] is False
    assert sha(record['checkpoint']) == record['checkpoint_sha256']
    for path, expected in record['source_hashes'].items():
        assert sha(ROOT / path) == expected, ('source changed since diagnostic run', path)
    cache = folder / 'validation/per_frame_predictions.jsonl'
    ordered = []
    counts = Counter()
    empty = 0
    num_boxes = 0
    with cache.open() as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            assert row['sample_index'] == index and row['formal_result'] is False
            assert row['model_source'] == record['checkpoint']
            truth = gt[row['file']]
            assert row['seq_id'] == truth['seq_id'] and row['frame_id'] == truth['frame_id']
            assert row['original_hw'] == [truth['height'], truth['width']]
            boxes, classes, scores = row['boxes_xyxy'], row['classes'], row['confidences']
            assert len(boxes) == len(classes) == len(scores)
            empty += not boxes
            num_boxes += len(boxes)
            for box, cls, score in zip(boxes, classes, scores):
                assert len(box) == 4 and all(math.isfinite(x) for x in box)
                assert cls == 0 and math.isfinite(score) and .001 - 1e-7 <= score <= 1
                x1, y1, x2, y2 = box
                assert -.01 <= x1 <= x2 <= truth['width'] + .01
                assert -.01 <= y1 <= y2 <= truth['height'] + .01
            ordered.append([row['seq_id'], row['frame_id'], row['file'], row['original_hw'], truth['label_sha256']])
            counts[Path(row['file']).stem] += 1
    coverage = read(folder / 'validation/sample_coverage.json')
    assert len(ordered) == coverage['num_samples'] == 4795
    assert len(counts) == coverage['num_unique_keys'] == 4763
    assert {k: v for k, v in counts.items() if v > 1} == coverage['duplicates']
    omitted = sorted(set(Path(p).stem for p in gt) - counts.keys())
    assert len(omitted) == 32
    protocol = {k: v for k, v in record['resolved_evaluation_args'].items()
                if k not in ['data', 'project', 'name', 'exist_ok']}
    protocol['data_yaml_sha256'] = sha(folder / 'data.yaml')
    provenance = {
        'id': record['id'], 'formal_result': False, 'status': 'diagnostic_cache_consistency_checked',
        'checkpoint_sha256': record['checkpoint_sha256'], 'original_run_record_sha256': sha(folder / 'run.json'),
        'raw_dataset_manifest_sha256': raw_split_sha,
        'ordered_evaluated_samples_sha256': digest(ordered),
        'ordered_samples_hash_scope': 'seq_id, frame_id, absolute file path, original_hw, label_sha256; includes duplicate requests in order',
        'prediction_cache_sha256': sha(cache), 'coverage_sha256': sha(folder / 'validation/sample_coverage.json'),
        'evaluation_config_sha256': digest(protocol), 'evaluation_config': protocol,
        'source_hashes': record['source_hashes'], 'samples': len(ordered), 'unique_keyframes': len(counts),
        'empty_prediction_samples': empty, 'predicted_boxes': num_boxes, 'omitted_keyframes': omitted,
        'meaning': 'Checksums and cache schema verified; AP was not reclassified as a formal independent test result',
    }
    provenance['run_identity_sha256'] = digest({k: provenance[k] for k in [
        'id', 'checkpoint_sha256', 'ordered_evaluated_samples_sha256', 'evaluation_config_sha256', 'source_hashes']})
    write(folder / 'provenance.json', provenance)
    return {k: provenance[k] for k in ['id', 'samples', 'unique_keyframes', 'empty_prediction_samples',
                                      'predicted_boxes', 'prediction_cache_sha256', 'ordered_evaluated_samples_sha256']}


def main():
    if (WORK / 'series_b/protocol.json').exists():
        raise RuntimeError('This validator certifies the preapproval audit phase only. Active Series B uses its frozen protocol, strict run checks and new_protocol.refresh.')
    source = read(WORK / 'evidence/source_manifest.json')
    for name, expected in source['files'].items():
        assert sha(ROOT / name) == expected, ('source manifest mismatch', name)
    with tarfile.open(WORK / 'evidence/source_snapshot.tar.gz', 'r:gz') as archive:
        archived = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                    for m in archive.getmembers() if m.isfile()}
    assert archived == source['files'], 'source snapshot is not identical to recorded source files'

    data = read(WORK / 'evidence/data/summary.json')
    manifests = {}
    for dataset, splits in data.items():
        for split, info in splits.items():
            if not isinstance(info, dict) or 'manifest_sha256' not in info:
                continue
            file = WORK / f'evidence/data/{dataset}_{split}_frames.jsonl'
            assert sha(file) == info['manifest_sha256']
            rows = [json.loads(s) for s in file.read_text().splitlines()]
            assert len(rows) == info['num_frames']
            assert sum(len(r['boxes']) for r in rows) == info['num_boxes']
            manifests[(dataset, split)] = rows

    gt = {r['file']: r for r in manifests[('DAUB', 'val')]}
    cache_checks = [check_predictions(p.parent, gt, data['DAUB']['val']['manifest_sha256'])
                    for p in sorted((WORK / 'runs').glob('REEVAL_DIAGNOSTIC_*/run.json'))]
    assert len(cache_checks) == 2
    assert len({r['ordered_evaluated_samples_sha256'] for r in cache_checks}) == 1

    common_path = WORK / 'evidence/data/DAUB_val_temporal_common.json'
    common = read(common_path)
    sequences = defaultdict(list)
    for r in manifests[('DAUB', 'val')]:
        sequences[r['seq_id']].append(r['file'])
    assert len(common) == 4683 and len({r['file'] for r in common}) == 4683
    for row in common:
        i = row['zero_based_index']
        seq = sequences[row['seq_id']]
        assert i >= 16 and seq[i] == row['file']
        for stride in [1, 2, 4]:
            assert row['clips'][str(stride)] == [seq[i - j * stride] for j in range(4, -1, -1)]

    registry = read(WORK / 'run_registry.json')
    ids = {r['id'] for r in registry['tasks']}
    assert len(ids) == len(registry['tasks'])
    for r in registry['tasks']:
        assert r['status'] in STATES and not r['formal_result']
        assert all(d in ids or d in registry['gates'] for d in r['dependencies'])
        if r['status'] == 'complete':
            assert r['artifacts'] and all((WORK / p).exists() for p in r['artifacts'])
    training = [r for r in registry['tasks'] if r['kind'] == 'training']
    assert len(training) == 13 and all(r['status'] == 'blocked' for r in training)
    assert all('TEST_' + r['id'] in ids for r in training)
    assert registry['new_formal_trainings_started'] == registry['formal_tests_verified'] == 0
    assert len(read(WORK / 'resolved_experiments.json')['tasks']) == 13

    with (WORK / 'results/metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    for r in metrics:
        assert r['formal'] == 'False' and math.isfinite(float(r['metric_value']))
        if r['metric_name'].startswith('AP'):
            assert r['unit'] == 'percent' and 0 <= float(r['metric_value']) <= 100
    csv_files = sorted((WORK / 'results/tables').glob('*.csv'))
    for file in csv_files:
        with file.open() as f:
            rows = list(csv.DictReader(f))
        assert rows and file.with_suffix('.tex').is_file()
    assert 'we conducted' not in (WORK / 'writing/reviewer_response_en.md').read_text().lower()
    broken_links = []
    for file in WORK.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', file.read_text()):
            if link.startswith(('http:', 'https:', '#')):
                continue
            target = link.split('#', 1)[0].strip('<>')
            if target and not (file.parent / target).exists():
                broken_links.append([str(file), link])
    assert not broken_links, broken_links

    scripts = {str(p.relative_to(ROOT)): sha(p) for p in sorted((WORK / 'scripts').glob('*.py'))}
    write(WORK / 'evidence/execution_scripts_manifest.json', {
        'scope': 'Current added audit/report scripts; original model code is in source_manifest.json',
        'files': scripts, 'scripts_sha256': digest(scripts),
        'warning': 'Current script manifest is not claimed as an immutable historical training version'})
    write(WORK / 'evidence/literature/source_index.json', {
        'SAIST': {'url': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.pdf',
                  'pdf_sha256': sha(WORK / 'evidence/literature/SAIST_CVPR2025.pdf'),
                  'method_pages_read': '9551-9554', 'extraction_tool': 'pdftotext -layout'},
        'other_primary_sources': 'Primary links and consulted sections are recorded in writing/technical_comparison.md'})

    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'artifact_consistency_passed': True,
              'scientific_completion': False, 'formal_tests_verified': 0,
              'source_files_verified': len(source['files']), 'source_archive_verified': True,
              'data_manifests_verified': len(manifests), 'prediction_caches': cache_checks,
              'P1_common_frames_verified': len(common), 'P1_common_manifest_sha256': sha(common_path),
              'metrics_rows': len(metrics), 'csv_latex_pairs': len(csv_files),
              'registry_dependencies_valid': True, 'markdown_local_links_valid': True,
              'training_slots_blocked': len(training), 'scripts_sha256': digest(scripts)}
    write(WORK / 'evidence/artifact_validation.json', report)
    # Exclude the checksum file itself and interpreter/font caches; preserve all experiment evidence.
    files = [p for p in sorted(WORK.rglob('*')) if p.is_file()
             and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts
             and 'matplotlib_config' not in p.parts]
    manifest = {str(p.relative_to(WORK)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in files}
    write(WORK / 'evidence/artifact_manifest.json', {'files': manifest, 'manifest_sha256': digest(manifest)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
