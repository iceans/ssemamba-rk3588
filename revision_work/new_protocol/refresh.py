"""Refresh status and tables from immutable configurations and saved run results."""
import csv
import os
import statistics
from pathlib import Path

from .common import ROOT, WORK, SERIES, now, read_json, atomic_json, digest


def optional(path):
    return read_json(path) if Path(path).is_file() else {}


def table(name, rows):
    target = WORK / 'results/tables' / name
    fields = list(rows[0])
    with target.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    def escape(value):
        if value is None or value == '':
            return '--'
        if isinstance(value, float):
            return f'{value:.5f}'
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    lines = ['% Series B; missing values are not zero.', r'\begin{tabular}{' + 'l' * len(fields) + '}',
             r'\hline', ' & '.join(map(escape, fields)) + r' \\', r'\hline']
    lines += [' & '.join(escape(row[k]) for k in fields) + r' \\' for row in rows]
    lines += [r'\hline', r'\end{tabular}']
    target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')


def refresh():
    lock = read_json(SERIES / 'protocol.json')
    tasks = sorted([read_json(p) for p in (SERIES / 'configs').glob('*.json')], key=lambda r: r['priority'])
    records, tests, registry_tasks = {}, {}, []
    metrics_path = WORK / 'history/preapproval/results/metrics.csv'
    with metrics_path.open() as f:
        metrics = list(csv.DictReader(f))
    metric_fields = list(metrics[0])
    for task in tasks:
        out = Path(task['output_directory'])
        record = optional(out / 'run.json')
        test = optional(out / 'TEST/run.json')
        records[task['id']], tests[task['id']] = record, test
        state = record.get('status', 'ready' if lock['formal_training_locked'] else 'pending')
        if test.get('status') == 'verified':
            state = 'verified'
            for key, name in [('metrics/mAP50(B)', 'AP50'), ('metrics/mAP50-95(B)', 'AP50-95')]:
                metrics.append(dict(experiment_id='TEST_' + task['id'], model=task['model'], dataset='DAUB',
                    split_hash=test['split_hash'], seed=task['seed'], checkpoint=test['checkpoint'],
                    protocol_id=lock['protocol_id'], metric_name=name, metric_value=100 * test['metrics'][key],
                    unit='percent', provenance=str(out / 'TEST/run.json'), status='verified', formal='True'))
        registry_tasks.append({'id': task['id'], 'kind': 'training', 'status': state, 'seed': task['seed'],
            'config_sha256': task['config_sha256'], 'source_sha256': lock['source_sha256'],
            'output_directory': str(out), 'command': task['command'], 'dependencies': [],
            'formal_result': state == 'verified', 'epoch_completed': record.get('epoch_completed', 0),
            'progress': optional(out / 'progress.json')})
        registry_tasks.append({'id': 'TEST_' + task['id'], 'kind': 'formal_test',
            'status': test.get('status', 'pending'), 'dependencies': [task['id']],
            'formal_result': test.get('status') == 'verified', 'artifacts': [str(out / 'TEST')]})
    def ap(task_id, metric='metrics/mAP50(B)'):
        value = tests.get(task_id, {})
        return 100 * value['metrics'][metric] if value.get('status') == 'verified' else None
    pairs = {'full': ['B_FULL_0', 'R1', 'R3'], 'STF-3DConv': ['B_3DCONV_0', 'R2', 'R4']}
    repeats = []
    for model, ids in pairs.items():
        for seed, task_id in zip(lock['COMMON_SEEDS'], ids):
            repeats.append({'model': model, 'seed_or_summary': seed, 'AP50_percent': ap(task_id),
                            'AP50_95_percent': ap(task_id, 'metrics/mAP50-95(B)'), 'sample_std_AP50': None,
                            'training_hours': records[task_id].get('elapsed_training_s', 0) / 3600 or None,
                            'status': tests[task_id].get('status', records[task_id].get('status', 'pending'))})
        values = [ap(i) for i in ids]
        complete = all(v is not None for v in values)
        repeats.append({'model': model, 'seed_or_summary': 'mean_ddof1',
                        'AP50_percent': statistics.mean(values) if complete else None,
                        'AP50_95_percent': statistics.mean([ap(i, 'metrics/mAP50-95(B)') for i in ids]) if complete else None,
                        'sample_std_AP50': statistics.stdev(values) if complete else None,
                        'training_hours': None, 'status': 'verified' if complete else 'pending'})
    table('stf_repeats', repeats)
    differences = []
    for seed, a, b in zip(lock['COMMON_SEEDS'], pairs['full'], pairs['STF-3DConv']):
        differences.append({'seed': seed, 'full_minus_3d_AP50_pp': ap(a) - ap(b) if ap(a) is not None and ap(b) is not None else None})
    complete_pairs = all(r['full_minus_3d_AP50_pp'] is not None for r in differences)
    differences.append({'seed': 'mean', 'full_minus_3d_AP50_pp': statistics.mean([r['full_minus_3d_AP50_pp'] for r in differences]) if complete_pairs else None})
    table('stf_paired_differences', differences)
    table('components', [
        {'dataset': 'DAUB', 'design': 'factorial_partial', 'LFSS': l, 'SAAM': s, 'AP50_percent': ap(t) if t else None,
         'status': tests[t].get('status', 'pending') if t else 'blocked_old_row_invalid_extra_training_not_approved'}
        for l, s, t in [(0, 0, None), (1, 0, None), (0, 1, 'E1'), (1, 1, 'B_FULL_0')]] + [
        {'dataset': 'IRDST', 'design': 'incremental_only', 'LFSS': l, 'SAAM': s, 'AP50_percent': None,
         'status': 'blocked_historical_provenance'} for l, s in [(0, 0), (1, 0), (1, 1)]])
    table('loss_sensitivity', [{'id': t['id'], 'box': t['loss'][0], 'cls': t['loss'][1], 'dfl': t['loss'][2],
                               'AP50_percent': ap(t['id']), 'AP50_95_percent': ap(t['id'], 'metrics/mAP50-95(B)'),
                               'status': tests[t['id']].get('status', records[t['id']].get('status', 'pending'))}
                              for t in tasks if t['id'] in ['B_FULL_0', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']])
    neighborhoods = []
    for task_id, kernel in [('E2', 1), ('B_FULL_0', 3), ('E3', 5)]:
        architecture = optional(SERIES / 'runs' / task_id / 'architecture.json')
        if not architecture:
            architecture = optional(SERIES / 'preflight' / (task_id + '.json')).get('architecture', {})
        neighborhoods.append({'id': task_id, 'kernel': kernel, 'n': 4, 'parameters': architecture.get('parameters'),
                              'FLOPs': None, 'AP50_percent': ap(task_id),
                              'status': tests[task_id].get('status', records[task_id].get('status', 'pending'))})
    neighborhoods.append({'id': 'old_order_only', 'kernel': None, 'n': None, 'parameters': None, 'FLOPs': None,
                          'AP50_percent': None, 'status': 'invalid_or_untraceable_order_only_definition'})
    table('scan_neighborhood', neighborhoods)
    temporal = []
    for name, task_id in [('full', 'B_FULL_0'), ('STF-3DConv', 'B_3DCONV_0'), ('DFAR', None)]:
        base = optional(SERIES / 'runs' / task_id / 'P1_s1/run.json') if task_id else {}
        for stride in [1, 2, 4]:
            record = optional(SERIES / 'runs' / task_id / f'P1_s{stride}/run.json') if task_id else {}
            good = record.get('status') == 'verified'
            values = record.get('metrics', {})
            if good:
                for key, label in [('metrics/mAP50(B)', 'AP50'), ('metrics/mAP50-95(B)', 'AP50-95')]:
                    metrics.append(dict(experiment_id=f'P1_{name}_s{stride}', model=name, dataset='DAUB_common_i_ge_16',
                        split_hash=record['common_keyframe_hash'], seed=0, checkpoint=record['checkpoint'],
                        protocol_id=lock['protocol_id'], metric_name=label, metric_value=100 * values[key],
                        unit='percent', provenance=str(SERIES / 'runs' / task_id / f'P1_s{stride}/run.json'), status='verified', formal='True'))
            temporal.append({'model': name, 'stride': stride, 'common_frames': 4683,
                'common_keyframe_hash': record.get('common_keyframe_hash'),
                'AP50_percent': 100 * values['metrics/mAP50(B)'] if good else None,
                'AP50_95_percent': 100 * values['metrics/mAP50-95(B)'] if good else None,
                'delta_AP50_pp': 100 * (values['metrics/mAP50(B)'] - base['metrics']['metrics/mAP50(B)']) if good and base.get('status') == 'verified' else None,
                'status': record.get('status', 'pending' if task_id else 'blocked_source_provenance')})
            registry_tasks.append({'id': f'P1_{name}_s{stride}', 'kind': 'inference',
                                   'status': record.get('status', 'pending' if task_id else 'blocked'),
                                   'dependencies': [task_id] if task_id else ['DFAR_provenance'], 'formal_result': good})
    table('temporal_robustness', temporal)
    sizes = []
    costs = []
    for name, task_id in [('full', 'B_FULL_0'), ('STF-3DConv', 'B_3DCONV_0')]:
        test = tests[task_id]
        good = test.get('status') == 'verified'
        if good:
            metrics.append(dict(experiment_id='P2_' + name, model=name, dataset='DAUB', split_hash=test['split_hash'], seed=0,
                checkpoint=test['checkpoint'], protocol_id=lock['protocol_id'], metric_name='false_positives_conf025_iou05',
                metric_value=test['false_positives_per_frame'], unit='false_positives/frame',
                provenance=str(SERIES / 'runs' / task_id / 'TEST/run.json'), status='verified', formal='True'))
        for group, count in [('d_le_3', 0), ('3_lt_d_le_7', 0), ('d_gt_7', 4795)]:
            sizes.append({'model': name, 'dataset': 'DAUB', 'group': group, 'GT_boxes': count,
                          'AP50_percent': ap(task_id) if count else None,
                          'AP50_95_percent': ap(task_id, 'metrics/mAP50-95(B)') if count else None,
                          'all_frame_FP_per_frame_conf025': test.get('false_positives_per_frame') if good else None,
                          'status': 'empty_bin' if count == 0 else test.get('status', 'pending')})
        registry_tasks.append({'id': 'P2_' + name, 'kind': 'inference', 'status': 'evaluated' if good else 'pending',
                               'dependencies': ['TEST_' + task_id], 'formal_result': good,
                               'limitation': 'Size/false-alarm metrics available when test completes; scene metadata still missing'})
        cost = optional(SERIES / 'runs' / task_id / 'P4/run.json')
        if cost.get('status') == 'verified':
            for key, unit in [('mean_latency_ms', 'ms/clip'), ('throughput_clips_s', 'clips/s'),
                              ('peak_allocated_bytes', 'bytes'), ('parameters', 'parameters')]:
                metrics.append(dict(experiment_id='P4_' + name, model=name, dataset='synthetic_fixed_input', split_hash='', seed=0,
                    checkpoint=cost['checkpoint'], protocol_id=lock['protocol_id'], metric_name=key, metric_value=cost[key],
                    unit=unit, provenance=str(SERIES / 'runs' / task_id / 'P4/run.json'), status='verified', formal='True'))
        costs.append({'model': name, 'parameters': cost.get('parameters'), 'AP50_percent': ap(task_id),
                      'AP50_95_percent': ap(task_id, 'metrics/mAP50-95(B)'),
                      'latency_ms': cost.get('mean_latency_ms'), 'clips_s': cost.get('throughput_clips_s'),
                      'peak_allocated_MiB': cost['peak_allocated_bytes'] / 2**20 if cost.get('status') == 'verified' else None,
                      'FLOPs': None, 'status': cost.get('status', 'pending')})
        cost_state = cost.get('status', 'pending')
        if cost_state == 'verified' and not good:
            cost_state = 'evaluated'
        registry_tasks.append({'id': 'P4_' + name, 'kind': 'inference', 'status': cost_state,
                               'dependencies': [task_id, 'no_competing_compute'], 'formal_result': cost.get('status') == 'verified'})
    sizes.append({'model': 'IFSS_noninterleaved', 'dataset': 'DAUB', 'group': 'all', 'GT_boxes': 4795,
                  'AP50_percent': None, 'AP50_95_percent': None, 'all_frame_FP_per_frame_conf025': None,
                  'status': 'blocked_invalid_or_missing_order_only_checkpoint'})
    registry_tasks.append({'id': 'P2_IFSS_noninterleaved', 'kind': 'inference', 'status': 'blocked',
                           'dependencies': ['valid_order_only_checkpoint'], 'formal_result': False})
    table('size_scene_breakdown', sizes)
    table('precision_runtime', costs)
    external_lock = optional(SERIES / 'external/protocol.json')
    external_ready = external_lock.get('status') == 'verified'
    external_rows = []
    for model, task_id in [('full', 'B_FULL_0'), ('STF-3DConv', 'B_3DCONV_0'), ('DFAR', None)]:
        path = SERIES / 'runs' / task_id / 'P3/run.json' if task_id else None
        record = optional(path) if path else {}
        good = record.get('status') == 'verified'
        state = record.get('status', 'pending' if external_ready and task_id else 'blocked')
        external_rows.append({'model': model, 'dataset': external_lock.get('dataset', 'not_frozen'),
                              'split': external_lock.get('split'), 'source': 'DAUB', 'seed': 0 if task_id else None,
                              'checkpoint': record.get('checkpoint'),
                              'AP50_percent': 100 * record['metrics']['metrics/mAP50(B)'] if good else None,
                              'AP50_95_percent': 100 * record['metrics']['metrics/mAP50-95(B)'] if good else None,
                              'FP_per_frame_conf025': record.get('false_positives_per_frame') if good else None,
                              'frames': external_lock.get('frames'), 'status': state})
        registry_tasks.append({'id': 'P3_' + model, 'kind': 'inference', 'status': state,
                               'dependencies': ['external_dataset_freeze', task_id or 'DFAR_provenance'],
                               'formal_result': good, 'artifacts': [str(path)] if path else [],
                               'limitation': external_lock.get('independence_limitation')})
        if good:
            for key, label in [('metrics/mAP50(B)', 'AP50'), ('metrics/mAP50-95(B)', 'AP50-95')]:
                metrics.append(dict(experiment_id='P3_' + model, model=model, dataset=record['dataset'],
                    split_hash=record['split_hash'], seed=0, checkpoint=record['checkpoint'],
                    protocol_id=record['protocol_id'], metric_name=label, metric_value=100 * record['metrics'][key],
                    unit='percent', provenance=str(path), status='verified', formal='True'))
            metrics.append(dict(experiment_id='P3_' + model, model=model, dataset=record['dataset'],
                split_hash=record['split_hash'], seed=0, checkpoint=record['checkpoint'],
                protocol_id=record['protocol_id'], metric_name='false_positives_conf025_iou05',
                metric_value=record['false_positives_per_frame'], unit='false_positives/frame',
                provenance=str(path), status='verified', formal='True'))
    table('external_generalization', external_rows)
    external_verified = sum(r['status'] == 'verified' for r in external_rows)
    external_note = (f"P3 已冻结 {external_lock['dataset']} 官方测试划分，共 {external_lock['frames']} 帧；两项新模型外部测试已验收 {external_verified}/2，DFAR 来源仍缺。原始背景素材是否经变换复用尚无法全面确认，限制见 series_b/external/protocol.json。"
                     if external_ready else 'P3 目标数据尚未冻结；官方清单和下载核查见 evidence/external/。')
    with (WORK / 'results/metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics)
    counts = {'started': sum(bool(r) for r in records.values()),
              'trained': sum(r.get('status') in ['trained', 'evaluated', 'verified', 'complete'] for r in records.values()),
              'running': sum(r.get('status') == 'running' for r in records.values()),
              'failed': sum(r.get('status') == 'failed' for r in records.values()),
              'test_verified': sum(r.get('status') == 'verified' for r in tests.values())}
    registry = {'status': 'running' if counts['running'] else 'partial' if counts['trained'] == 15 else 'ready',
                'updated_at': now(), 'formal_training_budget': 15, 'approval': lock['approval'],
                'counts': counts, 'formal_reused': 0, 'source_sha256': lock['source_sha256'], 'tasks': registry_tasks,
                'historical_phase': 'history/preapproval/run_registry.json',
                'gates': {'DFAR_provenance': 'blocked', 'no_competing_compute': 'scheduler_enforced',
                          'valid_order_only_checkpoint': 'blocked',
                          'external_dataset_freeze': 'verified' if external_ready else 'blocked'}}
    atomic_json(SERIES / 'registry.json', registry)
    atomic_json(WORK / 'run_registry.json', registry)
    lines = ['# SSE-Mamba 新协议执行状态', '', f"更新时间：{registry['updated_at']}。用户已批准方案 B 的 15 次完整训练。", '',
             f"正式训练：已启动 {counts['started']}/15，训练结束 {counts['trained']}/15，运行中 {counts['running']}，失败 {counts['failed']}；正式 TEST 已验收 {counts['test_verified']}/15。", '',
             '当前进度以各 run 的 progress.json 为准。所有训练为 100 epochs，选固定最后一轮 EMA；不以 test 指标选权、挑种子或提前停止。', '',
             '| 任务 | 状态 | epoch | 当前 batch |', '|---|---|---:|---:|']
    for task in tasks:
        record = records[task['id']]
        progress = optional(Path(task['output_directory']) / 'progress.json')
        lines.append(f"| {task['id']} | {record.get('status', 'ready')} | {progress.get('epoch', record.get('epoch_completed', 0))}/100 | {progress.get('batch', 0)}/1123 |")
    lines += ['', '预检通过：新模型严格结构匹配、batch=8 前向反向、数据时序/标注和断点恢复。GPU 检查证据：series_b/preflight/result.json。', '',
              external_note, '',
              '保留缺口：两个旧组件基线补训不在 15 次预算内；有效非交织权重、DFAR 来源及场景标签尚缺；投稿源码和审稿验收文件未找到。', '',
              '复现入口：series_b/README.md。旧诊断数值单列，原状态已归档到 history/preapproval/。']
    (WORK / 'STATUS.md').write_text('\n'.join(lines) + '\n')
    report = ['# 执行报告：已批准的新协议正在执行', '',
              f"新增训练预算 15；启动 {counts['started']}、训练结束 {counts['trained']}、失败 {counts['failed']}；正式测试验收 {counts['test_verified']}。正式复用 0。", '',
              '全部新结果使用 series_b/protocol.json；配置、代码摘要、数据清单、独立初始化及最终 checkpoint 选择规则均记录。当前各任务进度见 STATUS.md。', '',
              '已完成的前期工作：40 份重点权重审计、数据/索引检查、两份历史诊断重评、旧候选运行成本测量、六篇文献比较和写作草稿。旧诊断 AP50 为 fullsize 87.588%、3D 94.868%，不参与新协议正式统计。', '',
              '新协议 GPU smoke 已通过，完整模型 batch=8 的峰值约 6.53 GiB。训练耗时按实际 epoch 记录，完整训练尚未完成时不宣称实验已完成。', '',
              '15 次包括两模型各 3 个种子和 E1–E9；暂未补两个无效旧组件行，不能称 DAUB 四组合已完整。非交织行、强基线来源仍有缺口。', '',
              external_note, '',
              '结果长表：results/metrics.csv；正式行须 status=verified 且 formal=True。表格由已保存结果生成，缺失 AP/标准差留空。', '',
              'writing/ 中的论文片段和回复是草稿；稿件/验收文件缺失，未编译、未对外提交，未把写稿等同于完成审稿要求。', '',
              '复现命令与恢复方式见 series_b/README.md；每次恢复验证配置和代码哈希，并严格恢复模型、EMA、优化器、待累计梯度和随机状态。']
    (WORK / 'FINAL_REPORT.md').write_text('\n'.join(report) + '\n')
    verified_temporal = [r for r in temporal if r['status'] == 'verified']
    if len(verified_temporal) == 6:
        signature = digest(verified_temporal)
        marker = SERIES / 'temporal_figure.json'
        if optional(marker).get('signature') != signature:
            os.environ.setdefault('MPLCONFIGDIR', str(SERIES / 'matplotlib_config'))
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            for name in ['full', 'STF-3DConv']:
                rows = [r for r in verified_temporal if r['model'] == name]
                ax.plot([r['stride'] for r in rows], [r['AP50_percent'] for r in rows], marker='o', label=name)
            ax.set(xlabel='Temporal sampling stride (training stride = 1)', ylabel='AP50 (%)', xticks=[1, 2, 4],
                   title='DAUB: 4,683 common keyframes; fixed seed 0')
            ax.legend()
            fig.tight_layout()
            for extension in ['svg', 'pdf']:
                fig.savefig(WORK / f'results/figures/temporal_sampling_mismatch.{extension}')
            plt.close(fig)
            atomic_json(marker, {'signature': signature, 'rows': verified_temporal, 'DFAR': 'not verified'})
    prose = ['# Series B results draft — generated from verified tests', '',
             f"The approved series contains 15 independent training configurations; {counts['test_verified']} final-epoch tests have passed verification.", '',
             'Training uses the fixed protocol in series_b/protocol.json. Historical diagnostic values are excluded. Missing cells remain unavailable. Manuscript locations and reviewer acceptance remain TODO.', '']
    for row in repeats:
        if row['seed_or_summary'] == 'mean_ddof1' and row['status'] == 'verified':
            prose.append(f"Across the three predefined seeds, {row['model']} achieved AP50 {row['AP50_percent']:.3f}% with sample standard deviation {row['sample_std_AP50']:.3f} percentage points, and mean AP50–95 {row['AP50_95_percent']:.3f}%. See results/tables/stf_repeats.csv.")
    prose += ['', 'All six loss perturbations must be reported with the original default row. No test-selected loss weights replace the main result. The two additional component-repair trainings were not approved, so this series alone does not complete the DAUB LFSS×SAAM factorial table.', '',
              'The valid-model temporal mismatch results, if available, use a common set at every stride. Missing DFAR, order-only and external-data evidence is not inferred from these runs.']
    if external_ready:
        prose += ['', f"The external protocol was frozen for the official NUDT-MIRSDT test split before target-model scoring ({external_lock['frames']} keyframes). Source-domain seed0 final-epoch EMA weights and the predefined confidence threshold0.25 are retained. Exact decoded-image overlap was checked against DAUB train and test; transformed or reused background sources cannot be comprehensively ruled out."]
        for row in external_rows:
            if row['status'] == 'verified':
                prose.append(f"On that frozen external split, {row['model']} achieved AP50 {row['AP50_percent']:.3f}%, AP50–95 {row['AP50_95_percent']:.3f}%, and {row['FP_per_frame_conf025']:.5f} false positives per frame at the predefined source threshold.")
    (WORK / 'writing/series_b_results_en.md').write_text('\n'.join(prose) + '\n')
    return counts


if __name__ == '__main__':
    print(refresh())
