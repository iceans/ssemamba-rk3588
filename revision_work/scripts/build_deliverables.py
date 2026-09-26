"""Generate traceable audit deliverables; never promotes diagnostic metrics to formal results."""
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'revision_work'
NOW=datetime.now(timezone.utc).isoformat()


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,obj):
    p=WORK/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def text(path,value):
    p=WORK/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value.strip()+'\n')


def table(name,rows,fields=None):
    p=WORK/'results/tables'/name;p.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0])
    with p.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def esc(v):
        if v is None or v=='':return '--'
        if isinstance(v,float):v=f'{v:.5f}'
        return str(v).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&').replace('#','\\#')
    lines=['% Generated from saved evidence. -- means unavailable; not zero.','\\begin{tabular}{'+'l'*len(fields)+'}',
           '\\hline',' & '.join(map(esc,fields))+' \\\\','\\hline']
    lines+=[' & '.join(esc(r.get(k)) for k in fields)+' \\\\' for r in rows]
    lines+=['\\hline','\\end{tabular}']
    p.with_suffix('.tex').write_text('\n'.join(lines)+'\n')


def main():
    if (WORK/'series_b/protocol.json').exists():
        raise RuntimeError('Series B has been approved. Use python -m revision_work.new_protocol.refresh; preapproval reports are archived and must not overwrite active runs.')
    historical=json.loads((WORK/'evidence/historical_run_inventory.json').read_text())
    checkpoints=json.loads((WORK/'evidence/checkpoint_audit.json').read_text())
    data=json.loads((WORK/'evidence/data/summary.json').read_text())
    sample=json.loads((WORK/'evidence/sample_audit.json').read_text())
    cpu=json.loads((WORK/'evidence/preflight_cpu.json').read_text())
    gpu=json.loads((WORK/'evidence/preflight_cuda_1.json').read_text())
    evaluations=[json.loads(p.read_text()) for p in sorted((WORK/'runs').glob('REEVAL_DIAGNOSTIC_*/run.json'))]
    codes=[ROOT/'tsgmamba_train.py',ROOT/'fpstest.py',ROOT/'pyproject.toml']
    for folder in ['ultralytics/nn','ultralytics/data','ultralytics/engine','ultralytics/utils','ultralytics/models/yolo/detect']:
        codes+=sorted((ROOT/folder).rglob('*.py'))
    codes+=sorted((ROOT/'ultralytics/cfg').rglob('*.yaml'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(codes))}
    code_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    write_json('evidence/source_manifest.json',{'created_at':NOW,'files':manifest,'code_sha256':code_hash,'git_commit':None,
                'git_reason':'git status failed; .git is an empty directory in this execution environment'})
    archive=WORK/'evidence/source_snapshot.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            for p in sorted(set(codes)):tar.add(p,arcname=str(p.relative_to(ROOT)))
    import importlib.metadata as metadata
    packages={}
    for n in ['torch','torchvision','numpy','scipy','timm','einops','triton','opencv-python','PyYAML','matplotlib']:
        try:packages[n]=metadata.version(n)
        except metadata.PackageNotFoundError:packages[n]=None
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,
         'cuda_build':'11.8','driver':'535.288.01','gpus':['NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 4090'],
         'gpu_access':'requires execution outside sandbox; verified with real CUDA kernels',
         'selective_scan_actual_backend':'selective_scan_cuda_oflex (current implementation ignores requested backend)',
         'dcn':'torchvision.ops.DeformConv2d; exercised successfully',
         'ssm_core_extension':'installed; import passed','mamba_ssm':'not installed; not required by this code path',
         'disk_free_bytes':shutil.disk_usage(ROOT).free,'global_environment_changed':False}
    write_json('evidence/environment.json',env)
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    (WORK/'evidence/pip_freeze.txt').write_text(freeze.stdout+freeze.stderr)
    # Every discovered candidate gets a status; the status is about reuse under this taskbook.
    baseline=[]
    for r in checkpoints:
        path=Path(r['checkpoint']);args=r.get('train_args',{});name=path.parent.parent.name
        reasons=[]
        pt=args.get('pretrained');init=args.get('model','')
        dataset='IRDST' if '/IRDST/' in str(path) else 'DAUB'
        initialization_paths=[v for v in [pt,init] if isinstance(v,str) and v.endswith('.pt')]
        finetuned=any('/'+dataset+'/' in v for v in initialization_paths)
        if finetuned:reasons.append('Initialization points to an already trained same-dataset checkpoint; not an independent ablation baseline')
        elif initialization_paths:reasons.append('Checkpoint initialization requires a common, traceable pretraining rule; other-dataset pretraining alone is not proof of invalidity')
        if args.get('epochs',100)<100:reasons.append(f"Only {args.get('epochs')} configured epochs; historical full-training protocol is unresolved")
        if '/DAUB/' in str(path):reasons.append('Current DAUB val points to test/images; checkpoint selection is not independent of this set')
        if r.get('strict_reconstruction')!='passed':reasons.append('Current constructor cannot strictly reconstruct checkpoint; full serialized graph can still be inspected separately')
        reasons.append('No historical source version and frozen split hashes tied to this run')
        baseline.append({'id':str(path.relative_to(ROOT)).replace('/','__'),'model_candidate':name,
                         'dataset':'IRDST' if '/IRDST/' in str(path) else 'DAUB',
                         'status':'invalid' if finetuned or args.get('epochs',100)<100 else 'artifact_found',
                         'status_scope':'reuse as a formal independent taskbook baseline, not deletion or proof of a corrupt weight file',
                         'checkpoint':str(path),'sha256':r.get('sha256'),'historical_seed':args.get('seed'),
                         'valid_common_seed':False,'strict_constructor':r.get('strict_reconstruction'),
                         'provenance':{'train_args':args,'embedded_metrics':r.get('train_metrics'),
                                      'source_version':r.get('git')},'reasons':reasons})
    correspondence={
        'single_frame_YOLOv8n':['yolov8n_813'],'five_frame_stack':['viideo_None','video_None'],
        'SSTE':['sste_stage'],'SSTE_LFSS':['sst_stage_lfss','stestage_lfss'],
        'SSTE_LFSS_SAAM':['fullsize_model','frame5','my_2timMix_945'],
        'STF_3DConv':['conv3d-TIMSSM'],'STF_without_DCN':['wodwconv','wodwconv_w_dcn','dcn'],
        'STF_without_DWConv':['wodwconv','wodwconv_w_dcn','dcn'],'IFSS_without_STF':['vssblock'],
        'IFSS_without_SFIScan':[],'LFSS_without_SFIScan':['LFSSBlock_wo_sfi'],
        'IFSS_SS2D':['IFSSblockVSS'],'LFSS_SS2D':['lfss_vssblock'],
        'shallow_default':['frame5'],'deep_stride8':['deepstage_stride8_8876'],
        'deep_stride16':['deepstage_stride16_9114'],'deep_two_levels':['deepssstage_8790']}
    requested=[]
    for dataset in ['DAUB','IRDST']:
        for label,names in correspondence.items():
            candidates=[r['id'] for r in baseline if r['dataset']==dataset and r['model_candidate'] in names]
            requested.append({'row':label,'dataset':dataset,'status':'artifact_found' if candidates else 'missing',
                              'candidate_ids':candidates,'paper_row_mapping_verified':False,
                              'mapping_note':'Name-based search candidates only; paper and historical source are missing. Do not infer ablation boundaries from directory names.'})
    dfar=json.loads((WORK/'evidence/dfar_audit.json').read_text())
    write_json('baseline_registry.json',{'created_at':NOW,'formal_verified_reusable':0,'candidates':baseline,
               'requested_rows':requested,'DFAR':{'status':'artifact_found',**dfar},
               'paper_reported':[{'row':'STF_3DConv','AP50_percent':94.87,'source':'codex/TASKBOOK.md section 3.2B',
                                  'status':'paper_reported','original_manuscript_available':False}]})
    # Lock observed facts, explicitly leave the formal training protocol unlocked.
    protocol={'status':'blocked','formal_training_locked':False,'created_at':NOW,'code_sha256':code_hash,
              'BASE_SEED':None,'COMMON_SEEDS':None,'MISSING_SEEDS':None,
              'historical_observed_seeds':{'fullsize_model':0,'conv3d-TIMSSM':0},
              'reason':'Known seed values belong to runs without a valid comparable independent protocol; do not count them as valid third seeds',
              'datasets':data,'evaluation_diagnostic':{'imgsz':640,'frame_num':5,'channels':8,'batch':8,
                    'conf':.001,'iou':.7,'max_det':300,'half':False,'rect':False,'workers':0,'split':'val',
                    'physical_set':'DAUB/yolo_format/test/images','evaluator':'current local DetectionValidator',
                    'checkpoint_selection':'historical best.pt; diagnostic only'},
              'historical_discrepancies':{'candidate_full_epochs':20,'candidate_3dconv_epochs':20,
                    'candidate_init':'DAUB trained complete-model checkpoint',
                    'optimizer_argument':'auto','lr0_argument':.0001,'cos_lr':False,'loss':[7.5,.5,1.5],
                    'physical_batch':8,'nbs':64,'steady_state_accumulation':8,'steady_state_effective_batch':64,
                    'actual_optimizer':'not recoverable from stripped optimizer state; current auto rule is insufficient historical proof',
                    'lfss_supplement_layers':{'current_constructor':1,'historical_frame5_and_3dconv':4},
                    'frame_tensor_order':sample['channel_semantics']},
              'new_training_proposal_not_approved':{'epochs':100,'imgsz':640,'frame_num':5,'batch':8,'nbs':64,
                   'optimizer':'Adam','lr0':.001,'cos_lr':True,'lrf':.01,'amp':False,
                   'box':7.5,'cls':.5,'dfl':1.5,'seeds':[0,2027,3407],
                   'initialization':'same random initialization rule, no DAUB-trained complete-model initialization',
                   'checkpoint_selection':'fixed final epoch 100; no test-driven checkpoint selection',
                   'base_graph_candidate':'matched frame5 / conv3d-TIMSSM graphs; user must confirm original Ours(n) definition',
                   'loader':'proposal: one clip per keyframe, [g_i-4,g_i-3,g_i-2,g_i-1,g_i,R_i,G_i,B_i], earliest-frame replication at sequence start; not approved',
                   'count':15,'extra_over_default_13':2}}
    write_json('protocol_lock.json',protocol)
    config_full=next((WORK/'evidence/checkpoints').glob('*DAUB__frame5__weights__best.yaml'))
    config_3d=next((WORK/'evidence/checkpoints').glob('*DAUB__conv3d-TIMSSM__weights__best.yaml'))
    resolved=[];runs=[]
    def run(id,kind,status='blocked',deps=None,reason='',artifacts=None):
        rec={'id':id,'kind':kind,'status':status,'dependencies':deps or [],'reason':reason,'artifacts':artifacts or [],
             'updated_at':NOW,'formal_result':False,'code_sha256':code_hash}
        runs.append(rec);return rec
    for id in ['R1','R2','R3','R4']+[f'E{i}' for i in range(1,10)]:
        three=id in ['R2','R4']
        changes={};mapping='baseline model and additional common seed'
        if id=='E1':
            mapping='Replace LFSS CCCBlock using historical SSTE baseline C2f convention while retaining Detect.aligned (SAAM)'
            changes={'head':'candidate SSTE head removes CCCBlock and adapts next C2f input channels; requires baseline definition verification'}
        elif id in ['E2','E3']:
            k=1 if id=='E2' else 5;mapping='revision_work/scripts/preflight.py:set_lfss_kernel'
            changes={'model.21.op.my_conv.*.conv.kernel_size':[k,k],'padding':[k//2,k//2],
                     'n':'retain 4 for historical matched candidate, not current constructor default 1',
                     'fixed':['channels','groups','stride','dilation','bias setting','BN','activation','residual','other convolutions']}
        elif id.startswith('E') and int(id[1:])>=4:
            i=int(id[1:])-4;key=['box','box','cls','cls','dfl','dfl'][i];value=[3.75,15.,.25,1.,.75,3.][i]
            mapping='YOLO.train keyword -> model.args -> v8DetectionLoss.hyp';changes={key:value}
        reason='Formal baseline, initialization, loader and checkpoint-selection protocol unresolved; companion JSON manifest missing'
        rec={'id':id,'status':'blocked','manifest_source':'TASKBOOK only; experiment_manifest.json missing',
             'config_file':str(config_3d if three else config_full),'config_is':'extracted historical candidate, not an approved resolved training config',
             'mapping':mapping,'config_diff':changes,'seed':None,'checkpoint':None,
             'dependencies':['GATE_PROTOCOL','GATE_MODEL_DEFINITION'],'command':None,
             'command_unavailable_reason':reason,'output_directory':str(WORK/'runs'/id),
             'split_hash':data['DAUB']['train']['manifest_sha256'],'code_sha256':code_hash}
        resolved.append(rec);run(id,'training',deps=rec['dependencies'],reason=reason)
        run('TEST_'+id,'formal_test',deps=[id,'GATE_PROTOCOL'],reason='Waiting for a valid trained or reused checkpoint and frozen test protocol')
    for model in ['full','stf_3dconv','DFAR']:
        for stride in [1,2,4]:run(f'P1_{model}_s{stride}','inference',deps=['GATE_PROTOCOL','GATE_SOURCE_'+model],reason='Source checkpoint/protocol provenance gate unresolved; 4683-frame common manifest is ready')
    for model in ['full','stf_3dconv','IFSS_noninterleaved']:
        run(f'P2_{model}','inference',deps=['GATE_PROTOCOL','GATE_SOURCE_'+model,'GATE_GROUP_EVALUATOR'],reason='No verified model row; DAUB <=3 and (3,7] box-size groups are empty; scene metadata unverified')
    for model in ['full','stf_3dconv','DFAR']:
        run(f'P3_{model}','inference',deps=['GATE_PROTOCOL','GATE_SOURCE_'+model,'GATE_EXTERNAL'],reason='Frozen external dataset and source checkpoint provenance not ready')
    for model in ['full','stf_3dconv']:
        run(f'P4_{model}','inference',deps=['GATE_PROTOCOL','GATE_SOURCE_'+model],reason='Candidate latency measured under new protocol; formal AP/graph comparability remains blocked',
            artifacts=['evidence/preflight_cuda_1.json','benchmark_protocol.json'])
    for r in evaluations:
        run(r['id'],'diagnostic_inference',r['status'],reason='Not an independent test or verified baseline',
            artifacts=[str((WORK/'runs'/r['id']).relative_to(WORK))])
    run('AUDIT_CHECKPOINTS','audit','complete',artifacts=['evidence/checkpoint_audit.json'])
    run('AUDIT_DATA','audit','complete',artifacts=['evidence/data/summary.json','evidence/sample_audit.json'])
    run('PREFLIGHT_CPU','correctness','complete',artifacts=['evidence/preflight_cpu.json'])
    run('PREFLIGHT_GPU','correctness','complete',artifacts=['evidence/preflight_cuda_1.json'])
    writing_files=['writing/technical_comparison.md','writing/reviewer_response_en.md','writing/manuscript_revision_snippets.md','writing/reviewer_coverage.md']
    run('WRITING_DRAFTS','writing','complete' if all((WORK/p).is_file() for p in writing_files) else 'pending',
        reason='Draft artifacts only; reviewer acceptance and manuscript application are separate',artifacts=writing_files)
    run('WRITING_ACCEPTANCE','writing','blocked',deps=['WRITING_DRAFTS'],reason='Exact reviewer mapping, manuscript edit and formal result paragraphs require missing source documents and verified experiments')
    gates={
        'GATE_PROTOCOL':'Independent baseline training, seed, loader and checkpoint-selection protocol unresolved',
        'GATE_MODEL_DEFINITION':'Authoritative manifest and original Ours(n)/E1 definition missing',
        'GATE_SOURCE_full':'No verified full model source checkpoint',
        'GATE_SOURCE_stf_3dconv':'No verified comparable STF-3DConv source checkpoint',
        'GATE_SOURCE_DFAR':'Strict loading passed; checkpoint-tied training provenance missing',
        'GATE_SOURCE_IFSS_noninterleaved':'No valid same-stream order-only checkpoint established',
        'GATE_GROUP_EVALUATOR':'Ignore-aware grouped AP, fixed working threshold and scene metadata not established',
        'GATE_EXTERNAL':'One official external dataset/split and overlap audit not frozen'}
    write_json('resolved_experiments.json',{'resolution_complete':False,'tasks':resolved})
    write_json('run_registry.json',{'status':'blocked','created_at':NOW,'formal_training_budget':13,
              'new_formal_trainings_started':0,'new_formal_trainings_completed':0,'formal_reused':0,
              'formal_tests_verified':0,'diagnostic_evaluations_completed':len(evaluations),
              'training_failures':0,'incomplete_training_slots':13,'tasks':runs,
              'gates':{k:{'status':'blocked','reason':v} for k,v in gates.items()}})
    write_json('taskbook_derived_manifest.json',{'authoritative':False,'source':'codex/TASKBOOK.md',
                'missing_authoritative_file':'codex/experiment_manifest.json','training_ids':[r['id'] for r in resolved],
                'note':'Derived slot inventory only; never interpreted as training CLI arguments'})
    # Raw long table with distinct provenance for every number.
    metrics=[]
    def metric(id,model,dataset,seed,checkpoint,protocol,name,value,unit,prov,split=None,status='diagnostic'):
        metrics.append(dict(experiment_id=id,model=model,dataset=dataset,split_hash=split or '',seed=seed,
                            checkpoint=checkpoint,protocol_id=protocol,metric_name=name,metric_value=value,
                            unit=unit,provenance=prov,status=status,formal=False))
    for r in checkpoints:
        name=Path(r['checkpoint']).parent.parent.name;ds='IRDST' if '/IRDST/' in r['checkpoint'] else 'DAUB'
        for key,short in [('metrics/mAP50(B)','AP50'),('metrics/mAP50-95(B)','AP50-95')]:
            if key in (r.get('train_metrics') or {}):
                metric('HIST_'+str(Path(r['checkpoint']).relative_to(ROOT)).replace('/','__'),name,ds,
                    r.get('train_args',{}).get('seed'),r['checkpoint'],'historical_unlocked',short,
                    100*r['train_metrics'][key],'percent','checkpoint-embedded training validation; historical split hash unavailable')
    metric('PAPER_STF_3DCONV','STF-3DConv','DAUB',None,None,'paper_reported','AP50',94.87,'percent',
           'codex/TASKBOOK.md section 3.2B; manuscript unavailable',status='paper_reported')
    diag=[]
    for r in evaluations:
        name=Path(r['checkpoint']).parent.parent.name
        old=next(c for c in checkpoints if c['checkpoint']==r['checkpoint'])
        measured=r.get('metrics',{})
        for key,short in [('metrics/mAP50(B)','AP50'),('metrics/mAP50-95(B)','AP50-95')]:
            if key in measured:metric(r['id'],name,'DAUB',0,r['checkpoint'],'diagnostic_current_legacy_val',short,
                    measured[key]*100,'percent',str(WORK/'runs'/r['id']/'run.json'),data['DAUB']['val']['manifest_sha256'])
        if measured:
            diag.append({'model':name,'historical_val_AP50_percent':old['train_metrics']['metrics/mAP50(B)']*100,
                         'diagnostic_AP50_percent':measured['metrics/mAP50(B)']*100,
                         'diagnostic_AP50_95_percent':measured['metrics/mAP50-95(B)']*100,
                         'delta_AP50_pp':(measured['metrics/mAP50(B)']-old['train_metrics']['metrics/mAP50(B)'])*100,
                         'samples':4795,'unique_keyframes':4763,'formal':'no'})
    table('diagnostic_reevaluation',diag)
    costs=[]
    for r in gpu.get('historical_serialized_forward_diagnostic',[]):
        b=r.get('benchmark',{})
        if b.get('status')!='evaluated':continue
        name=r['name'];ck=str(ROOT/'runs/compare_result/ablation/Ablation/DAUB'/name/'weights/best.pt')
        for key,unit in [('mean_latency_ms','ms/clip'),('throughput_clips_s','clips/s'),('peak_allocated_bytes','bytes'),('params','parameters')]:
            metric('COST_DIAG_'+name,name,'synthetic',None,ck,'new_uniform_b1_fp32_640',key,b[key],unit,
                   'evidence/preflight_cuda_1.json; fixed serialized candidate graph')
        costs.append({'model':name,'params':b['params'],'mean_latency_ms':b['mean_latency_ms'],
                      'clips_per_second':b['throughput_clips_s'],'peak_allocated_MiB':b['peak_allocated_bytes']/2**20,
                      'FLOPs':None,'status':'candidate_graph_only'})
    table('candidate_runtime',costs)
    fields=['experiment_id','model','dataset','split_hash','seed','checkpoint','protocol_id','metric_name','metric_value','unit','provenance','status','formal']
    with (WORK/'results/metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(metrics)
    table('stf_repeats',[{'model':m,'verified_seeds':0,'AP50_mean_percent':None,'AP50_sample_std_percent':None,'status':'blocked'} for m in ['full','STF-3DConv']])
    table('components',[{'dataset':d,'design':'factorial_pending' if d=='DAUB' else 'incremental_only',
                         'SSTE':1,'LFSS':l,'SAAM':a,'AP50_percent':None,'status':'blocked'}
                        for d in ['DAUB','IRDST'] for l,a in ([(0,0),(1,0),(0,1),(1,1)] if d=='DAUB' else [(0,0),(1,0),(1,1)])])
    table('scan_neighborhood',[{'variant':f'LFSS_kernel_{r["kernel"]}','kernel':r['kernel'],'n':r['n'],
                                'LFSS_block_params':r['params'],'AP50_percent':None,'status':'correctness_passed_training_blocked'} for r in gpu['historical_lfss_kernels']]+[
                               {'variant':'LFSS_current_wosfi','kernel':None,'n':None,'LFSS_block_params':None,'AP50_percent':None,'status':'invalid_order_only_definition'}])
    table('loss_sensitivity',[{'id':i,'box':v[0],'cls':v[1],'dfl':v[2],'AP50_percent':None,'status':'loss_plumbing_passed_training_blocked'}
                             for i,v in zip(['default']+[f'E{i}' for i in range(4,10)],[(7.5,.5,1.5),(3.75,.5,1.5),(15.,.5,1.5),(7.5,.25,1.5),(7.5,1.,1.5),(7.5,.5,.75),(7.5,.5,3.)])])
    table('temporal_robustness',[{'model':m,'stride':s,'common_frames':4683,'AP50_percent':None,'AP50_95_percent':None,'delta_pp':None,'status':'blocked'}
                               for m in ['full','STF-3DConv','DFAR'] for s in [1,2,4]])
    table('size_scene_breakdown',[{'dataset':d,'split':'val','group':g,'boxes':data[d]['val']['groups'].get(g,0),
                                  'AP50_percent':None,'AP50_95_percent':None,'status':'empty' if not data[d]['val']['groups'].get(g,0) else 'evaluation_blocked'}
                                  for d in ['DAUB','IRDST'] for g in ['d_le_3','3_lt_d_le_7','d_gt_7']])
    table('external_generalization',[{'model':m,'dataset':'unlocked','source':'DAUB_candidate','AP50_percent':None,'status':'blocked'} for m in ['full','STF-3DConv','DFAR']])
    groups={'frozen_before_group_model_scores':True,'definition':'maximum annotated bbox side in original-image pixels',
            'thresholds_px':[3,7],'data_summary':{d:data[d]['val']['groups'] for d in data},
            'DAUB_action':'retain empty bins explicitly; do not invent single-pixel or <=7px detections',
            'IRDST_action':'retain nonempty (3,7] and >7 groups; <=3 empty',
            'scene_groups':None,'scene_reason':'no verified per-sequence scene labels available',
            'group_AP_status':'not computed; reliable ignore-aware evaluator still required',
            'false_alarm_working_threshold':None,'threshold_reason':'source-domain predeclared working threshold is not traceable; do not choose on test'}
    write_json('grouping_lock.json',groups)
    # Use a scientific plot only for measured data, never for missing AP cells.
    os.environ.setdefault('MPLCONFIGDIR',str(WORK/'evidence/matplotlib_config'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    bins=['<=3 px','(3,7] px','>7 px']
    for ax,d in zip(axes,['DAUB','IRDST']):
        values=[data[d]['val']['groups'].get(g,0) for g in ['d_le_3','3_lt_d_le_7','d_gt_7']]
        ax.bar(bins,values,color='#33658a');ax.set_title(d+' validation / held-out directory');ax.set_ylabel('Annotated boxes')
        for j,v in enumerate(values):ax.text(j,v,str(v),ha='center',va='bottom')
        ax.set_ylim(0,max(values)*1.18)
    fig.suptitle('Original-image annotated box maximum side; no model scores used',fontsize=10)
    fig.tight_layout();(WORK/'results/figures').mkdir(exist_ok=True)
    fig.savefig(WORK/'results/figures/annotation_size_groups.svg');fig.savefig(WORK/'results/figures/annotation_size_groups.pdf');plt.close(fig)
    text('inventory.md',f'''# 资源清单

记录时间：{NOW}。仓库目录：`{ROOT}`。

未发现当前目录及祖先目录或项目相关发布副本中的 AGENTS.md。`.git/` 在当前环境为空，`git status` 失败，因此无法核验 commit/未提交改动或创建分支；没有执行 init/reset/clean。保留全部原文件，当前源文件摘要为 `{code_hash}`，备份见 `evidence/source_snapshot.tar.gz`。

真实训练入口是 `tsgmamba_train.py`，目前激活的是 JJB、7 帧、resume=True 的任务，不能直接运行用于此次实验。模型配置为 Ultralytics YAML + `YOLO.train(**kwargs)`。当前验证入口是 `YOLO.val` / `ultralytics/models/yolo/detect/val.py`。`fpstest.py` 可查到旧速度口径，但其注释与输入参数存在不一致。

Python `{sys.executable}`，{platform.python_version()}；PyTorch {packages['torch']}，torchvision {packages['torchvision']}，CUDA build 11.8。两张 RTX 4090，每张约 24 GiB，驱动 535.288.01；核查时无计算任务，GPU 0 有桌面进程。GPU 1 已执行真实 SSM/DCN 前向反向与推理。CUDA 在沙箱内不可用，在已获准的沙箱外运行可用。未升级或安装全局依赖。完整环境见 `evidence/environment.json` 和 `pip_freeze.txt`。工作盘剩余约 {env['disk_free_bytes']/2**30:.1f} GiB。

发现 720 份 args.yaml、928 个 pt/pth 文件（含重复副本，不能视为独立有效实验）；对 40 个重点 checkpoint 做了哈希、参数图、训练配置、严格构造检查，其中 {sum(r.get('strict_reconstruction')=='passed' for r in checkpoints)} 个通过当前构造器的张量严格匹配，但没有一个已通过完整科学复用验收。详见 baseline_registry.json。

DAUB：`/home/dell/lxs/Anti_UAV_dataset/DAUB/yolo_format`，train 8982 帧/10 序列，test 4795 帧/7 序列；YOLO 归一化框。当前 val 指向 test，没有独立 val。IRDST：`/home/dell/lxs/Anti_UAV_dataset/IRDST/yolo_dataset_png/images`，train 20398 帧/42 序列，val 20258 帧/43 序列；labels 在同级 labels 目录。没有配置 test。划分序列 ID 无交集；尚不能排除改名或派生背景重叠。逐帧标注哈希、原尺寸、框、序列和 P1 共同集合见 evidence/data/。根目录同名 DAUB 和 IRDST 实际是 PNG 图，不是数据目录。

DFAR：`/home/dell/lxs/Anti_UAV_compare_method/DFAR-main`，DAUB 候选权重 `logs_updated/DAUB/ep010-loss1.974-val_loss1.744.pth` 的 628 项 state_dict 严格加载通过；对应训练 provenance、论文使用的是哪个 checkpoint 尚未锁定，脚本 seed=2023 不能当作该 checkpoint 的有效历史 seed。

相关发布副本：`/home/dell/lxs/tsgmamba/tsgmamba-release`，提供 README/train/val，但不是可追溯的投稿版本。UAVSwarm、STtran、ITSDT 视频目录存在；尚未认定为适用于本任务且独立的第三红外视频测试协议。指定相关目录未发现 NUDT-MIRSDT/IRSTD-UAV 已适配数据。

缺失：`codex/experiment_manifest.json`、`codex/REVIEWER_ACCEPTANCE.md`、投稿论文 PDF/tex/bib、与有效原始训练对应的不可变源代码/配置和划分摘要。论文图片和 PR 数组存在，Ours.npy 只有 precision/recall，不能追溯 checkpoint 或逐帧预测来源。没有向无关目录或账户搜索。
''')
    text('preflight_report.md',f'''# 正确性检查

CPU：索引恢复和真实 v8DetectionLoss 的 7 组系数检查全部通过，包含一次反向传播。恒等 SSM 的 interleaved 与正确 blocked 恢复都得到 [5.5,11,16.5]；错误 even/odd 恢复得到 [15,15.5,2.5]。实际 IFSS/LFSS 当前求和而非平均，测试明确区分这一比例。非方形、多 batch/channel、转置、反向、merge 均已检查。未比较真实因果 SSM 在不同顺序下的数值相等性。

GPU：当前 LFSS n=1 与历史 LFSS n=4 的 kernel=1/3/5 均做了真实 SSM 前向、有限值和反向检查，只改补充分支卷积的 kernel/padding。历史 n=4 的块参数分别为 72576/76672/84864。STF 使用 4 次 DCN 对齐，3DConv 输出尺寸匹配。两份完整序列化历史模型的 640×640 前向成功；这不等于通过当前构造器或历史协议复现。

真实 DAUB 小样本：{len(sample['sample_checks'])} 个首尾/跨序列边界位置，加载帧都在同一序列，关键帧标注吻合，坐标回原图误差小于 0.01 px。完整索引审计发现 4795 次请求只覆盖 4763 个不同关键帧，遗漏 32 帧，7 个关键帧重复。当前 Format 反转整个 8 通道张量，使时间分支顺序成为 i,i-1,...,i-4；STF 最后时间槽对应最早帧。不能悄悄修改后称为原协议。

E1 尚未通过完整变体检查：缺少可信的 LFSS×SAAM 原始定义，候选 SSTE checkpoint 虽无 CCCBlock，仍存有 Detect.aligned 模块，历史 forward 是否调用不能仅凭 state_dict 判断。

未进行任何正式训练。CPU 第一次 smoke 因测试脚本缺少 inchannel 参数失败，已修正并重跑；不计作模型训练失败。沙箱 CUDA 不可用的尝试同样不计入正式训练。全部检查日志保留在 evidence/。
''')
    text('commands.md',f'''# 已执行与恢复命令

工作目录：`{ROOT}`；解释器：`{sys.executable}`。

以下都是实际入口，无未解析占位符。审计脚本会重新生成自身报告；诊断重评目录若已存在会拒绝覆盖，重启时先读取 run_registry.json 和 runs/*/run.json，并用 `nvidia-smi` / `ps` 检查进程。

```bash
python revision_work/scripts/audit_checkpoints.py
python revision_work/scripts/audit_data.py
python revision_work/scripts/audit_samples.py
python revision_work/scripts/preflight.py --device cpu
python revision_work/scripts/preflight.py --device cuda:1
python revision_work/scripts/diagnostic_reeval.py --model fullsize_model --device 1
python revision_work/scripts/diagnostic_reeval.py --model conv3d-TIMSSM --device 1
python revision_work/scripts/build_deliverables.py
python revision_work/scripts/finalize_artifacts.py
```

GPU 命令必须处于可以访问 NVIDIA 设备的执行上下文，已使用沙箱外权限运行。所有新文件均写入 revision_work。dataset 原目录没有被修改，必要的新缓存重定向到 evidence/。

E1–E9、R1–R4 尚无可诚信给出的正式启动命令：有效基准、准确网络定义、训练与选权协议未锁定，而且缺少语义清单。resolved_experiments.json 对 command 使用 null 并给出原因，不能将其误当 ready。当前 tsgmamba_train.py 会恢复 JJB 训练，禁止用它代替本任务入口。
''')
    text('STATUS.md',f'''# SSE-Mamba 实验状态

更新：{NOW}。状态：**正式训练被协议/历史定义问题阻塞，审计和诊断已完成**。

- 已完成：环境/数据/权重清单；40 个重点 checkpoint 审计；两种扫描索引与损失检查；GPU kernel=1/3/5 与 STF/3DConv 前向反向；两份历史权重诊断重评及逐帧预测；两个模型的新统一速度/显存测量；分组计数；CSV/LaTeX 表壳和有证据的诊断表。
- 正式训练：启动 0 / 完成 0；正式复用 0；正式测试验收 0；13 个 E/R 槽位及对应 TEST 槽位 blocked。
- 主要问题：候选消融为完整模型权重初始化后 20 epoch；当前构造器/历史网络不一致；DAUB 测试目录用于选 best；加载器存在重复/遗漏；LFSS wosfi 当前并非纯顺序对照；完整模型重评与内嵌值差约 7.41 个百分点。
- 缺失：experiment_manifest.json、REVIEWER_ACCEPTANCE.md、投稿稿件和真实原始训练来源。未将任何审稿意见标记已处理。
- 下一步：优先补齐上述文件和可信基准；如果没有有效历史种子，要明确批准新协议和至少 +2 次完整训练（15 次合计）。完整四组合另需评估两个旧组件行补训的 +2 成本。

请先读 FINAL_REPORT.md、implementation_audit.md、remediation_options.md。没有后台训练队列；诊断进程已结束。
''')
    report_lines=['# 执行报告（未完成正式实验）','',
        '已执行任务书允许且不依赖有效历史协议的审计、必要测试、历史权重诊断重评和运行成本测量。正式训练尚未开始，不能将本目录当作已完成的重投稿件证据包。','',
        '| 项目 | 实际数量 / 状态 |','|---|---|',
        '| 新增完整训练 | 0 |','| 正式复用 | 0 |','| 失败的完整训练 | 0 |',
        '| 未完成的默认训练 | 13（blocked） |','| 已完成诊断重评 | 2（推理任务） |',
        '| 已完成成本测量 | 2 个候选图，50 warm-up + 950 次计时 |',
        '| 正式 TEST 验收 | 0 |','',
        '诊断 AP 使用当前历史加载器、同一 DAUB test/images 目录、conf=0.001、NMS IoU=0.7、640、五帧、FP32。原 best checkpoint 在此目录参与验证选取；另外仅覆盖 4763 个不同关键帧。因此以下不是独立测试或有效消融合并统计。','',
        '| 候选权重 | 内嵌验证 AP50 (%) | 本次诊断 AP50 (%) | 本次 AP50–95 (%) |','|---|---:|---:|---:|']
    for r in diag:report_lines.append(f"| {r['model']} | {r['historical_val_AP50_percent']:.3f} | {r['diagnostic_AP50_percent']:.3f} | {r['diagnostic_AP50_95_percent']:.3f} |")
    report_lines+=['','3DConv 接近内嵌指标，仅说明当前路径在该 checkpoint 上数值接近。完整模型的显著差异尚未归因完成；数据、代码和候选图的不一致都有证据，没有调阈值追分，也没有把表现不佳当作额外训练授权。',
        '', '数据：DAUB train/test 8982/4795 帧，IRDST train/val 20398/20258 帧。已保存序列和标注摘要。DAUB P1 的共同集合为 4683 帧。DAUB 所有框最大边 >7 px；IRDST val 的 (3,7] 为 618 框、>7 为 20145 框、<=3 为 0。未伪造空组 AP、单像素覆盖或场景标签。',
        '', '原结论尚未得到新的正式受控证据支持：三 seed 显著性、SSM 独立必要性、扫描顺序增益、四组件组合、损失敏感性和外部泛化均未完成。实际代码表明 STF-3DConv 仍保留 IFSS/LFSS 内 SSM，不能称全网络 CNN 对 Mamba；某些候选图还改变 neck 路由/深度。',
        '', '复现入口见 commands.md。机器可读状态见 run_registry.json，基准证据见 baseline_registry.json，所有数值长表见 results/metrics.csv。formal 列全部为 False。CSV/LaTeX 表中的空单元为未知，不是零。图只绘制真实标注统计。',
        '', '论文源码缺失，提供 writing/manuscript_revision_snippets.md 与英文回复骨架；没有修改或编译原稿。验收文件缺失，无法忠实映射完整 R1.1–R2.4，不将写稿等同于完成实验。',
        '', '继续执行需要：补齐任务配套文件和可信原始来源；或批准 remediation_options.md 中明确的新协议/追加训练成本。当前没有运行中的训练，未对外提交或发送邮件。']
    text('FINAL_REPORT.md','\n'.join(report_lines))
    print(json.dumps({'created_at':NOW,'code_sha256':code_hash,'metrics_rows':len(metrics),'run_slots':len(runs),
                      'formal_trainings':0,'diagnostic_evaluations':len(evaluations)},indent=2))


if __name__=='__main__':main()
