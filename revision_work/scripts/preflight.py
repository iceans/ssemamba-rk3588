"""Necessary deterministic CPU/GPU checks. No formal training or test metrics."""
import argparse
import copy
import json
import sys
import time
import traceback
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn as nn
from ultralytics.nn.modules import tgsmamba as tg
from ultralytics.nn.modules import timeguid as ti
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import scale_boxes
from scan_pairing import pack,restore


def set_lfss_kernel(module,kernel):
    """Change only local supplemental depthwise kernels, retaining n and all other settings."""
    if kernel not in (1,3,5):raise ValueError(kernel)
    targets=[]
    for name,m in module.named_modules():
        if isinstance(m,tg.SS2D1conv):
            for path,conv in list(m.my_conv.named_modules()):
                if isinstance(conv,nn.Conv2d):
                    if conv.groups != conv.in_channels or conv.in_channels != conv.out_channels:
                        raise ValueError('Supplement is not depthwise')
                    if conv.stride!=(1,1) or conv.dilation!=(1,1):
                        raise ValueError('Unexpected baseline stride/dilation')
                    replacement=nn.Conv2d(conv.in_channels,conv.out_channels,kernel,stride=conv.stride,
                        padding=kernel//2,dilation=conv.dilation,groups=conv.groups,
                        bias=conv.bias is not None,padding_mode=conv.padding_mode,
                        device=conv.weight.device,dtype=conv.weight.dtype)
                    parent=m.my_conv
                    parts=path.split('.')
                    for part in parts[:-1]:parent=getattr(parent,part)
                    setattr(parent,parts[-1],replacement)
                    targets.append(name+'.my_conv.'+path)
    if not targets:raise ValueError('No LFSS supplement found')
    return targets


def scan_checks():
    checks=[]
    main=torch.tensor([10.,20.,30.]).reshape(1,1,3)
    supp=torch.tensor([1.,2.,3.]).reshape(1,1,3)
    for order in ['interleaved','blocked']:
        out=restore(pack(main,supp,order),3,order)
        torch.testing.assert_close(out,torch.tensor([5.5,11.,16.5]).reshape(1,1,3))
        checks.append({'name':order+'_identity','output':out.flatten().tolist(),'passed':True})
    wrong=restore(pack(main,supp,'blocked'),3,'interleaved')
    assert not torch.allclose(wrong,(main+supp)/2)
    checks.append({'name':'blocked_with_even_odd_pairing_counterexample','output':wrong.flatten().tolist(),
                   'passed':True,'meaning':'Counterexample for the commented blocked IFSS alternative with unchanged reduction; not historical provenance.'})
    for h,w in [(2,3),(3,5)]:
        main=torch.arange(2*4*h*w,dtype=torch.float).view(2,4,h,w)
        supp=main*0.1+1
        for direction in [0,1,2,3]:
            a=tg.cross_scan_fwd(main,scans=0)[:,direction]
            b=tg.cross_scan_fwd(supp,scans=0)[:,direction]
            for order in ['interleaved','blocked']:
                out=restore(pack(a,b,order),h*w,order)
                if direction in [2,3]:out=out.flip(-1)
                if direction in [1,3]:out=out.view(2,4,w,h).transpose(2,3)
                else:out=out.view(2,4,h,w)
                torch.testing.assert_close(out,(main+supp)/2)
        scans=tg.cross_scan_fn(main,scans=0,force_torch=True).view(2,4,4,h,w)
        merged=tg.cross_merge_fn(scans,scans=0,force_torch=True).view_as(main)
        torch.testing.assert_close(merged/4,main)
    checks.append({'name':'non_square_batch_channel_four_direction_roundtrip','passed':True})
    # Execute actual production core functions with identity selective scan only.
    main=torch.arange(2*4*3*5,dtype=torch.float).view(2,4,3,5)+1
    supp=main/10
    lengths=[]
    def identity_scan(u,*args,**kwargs):lengths.append(u.shape[-1]);return u
    mod=ti.SS2DguideMix(inchannel=2,d_model=4,ssm_ratio=1.,channel_first=True)
    mod.out_norm=nn.Identity()
    with patch.object(ti,'selective_scan_fn',identity_scan):
        out=mod.forward_corev2([main,supp],scan_force_torch=True)
    torch.testing.assert_close(out,main+supp)
    checks.append({'name':'actual_IFSS_identity_core','passed':True,'sequence_length':lengths[-1],
                   'reduction':'sum; normalized by 2 only for mean-index comparison'})
    class Supply(nn.Module):
        def forward(self,x):return supp
    mod=tg.SS2D1conv(inchannel=4,d_model=4,ssm_ratio=1.,channel_first=True);mod.out_norm=nn.Identity();mod.my_conv=Supply()
    with patch.object(tg,'selective_scan_fn',identity_scan):out=mod.forward_corev2(main,scan_force_torch=True)
    torch.testing.assert_close(out,main+supp)
    checks.append({'name':'actual_LFSS_identity_core','passed':True,'sequence_length':lengths[-1],'reduction':'sum'})
    mod=tg.SS2D1conv_wosfi(inchannel=4,d_model=4,ssm_ratio=1.,channel_first=True);mod.out_norm=nn.Identity();mod.my_conv=Supply()
    with patch.object(tg,'selective_scan_fn',identity_scan):out=mod.forward_corev2(main,scan_force_torch=True)
    torch.testing.assert_close(out,main)
    checks.append({'name':'actual_wosfi_is_not_order_only','passed':True,'sequence_length':lengths[-1],
                   'expected_order_only_length':30,'scientific_status':'invalid as an order-only comparison under current code'})
    return checks


def loss_checks():
    class Model(nn.Module):
        def __init__(self,gains):
            super().__init__();self.anchor=nn.Parameter(torch.zeros(1));head=nn.Module()
            head.stride=torch.tensor([8.,16.,32.]);head.nc=1;head.reg_max=16
            self.model=nn.ModuleList([head]);self.args=SimpleNamespace(cube=True,box=gains[0],cls=gains[1],dfl=gains[2])
    torch.manual_seed(0)
    feats=[torch.randn(2,65,s,s) for s in [8,4,2]]
    batch={'img':torch.zeros(2,8,64,64),'batch_idx':torch.tensor([0,1]),'cls':torch.zeros(2,1),
           'bboxes_t5':torch.tensor([[.5,.5,.25,.25],[.3,.4,.2,.3]])}
    base=(7.5,.5,1.5); gains=[base,(3.75,.5,1.5),(15.,.5,1.5),(7.5,.25,1.5),(7.5,1.,1.5),(7.5,.5,.75),(7.5,.5,3.)]
    results=[];reference=None
    for g in gains:
        inp=[f.clone().requires_grad_() for f in feats]
        total,components=v8DetectionLoss(Model(g))(inp,batch)
        total.sum().backward()
        assert torch.isfinite(components).all() and all(torch.isfinite(f.grad).all() for f in inp)
        if reference is None:reference=components.clone();assert (reference>0).all()
        torch.testing.assert_close(components,reference*torch.tensor(g)/torch.tensor(base))
        results.append({'gains':g,'weighted_components':components.tolist(),'passed':True})
    return results


def kernel_checks(device):
    torch.manual_seed(0)
    model=tg.CCCBlock(inchannel=8,hidden_dim=8).to(device)
    results=[]
    for k in [1,3,5]:
        mod=copy.deepcopy(model);paths=set_lfss_kernel(mod,k)
        x=torch.randn(2,8,6,10,device=device,requires_grad=True)
        y=mod(x); y.square().mean().backward()
        assert y.shape==x.shape and torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        results.append({'kernel':k,'paths':paths,'n':len(mod.op.my_conv),'shape':list(y.shape),
                        'params':sum(p.numel() for p in mod.parameters()),'passed':True})
    return results


def gpu_checks(device):
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this execution context')
    torch.cuda.set_device(device)
    results={'lfss_kernels':kernel_checks(device)}
    stem=ti.TimeMambaStem(5,16).to(device)
    aligned_calls=[]
    hook=stem.alignment.dcn.register_forward_hook(lambda *a:aligned_calls.append(True))
    x=torch.randn(2,5,32,48,device=device,requires_grad=True);y=stem(x);y.square().mean().backward()
    hook.remove()
    assert len(aligned_calls)==4 and torch.isfinite(x.grad).all()
    results['stf']={'shape':list(y.shape),'dcn_calls':len(aligned_calls),'passed':True}
    conv=ti.Conv3dBlock(1,16,3,2,1).to(device)
    x=torch.randn(2,5,32,48,device=device,requires_grad=True);y=conv(x);y.square().mean().backward()
    assert y.shape==(2,16,16,24) and torch.isfinite(x.grad).all()
    results['stf_3dconv']={'shape':list(y.shape),'passed':True}
    del stem,conv,x,y
    # Check the serialized historical LFSS with its actual four-layer supplement.
    path=ROOT/'runs/compare_result/ablation/Ablation/DAUB/frame5/weights/best.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    block=(ck.get('ema') or ck['model']).model[21].float().to(device)
    block.requires_grad_(True)
    historical_kernels=[]
    for k in [1,3,5]:
        mod=copy.deepcopy(block);paths=set_lfss_kernel(mod,k)
        inp=torch.randn(2,mod.first_reshape.in_channels,6,10,device=device,requires_grad=True)
        output=mod(inp);output.square().mean().backward()
        assert len(mod.op.my_conv)==4 and torch.isfinite(inp.grad).all()
        historical_kernels.append({'kernel':k,'n':len(mod.op.my_conv),'changed_convs':paths,
                                   'params':sum(p.numel() for p in mod.parameters()),
                                   'shape':list(output.shape),'passed':True})
        del mod,inp,output
    del block,ck
    results['historical_lfss_kernels']=historical_kernels
    results['gpu_name']=torch.cuda.get_device_name(device)
    results['gpu_process_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                                            '--format=csv,noheader'],text=True)
    protocol={'batch':1,'precision':'float32','channels':8,'frames':5,'resolution':[640,640],
              'warmup':50,'timed_iterations':950,'include_preprocess':False,'include_nms':False,
              'synchronize_before_after':True,'fuse':False,'device':device,
              'scope':'new uniform protocol for serialized candidate graphs; not historical FPS reproduction'}
    (ROOT/'revision_work/benchmark_protocol.json').write_text(json.dumps(protocol,indent=2))
    results['historical_serialized_forward_diagnostic']=[]
    for name in ['fullsize_model','conv3d-TIMSSM']:
        path=ROOT/'runs/compare_result/ablation/Ablation/DAUB'/name/'weights/best.pt'
        record={'name':name,'formal_test':False,'reason':'historical definition/protocol gates have not passed'}
        try:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            net=(ck.get('ema') or ck['model']).float().to(device).eval()
            # Test the fully serialized graph; do not fill missing parameters in a new graph.
            inp=torch.zeros(1,8,640,640,device=device)
            torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            with torch.no_grad():value=net(inp)
            torch.cuda.synchronize(device)
            tensor=value[0] if isinstance(value,tuple) else value
            assert torch.isfinite(tensor).all()
            record.update(passed=True,shape=list(tensor.shape),elapsed_s=time.perf_counter()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            # Benchmark only after both diagnostic evaluation jobs have finished.
            import os
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            others=[int(p.strip()) for p in processes.splitlines() if p.strip().isdigit() and int(p.strip())!=os.getpid()]
            if others:
                record['benchmark']={'status':'blocked','competing_pids':others}
            else:
                with torch.no_grad():
                    for _ in range(protocol['warmup']):net(inp)
                    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                    measurements=[]
                    for _ in range(protocol['timed_iterations']):
                        torch.cuda.synchronize(device);t0=time.perf_counter();net(inp);torch.cuda.synchronize(device)
                        measurements.append(time.perf_counter()-t0)
                ordered=sorted(measurements);mean=sum(measurements)/len(measurements)
                record['benchmark']={'status':'evaluated','mean_latency_ms':mean*1000,'throughput_clips_s':1/mean,
                                     'median_latency_ms':ordered[len(ordered)//2]*1000,
                                     'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                                     'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                                     'params':sum(p.numel() for p in net.parameters()),
                                     'flops':None,'flops_note':'No audited custom SSM/DCN FLOPs handlers',
                                     'timings_s':measurements}
            del net,ck,inp,value,tensor
        except Exception:record.update(passed=False,error=traceback.format_exc())
        results['historical_serialized_forward_diagnostic'].append(record)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(2)
    report={'device':args.device,'torch':torch.__version__,'formal_training':False,'formal_metrics':False}
    start=time.perf_counter()
    if args.device=='cpu':
        for name,func in [('scan',scan_checks),('loss',loss_checks)]:
            try:report[name]={'passed':True,'checks':func()}
            except Exception:report[name]={'passed':False,'error':traceback.format_exc()}
        original=torch.tensor([[10.,20.,30.,40.]])
        transformed=original*2+torch.tensor([0.,10.,0.,10.])
        restored=scale_boxes((110,160),transformed,(50,80),ratio_pad=((2.,2.),(0.,10.)))
        torch.testing.assert_close(restored,original)
        report['coordinate_roundtrip']={'passed':True}
    else:report.update(gpu_checks(args.device))
    report['elapsed_s']=time.perf_counter()-start
    out=ROOT/'revision_work/evidence'/('preflight_'+args.device.replace(':','_')+'.json')
    if out.exists():
        from shutil import copyfile
        copyfile(out,out.with_suffix('.previous.json'))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if any(isinstance(v,dict) and v.get('passed') is False for v in report.values()):sys.exit(1)


if __name__=='__main__':main()
