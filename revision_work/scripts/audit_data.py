"""Read-only sequence, annotation and split audit; never writes dataset caches."""
import collections
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'revision_work/evidence/data'
EXTS = {'.bmp','.png','.jpg','.jpeg','.tif','.tiff'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def frame_row(p):
    lp = Path(str(p).replace('/images/','/labels/')).with_suffix('.txt')
    raw = lp.read_bytes() if lp.is_file() else None
    with Image.open(p) as im:
        w,h = im.size
    seq,idx = p.stem.rsplit('_',1)
    boxes=[];errors=[]
    if raw is None:
        errors.append('missing_annotation')
    else:
        for line in raw.decode().splitlines():
            try:
                values = [float(v) for v in line.split()]
                if len(values)!=5 or not all(math.isfinite(v) for v in values):
                    raise ValueError('invalid YOLO box')
                cls,x,y,bw,bh=values
                if int(cls)!=cls or cls<0 or not(0<=x<=1 and 0<=y<=1 and 0<bw<=1 and 0<bh<=1):
                    raise ValueError('invalid class or normalized coordinates')
                boxes.append({'class':int(cls),'xyxy':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h],
                              'max_side_px':max(bw*w,bh*h)})
            except ValueError as exc:errors.append(str(exc))
    return {'seq_id':seq,'frame_id':int(idx),'file':str(p),'width':w,'height':h,
            'file_size':p.stat().st_size,'label_path':str(lp),
            'label_sha256':hashlib.sha256(raw).hexdigest() if raw is not None else None,
            'boxes':boxes,'errors':errors}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={}
    for name in ['DAUB','IRDST']:
        cfg=ROOT/'ultralytics/cfg/datasets'/f'{name}.yaml'
        data=yaml.safe_load(cfg.read_text()); base=Path(data['path']); result={}
        sets={}
        for split in ['train','val','test']:
            if split not in data:continue
            folder=base/data[split]
            files=sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTS)
            with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(frame_row,files))
            seqs=collections.defaultdict(list)
            for row in rows:seqs[row['seq_id']].append(row['frame_id'])
            seqrows=[{'seq_id':s,'frames':len(v),'first_frame':min(v),'last_frame':max(v),
                      'missing_ids':sorted(set(range(min(v),max(v)+1))-set(v))} for s,v in sorted(seqs.items())]
            groups=collections.Counter('d_le_3' if b['max_side_px']<=3 else '3_lt_d_le_7' if b['max_side_px']<=7 else 'd_gt_7'
                                       for r in rows for b in r['boxes'])
            common=[]
            for seq in seqrows:
                sr=[r for r in rows if r['seq_id']==seq['seq_id']]
                for i,r in enumerate(sr):
                    if i>=16:
                        common.append({'seq_id':r['seq_id'],'zero_based_index':i,'frame_id':r['frame_id'],
                                       'file':r['file'],'clips':{str(s):[sr[i-j*s]['file'] for j in range(4,-1,-1)] for s in [1,2,4]}})
            target=OUT/f'{name}_{split}_frames.jsonl'
            with target.open('w') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            (OUT/f'{name}_{split}_sequences.json').write_text(json.dumps(seqrows,indent=2))
            if split!='train':
                (OUT/f'{name}_{split}_temporal_common.json').write_text(json.dumps(common,indent=2))
            result[split]={'path':str(folder),'num_frames':len(rows),'num_sequences':len(seqs),
                           'num_boxes':sum(len(r['boxes']) for r in rows),'groups':dict(groups),
                           'manifest_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                           'sequence_list_sha256':digest(seqrows),'frames_with_errors':sum(bool(r['errors']) for r in rows),
                           'empty_frames':sum(not r['boxes'] for r in rows),
                           'common_stride_frames':len(common),'excluded_short_sequences':sum(s['frames']<17 for s in seqrows),
                           'hash_scope':'paths, image sizes and dimensions, full label bytes; not full image content'}
            sets[split]=seqs
            print(name,split,result[split],flush=True)
        result['sequence_id_overlap']={f'{a}__{b}':sorted(set(sets[a])&set(sets[b])) for a in sets for b in sets if a<b}
        result['content_overlap_status']='Sequence/frame IDs checked. Renamed/derived background overlap not yet established.'
        result['test_is_used_as_val']=(data.get('val')=='test/images')
        summary[name]=result
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
