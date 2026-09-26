import time, glob, os, numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
from rknnlite.api import RKNNLite

MODEL="/home/Tronlong/rknn_deploy/best_int8.rknn"
DIR="/home/Tronlong/rknn_deploy/img640"
IMG_SIZE=640; CONF, IOU = 0.25, 0.45

def sig(z): return 1.0/(1.0+np.exp(-z))
def decode_fast(outputs, conf=CONF):
    B=[]; S=[]
    for i in range(3):
        box=np.asarray(outputs[2*i],np.float32); lg=np.asarray(outputs[2*i+1],np.float32)
        h,w=lg.shape[2],lg.shape[3]; cls=sig(lg).reshape(-1)
        idx=np.nonzero(cls>=conf)[0]
        if idx.size==0: continue
        bf=box[0].reshape(4,16,h,w).transpose(2,3,0,1).reshape(h*w,4,16)[idx]
        e=np.exp(bf-bf.max(2,keepdims=True)); p=e/e.sum(2,keepdims=True)
        dist=(p*np.arange(16)).sum(2); gy,gx=idx//w,idx%w; st=IMG_SIZE//w
        B.append(np.stack([(gx+0.5-dist[:,0])*st,(gy+0.5-dist[:,1])*st,(gx+0.5+dist[:,2])*st,(gy+0.5+dist[:,3])*st],1))
        S.append(cls[idx])
    if not B: return np.zeros((0,4),np.float32), np.zeros(0,np.float32)
    return np.concatenate(B), np.concatenate(S)
def nms(boxes,scores,thr=IOU):
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        ovr=inter/(areas[i]+areas[order[1:]]-inter+1e-12)
        order=order[1+np.where(ovr<=thr)[0]]
    return np.array(keep,dtype=int)

def make(core):
    r=RKNNLite(); assert r.load_rknn(MODEL)==0; assert r.init_runtime(core_mask=core)==0; return r

def one(path, r):
    a=time.perf_counter(); frame=cv2.imread(path); b=time.perf_counter()
    xx=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.uint8)[None]; c=time.perf_counter()
    o=r.inference(inputs=[xx],data_format=["nhwc"]); d=time.perf_counter()
    bxs,s=decode_fast(o)
    if len(bxs):
        k=nms(bxs,s)
        for bb in bxs[k]: cv2.rectangle(frame,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(0,0,255),2)
    e=time.perf_counter()
    return (b-a, c-b, d-c, e-d)

files=sorted(glob.glob(os.path.join(DIR,"*.jpg")))
print("images:", len(files), "size 640x640")

# single core
r=make(RKNNLite.NPU_CORE_0)
for f in files[:5]: one(f,r)
n=0; T=[0,0,0,0]; t0=time.perf_counter()
for f in files:
    dt=one(f,r)
    for j in range(4): T[j]+=dt[j]
    n+=1
tot=time.perf_counter()-t0
print(f"[single core] {n} imgs, {tot:.2f}s -> {n/tot:.1f} FPS end-to-end")
print(f"   imread {T[0]/n*1000:.2f} | preprocess {T[1]/n*1000:.2f} | infer {T[2]/n*1000:.2f} | decode+nms+draw {T[3]/n*1000:.2f} ms")
r.release()

# 3-core parallel
rk=[make(m) for m in (RKNNLite.NPU_CORE_0,RKNNLite.NPU_CORE_1,RKNNLite.NPU_CORE_2)]
for f in files[:5]:
    for rr in rk: one(f,rr)
def worker(i):
    rr=rk[i%3]
    for f in files[i::3]: one(f,rr)
t0=time.perf_counter()
with ThreadPoolExecutor(max_workers=3) as ex: list(ex.map(worker, range(3)))
tot=time.perf_counter()-t0
print(f"[3-core parallel] {n} imgs, {tot:.2f}s -> {n/tot:.1f} FPS (throughput)")
for rr in rk: rr.release()
