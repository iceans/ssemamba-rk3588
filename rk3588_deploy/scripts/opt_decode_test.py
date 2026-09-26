import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_int8.rknn"
IMG = "/home/Tronlong/rknn_deploy/test_uav.bmp"
IMG_SIZE = 640; CONF = 0.5

r = RKNNLite(); r.load_rknn(MODEL); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
fr = cv2.imread(IMG)
x = cv2.cvtColor(cv2.resize(fr,(640,640),interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
outs = r.inference(inputs=[x], data_format=["nhwc"])

def sig(z): return 1.0/(1.0+np.exp(-z))

def decode_full(outputs):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2*i], np.float32); lg = np.asarray(outputs[2*i+1], np.float32)
        n,c,h,w = box.shape
        y = box.reshape(n,4,c//4,h,w); e = np.exp(y-y.max(2,keepdims=True)); y = e/e.sum(2,keepdims=True)
        dist = (y*np.arange(c//4).reshape(1,1,c//4,1,1)).sum(2)
        col,row = np.meshgrid(np.arange(w),np.arange(h))
        grid = np.concatenate((col.reshape(1,1,h,w),row.reshape(1,1,h,w)),1)
        s = np.array([IMG_SIZE//w,IMG_SIZE//h]).reshape(1,2,1,1)
        xyxy = np.concatenate(((grid+0.5-dist[:,0:2])*s,(grid+0.5+dist[:,2:4])*s),1)
        boxes.append(xyxy.reshape(4,-1).T); scores.append(sig(lg).reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)

def decode_fast(outputs, conf=CONF):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2*i], np.float32); lg = np.asarray(outputs[2*i+1], np.float32)
        h,w = lg.shape[2], lg.shape[3]
        cls = sig(lg).reshape(-1)
        idx = np.nonzero(cls>=conf)[0]
        if idx.size==0: continue
        bf = box[0].reshape(4,16,h,w).transpose(2,3,0,1).reshape(h*w,4,16)[idx]
        e = np.exp(bf-bf.max(2,keepdims=True)); p = e/e.sum(2,keepdims=True)
        dist = (p*np.arange(16)).sum(2)
        gy, gx = idx//w, idx%w; st = IMG_SIZE//w
        boxes.append(np.stack([(gx+0.5-dist[:,0])*st,(gy+0.5-dist[:,1])*st,
                               (gx+0.5+dist[:,2])*st,(gy+0.5+dist[:,3])*st],1))
        scores.append(cls[idx])
    return np.concatenate(boxes), np.concatenate(scores)

def timeit(fn,n=200):
    for _ in range(5): fn()
    t=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t)/n*1000

tf = timeit(lambda: decode_full(outs)); tfa = timeit(lambda: decode_fast(outs))
bf, sf = decode_full(outs); bq, sq = decode_fast(outs)
print(f"decode_full (全部8400 anchor): {tf:.2f} ms, boxes={len(bf)}")
print(f"decode_fast (先过阈值再DFL)   : {tfa:.2f} ms, boxes={len(bq)}")
print("survivors @conf0.5:", (sf>=CONF).sum())
if len(bq):
    kf = sf.argsort()[::-1][:3]; kq = sq.argsort()[::-1][:3]
    print("full top:", [ (round(float(sf[k]),3), bf[k].round(1).tolist()) for k in kf ])
    print("fast top:", [ (round(float(sq[k]),3), bq[k].round(1).tolist()) for k in kq ])
r.release()
