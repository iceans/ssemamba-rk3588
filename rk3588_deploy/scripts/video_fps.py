import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_int8.rknn"
VIDEO = "/home/Tronlong/RK3588_uav/1.mp4"
IMG_SIZE = 640
CONF, IOU = 0.5, 0.45

def dfl(pos):
    n, c, h, w = pos.shape
    y = pos.reshape(n, 4, c // 4, h, w)
    e = np.exp(y - y.max(2, keepdims=True))
    y = e / e.sum(2, keepdims=True)
    return (y * np.arange(c // 4).reshape(1, 1, c // 4, 1, 1)).sum(2)

def decode(outputs):
    boxes, scores = [], []
    for i in range(3):
        box = np.asarray(outputs[2*i], dtype=np.float32)
        cls = 1.0 / (1.0 + np.exp(-np.asarray(outputs[2*i+1], dtype=np.float32)))
        n, c, h, w = box.shape
        pos = dfl(box)
        col, row = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.concatenate((col.reshape(1,1,h,w), row.reshape(1,1,h,w)), 1)
        s = np.array([IMG_SIZE//w, IMG_SIZE//h]).reshape(1,2,1,1)
        xyxy = np.concatenate(((grid+0.5-pos[:,0:2])*s, (grid+0.5+pos[:,2:4])*s), 1)
        boxes.append(xyxy.reshape(4,-1).T); scores.append(cls.reshape(-1))
    return np.concatenate(boxes), np.concatenate(scores)

def nms(boxes, scores, thr=IOU):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        ovr=inter/(areas[i]+areas[order[1:]]-inter+1e-12)
        order=order[1+np.where(ovr<=thr)[0]]
    return np.array(keep,dtype=int)

cap = cv2.VideoCapture(VIDEO)
print("video:", VIDEO, "size=", int(cap.get(3)), "x", int(cap.get(4)), "src_fps=", cap.get(cv2.CAP_PROP_FPS))
r = RKNNLite(); r.load_rknn(MODEL); r.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# warmup
for _ in range(5):
    ok, fr = cap.read()
    xx = cv2.cvtColor(cv2.resize(fr,(640,640)), cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
    r.inference(inputs=[xx], data_format=["nhwc"])

n=0; tr=ti=td=tp=0.0; t0=time.perf_counter()
while n < 300:
    a=time.perf_counter()
    ok, frame = cap.read()
    if not ok: break
    b=time.perf_counter(); tr += b-a
    xx = cv2.cvtColor(cv2.resize(frame,(640,640),interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB).astype(np.uint8)[None]
    c=time.perf_counter(); tp += c-b
    o = r.inference(inputs=[xx], data_format=["nhwc"])
    d=time.perf_counter(); ti += d-c
    boxes, scores = decode(o)
    m = scores>=CONF; boxes, scores = boxes[m], scores[m]
    if len(boxes): nms(boxes, scores)
    e=time.perf_counter(); td += e-d
    n+=1
dt = time.perf_counter()-t0
print(f"frames={n}  total={dt:.2f}s  ->  {n/dt:.1f} FPS  (single core, video read+pre+infer+decode+nms)")
print(f"  read {tr/n*1000:.2f}ms | preprocess {tp/n*1000:.2f}ms | infer {ti/n*1000:.2f}ms | decode+nms {td/n*1000:.2f}ms")
cap.release(); r.release()
