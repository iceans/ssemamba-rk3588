"""int8 RKNN FPS benchmark on RK3588 (run on board with miniconda rknn env)."""
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from rknnlite.api import RKNNLite

MODEL = "/home/Tronlong/rknn_deploy/best_int8.rknn"
IMG = "/home/Tronlong/RK3588_uavtest/detect_results/0302_detect.jpg"
N = 100


def prepare_input():
    img = cv2.imread(IMG)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    return img.astype(np.uint8)[None]  # uint8 [1,640,640,3], runtime normalizes /255


def make_rknn(core_mask):
    r = RKNNLite()
    assert r.load_rknn(MODEL) == 0
    assert r.init_runtime(core_mask=core_mask) == 0
    return r


def bench(name, rknn, img, n=N):
    for _ in range(10):
        rknn.inference(inputs=[img], data_format=["nhwc"])
    t = time.time()
    for _ in range(n):
        rknn.inference(inputs=[img], data_format=["nhwc"])
    dt = time.time() - t
    print(f"{name}: {dt/n*1000:.2f} ms/infer, {n/dt:.2f} FPS")


def main():
    img = prepare_input()
    print("input:", img.shape, img.dtype)
    bench("single core (NPU_CORE_0)      ", make_rknn(RKNNLite.NPU_CORE_0), img)
    bench("3-core combined (NPU_CORE_0_1_2)", make_rknn(RKNNLite.NPU_CORE_0_1_2), img)

    rk = [make_rknn(m) for m in (RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2)]
    for r in rk:
        for _ in range(10):
            r.inference(inputs=[img], data_format=["nhwc"])

    def worker(r):
        for _ in range(N):
            r.inference(inputs=[img], data_format=["nhwc"])

    t = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(worker, rk))
    dt = time.time() - t
    print(f"3-instance parallel (3 cores): {N*3/dt:.2f} FPS (total throughput)")


if __name__ == "__main__":
    main()
