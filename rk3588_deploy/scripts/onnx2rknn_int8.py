"""ONNX -> RKNN (int8, asymmetric_quantized-8) for RK3588, calibrated on DAUB train set."""
from rknn.api import RKNN

ONNX = "/home/dell/lxs/rk3588_deploy/best_split.onnx"
CALIB = "/home/dell/lxs/rk3588_deploy/calib.txt"
RKNN_OUT = "/home/dell/lxs/rk3588_deploy/best_int8.rknn"


def main():
    rknn = RKNN(verbose=True)

    print("[1/4] config ...")
    ret = rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="asymmetric_quantized-8",
        quantized_algorithm="normal",
    )
    if ret != 0:
        raise RuntimeError(f"config failed {ret}")

    print("[2/4] load onnx ...")
    ret = rknn.load_onnx(model=ONNX)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed {ret}")

    print("[3/4] build (int8, calibrate on DAUB train) ...")
    ret = rknn.build(do_quantization=True, dataset=CALIB)
    if ret != 0:
        raise RuntimeError(f"build failed {ret}")

    print("[4/4] export ...")
    ret = rknn.export_rknn(RKNN_OUT)
    if ret != 0:
        raise RuntimeError(f"export failed {ret}")

    print("saved:", RKNN_OUT)
    rknn.release()


if __name__ == "__main__":
    main()
