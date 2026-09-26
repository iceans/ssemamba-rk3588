"""ONNX -> RKNN fp16 for RK3588 (frame7 TsgMamba).

Notes
-----
* mean/std are [0]*10 / [255]*10 because the 10-channel input is fed as raw
  [0,255] and the model was trained on /255-normalised inputs.
* `disable_rules=['reduce_reshape_op_around_split']` works around an RKNN
  toolkit optimizer bug triggered by the DCN reshape ("Can not reshape from
  [1,18,320,320] to [1,320,1,320]").
* The 4 DeformConv GridSample ops have no NPU lowering; RKNN emits them as
  CPU/custom operators ("No lowering found ... use CustomOperatorLower").
  Whether rknn-toolkit-lite2 on the board can execute them must be verified
  on-device. If not, export with `--dcn conv` in export_onnx.py instead.

Usage:
    python onnx2rknn_fp16.py --onnx ../onnx/frame7_640.onnx --out ../model/frame7_640_fp16.rknn
"""
import argparse
import os

from rknn.api import RKNN


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--onnx", default=os.path.join(here, "../onnx/frame7_640.onnx"))
    ap.add_argument("--out", default=os.path.join(here, "../model/frame7_640_fp16.rknn"))
    ap.add_argument("--nc", type=int, default=10, help="input channels")
    args = ap.parse_args()

    rknn = RKNN(verbose=True)
    assert rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        mean_values=[[0] * args.nc],
        std_values=[[255] * args.nc],
        disable_rules=["reduce_reshape_op_around_split"],
    ) == 0, "config failed"
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=False) == 0, "build failed"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    assert rknn.export_rknn(args.out) == 0, "export failed"
    print("saved:", args.out, f"({os.path.getsize(args.out)/1e6:.1f} MB)")
    rknn.release()


if __name__ == "__main__":
    main()
