"""Export frame7 TsgMamba best.pt -> split-head ONNX for RK3588 RKNN conversion.

The original model cannot be exported directly:
  * its selective-scan uses a CUDA-only autograd.Function (SelectiveScanCuda);
  * its DCNAlignment uses torchvision DeformConv2d, which has no ONNX symbolic.

This script patches both at export time (non-invasively, nothing in the model
source is modified):

  * selective_scan_fn  -> exact associative (Hillis-Steele) parallel scan (pscan.py)
  * DeformConv2d.forward -> vectorized DCNv2 built from GridSample + MatMul (dcn_vect.py)
  * torch.einsum "b k d l, k c d -> b k c l" -> batched MatMul (helps RKNN)

Input : [1, 10, H, W]  (3 = RGB current frame, 7 = temporal grayscale frames)
Output: [box0, cls0, box1, cls1, box2, cls2]  (raw logits, decode in post-process)

Usage:
    python export_onnx.py --imgsz 640 --out ../onnx/frame7_640.onnx
    python export_onnx.py --imgsz 640 --dcn conv   # approximate DCN with standard conv
"""
import argparse
import os
import sys
import time
import types

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))   # pscan.py / dcn_vect.py
sys.path.insert(0, REPO)                        # local ultralytics

from pscan import selective_scan_parallel       # noqa: E402


def install_patches(dcn_mode: str):
    # 1) DCN: faithful (grid_sample) or approximate (standard conv)
    import torchvision.ops as ops
    if dcn_mode == "grid":
        from dcn_vect import deform_conv2d_forward
        ops.DeformConv2d.forward = deform_conv2d_forward
    else:
        import torch.nn.functional as F

        def _conv(self, input, offset, mask=None):
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        ops.DeformConv2d.forward = _conv

    # 2) CUDA-only selective scan -> parallel torch scan
    import ultralytics.nn.modules.tgsmamba as tg
    import ultralytics.nn.modules.timeguid as tgd
    tg.selective_scan_fn = selective_scan_parallel
    if hasattr(tgd, "selective_scan_fn"):
        tgd.selective_scan_fn = selective_scan_parallel

    # 3) batched projection einsum -> MatMul (RKNN-friendly)
    orig = torch.einsum

    def _einsum(eq, *a):
        if eq.replace(" ", "") == "bkdl,kcd->bkcl":
            return torch.matmul(a[1], a[0])
        return orig(eq, *a)

    torch.einsum = _einsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(
        REPO, "runs/compare_result/ablation/Ablation/DAUB/frame7/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(REPO, "rk3588_frame7_deploy/onnx/frame7_640.onnx"))
    ap.add_argument("--dcn", choices=["grid", "conv"], default="grid")
    args = ap.parse_args()

    install_patches(args.dcn)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    net = model.model.float().eval()

    detect = net.model[-1]
    def split_forward(self, x):
        return [t for i in range(self.nl) for t in (self.cv2[i](x[i]), self.cv3[i](x[i]))]
    detect.forward = types.MethodType(split_forward, detect)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dummy = torch.zeros(1, 10, args.imgsz, args.imgsz)
    t0 = time.time()
    torch.onnx.export(
        net, dummy, args.out, opset_version=16,
        input_names=["images"],
        output_names=["box0", "cls0", "box1", "cls1", "box2", "cls2"],
        do_constant_folding=True, dynamic_axes=None,
    )
    print(f"exported {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, {time.time()-t0:.1f}s, dcn={args.dcn})")


if __name__ == "__main__":
    main()
