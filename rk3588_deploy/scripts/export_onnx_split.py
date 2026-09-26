"""Export best.pt -> split-head ONNX (6 outputs) for RK3588 RKNN conversion.

Output layout (matches board's func.py yolov8_post_process, pair_per_branch=2):
    [box0, cls0, box1, cls1, box2, cls2]
    box_i: [1, 64, h, w]  raw DFL logits (cv2 output, BEFORE softmax)
    cls_i: [1,  1, h, w]  class prob (cv3 output AFTER sigmoid)
"""
import types
import torch
from ultralytics import YOLO

PT_PATH = "/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/yolov8n_813/weights/best.pt"
OUT_PATH = "/home/dell/lxs/rk3588_deploy/best_split.onnx"
IMG_SIZE = 640


def split_forward(self, x):
    outs = []
    for i in range(self.nl):
        box = self.cv2[i](x[i])   # [1, 64, h, w] raw DFL logits
        cls = self.cv3[i](x[i])   # [1, 1, h, w] raw cls logits (sigmoid in postprocess)
        outs.append(box)
        outs.append(cls)
    return outs


def main():
    model = YOLO(PT_PATH)
    net = model.model
    net.eval()

    detect = net.model[-1]
    assert type(detect).__name__ == "Detect", f"last module is {type(detect).__name__}"
    print(f"Detect: nc={detect.nc} reg_max={detect.reg_max} nl={detect.nl} no={detect.no}")

    detect.forward = types.MethodType(split_forward, detect)

    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        outs = net(dummy)
        for o in outs:
            print("trace output:", tuple(o.shape))

    torch.onnx.export(
        net,
        dummy,
        OUT_PATH,
        opset_version=12,
        input_names=["images"],
        output_names=["box0", "cls0", "box1", "cls1", "box2", "cls2"],
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print("saved:", OUT_PATH)


if __name__ == "__main__":
    main()
