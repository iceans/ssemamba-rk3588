import torch
from thop import profile
from thop import clever_format
from ultralytics import YOLO
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 1. 实例化你的模型 (假设你的模型类叫 MyCustomYOLO)
# model = MyCustomYOLO()
# 2. 加载权重
# weights = torch.load('path/to/your/best.pt', map_location='cpu')
# model.load_state_dict(weights['model'] if 'model' in weights else weights)
# model_wrapper = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/my_2timMix_945/weights/best.pt')
# model_wrapper = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/ablation/Ablation/DAUB/deepssstage_8790/weights/best.pt')
# model_wrapper = YOLO('./ultralytics/cfg/models/v8/yolov8n_mamba.yaml')
# model_wrapper = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml')
# model_wrapper = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/ablation/Ablation/DAUB/wodwconv_w_dcn/weights/best.pt')
model_wrapper = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/detect/train90/weights/best.pt')
pytorch_model = model_wrapper.model.eval()

# 深拷贝一个模型专门用来算计算量
# 注意：有时深拷贝也会把 buffer 拷过去，所以最好结合第一种方法的清理使用，
# 但通常对于部分动态挂载的 hook，深拷贝能解决冲突。
model_for_profiling = copy.deepcopy(pytorch_model)
device = torch.device('cuda:0' )
model_for_profiling = model_for_profiling.to(device)
# (同样建议加上方法一的清理循环以防万一)
for m in model_for_profiling.modules():
    m._buffers.pop("total_ops", None)
    m._buffers.pop("total_params", None)

dummy_input = torch.randn(2, 10, 640, 640).to(next(model_for_profiling.parameters()).device)
flops, params = profile(model_for_profiling, inputs=(dummy_input, ), verbose=False)
# ... 打印结果 ...
flops_str, params_str = clever_format([flops, params], "%.3f")
print(f"总计算量 (FLOPs): {flops_str}")
print(f"总参数量 (Params): {params_str}")