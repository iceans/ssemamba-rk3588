#--------------------------------------------------------------#
# fps_test.py  (for Ultralytics YOLO, 如 yolov8/yolo11)  —— 单帧检测器
# 用与前面 IR 方法相同的标准测纯推理 FPS。
#
# 口径(与 STME/SSTNet/Tridos/CoMoE/SCTransNet/DFAR 脚本一致):
#   - 只对模型前向 net(x) 计时, 不含 preprocess / NMS / postprocess
#     (即只测 Ultralytics 里那条 "x.x ms inference", 不含 0.7ms 前处理和 0.2ms 后处理);
#   - dummy 单帧输入 (B, 3, H, W), 放在 GPU 上反复前向, 最纯净;
#   - 前后 cuda.synchronize(); 前 warmup 次不计时;
#   - 输出平均 FPS、平均/中位数延迟、标准差。
#
# 依赖:  pip install ultralytics
#
# 用法:
#   # 用你自己训练好的权重(推荐, 与对比实验同一模型)
#   CUDA_VISIBLE_DEVICES=0 python fps_test.py --weights runs/detect/train/weights/best.pt --imgsz 640
#   # 官方预训练(会自动下载)
#   CUDA_VISIBLE_DEVICES=0 python fps_test.py --weights yolov8n.pt
#   # FP16 / 改尺寸与其它方法对齐
#   CUDA_VISIBLE_DEVICES=0 python fps_test.py --weights best.pt --half --imgsz 512
#--------------------------------------------------------------#
import argparse
import time

import numpy as np
import torch


def build_model(weights, device, half, fuse=True):
    from ultralytics import YOLO
    yolo = YOLO(weights)
    net = yolo.model.to(device).eval()
    if fuse:
        try:
            net.fuse()           # Conv+BN 融合, 部署时的真实推理状态
        except Exception:
            pass
    if half:
        net = net.half()
    return yolo, net


def make_dummy_input(batch, size, half, device):
    x = torch.randn(batch, 8, size, size, device=device)
    return x.half() if half else x


def report_params_flops(yolo, imgsz):
    # Ultralytics 自带统计, 最省事且准确
    try:
        n_layers, n_params, n_grad, flops = yolo.info(imgsz=imgsz, verbose=False)
        print(f'Params: {n_params/1e6:.3f} M   GFLOPs: {flops:.3f} (@ {imgsz})')
    except Exception as e:
        print(f'[跳过 Params/FLOPs 统计] {e}')


@torch.no_grad()
def benchmark(model, get_input, max_iter, num_warmup, log_interval):
    per_iter_times = []
    pure_inf_time = 0.0
    for i in range(max_iter):
        x = get_input()
        torch.cuda.synchronize()
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= num_warmup:
            pure_inf_time += elapsed
            per_iter_times.append(elapsed)
        if (i + 1) % log_interval == 0:
            cur_fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'[{i + 1:>4}/{max_iter}]  fps: {cur_fps:6.1f} img/s  '
                  f'latency: {1000.0 / cur_fps:6.2f} ms/img', flush=True)
    fps = (max_iter - num_warmup) / pure_inf_time
    times_ms = np.array(per_iter_times) * 1000.0
    return fps, times_ms


def main():
    p = argparse.ArgumentParser(description='Ultralytics YOLO FPS benchmark')
    p.add_argument('--weights', default='/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/my_s_9316/weights/best.pt', help='.pt 权重路径')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--log-interval', type=int, default=100)
    p.add_argument('--half', action='store_true')
    p.add_argument('--no-fuse', action='store_true', help='不做 Conv+BN 融合')
    p.add_argument('--no-info', action='store_true', help='跳过 Params/FLOPs')
    p.add_argument('--rect', type=bool, default=False)
    p.add_argument('--frame_num', type=int, default=7)
    args = p.parse_args()
    # rect = False, frame_num = 5
    assert torch.cuda.is_available(), '需要 CUDA GPU'
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    print(f'=== YOLO ({args.weights}) | 精度: {"FP16" if args.half else "FP32"} | '
          f'输入: {args.imgsz} | batch: {args.batch} ===')

    yolo, net = build_model(args.weights, device, args.half, fuse=not args.no_fuse)

    if not args.no_info:
        report_params_flops(yolo, args.imgsz)

    fixed = make_dummy_input(args.batch, args.imgsz, args.half, device)
    get_input = lambda: fixed

    warmup = min(args.warmup, args.iters // 4)
    fps, times_ms = benchmark(net, get_input, args.iters, warmup, args.log_interval)

    print('\n================ 结果 ================')
    print(f'平均 FPS        : {fps:.1f} img/s')
    print(f'平均延迟(mean)  : {times_ms.mean():.3f} ms/img')
    print(f'中位数延迟(p50) : {np.median(times_ms):.3f} ms/img')
    print(f'标准差(std)     : {times_ms.std():.3f} ms')
    print(f'统计样本数      : {len(times_ms)} (已排除 {warmup} 次预热)')
    print('=====================================')


if __name__ == '__main__':
    main()