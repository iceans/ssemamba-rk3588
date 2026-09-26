# import sys
# try:
#     from ultralytics.models.yolo import model
#     print('1')
# except Exception as e:
#     print(f"error:{e}")
#     import traceback
#     traceback.print_exc()
#     sys.exit(1)
from ultralytics import YOLO, RTDETR
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Load a model
# model = YOLO("/home/dell/lxs/tsgmamba/ultralytics-main/cfg_new/files/yolov8-spectrans-p3.yaml")
# # model = YOLO("./ultralytics/cfg/models/v8/yolov8n.yaml")
# # model.load('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/my_2timx_1280_947/weights/best.pt')
# model.load('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/IRDST/my_1280_944/weights/best.pt')
# model.load('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/DAUB/my_2timMix_945/weights/best.pt')
# model = RTDETR("./ultralytics/cfg/models/v8/yolov8-rtdetr.yaml")# 从头训练
# model = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/IRDST/my_2timMix_c2spa_9531/weights/best.pt')
model = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/detect/train126/weights/last.pt')
# model = YOLO("./ultralytics/cfg/models/v8/yolov8.yaml") yolomambav8-RTDETR yolov8nmamba               # 加载预训练模型训练
# model = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/detect/train75/weights/best.pt')     # 从断点训练，resume设为True，默认为False
# model = YOLO('/home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/ablation/Ablation/DAUB/frame3/weights/best.pt')
# /home/dell/lxs/tsgmamba/ultralytics-main/runs/compare_result/IRDST/my_2timMix_1280_887/weights/best.pt
# Use the modelDAUB
# model.train(data="./ultralytics/datasets/iruav.yaml", batch=128, lr0=0.04, epochs=100, device=[2,3], patience=20, save_json=True, ,workers=1,resume=False, cube=True, dwa=False, val=True, gray=True, mod=False)
# model.train(data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/STtrans STtrans_l ITSDT  UAVSwarm.yaml,UAVSwarm", batch=16, lr0=0.04, epochs=50, device=[0,1], patience=20, save_json=False, resume=False, cube=True, dwa=False, val=True, gray=True, mod=False)
# model.train(data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/DAUB IRDST AntiUAV300 .yaml", batch=16,imgsz=640, lr0=0.01, epochs=50, device=[0,1], patience=20, frame_num=3,save_json=False, resume=False, amp=False, cube=True, val=True, gray=True, mod=False)
model.train(data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/JJB.yaml", batch=4,imgsz=640,  epochs=100, augment = True,device=[0,1], patience=40,
            frame_num=7,save_json=False, resume=True, amp=False, cube=True, val=True, gray=False, mod=False,lr0=0.0001)
model.val(data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/JJB.yaml",imgsz=640,rect=False,frame_num=7,device=[0,1],
          )
# #
#             box=5,  # 默认 7.5。你的 mAP 已经极高，回归任务已经过拟合，大幅降低该权重，释放梯度空间。
#             cls=1.5,  # 默认 0.5。强行提升 6 倍分类权重，逼迫网络学会区分真目标与高频背景。
            # kobj=5.0,  # 默认 1.0。强化 Objectness 预测，提高对纯背景区域的置信度压制力。
#             label_smoothing=0.1,  # 引入标签平滑，防止网络在面对极端热噪声时产生绝对确信的错误激活。

            # --- 增强配置 ---
            # mosaic=0.0,  # 【强制关闭】红外弱小目标极小，Mosaic 拼接会产生大量人造的高频拼接边缘，导致频域模块学习崩溃。
            # mixup=0.0,
#         mosaic=0.5,
#         mixup=0.0,              # 关闭 Mixup AntiUAV300 DAUB UAVSwarm ITSDT
#         copy_paste=0.5,         # 开启 Copy-PasteLearnableFrequencyModulator
#         degrees=0.0,            # 小目标对旋转敏感，除非是航拍俯视，否则慎用旋转
#         优化器
#         optimizer='AdamW',IRDST
#         lr0=0.0001,#lr0=0.01,
            # freeze = 21,
        # momentum=0.937,weight_decay=0.0005,

        # lr0=0.01
        # cos_lr=True,            # 余弦退火
        # close_mosaic=20,
        # box=10,
        # cls=0.3,
        # dfl=1.0
        #     )
# model.val(data="/home/dell/lxs/tsgmamba/ultralytics-main/ultralytics/cfg/datasets/DAUB.yaml",imgsz=640,rect=False,frame_num=7,device=[1],
#         # conf=0.5,
#         # iou=0.5,
#           )
#STtrans STtrans_l ITSDT DAUB  UAVSwarm AntiUAV300 IRDST
# model.train(data="./ultralytics/datasets/uavswarm.yaml", batch=128, lr0=0.02, epochs=300, device=[0,1], patience=20, save_json=True, resume=False, cube=True, dwa=False, val=True, gray=True, mod=False)