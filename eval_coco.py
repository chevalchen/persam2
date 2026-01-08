import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data.dataset import FSSDataset

def calculate_iou(pred_mask, gt_mask):
    """计算单个样本的交并比"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0  # 如果全为背景且预测正确
    return intersection / union

def evaluate():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation Script for MPASAM2")
    parser.add_argument("--predict_dir", type=str, required=True, help="推理结果保存的路径")
    parser.add_argument("--benchmark", type=str, default="paco_part", choices=["paco_part", "pascal_part", "fss", "coco", "lvis", "pascal"])
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()

    # 1. 加载数据集以获取真值
    FSSDataset.initialize(img_size=args.img_size, datapath=args.data_root, use_original_imgsize=True)
    dataloader = FSSDataset.build_dataloader(args.benchmark, 1, 4, args.fold, args.split, shot=1)

    class_ious = {} # 存储每个类别的 IoU 列表
    fg_ious = []    # 存储每个样本的前景 IoU
    bg_ious = []    # 存储每个样本的背景 IoU

    print(f"Evaluating {args.benchmark} results from {args.predict_dir}...")

    for i, batch in enumerate(tqdm(dataloader)):
        query_name = batch['query_name'][0]
        class_id = batch['class_id'].item()
        
        # 处理文件名以匹配推理保存的格式
        safe_name = query_name.replace('/', '_').replace('\\', '_').replace('.jpg', '').replace('.png', '')
        pred_path = os.path.join(args.predict_dir, f"{safe_name}.png")

        if not os.path.exists(pred_path):
            continue

        # 读取预测掩码并二值化
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = (pred_mask > 128).astype(np.uint8)

        # 获取真值掩码 (Ground Truth)
        gt_mask = batch['query_mask'][0].numpy().astype(np.uint8)
        
        # 对齐尺寸 (如果推理结果和GT尺寸不一致)
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- 计算 mIoU 相关数据 ---
        iou = calculate_iou(pred_mask, gt_mask)
        if class_id not in class_ious:
            class_ious[class_id] = []
        class_ious[class_id].append(iou)

        # --- 计算 FB-IoU 相关数据 ---
        # 前景 IoU
        fg_ious.append(iou)
        # 背景 IoU
        bg_iou = calculate_iou(1 - pred_mask, 1 - gt_mask)
        bg_ious.append(bg_iou)

    # 2. 计算最终指标
    # mIoU: 先计算每个类的平均 IoU，再对所有类取平均
    mean_class_ious = [np.mean(ious) for ious in class_ious.values()]
    miou = np.mean(mean_class_ious) if mean_class_ious else 0

    # FB-IoU: 所有样本的前景 IoU 和背景 IoU 的均值
    fbiou = (np.mean(fg_ious) + np.mean(bg_ious)) / 2 if fg_ious else 0

    print("\n" + "="*30)
    print(f"Results for Fold {args.fold}:")
    print(f"mIoU:  {miou:.4f}")
    print(f"FB-IoU: {fbiou:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()