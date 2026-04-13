#!/usr/bin/env python3
import os
import glob
import argparse
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class MPASAM2:
    """
    Mutil-Peak Autopointer for SAM2:
    - single extraction of foreground features from reference
    - produce: dense_fg_feats (list of [N,C]), prompt_centers [B,K,C,1,1], mean target_embedding [B,1,C]
    - similarity computed from dense_fg_feats -> multi-peak maps
    - cal_point uses prompt_centers to select K positive points + 1 negative
    """

    def __init__(self, sam2_checkpoint: str, sam2_cfg: str, device: Optional[str] = None,
                 num_prompt_centers: int = 3):
        self.model = build_sam2(sam2_cfg, sam2_checkpoint)
        self.predictor = SAM2ImagePredictor(self.model)
        self.device = device or self.model.device
        self.model.eval()

        # Derived from reference:
        self.dense_fg_feats: Optional[List[torch.Tensor]] = None  # list[B] of [N, C]
        self.prompt_centers: Optional[torch.Tensor] = None        # [B, K, C, 1, 1]
        self.target_embedding: Optional[torch.Tensor] = None     # [B,1,C]

        self.num_prompt_centers = max(1, int(num_prompt_centers))

        # last chosen points for visualization
        self.last_points: Optional[torch.Tensor] = None
        self.last_labels: Optional[torch.Tensor] = None

    def set_reference(self, ref_image: np.ndarray, ref_mask: np.ndarray) -> None:
        """
        Extract foreground features once, then derive:
          - dense_fg_feats: list of [N, C] (normalized)
          - prompt_centers: [B, K, C, 1, 1]
          - target_embedding: [B,1,C] (mean of fg_feats)
        """
        self.predictor.set_image(ref_image)
        ref_features = self.predictor._features
        mask = torch.as_tensor(ref_mask, dtype=torch.float32, device=self.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        proc_mask = self.predictor._transforms.transform_masks(mask)

        feats = ref_features["image_embed"]
        if feats.dim() == 3:
            feats = feats.unsqueeze(0)  # [B, C, Hf, Wf]
        B, C, Hf, Wf = feats.shape

        mask_low = F.interpolate(proc_mask.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        mask_low = (mask_low > 0).squeeze(1)  # [B, Hf, Wf]

        feats_perm = feats.permute(0, 2, 3, 1).detach()  # [B, Hf, Wf, C]
        dense_list = []
        means = []
        for b in range(B):
            fg = feats_perm[b][mask_low[b]]  # [N, C] (may be 0)
            if fg.numel() == 0:
                fg = feats_perm[b].reshape(-1, C)
            fg = F.normalize(fg, p=2, dim=1).to(self.device)  # normalize vectors
            dense_list.append(fg)
            means.append(fg.mean(0, keepdim=True))  # [1, C]

        self.dense_fg_feats = dense_list
        self.target_embedding = torch.cat(means, dim=0).unsqueeze(1).to(self.device)  # [B,1,C]

        self.prompt_centers = self._compute_prompt_centers(self.dense_fg_feats, self.num_prompt_centers)

    def _compute_prompt_centers(self, dense_list: List[torch.Tensor], target_k: int) -> torch.Tensor:
        """
        Build [B, K, C, 1, 1] tensor of centers.
        If a batch item has fewer points than K, fallback to repeating the mean.
        """
        centers_per_batch = []
        for fg in dense_list:
            N = fg.shape[0]
            if N == 0:
                # fallback: zero center
                center = fg.new_zeros((1, fg.shape[1]))
                center = F.normalize(center, p=2, dim=1)
                centers = center
            elif N < target_k:
                mean = fg.mean(0, keepdim=True)
                centers = mean.repeat(target_k, 1)
            else:
                k = max(1, min(target_k, N))
                km = KMeans(n_clusters=k, random_state=0).fit(fg.cpu().numpy())
                centers = torch.tensor(km.cluster_centers_, dtype=fg.dtype, device=fg.device)
                if k < target_k:
                    pad = centers[0:1].repeat(target_k - k, 1)
                    centers = torch.cat([centers, pad], dim=0)
            centers = F.normalize(centers, p=2, dim=1)  # [K, C]
            centers = centers.unsqueeze(-1).unsqueeze(-1)  # [K, C, 1, 1]
            centers_per_batch.append(centers)
        # stack to [B, K, C, 1, 1]
        prompt_tensor = torch.stack(centers_per_batch, dim=0)
        return prompt_tensor
    
    def predict(self, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dense_fg_feats is None or self.prompt_centers is None:
            raise RuntimeError("Reference not set. Call set_reference first.")

        centers_kmeans = self.prompt_centers

        self.predictor.set_image(test_image)

        cached_features = self.predictor._features
        orig_hw = self.predictor._orig_hw[0]  # (H, W)
        test_feat = cached_features["image_embed"]  # [B, C, Hf, Wf]
        B, C, Hf, Wf = test_feat.shape

        test_norm = F.normalize(test_feat, p=2, dim=1)  # [B, C, Hf, Wf]
        test_flat = test_norm.view(B, C, Hf * Wf)       # [B, C, Hf*Wf]

        sim_maps = []
        for b in range(B):
            ref = self.dense_fg_feats[b]                 # [N, C]
            # [N, C] @ [C, Hf*Wf] -> [N, Hf*Wf]
            sim_dense = torch.matmul(ref, test_flat[b])  # [N, Hf*Wf]
            sim_agg = sim_dense.mean(0)                  # [Hf*Wf]
            sim_agg = sim_agg.view(1, 1, Hf, Wf)         # [1,1,Hf,Wf]
            sim_maps.append(sim_agg)
        sim = torch.cat(sim_maps, dim=0)  # [B,1,Hf,Wf]

        sim_up = self.predictor._transforms.postprocess_masks(sim, orig_hw=orig_hw)  # [B,1,H,W]
        sim_orig = sim_up.squeeze(1)  # [B, H, W]

        attn_sim_list = []
        for b in range(B):
            sim_b = sim_orig[b]
            sim_std = torch.std(sim_b)
            if sim_std == 0:
                sim_std = 1.0
            sim_b = (sim_b - sim_b.mean()) / sim_std
            sim_b_64 = F.interpolate(sim_b.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
            attn_sim_b = sim_b_64.sigmoid_().unsqueeze(0).flatten(3)
            attn_sim_list.append(attn_sim_b)
        attn_sim = torch.cat(attn_sim_list, dim=0)


        #   KMeans-only prompt
        auto_coords_k, auto_labels_k = self.cal_point(cached_features, centers_kmeans, orig_hw)

        masks_k, scores_k, logits_k = self.predictor.predict(
            point_coords=auto_coords_k,
            point_labels=auto_labels_k,
            multimask_output=True,
            attn_sim=attn_sim,
            target_embedding=self.target_embedding
        )
        best_idx_k = int(np.argmax(scores_k))
        best_score_k = scores_k[best_idx_k]

        self.last_points = auto_coords_k.clone()
        self.last_labels = auto_labels_k.clone()
        masks, scores, logits = masks_k, scores_k, logits_k
        best_idx = int(np.argmax(scores))
        best_logits = logits[best_idx][None, ...]

        masks_ref1, scores_ref1, logits_ref1 = self.predictor.predict(
            point_coords=self.last_points,
            point_labels=self.last_labels,
            mask_input=best_logits,
            multimask_output=True,
        )

        return masks_ref1, scores_ref1, logits_ref1

    def cal_point(self, test_features: Dict[str, torch.Tensor],
                  prompt_centers: torch.Tensor,
                  original_image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = test_features["image_embed"].device
        B, C, Hf, Wf = test_features["image_embed"].shape
        orig_h, orig_w = original_image_size

        test_feat = F.normalize(test_features["image_embed"], p=2, dim=1)  # [B, C, Hf, Wf]
        prompt = prompt_centers.to(device)  # [B, K, C, 1, 1]
        K = prompt.shape[1]

        sim_map_list = []
        for k in range(K):
            p = prompt[:, k]  # [B, C, 1, 1]
            sim = F.cosine_similarity(test_feat, p, dim=1)  # [B, Hf, Wf]
            sim_map_list.append(sim.unsqueeze(1))
        sim_map = torch.cat(sim_map_list, dim=1)  # [B, K, Hf, Wf]

        sim_agg = sim_map.mean(dim=1)  # [B, Hf, Wf]

        sim_up_multi = F.interpolate(sim_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False)  # [B, K, H, W]
        sim_up_agg = F.interpolate(sim_agg.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)  # [B, H, W]

        coords_batch = []
        labels_batch = []
        for b in range(B):
            h, w = sim_up_agg[b].shape
            coords = []
            labels = []
            for k in range(sim_up_multi.shape[1]):
                sim_k = sim_up_multi[b, k]
                flat_k = sim_k.flatten()
                pos_idx = torch.argmax(flat_k)
                pos_y, pos_x = divmod(int(pos_idx.item()), w)
                pos_x = float(max(0, min(w - 1, pos_x)))
                pos_y = float(max(0, min(h - 1, pos_y)))
                coords.append([pos_x, pos_y])
                labels.append(1)
            # negative: global min on aggregated map
            flat_g = sim_up_agg[b].flatten()
            neg_idx = torch.argmin(flat_g)
            ny = int(neg_idx // w)
            nx = int(neg_idx % w)
            neg_x = float(max(0, min(w - 1, nx)))
            neg_y = float(max(0, min(h - 1, ny)))
            coords.append([neg_x, neg_y])
            labels.append(0)

            coords_batch.append(torch.tensor(coords, device=device, dtype=torch.float32).unsqueeze(0))  # [1, K+1, 2]
            labels_batch.append(torch.tensor(labels, device=device, dtype=torch.long).unsqueeze(0))     # [1, K+1]

        auto_point_coords = torch.cat(coords_batch, dim=0)  # [B, K+1, 2]
        auto_point_labels = torch.cat(labels_batch, dim=0)  # [B, K+1]

        return auto_point_coords, auto_point_labels

    def save_vis(self,
                    image: np.ndarray,
                    mask: np.ndarray,
                    output_path: str,
                    pos_icon_path: str = "icon/click3.png",
                    neg_icon_path: str = "icon/click4.png"):
        if self.last_points is None or self.last_labels is None:
            return

        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        H, W = image.shape[:2]
        alpha = 0.5
        overlay = image.copy().astype(float)
        overlay[mask > 0] = (
            alpha * np.array([0, 255, 0]) + (1 - alpha) * overlay[mask > 0]
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(overlay.astype(np.uint8))
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
        ax.axis("off")

        pos_icon = plt.imread(pos_icon_path) if os.path.exists(pos_icon_path) else None
        neg_icon = plt.imread(neg_icon_path) if os.path.exists(neg_icon_path) else None

        points = self.last_points[0]
        labels = self.last_labels[0]

        for i in range(points.shape[0]):
            x = float(points[i, 0].item())
            y = float(points[i, 1].item())
            label = labels[i].item()
            icon_img = pos_icon if label == 1 else neg_icon
            if icon_img is None:
                continue
            # dynamic zoom to keep icon proportional across varying image sizes
            base_ratio = 0.01  # icon covers ~5% of image height
            icon_scale = base_ratio * (H / icon_img.shape[0])
            icon_box = OffsetImage(icon_img, zoom=icon_scale)
            ab = AnnotationBbox(icon_box, (x, y), frameon=False)
            ax.add_artist(ab)

        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def inference(ptr: MPASAM2,
              test_image: np.ndarray,
              vis_output_path: Optional[str] = None,
              mask_output_path: Optional[str] = None):
    masks, scores, logits = ptr.predict(test_image)
    best_idx = int(np.argmax(scores))
    final_mask = masks[best_idx]
    if vis_output_path:
        ptr.save_vis(test_image, final_mask, vis_output_path)
    if mask_output_path:
        masks_uint8 = (final_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_output_path, masks_uint8)
    return masks, scores


def main():
    parser = argparse.ArgumentParser(description="MPA-SAM2 - Multi-Prompt Auto Segmentation ")
    parser.add_argument("--sam2_checkpoint", type=str, required=True)
    parser.add_argument("--model_cfg", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--class_name", type=str, default="cat_head")
    parser.add_argument("--ref_idx", type=str, default="00")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_prompt_centers", type=int, default=2)
    args = parser.parse_args()

    images_root = os.path.join(args.data_root, "Images")
    if args.class_name:
        classes = [args.class_name]
    else:
        classes = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])

    os.makedirs(args.output_dir, exist_ok=True)

    for class_name in classes:
        img_dir = os.path.join(images_root, class_name)
        mask_dir = os.path.join(args.data_root, "Annotations", class_name)
        ref_img_path = os.path.join(img_dir, f"{args.ref_idx}.jpg")
        ref_mask_path = os.path.join(mask_dir, f"{args.ref_idx}.png")
        if not os.path.exists(ref_img_path) or not os.path.exists(ref_mask_path):
            print(f"Skipping {class_name}: reference missing")
            continue

        ptr = MPASAM2(args.sam2_checkpoint, args.model_cfg,
                             num_prompt_centers=args.num_prompt_centers)

        ref_img = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)

        ptr.set_reference(ref_img, ref_mask)

        out_class_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        test_images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        for test_path in tqdm(test_images, desc=f"Infer {class_name}"):
            img_name = os.path.basename(test_path)
            if img_name == f"{args.ref_idx}.jpg":
                continue
            test_img = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
            base = os.path.splitext(img_name)[0]
            vis_path = os.path.join(out_class_dir, base + "_vis.jpg")
            mask_path = os.path.join(out_class_dir, base + ".png")
            inference(ptr, test_img, vis_output_path=vis_path, mask_output_path=mask_path)

    print("Done.")


if __name__ == "__main__":
    main()
