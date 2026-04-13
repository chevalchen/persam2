#!/usr/bin/env python3
import os
import argparse
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Check if sam2 is available, otherwise this script won't run.
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import the decoupled dataset loader
try:
    from data.dataset import FSSDataset
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
    from data.dataset import FSSDataset

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

        # align mask to feature resolution
        mask_low = F.interpolate(proc_mask.float(), size=(Hf, Wf), mode="bilinear", align_corners=False)
        mask_low = (mask_low > 0).squeeze(1)  # [B, Hf, Wf]

        # one-time extraction of foreground features per batch
        feats_perm = feats.permute(0, 2, 3, 1).detach()  # [B, Hf, Wf, C]
        dense_list = []
        means = []
        for b in range(B):
            fg = feats_perm[b][mask_low[b]]  # [N, C] (may be 0)
            if fg.numel() == 0:
                # fallback to global mean over full feature map
                fg = feats_perm[b].reshape(-1, C)
            fg = F.normalize(fg, p=2, dim=1).to(self.device)  # normalize vectors
            dense_list.append(fg)
            means.append(fg.mean(0, keepdim=True))  # [1, C]

        self.dense_fg_feats = dense_list
        self.target_embedding = torch.cat(means, dim=0).unsqueeze(1).to(self.device)  # [B,1,C]

        # derive prompt centers from dense features
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
                # run KMeans on CPU
                k = max(1, min(target_k, N))

                km = KMeans(n_clusters=k, random_state=0).fit(fg.cpu().numpy())
                centers = torch.tensor(km.cluster_centers_, dtype=fg.dtype, device=fg.device)
                if k < target_k:
                    # pad by repeating first center
                    pad = centers[0:1].repeat(target_k - k, 1)
                    centers = torch.cat([centers, pad], dim=0)
            centers = F.normalize(centers, p=2, dim=1)  # [K, C]
            centers = centers.unsqueeze(-1).unsqueeze(-1)  # [K, C, 1, 1]
            centers_per_batch.append(centers)
        # stack to [B, K, C, 1, 1]
        prompt_tensor = torch.stack(centers_per_batch, dim=0)
        return prompt_tensor
    

    def predict(self, test_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main inference:
         - compute multi-peak similarity from dense_fg_feats
         - auto select points from prompt_centers
         - call predictor.predict with multimask_output
        """
        if self.dense_fg_feats is None or self.prompt_centers is None:
            raise RuntimeError("Reference not set. Call set_reference first.")

        centers_kmeans = self.prompt_centers

        self.predictor.set_image(test_image)

        cached_features = self.predictor._features
        orig_hw = self.predictor._orig_hw[0]  # (H, W)

        topk_points, topk_labels = self.cal_point(cached_features, centers_kmeans, orig_hw)

        masks, scores, logits = self.predictor.predict(
            point_coords=topk_points,
            point_labels=topk_labels,
            multimask_output=True
        )
        self.last_points = topk_points.clone()
        self.last_labels = topk_labels.clone()
        best_idx = int(np.argmax(scores))
        best_logits = logits[best_idx][None, ...]

        masks_ref1, scores_ref1, logits_ref1 = self.predictor.predict(
            point_coords=topk_points,
            point_labels=topk_labels,
            mask_input=best_logits,
            multimask_output=True,
        )

        return masks_ref1, scores_ref1, logits_ref1

    def cal_point(self, test_features: Dict[str, torch.Tensor],
                  prompt_centers: torch.Tensor,
                  original_image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select points:
         - for each cluster center (K) pick argmax location in the test feature map
         - add one negative point by argmin on aggregated sim (or choose far-away)
        Returns:
         - coords: [B, K+1, 2] float tensor (x, y)
         - labels: [B, K+1] long tensor (1 for positives, 0 for negative)
        """
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
        # 点坐标张量和点的属性标签张量
        point_coords = torch.cat(coords_batch, dim=0)  # [B, K+1, 2]
        point_labels = torch.cat(labels_batch, dim=0)  # [B, K+1]

        return point_coords, point_labels

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
            
            if pos_icon is not None and label == 1:
                icon_img = pos_icon
                base_ratio = 0.01
                icon_scale = base_ratio * (H / icon_img.shape[0])
                icon_box = OffsetImage(icon_img, zoom=icon_scale)
                ab = AnnotationBbox(icon_box, (x, y), frameon=False)
                ax.add_artist(ab)
            elif neg_icon is not None and label == 0:
                icon_img = neg_icon
                base_ratio = 0.01
                icon_scale = base_ratio * (H / icon_img.shape[0])
                icon_box = OffsetImage(icon_img, zoom=icon_scale)
                ab = AnnotationBbox(icon_box, (x, y), frameon=False)
                ax.add_artist(ab)
            else:
                color = 'r' if label==1 else 'b'
                ax.plot(x, y, marker='*', c=color, markersize=10)

        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MPASAM2 Inference on FSS Datasets")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to SAM2 config")
    parser.add_argument("--num_prompt_centers", type=int, default=3)
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--benchmark", type=str, default="paco_part", choices=["paco_part", "pascal_part", "fss", "coco", "lvis", "pascal"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--use_original_imgsize", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--vis", action="store_true", help="Save visualizations")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Dataset
    FSSDataset.initialize(img_size=args.img_size, datapath=args.data_root, use_original_imgsize=args.use_original_imgsize)
    dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, 4, args.fold, args.split, args.shot)

    # Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ptr = MPASAM2(args.sam2_checkpoint, args.model_cfg, device=device, num_prompt_centers=args.num_prompt_centers)

    print(f"Starting inference on {args.benchmark} (Fold {args.fold}, Split {args.split})...")

    for i, batch in enumerate(tqdm(dataloader)):
        # Retrieve batch data
        query_img = batch['query_img']      # [B, 3, H, W]
        # query_mask = batch['query_mask']    # [B, H, W] or [B, H, W]
        support_imgs = batch['support_imgs'] # [B, S, 3, H, W]
        support_masks = batch['support_masks'] # [B, S, H, W]
        query_name = batch['query_name']     # list of strings
        
        for b in range(query_img.shape[0]):
            
            q_img = query_img[b].permute(1, 2, 0).numpy()
            q_img = (q_img * 255).astype(np.uint8)

            s_img = support_imgs[b][0].permute(1, 2, 0).numpy()
            s_img = (s_img * 255).astype(np.uint8)
            
            s_mask = support_masks[b][0].numpy()
            if s_mask.max() <= 1.0:
                s_mask = (s_mask * 255).astype(np.uint8)
            else:
                s_mask = s_mask.astype(np.uint8)

            # Inference
            try:
                ptr.set_reference(s_img, s_mask)
                masks, scores, logits = ptr.predict(q_img)
                
                best_idx = int(np.argmax(scores))
                final_mask = masks[best_idx]
                
                q_n = query_name[b]
                safe_name = q_n.replace('/', '_').replace('\\', '_').replace('.jpg', '').replace('.png', '')
                if not safe_name: 
                    safe_name = f"{i}_{b}"
                
                # Save Visualization
                if args.vis:
                    vis_path = os.path.join(args.output_dir, f"{safe_name}_vis.jpg")
                    ptr.save_vis(q_img, final_mask, vis_path)
                
                # Save Mask
                mask_path = os.path.join(args.output_dir, f"{safe_name}.png")
                final_mask_uint8 = (final_mask > 0).astype(np.uint8) * 255
                cv2.imwrite(mask_path, final_mask_uint8)

            except Exception as e:
                print(f"Failed on {query_name[b]}: {e}")
                continue
                
    print("Inference finished.")

if __name__ == "__main__":
    main()