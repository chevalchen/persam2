#!/usr/bin/env python3
import os
import glob
import argparse
import pickle
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import Dataset
from PIL import Image

try:
    from pycocotools import mask as mask_util
    def polygons_to_bitmask(polygons, height, width):
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        return mask_util.decode(rle)
except ImportError:
    print("Warning: pycocotools is not installed. PACO/LVIS mask handling may fail.")
    def polygons_to_bitmask(polygons, height, width):
        raise NotImplementedError("pycocotools is required for PACO/LVIS mask handling.")
    class DummyMaskUtil:
        def decode(self, segm):
            raise NotImplementedError("pycocotools is required for PACO/LVIS mask handling.")
    mask_util = DummyMaskUtil()

warnings.filterwarnings("ignore")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class MPASAM2:
    """
    Mutil-Peak Autopointer for SAM2:
    ... (MPASAM2 class definition as provided by the user)
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
        mask_low = (mask_low > 0.5).squeeze(1)  # [B, Hf, Wf]

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
                # sklearn requires numpy float64 or float32. Use float32
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
        # ... (predict method as provided by the user)
        if self.dense_fg_feats is None or self.prompt_centers is None:
            raise RuntimeError("Reference not set. Call set_reference first.")

        self.predictor.set_image(test_image)
        cached_features = self.predictor._features
        orig_hw = self.predictor._orig_hw[0]  # (H, W)
        test_feat = cached_features["image_embed"]  # [B, C, Hf, Wf]
        B, C, Hf, Wf = test_feat.shape

        # normalize test features
        test_norm = F.normalize(test_feat, p=2, dim=1)  # [B, C, Hf, Wf]
        test_flat = test_norm.view(B, C, Hf * Wf)       # [B, C, Hf*Wf]

        # similarity aggregated from dense reference pixels (mean over ref pixels)
        sim_maps = []
        for b in range(B):
            ref = self.dense_fg_feats[b]                 # [N, C]
            # [N, C] @ [C, Hf*Wf] -> [N, Hf*Wf]
            sim_dense = torch.matmul(ref, test_flat[b])  # [N, Hf*Wf]
            sim_agg = sim_dense.mean(0)                  # [Hf*Wf]
            sim_agg = sim_agg.view(1, 1, Hf, Wf)         # [1,1,Hf,Wf]
            sim_maps.append(sim_agg)
        sim = torch.cat(sim_maps, dim=0)  # [B,1,Hf,Wf]

        # postprocess to original size
        sim_up = self.predictor._transforms.postprocess_masks(sim, orig_hw=orig_hw)  # [B,1,H,W]
        sim_orig = sim_up.squeeze(1)  # [B, H, W]

        # attn_sim for predictor guidance (64x64 flattened)
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

        # choose auto points from prompt_centers
        auto_point_coords, auto_point_labels = self.cal_point(cached_features, self.prompt_centers, orig_hw)

        self.last_points = auto_point_coords.clone()
        self.last_labels = auto_point_labels.clone()

        masks, scores, logits = self.predictor.predict(
            point_coords=auto_point_coords,
            point_labels=auto_point_labels,
            multimask_output=True,
            attn_sim=attn_sim,
            target_embedding=self.target_embedding
        )
        best_idx = int(np.argmax(scores))
        best_logits = logits[best_idx][None, ...]

        masks_ref1, scores_ref1, logits_ref1 = self.predictor.predict(
            point_coords=auto_point_coords,
            point_labels=auto_point_labels,
            mask_input=best_logits,
            multimask_output=True,
        )
        return masks_ref1, scores_ref1, logits_ref1

    def cal_point(self, test_features: Dict[str, torch.Tensor],
                  prompt_centers: torch.Tensor,
                  original_image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # ... (cal_point method as provided by the user)
        device = test_features["image_embed"].device
        B, C, Hf, Wf = test_features["image_embed"].shape
        orig_h, orig_w = original_image_size

        test_feat = F.normalize(test_features["image_embed"], p=2, dim=1)  # [B, C, Hf, Wf]
        # compute per-cluster similarity maps at feature resolution
        prompt = prompt_centers.to(device)  # [B, K, C, 1, 1]
        K = prompt.shape[1]

        sim_map_list = []
        for k in range(K):
            p = prompt[:, k]  # [B, C, 1, 1]
            sim = F.cosine_similarity(test_feat, p, dim=1)  # [B, Hf, Wf]
            sim_map_list.append(sim.unsqueeze(1))
        sim_map = torch.cat(sim_map_list, dim=1)  # [B, K, Hf, Wf]

        # aggregated map for negative selection
        sim_agg = sim_map.mean(dim=1)  # [B, Hf, Wf]

        # upsample to image resolution
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



def iou_compute(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """计算单个样本的 IoU"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, split, shot, **kwargs): # 移除 transform, use_original_imgsize
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        
        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return 1000 # 固定评估数量

    def __getitem__(self, idx):
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        # 返回 PIL Image 和 NumPy Mask (bool)
        query_mask = query_mask.numpy().astype(bool)
        support_masks = [m.numpy().astype(bool) for m in support_masks]

        batch = {'query_img': query_img,
                 'query_mask': query_mask, # np.ndarray bool
                 'support_imgs': support_imgs, # List[PIL Image]
                 'support_masks': support_masks, # List[np.ndarray bool]
                 'org_query_imsize': org_qry_imsize,
                 'class_id': class_sample}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        return class_ids

    def build_img_metadata_classwise(self):
        # assumed datapath/COCO2014/splits/val/fold{self.fold}.pkl exists
        path = os.path.join(self.base_path, 'splits', self.split, f'fold{self.fold}.pkl')
        with open(path, 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


class DatasetPACOPart(Dataset):
    def __init__(self, datapath, fold, split, shot, **kwargs): 
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 448
        self.benchmark = 'paco_part'
        self.shot = shot
        self.img_path = os.path.join(datapath, 'PACO-Part', 'coco')
        self.anno_path = os.path.join(datapath, 'PACO-Part', 'paco')
        self.box_crop = kwargs.get('box_crop', False) 

        self.class_ids_ori, self.cid2img, self.img2anno = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return 2500 

    def __getitem__(self, idx):
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        # return PIL Image & NumPy Mask (bool)
        query_mask = query_mask.numpy().astype(bool)
        support_masks = [m.numpy().astype(bool) for m in support_masks]

        batch = {'query_img': query_img,
                 'query_mask': query_mask, # np.ndarray bool
                 'support_imgs': support_imgs, # List[PIL Image]
                 'support_masks': support_masks, # List[np.ndarray bool]
                 'org_query_imsize': org_qry_imsize,
                 'class_id': self.class_ids_c[class_sample]}

        return batch

    def build_img_metadata_classwise(self):
        with open(os.path.join(self.anno_path, 'paco_part_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'paco_part_val.pkl'), 'rb') as f:
            test_anno = pickle.load(f)

        new_cid2img = {}
        for cid_id in test_anno['cid2img']:
            id_list = []
            if cid_id not in new_cid2img: new_cid2img[cid_id] = []
            for img in test_anno['cid2img'][cid_id]:
                img_id = list(img.keys())[0]
                if img_id not in id_list:
                    id_list.append(img_id)
                    new_cid2img[cid_id].append(img)
        test_anno['cid2img'] = new_cid2img

        train_cat_ids = list(train_anno['cid2img'].keys())
        test_cat_ids = [i for i in list(test_anno['cid2img'].keys()) if len(test_anno['cid2img'][i]) > self.shot]

        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [train_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_trn)]
        class_ids_val = [x for x in class_ids_val if x in test_cat_ids]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        img_metadata_classwise = train_anno if self.split == 'trn' else test_anno
        cid2img = img_metadata_classwise['cid2img']
        img2anno = img_metadata_classwise['img2anno']

        return class_ids, cid2img, img2anno

    def build_img_metadata(self):
        img_metadata = []
        for k in self.cid2img.keys():
            img_metadata += self.cid2img[k]
        return img_metadata

    def get_mask(self, segm, image_size):
        if isinstance(segm, list):
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2
            mask = segm
        else:
            raise NotImplementedError
        return torch.tensor(mask)

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]
        query = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
        query_id, query_name = list(query.keys())[0], list(query.values())[0]
        query_name = '/'.join( query_name.split('/')[-2:])
        query_img = Image.open(os.path.join(self.img_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = self.img2anno[query_id]

        query_obj_dict = {}
        for anno in query_annos:
            if anno['category_id'] == class_sample:
                obj_id = anno['obj_ann_id']
                if obj_id not in query_obj_dict:
                    query_obj_dict[obj_id] = {'obj_bbox': [], 'segms': []}
                query_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                query_obj_dict[obj_id]['segms'].append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...])

        sel_query_id = np.random.choice(list(query_obj_dict.keys()), 1, replace=False)[0]
        query_obj_bbox = query_obj_dict[sel_query_id]['obj_bbox'][0]
        query_part_masks = query_obj_dict[sel_query_id]['segms']
        query_mask = torch.cat(query_part_masks, dim=0)
        query_mask = query_mask.sum(0) > 0 

        support_names = []
        support_pre_masks = []
        support_boxes = []
        while True:
            support = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
            support_id, support_name = list(support.keys())[0], list(support.values())[0]
            support_name = '/'.join(support_name.split('/')[-2:])
            if query_name != support_name:
                support_names.append(support_name)
                support_annos = self.img2anno[support_id]

                support_obj_dict = {}
                for anno in support_annos:
                    if anno['category_id'] == class_sample:
                        obj_id = anno['obj_ann_id']
                        if obj_id not in support_obj_dict:
                            support_obj_dict[obj_id] = {'obj_bbox': [], 'segms': []}
                        support_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                        support_obj_dict[obj_id]['segms'].append(anno['segmentation'])

                sel_support_id = np.random.choice(list(support_obj_dict.keys()), 1, replace=False)[0]
                support_obj_bbox = support_obj_dict[sel_support_id]['obj_bbox'][0]
                support_part_masks = support_obj_dict[sel_support_id]['segms']

                support_boxes.append(support_obj_bbox)
                support_pre_masks.append(support_part_masks)

            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.img_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...])
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0
            support_masks.append(support_mask)

        if self.box_crop:
            query_img_np = np.asarray(query_img)
            x, y, w, h = [int(b) for b in query_obj_bbox]
            query_img_np = query_img_np[y:y+h, x:x+w]
            query_img = Image.fromarray(np.uint8(query_img_np))
            org_qry_imsize = query_img.size
            query_mask = query_mask[y:y+h, x:x+w]

            new_support_imgs = []
            new_support_masks = []
            for sup_img, sup_mask, sup_box in zip(support_imgs, support_masks, support_boxes):
                sup_img_np = np.asarray(sup_img)
                x, y, w, h = [int(b) for b in sup_box]
                sup_img_np = sup_img_np[y:y+h, x:x+w]
                sup_img = Image.fromarray(np.uint8(sup_img_np))
                new_support_imgs.append(sup_img)
                new_support_masks.append(sup_mask[y:y+h, x:x+w])

            support_imgs = new_support_imgs
            support_masks = new_support_masks

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


class DatasetLVIS(Dataset):
    def __init__(self, datapath, fold, split, shot, **kwargs): # 移除 transform, use_original_imgsize
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 10
        self.benchmark = 'lvis'
        self.shot = shot
        self.anno_path = os.path.join(datapath, "LVIS")
        self.base_path = os.path.join(datapath, "LVIS", 'coco')
        
        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return 2300 # 固定评估数量

    def __getitem__(self, idx):
        # 这里使用 idx 来随机选择一个类别进行采样
        idx %= len(self.class_ids)

        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(idx)

        # 返回 PIL Image 和 NumPy Mask (bool)
        query_mask = query_mask.numpy().astype(bool)
        support_masks = [m.numpy().astype(bool) for m in support_masks]

        batch = {'query_img': query_img,
                 'query_mask': query_mask, # np.ndarray bool
                 'support_imgs': support_imgs, # List[PIL Image]
                 'support_masks': support_masks, # List[np.ndarray bool]
                 'org_query_imsize': org_qry_imsize,
                 'class_id': self.class_ids_c[class_sample]}

        return batch

    def build_img_metadata_classwise(self):
        # 假设 lvis_train.pkl 和 lvis_val.pkl 存在
        with open(os.path.join(self.anno_path, 'lvis_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        train_cat_ids = list(train_anno.keys())
        val_cat_ids = [i for i in list(val_anno.keys()) if len(val_anno[i]) > self.shot]

        trn_nclass = len(train_cat_ids)
        val_nclass = len(val_cat_ids)

        nclass_val_spilt = val_nclass // self.nfolds

        class_ids_val = [val_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_val_spilt)]
        class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]

        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        nclass = trn_nclass if self.split == 'trn' else val_nclass
        img_metadata_classwise = train_anno if self.split == 'trn' else val_anno

        return nclass, class_ids, img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size):
        if isinstance(segm, list):
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)

    def load_frame(self, idx):
        # 保持 load_frame 逻辑不变
        class_sample = self.class_ids_ori[idx]
        query_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
        query_info = self.img_metadata_classwise[class_sample][query_name]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = query_info['annotations']
        segms = []

        for anno in query_annos:
            segms.append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...].float())
        query_mask = torch.cat(segms, dim=0)
        query_mask = query_mask.sum(0) > 0

        support_names = []
        support_pre_masks = []
        while True:
            support_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
            if query_name != support_name:
                support_names.append(support_name)
                support_info = self.img_metadata_classwise[class_sample][support_name]
                support_annos = support_info['annotations']

                support_segms = []
                for anno in support_annos:
                    support_segms.append(anno['segmentation'])
                support_pre_masks.append(support_segms)

            if len(support_names) == self.shot: break


        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.base_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...].float())
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0

            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


def main_validation():
    parser = argparse.ArgumentParser(description="MPASAM2 Few-Shot Segmentation Validation")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to SAM2 model checkpoint.")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to SAM2 model config file.")
    parser.add_argument("--datapath", type=str, default="./data", help="Root directory of the few-shot datasets (COCO2014, LVIS, PACO-Part).")
    parser.add_argument("--benchmark", type=str, choices=['coco', 'paco_part', 'lvis'], required=True, help="Dataset to evaluate.")
    parser.add_argument("--fold", type=int, default=0, help="Dataset fold for evaluation.")
    parser.add_argument("--nshot", type=int, default=1, choices=[1], help="Number of support shots (forced to 1 for this implementation).")
    parser.add_argument("--num_prompt_centers", type=int, default=3, help="Number of prompt centers for MPASAM2.")
    args = parser.parse_args()
    
    # 强制 1-shot
    args.nshot = 1

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_map = {
        'coco': DatasetCOCO,
        'paco_part': DatasetPACOPart,
        'lvis': DatasetLVIS,
    }

    if args.benchmark not in dataset_map:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    # 1. 初始化数据集
    print(f"Initializing {args.benchmark} (Fold {args.fold}) dataset...")
    DatasetClass = dataset_map[args.benchmark]
    dataset = DatasetClass(
        datapath=args.datapath,
        fold=args.fold,
        split='val',
        shot=args.nshot,
        # PACO-Part 默认开启 box_crop
        box_crop=True if args.benchmark == 'paco_part' else False 
    )

    # 2. 初始化模型
    ptr = MPASAM2(args.sam2_checkpoint, args.model_cfg, device=DEVICE,
                         num_prompt_centers=args.num_prompt_centers)
    
    # 3. 运行验证循环
    iou_scores = []
    
    for idx in tqdm(range(len(dataset)), desc=f"Validating {args.benchmark}"):
        batch = dataset[idx]

        query_img_pil = batch['query_img']
        query_mask_gt = batch['query_mask'] # np.ndarray bool

        # 确保 support set 只有一个
        ref_img_pil = batch['support_imgs'][0]
        ref_mask_gt = batch['support_masks'][0] # np.ndarray bool

        # 转换为 np.ndarray for MPASAM2 inputs
        ref_img = np.array(ref_img_pil, dtype=np.uint8)
        ref_mask = ref_mask_gt.astype(np.uint8) * 255 
        query_img = np.array(query_img_pil, dtype=np.uint8)

        # 3.1. 设置参考 (Set Reference)
        # MPASAM2 内部会进行预处理和特征提取
        ptr.set_reference(ref_img, ref_mask)

        # 3.2. 预测查询 (Predict Query)
        # 预测返回的 masks[0] 是最佳预测，大小与 query_img 相同
        try:
            masks, scores, logits = ptr.predict(query_img)
        except Exception as e:
            print(f"Error during prediction for episode {idx}: {e}")
            continue

        best_idx = int(np.argmax(scores))
        pred_mask_bool = masks[best_idx].astype(bool)
        
        # 3.3. 计算 IoU
        # 预测掩码 (pred_mask_bool) 和真实掩码 (query_mask_gt) 此时应具有相同的空间分辨率
        # 原始的 Dataset 实现中，Query Mask 是在原始分辨率或插值到统一尺寸后与 Query Image 对应的。
        # 由于我们移除了转换，这里的 pred_mask_bool 应该与 query_mask_gt 形状相同 (原始图像大小)。
        
        # 确保形状匹配 (如果预测步骤未正确处理分辨率变化，可能需要额外resize)
        if pred_mask_bool.shape != query_mask_gt.shape:
             # 如果形状不匹配，将预测结果resize到GT尺寸
            H, W = query_mask_gt.shape
            pred_mask_uint8 = (pred_mask_bool * 255).astype(np.uint8)
            pred_mask_resized = cv2.resize(pred_mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST) > 127
            iou = iou_compute(pred_mask_resized, query_mask_gt)
        else:
            iou = iou_compute(pred_mask_bool, query_mask_gt)

        iou_scores.append(iou)

    # 4. 结果汇报
    mean_iou = np.mean(iou_scores) * 100
    
    print("\n" + "="*50)
    print(f"Few-Shot Segmentation Results on {args.benchmark} (Fold {args.fold}):")
    print(f"Total Episodes: {len(iou_scores)}")
    print(f"Mean IoU (mIoU): {mean_iou:.2f}%")
    print("="*50)


if __name__ == "__main__":
    # 使用新的验证函数
    main_validation()