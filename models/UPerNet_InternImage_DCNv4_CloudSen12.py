import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import sys
sys.path.append(os.path.abspath("DCNv4/segmentation"))
from mmseg_custom.models.backbones import FlashInternImage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import tacoreader
from tqdm import tqdm
from mmengine.config import Config
from mmseg.models import build_segmentor

band_combinations = [
    [3, 4, 10],    # Green, Red, Cirrus
    # [3, 8, 11],    # Green, NIR, SWIR1
    # [4, 6, 11],    # Red, RedEdge2, SWIR1
    # [2, 11, 12],   # Blue, SWIR1, SWIR2    
    # [4, 3, 2]     # RGB
]


class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s2_l1c = self.dataset.read(self.indices[idx]).read(0)
        s2_label = self.dataset.read(self.indices[idx]).read(1)
        with rio.open(s2_l1c) as src, rio.open(s2_label) as dst:
            img = src.read(self.selected_bands, window=rio.windows.Window(0, 0, 512, 512)).astype(np.float32)
            label = dst.read(1, window=rio.windows.Window(0, 0, 512, 512)).astype(np.uint8)
        img = img / 3000.0
        img = np.transpose(img, (1, 2, 0))
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(label)

def train_and_evaluate(bands, combo_index, total_combos, result_path):
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    train_idx, test_idx = list(range(8000)), list(range(8000, 10000))
    train_set = CloudSegmentationDataset(taco_path, train_idx, bands)
    test_set = CloudSegmentationDataset(taco_path, test_idx, bands)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

    num_classes = 4

    cfg = Config.fromfile("DCNv4/segmentation/configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py")
    cfg.model.pretrained = None
    cfg.model.backbone.in_channels = 3
    cfg.model.decode_head.num_classes = num_classes
    if "auxiliary_head" in cfg.model:
        cfg.model.auxiliary_head.num_classes = num_classes

    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    if hasattr(cfg.model.backbone, 'norm_cfg'):
        cfg.model.backbone.norm_cfg = cfg.norm_cfg
    if hasattr(cfg.model.decode_head, 'norm_cfg'):
        cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    if 'auxiliary_head' in cfg.model and hasattr(cfg.model.auxiliary_head, 'norm_cfg'):
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    cfg.model.test_cfg = dict(mode='whole')

    model = build_segmentor(cfg.model)
    model.init_weights()
    model = model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(0):
        model.train(); total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"InternImage | Combo {combo_index+1}/{total_combos} | Epoch {epoch+1}/10"):
            imgs = imgs.to("cuda")
            labels = labels.long().unsqueeze(1).to("cuda")
            H, W = imgs.shape[2], imgs.shape[3]

            img_metas = [
                dict(
                    ori_shape=(H, W),
                    img_shape=(H, W),
                    pad_shape=(H, W),
                    batch_input_shape=(H, W),
                    scale_factor=1.0,
                    flip=False
                )
                for _ in range(imgs.size(0))
            ]

            losses = model(imgs, img_metas=img_metas, gt_semantic_seg=labels)
            loss = sum(v for v in losses.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[InternImage] Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")

    model.eval(); total = 0
    total_iou = [0]*4
    total_f1 = [0]*4

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="InternImage | Evaluation"):
            imgs = imgs.to("cuda")
            labels = labels.to("cuda")
            batch_size, _, H, W = imgs.shape

            img_metas_batch = [
                [dict(
                    ori_shape=(H, W),
                    img_shape=(H, W),
                    pad_shape=(H, W),
                    batch_input_shape=(H, W),
                    scale_factor=1.0,
                    flip=False
                )]
                for _ in range(batch_size)
            ]

            preds = []
            for i in range(batch_size):
                output = model.forward_test([imgs[i].unsqueeze(0)], [img_metas_batch[i]])
                pred_np = output[0]  # already (H, W), e.g., (512, 512)
                pred_tensor = torch.from_numpy(pred_np) if isinstance(pred_np, np.ndarray) else pred_np
                preds.append(pred_tensor)

            preds = torch.stack(preds)  # shape: [B, H, W]

            labels = labels.cpu()
            if labels.ndim == 4:
                labels = labels.squeeze(1)

            assert preds.shape == labels.shape, f"Mismatch: preds {preds.shape}, labels {labels.shape}"

            for c in range(4):
                pred_c = preds == c
                label_c = labels == c
                inter = (pred_c & label_c).sum().float()
                union = (pred_c | label_c).sum().float()
                tp = inter
                prec = tp / (pred_c.sum() + 1e-6)
                rec = tp / (label_c.sum() + 1e-6)
                total_iou[c] += (inter / (union + 1e-6)).item()
                total_f1[c] += (2 * prec * rec / (prec + rec + 1e-6)).item()
            total += 1


    mean_iou = [x / total for x in total_iou]
    mean_f1 = [x / total for x in total_f1]
    class_metrics = "\n  " + "\n  ".join(
        [f"Class {c}: IoU={mean_iou[c]:.4f}, F1={mean_f1[c]:.4f}" for c in range(len(mean_iou))]
    )
    metrics = f"{class_metrics}\n  Mean IoU: {np.mean(mean_iou):.4f}\n  Mean F1: {np.mean(mean_f1):.4f}"

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "a") as f:
        f.write(f"Backbone: InternImage_DCNv4_b | Combo {combo_index+1}/{total_combos} | Bands: {bands} | {metrics}\n")
    print(f"Done InternImage | {metrics}")

if __name__ == "__main__":
    result_file = f"results/UPerNet_InternImage_CloudSen12.txt"
    for i, bands in enumerate(band_combinations):
        train_and_evaluate(bands, i, len(band_combinations), result_file)
