import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import rasterio as rio
import numpy as np
from tqdm import tqdm
from mmseg.models.decode_heads import UPerHead
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class L8BiomePatchDataset(Dataset):
    def __init__(self, scene_dirs, patch_size=518, stride=518, band_mode='349'):
        self.scene_dirs = scene_dirs
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []

        if band_mode == 'rgb':
            self.band_indices = [2, 3, 4]
        elif band_mode == '349':
            self.band_indices = [3, 4, 9]
        elif band_mode == 'all':
            self.band_indices = list(range(1, 12))
        else:
            raise ValueError("band_mode must be one of ['rgb', '349', 'all']")

        for scene_dir in self.scene_dirs:
            scene_id = os.path.basename(scene_dir)
            img_path = os.path.join(scene_dir, f"{scene_id}.TIF")
            mask_path = os.path.join(scene_dir, f"{scene_id}_fixedmask.TIF")

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            with rio.open(img_path) as img:
                h, w = img.height, img.width

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_path, mask_path, x, y))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.patches[idx]

        with rio.open(img_path) as src:
            img = src.read(self.band_indices, window=rio.windows.Window(x, y, self.patch_size, self.patch_size)).astype(np.float32)

        with rio.open(mask_path) as dst:
            label_raw = dst.read(1, window=rio.windows.Window(x, y, self.patch_size, self.patch_size)).astype(np.uint8)

        mapping = {
            128: 0,  # Clear
            255: 1,  # Thick Cloud
            192: 2,  # Thin Cloud
            64:  3   # Shadow
        }

        label = np.full_like(label_raw, fill_value=255)
        for k, v in mapping.items():
            label[label_raw == k] = v

        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()

        return img, label

class DINOv2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.model.eval()

    def forward(self, x):
        feat = self.model.get_intermediate_layers(x, n=4)
        return [f.permute(0, 2, 1).reshape(f.shape[0], -1, int(f.shape[1]**0.5), int(f.shape[1]**0.5)) for f in feat]

class DINOv2_UPerNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = DINOv2Backbone()
        self.decode_head = UPerHead(
            in_channels=[768, 768, 768, 768],
            in_index=[0, 1, 2, 3],
            channels=512,
            num_classes=num_classes,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        )

    def forward(self, x, data_samples=None, mode='loss'):
        feats = self.backbone(x)
        if mode == 'loss':
            seg_logits = self.decode_head.forward(feats)
            for i, sample in enumerate(data_samples):
                sample.set_metainfo(dict(img_shape=seg_logits.shape[2:], ori_shape=seg_logits.shape[2:]))
            return self.decode_head.loss_by_feat(seg_logits, data_samples)
        elif mode == 'predict':
            return self.decode_head.predict_by_feat(feats, batch_img_metas=[dict(ori_shape=x.shape[2:])])
        elif mode == 'tensor':
            return self.decode_head.forward(feats)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

def train_one_epoch(model, loader, optimizer):
    model.train()
    model.backbone.eval()  # Keep DINOv2 frozen
    total_loss = 0
    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        data_samples = []
        for label in labels:
            H, W = label.shape[-2], label.shape[-1]
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
            sample.set_metainfo(dict(img_shape=(H, W), ori_shape=(H, W)))
            data_samples.append(sample)

        losses = model(imgs, data_samples, mode='loss')
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs, mode='tensor')
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            preds = logits.argmax(dim=1)
            mask = (labels >= 0) & (labels < num_classes)
            conf_mat += np.bincount(
                num_classes * labels[mask].cpu().numpy() + preds[mask].cpu().numpy(),
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"  Class {i}: IoU={iou:.4f}, F1={f1:.4f}\n")
    lines.append(f"  Mean IoU: {np.mean(ious):.4f}\n")
    lines.append(f"  Mean F1: {np.mean(f1s):.4f}\n")
    print("".join(lines))
    return lines

if __name__ == '__main__':
    root_dir = "data/l8_biome/l8biome"
    scene_dirs = []
    for cover_type in os.listdir(root_dir):
        full_cover_path = os.path.join(root_dir, cover_type)
        if os.path.isdir(full_cover_path):
            for scene_id in os.listdir(full_cover_path):
                scene_path = os.path.join(full_cover_path, scene_id)
                tif_file = os.path.join(scene_path, f"{scene_id}.TIF")
                mask_file = os.path.join(scene_path, f"{scene_id}_fixedmask.TIF")
                if os.path.exists(tif_file) and os.path.exists(mask_file):
                    scene_dirs.append(scene_path)

    print(f"Total scenes found: {len(scene_dirs)}")
    train_dirs, test_dirs = train_test_split(scene_dirs, test_size=0.2, random_state=42)

    os.makedirs("results", exist_ok=True)
    result_file = "results/UPerNet_DINOv2_L8Biome.txt"
    
    with open(result_file, "a") as f:
        for band_mode in ['rgb', '349']:
            print(f"\nRunning for band mode: {band_mode}")
            f.write(f"\n=== Evaluation for Band Mode: {band_mode} ===\n")

            train_ds = L8BiomePatchDataset(train_dirs, patch_size=518, stride=518, band_mode=band_mode)
            test_ds = L8BiomePatchDataset(test_dirs, patch_size=518, stride=518, band_mode=band_mode)
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

            model = DINOv2_UPerNet().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

            for epoch in range(10):
                loss = train_one_epoch(model, train_loader, optimizer)
                print(f"Epoch {epoch + 1} Loss: {loss:.4f}")
                f.write(f"Epoch {epoch + 1} Loss: {loss:.4f}\n")

            results = evaluate_test(model, test_loader)
            f.writelines(results)

