import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
from sklearn.model_selection import train_test_split
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from tqdm import tqdm
import logging

logging.disable(logging.CRITICAL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class L8BiomePatchDataset(Dataset):
    def __init__(self, scene_dirs, patch_size=512, stride=512, band_mode='349'):
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


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        data_samples = []
        for i in range(labels.shape[0]):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=labels[i])
            data_samples.append(sample)
        losses = model(imgs, data_samples, mode='loss')
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(
        num_classes * labels[mask].astype(int) + preds[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    class_names = ['Clear', 'Thick Cloud', 'Thin Cloud', 'Shadow']
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(512, 512), ori_shape=(512, 512))])
            preds = logits.softmax(dim=1).argmax(dim=1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten(), num_classes)

    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"  Class {i} ({class_names[i]}): IoU={iou:.4f}, F1={f1:.4f}\n")

    lines.append(f"  Mean IoU: {np.mean(ious):.4f}\n")
    lines.append(f"  Mean F1: {np.mean(f1s):.4f}\n")
    return lines


# --- Load scene dirs once ---
root_dir = "data/l8_biome/l8biome"
scene_dirs = []
for cover_type in os.listdir(root_dir):
    full_cover_path = os.path.join(root_dir, cover_type)
    if os.path.isdir(full_cover_path):
        for scene_id in os.listdir(full_cover_path):
            scene_path = os.path.join(full_cover_path, scene_id)
            if os.path.isdir(scene_path):
                tif_file = os.path.join(scene_path, f"{scene_id}.TIF")
                mask_file = os.path.join(scene_path, f"{scene_id}_fixedmask.TIF")
                if os.path.exists(tif_file) and os.path.exists(mask_file):
                    scene_dirs.append(scene_path)

print(f"Total scenes found: {len(scene_dirs)}")
train_dirs, test_dirs = train_test_split(scene_dirs, test_size=0.2, random_state=42)

# --- Result file ---
os.makedirs("results", exist_ok=True)
result_file = "results/Segformer_mitb5_L8Biome.txt"
with open(result_file, "a") as f:
    f.write("SegFormer mit_b5 Evaluation Results on L8Biome for Different Band Modes\n\n")

# --- Run all band modes ---
for band_mode in ['rgb', '349', 'all']:
    print(f"\n========== Running for band mode: {band_mode} ==========\n")

    train_ds = L8BiomePatchDataset(train_dirs, patch_size=512, stride=512, band_mode=band_mode)
    test_ds = L8BiomePatchDataset(test_dirs, patch_size=512, stride=512, band_mode=band_mode)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    cfg = Config(dict(
        model=dict(
            type='EncoderDecoder',
            backbone=dict(
                type='MixVisionTransformer',
                in_channels=len(train_ds.band_indices),
                embed_dims=64,
                num_stages=4,
                num_layers=[3, 4, 18, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                norm_cfg=dict(type='LN', requires_grad=True),
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
                )
            ),
            decode_head=dict(
                type='SegformerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=512,
                dropout_ratio=0.1,
                num_classes=4,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            ),
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )
    ))

    model = build_segmentor(cfg.model)
    model.decode_head.loss_decode.ignore_index = 255
    model.init_weights()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=6e-5, weight_decay=0.01
    )

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

    results = evaluate_test(model, test_loader)

    # Save all results to one file
    with open(result_file, "a") as f:
        f.write(f"\n=== Band Mode: {band_mode} ===\n")
        f.write(f"Epoch 10: Train Loss = {train_loss:.4f}\n")
        f.writelines(results)
