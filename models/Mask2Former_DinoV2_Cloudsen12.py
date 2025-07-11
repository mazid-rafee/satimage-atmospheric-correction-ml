# Added dinvov2.py in mmsegmentation/mmseg/models/backbones/
# Edited __init__.py in mmsegmentation/mmseg/models/backbones/ to include dinov2


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
import torchvision.transforms.functional as TF

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------
# Dataset
# -------------
class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.dataset.read(self.indices[idx])
        s2_l1c_path = record.read(0)
        s2_label_path = record.read(1)
        with rio.open(s2_l1c_path) as src, rio.open(s2_label_path) as dst:
            img = src.read(indexes=self.selected_bands).astype(np.float32)
            label = dst.read(1).astype(np.uint8)
        img = torch.from_numpy(img / 3000.0).float()
        label = torch.from_numpy(label).long()
        img = TF.resize(img, [518,518])
        label = TF.resize(label.unsqueeze(0), [518,518], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return img, label

# -------------
# Config
# -------------
cfg = Config.fromfile("mmsegmentation/configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py")
cfg.custom_imports = dict(imports=['mmseg.models.backbones.dinov2_backbone'], allow_failed_imports=False)
cfg.model.backbone = dict(type="DINOv2")
cfg.model.decode_head.in_channels = [768,768,768,768]
cfg.model.data_preprocessor = None

# -------------
# Model
# -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)

# -------------
# DataLoader
# -------------
taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
indices = list(range(0, 10000))
train_loader = DataLoader(
    CloudSegmentationDataset(taco_path, indices[:8000], [3,4,10]),
    batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(
    CloudSegmentationDataset(taco_path, indices[8000:], [3,4,10]),
    batch_size=16, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

# -------------
# Train
# -------------
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        data_samples = []
        for label in labels:
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
            sample.set_metainfo(dict(
                img_shape=label.shape, ori_shape=label.shape))
            data_samples.append(sample)
        losses = model(imgs, data_samples, mode="loss")
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# -------------
# Evaluate
# -------------
def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(518,518), ori_shape=(518,518))])
            preds = logits.argmax(dim=1)
            preds = torch.clamp(preds, 0, num_classes-1)
            mask = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
            flat = num_classes * labels[mask].cpu().numpy() + preds[mask].cpu().numpy()
            bincount = np.bincount(flat, minlength=num_classes**2)
            if bincount.size != num_classes**2:
                raise ValueError(f"Confusion matrix size mismatch, got {bincount.size}, expected {num_classes**2}")
            conf_mat += bincount.reshape(num_classes, num_classes)

    lines, ious, f1s = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i,i]
        FP = conf_mat[:,i].sum() - TP
        FN = conf_mat[i,:].sum() - TP
        iou = TP/(TP+FP+FN) if (TP+FP+FN) else 0
        prec = TP/(TP+FP) if (TP+FP) else 0
        rec = TP/(TP+FN) if (TP+FN) else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        ious.append(iou)
        f1s.append(f1)
        lines.append(f"Class {i}: IoU={iou:.4f}, F1={f1:.4f}\n")
    lines.append(f"Mean IoU: {np.mean(ious):.4f}\nMean F1: {np.mean(f1s):.4f}\n")
    print("".join(lines))
    return lines

# -------------
# Driver
# -------------
if __name__ == "__main__":
    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}: Train Loss = {loss:.4f}")
    results = evaluate_test(model, test_loader)
    os.makedirs("results", exist_ok=True)
    with open("results/Mask2Former_DINOv2_CloudSen12.txt", "a") as f:
        f.writelines(results)
