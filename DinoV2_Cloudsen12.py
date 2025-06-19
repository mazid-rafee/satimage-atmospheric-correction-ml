import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm
from mmseg.models.decode_heads import FCNHead
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from dinov2.models.vision_transformer import vit_base
import torchvision.transforms.functional as TF

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = (img - mean) / std

        label = torch.from_numpy(label)
        img = TF.resize(img, [518, 518])
        label = TF.resize(label.unsqueeze(0), [518, 518], interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

        return img, label


class DINOv2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.model.eval()

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(f"DINOv2 expects 3 channels (RGB). Got {x.shape[1]} channels.")

        feat = self.model.get_intermediate_layers(x, n=1)[0]
        B, N, D = feat.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(f"Feature map is not square: got N={N}, but HxW={H}x{W}")
        out = feat.transpose(1, 2).reshape(B, D, H, W)
        return out


class DINOv2_FCN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = DINOv2Backbone()
        self.decode_head = FCNHead(
            in_channels=768,
            channels=256,
            num_convs=2,
            kernel_size=3,
            num_classes=num_classes,
            dropout_ratio=0.1,
            norm_cfg=dict(type='BN', requires_grad=True),
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        )

    def forward(self, x, data_samples=None, mode='loss'):
        feat = self.backbone(x)
        if mode == 'loss':
            return self.decode_head.loss_by_feat(feat, data_samples)
        elif mode == 'predict':
            return self.decode_head.predict_by_feat(feat, batch_img_metas=[dict(ori_shape=x.shape[2:])])
        elif mode == 'tensor':
            out = self.decode_head.forward([feat])
            return out
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def train_one_epoch(model, loader, optimizer):
    model.train()
    model.backbone.eval()
    total_loss = 0

    for i, (imgs, labels) in enumerate(tqdm(loader, desc='Training')):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        data_samples = [
            SegDataSample(gt_sem_seg=PixelData(data=label.unsqueeze(0)))
            for label in labels
        ]

        losses = model(imgs, data_samples, mode='loss')
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


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
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    indices = list(range(0, 10000))
    train_ds = CloudSegmentationDataset(taco_path, indices[:8000], [3, 4, 10])
    test_ds = CloudSegmentationDataset(taco_path, indices[8000:], [3, 4, 10])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    model = DINOv2_FCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

    for epoch in range(1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

    results = evaluate_test(model, test_loader)
    os.makedirs("results", exist_ok=True)
    with open("results/Dinov2_Cloudsen12.txt", "a") as f:
        f.writelines(results)
