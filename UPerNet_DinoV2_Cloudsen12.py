import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm
from mmseg.models.decode_heads import UPerHead
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import torchvision.transforms.functional as TF
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        feat = self.model.get_intermediate_layers(x, n=4)
        outs = [f.permute(0, 2, 1).reshape(f.shape[0], -1, int(f.shape[1]**0.5), int(f.shape[1]**0.5)) for f in feat]
        return outs


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
                sample.set_metainfo(dict(
                    img_shape=seg_logits.shape[2:],
                    ori_shape=seg_logits.shape[2:]
                ))
            return self.decode_head.loss_by_feat(seg_logits, data_samples)
        elif mode == 'predict':
            return self.decode_head.predict_by_feat(feats, batch_img_metas=[dict(ori_shape=x.shape[2:])])
        elif mode == 'tensor':
            out = self.decode_head.forward(feats)
            return out
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def train_one_epoch(model, loader, optimizer):
    model.train()
    model.backbone.eval()
    total_loss = 0

    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        data_samples = []
        for label in labels:
            H, W = label.shape[-2], label.shape[-1]
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=label.unsqueeze(0))
            sample.set_metainfo(dict(
                img_shape=(H, W),
                ori_shape=(H, W)
            ))
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
    model = DINOv2_UPerNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)

    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    indices = list(range(10000))
    train_ds = CloudSegmentationDataset(taco_path, indices[:8000], [3, 4, 10])
    test_ds = CloudSegmentationDataset(taco_path, indices[8000:], [3, 4, 10])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch + 1} Loss: {loss:.4f}")

    results = evaluate_test(model, test_loader)
    os.makedirs("results", exist_ok=True)
    with open("results/UPerNet_Dinov2_Cloudsen12.txt", "a") as f:
        f.writelines(results)
