# export LD_LIBRARY_PATH=~/.conda/envs/internimage-env/lib:$LD_LIBRARY_PATH


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import rasterio as rio
import tacoreader
from tqdm import tqdm
from mmengine.config import Config
from mmseg.models import build_segmentor
from torchdiffeq import odeint

sys.path.append(os.path.abspath("DCNv4/segmentation"))
from mmseg_custom.models.backbones import FlashInternImage

import logging
logging.disable(logging.CRITICAL)



band_combinations = [
    [3, 4, 10]  # Green, Red, Cirrus
]

class ODEFunc(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, t, x):
        return self.net(x)

class ODEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.odefunc = ODEFunc(channels)

    def forward(self, x):
        t = torch.tensor([0, 1], dtype=torch.float32).to(x.device)
        return odeint(self.odefunc, x, t, method='rk4')[1].to(x.device)

class FlashInternImageWithODE(nn.Module):
    def __init__(self, model, in_channels):
        super().__init__()
        self.model = model
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 512, 512).to(next(model.parameters()).device)
            features = model.backbone(dummy_input)
        channels = features[-1].shape[1]
        self.ode_block = ODEBlock(channels)

    def forward(self, imgs, img_metas=None, gt_semantic_seg=None, return_loss=True):
        # if isinstance(imgs, list):
        #     imgs = imgs[0]
        x = self.model.backbone(imgs)
        x[-1] = self.ode_block(x[-1])
        if return_loss:
            return self.model.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.model.train_cfg)


    def forward_test(self, imgs, img_metas, **kwargs):

        x = self.model.backbone(imgs)
        x[-1] = self.ode_block(x[-1])
        seg_logits = self.model.decode_head.forward_test(x, img_metas, self.model.test_cfg)
        #Fix here
        seg_logits = [F.interpolate(logit.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0) for logit in seg_logits]
        seg_preds = [logit.argmax(dim=0).cpu() for logit in seg_logits]

        return seg_preds



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
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(label, dtype=torch.long)

def train_and_evaluate(bands, combo_index, total_combos, result_path):
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    train_idx, test_idx = list(range(8000)), list(range(8000, 10000))
    train_set = CloudSegmentationDataset(taco_path, train_idx, bands)
    test_set = CloudSegmentationDataset(taco_path, test_idx, bands)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    num_classes = 4
    cfg = Config.fromfile("DCNv4/segmentation/configs/ade20k/upernet_flash_internimage_b_512_160k_ade20k.py")
    cfg.model.pretrained = None
    cfg.model.backbone.in_channels = len(bands)
    cfg.model.decode_head.num_classes = num_classes
    if "auxiliary_head" in cfg.model:
        cfg.model.auxiliary_head.num_classes = num_classes

    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    if "auxiliary_head" in cfg.model:
        cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    cfg.model.test_cfg = dict(mode='whole')

    base_model = build_segmentor(cfg.model)
    base_model.init_weights()
    base_model = base_model.to("cuda")
    model = FlashInternImageWithODE(base_model, in_channels=len(bands)).to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting Training...")
    model.train()
    for epoch in range(10):
        epoch_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            imgs = imgs.to("cuda")
            labels = labels.to("cuda")

            img_metas_batch = [
                dict(
                    ori_shape=(512, 512),
                    img_shape=(512, 512),
                    pad_shape=(512, 512),
                    batch_input_shape=(512, 512),
                    scale_factor=1.0,
                    flip=False
                )
                for _ in range(imgs.size(0))
            ]

            optimizer.zero_grad()
            loss_dict = model(imgs, img_metas=img_metas_batch, gt_semantic_seg=labels.unsqueeze(1), return_loss=True)

            if isinstance(loss_dict, dict):
                loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            else:
                loss = loss_dict

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training Complete. Starting Evaluation...")
    model.eval()
    total = 0
    total_iou = [0] * num_classes
    total_f1 = [0] * num_classes

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluation"):
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
                output = model.forward_test(imgs[i].unsqueeze(0), img_metas_batch[i])
                pred_tensor = output[0]
                preds.append(pred_tensor)

            preds = torch.stack(preds)

            labels = labels.cpu()
            if labels.ndim == 4:
                labels = labels.squeeze(1)

            assert preds.shape == labels.shape, f"Mismatch: preds {preds.shape}, labels {labels.shape}"

            for c in range(num_classes):
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
        [f"Class {c}: IoU={mean_iou[c]:.4f}, F1={mean_f1[c]:.4f}" for c in range(num_classes)]
    )
    metrics = f"{class_metrics}\n  Mean IoU: {np.mean(mean_iou):.4f}\n  Mean F1: {np.mean(mean_f1):.4f}"

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "a") as f:
        f.write(f"Backbone: InternImage_DCNv4_b with ODE | Combo {combo_index+1}/{total_combos} | Bands: {bands} | {metrics}\n")
    print(f"Done InternImage | {metrics}")

if __name__ == "__main__":
    result_file = f"results/UPerNet_InternImage_CloudSen12.txt"
    for i, bands in enumerate(band_combinations):
        train_and_evaluate(bands, i, len(band_combinations), result_file)
