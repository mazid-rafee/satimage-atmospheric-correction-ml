import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import tacoreader
import segmentation_models_pytorch as smp
from tqdm import tqdm

MODE = "multiclass"  # "binary" or "multiclass"

band_combinations = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

encoder_configs = [
    {"name": "resnet50", "weights": "imagenet"},
    {"name": "se_resnet50", "weights": "imagenet"},
    {"name": "efficientnet-b0", "weights": "imagenet"},
    {"name": "timm-efficientnet-b0", "weights": "imagenet"},
    {"name": "efficientnet-b3", "weights": "imagenet"},
    {"name": "timm-efficientnet-b3", "weights": "imagenet"},
    {"name": "efficientnet-b5", "weights": "imagenet"},
    {"name": "mobileone_s2", "weights": "imagenet"},
    {"name": "mobilenet_v2", "weights": "imagenet"},
    {"name": "mit_b3", "weights": "imagenet"},
    {"name": "mit_b4", "weights": "imagenet"}
]

class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices, selected_bands, mode="binary"):
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        self.selected_bands = selected_bands
        self.mode = mode

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
        if self.mode == "binary":
            label = np.where(label == 0, 0, 1).astype(np.uint8)
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(label)

def train_and_evaluate(bands, combo_index, total_combos, encoder_name, encoder_weights, result_path, mode="binary"):
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    train_idx, test_idx = list(range(8000)), list(range(8000, 10000))
    train_set = CloudSegmentationDataset(taco_path, train_idx, bands, mode)
    test_set = CloudSegmentationDataset(taco_path, test_idx, bands, mode)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, num_workers=4)

    num_classes = 1 if mode == "binary" else 4

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=len(bands),
        classes=num_classes
    ).to("cuda")

    criterion = nn.BCEWithLogitsLoss() if mode == "binary" else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sigmoid = nn.Sigmoid(); threshold = 0.5

    for epoch in range(10):
        model.train(); total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"{encoder_name} | Combo {combo_index+1}/{total_combos} | Epoch {epoch+1}/10"):
            imgs = imgs.to("cuda")
            labels = labels.unsqueeze(1).float().to("cuda") if mode == "binary" else labels.long().to("cuda")
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"[{encoder_name}] Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")

    model.eval(); total = 0
    total_iou = [0]*4 if mode == "multiclass" else 0
    total_f1 = [0]*4 if mode == "multiclass" else 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"{encoder_name} | Evaluation"):
            imgs = imgs.to("cuda")
            if mode == "binary":
                labels = labels.unsqueeze(1).float().to("cuda")
                preds = sigmoid(model(imgs)) > threshold
                labels = labels.bool()
                inter = (preds & labels).sum((1, 2, 3)).float()
                union = (preds | labels).sum((1, 2, 3)).float()
                tp = inter; prec = tp / (preds.sum((1, 2, 3)) + 1e-6)
                rec = tp / (labels.sum((1, 2, 3)) + 1e-6)
                total_iou += (inter / (union + 1e-6)).mean().item()
                total_f1 += (2 * prec * rec / (prec + rec + 1e-6)).mean().item()
            else:
                labels = labels.to("cuda")
                preds = torch.argmax(model(imgs), dim=1)
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

    if mode == "binary":
        mean_iou = total_iou / total
        mean_f1 = total_f1 / total
        metrics = f"\n  Mean IoU: {mean_iou:.4f}\n  Mean F1: {mean_f1:.4f}"
    else:
        mean_iou = [x / total for x in total_iou]
        mean_f1 = [x / total for x in total_f1]
        class_metrics = "\n  " + "\n  ".join(
            [f"Class {c}: IoU={mean_iou[c]:.4f}, F1={mean_f1[c]:.4f}" for c in range(len(mean_iou))]
        )
        metrics = f"{class_metrics}\n  Mean IoU: {np.mean(mean_iou):.4f}\n  Mean F1: {np.mean(mean_f1):.4f}"


    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "a") as f:
        f.write(f"Encoder: {encoder_name} | Combo {combo_index+1}/{total_combos} | Bands: {bands} | Mode: {mode} | {metrics}\n")
    print(f"Done {encoder_name} | {metrics}")

if __name__ == "__main__":
    result_file = f"results/DeepLabV3Plus_CloudSen12_512px.txt"
    for enc in encoder_configs:
        for i, bands in enumerate(band_combinations):
            train_and_evaluate(bands, i, len(band_combinations), enc["name"], enc["weights"], result_file, mode=MODE)
