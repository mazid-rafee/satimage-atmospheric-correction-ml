import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm
from fastkan import FastKAN as KAN
import torch.nn.functional as F


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
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
        return img, label

# PSP Module
class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x] + [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        output = torch.cat(pyramids, dim=1)
        return self.bottleneck(output)


# ASPP Module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)

# CNN-KAN-ASPP Model
class CNN_KAN_Bottleneck(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.aspp = ASPP(in_channels=512, out_channels=256)
        self.psp = PSPModule(in_channels=512, pool_sizes=[1, 2, 3, 6], out_channels=256)
        self.kan = KAN([512, 256, 256, 512])


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)                             # [B, 512, H/16, W/16]
        x_aspp = self.aspp(x)                           # [B, 256, H/16, W/16]
        x_psp = self.psp(x)                             # [B, 256, H/16, W/16]
        x = torch.cat([x_aspp, x_psp], dim=1)           # [B, 512, H/16, W/16]

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)        # Flatten spatial
        x = self.kan(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return self.decoder(x)

# Training function
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(num_classes * labels[mask] + preds[mask], minlength=num_classes**2).reshape(num_classes, num_classes)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(), labels.cpu().numpy().ravel(), num_classes)
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
    return lines

# Main script
if __name__ == '__main__':
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    indices = list(range(10000))
    band_sets = {
        "Bands_All_1_to_13": list(range(1, 14))
    }

    os.makedirs("results", exist_ok=True)
    log_path = "results/CNN_KAN_Segmenter.txt"

    with open(log_path, "a") as log_file:
        for name, selected_bands in band_sets.items():
            train_dataset = CloudSegmentationDataset(taco_path, indices[:8000], selected_bands)
            test_dataset = CloudSegmentationDataset(taco_path, indices[8000:], selected_bands)

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

            model = CNN_KAN_Bottleneck(in_channels=len(selected_bands), num_classes=4).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(100):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")


                if (epoch + 1) in [10, 20, 50, 100]:
                    results = evaluate_test(model, test_loader)
                    print(f"\nEvaluation after Epoch {epoch + 1}:\n" + "".join(results))
                    log_file.write(f"\nEvaluation after Epoch {epoch + 1}:\n")
                    log_file.writelines(results)
                    log_file.write("\n")
