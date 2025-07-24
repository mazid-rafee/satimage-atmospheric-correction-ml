import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared(self.avg_pool(x))
        max_out = self.shared(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

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

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, out_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
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

class CNN_KAN_Segmenter(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.aspp = ASPP(512, 256)
        self.psp = PSPModule(512, [1, 2, 3, 6], 256)
        self.cbam = CBAM(512)
        self.kan = KAN([512, 256, 256, 512])

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.aux1 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.aux2 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.final_out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x_aspp = self.aspp(x)
        x_psp = self.psp(x)
        x = torch.cat([x_aspp, x_psp], dim=1)
        x = self.cbam(x)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.kan(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.dec1(x)
        aux_out1 = self.aux1(F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False))

        x = self.dec2(x)
        aux_out2 = self.aux2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

        x = self.dec3(x)
        x = self.dec4(x)
        out = self.final_out(x)

        return out, aux_out1, aux_out2


import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        main_out, aux1, aux2 = model(imgs)

        # Ensure aux outputs match label spatial size
        target_size = labels.shape[-2:]
        aux1 = F.interpolate(aux1, size=target_size, mode='bilinear', align_corners=False)
        aux2 = F.interpolate(aux2, size=target_size, mode='bilinear', align_corners=False)

        loss_main = loss_fn(main_out, labels)
        loss_aux1 = loss_fn(aux1, labels)
        loss_aux2 = loss_fn(aux2, labels)

        loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(num_classes * labels[mask] + preds[mask], minlength=num_classes**2).reshape(num_classes, num_classes)

def evaluate_val(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().ravel(), labels.cpu().numpy().ravel(), num_classes)
    ious = []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        iou = TP / (TP + FP + FN) if (TP + FP + FN) else 0.0
        ious.append(iou)
    return np.mean(ious)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)[0].argmax(1)
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

if __name__ == '__main__':
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    indices = list(range(10000))
    band_sets = {"Bands_All_1_to_13": list(range(1, 14))}
    os.makedirs("results", exist_ok=True)
    log_path = "results/CNN_KAN_Segmenter.txt"

    with open(log_path, "a") as log_file:
        for name, selected_bands in band_sets.items():
            train_indices = indices[:8500]
            val_indices = indices[8500:9000]
            test_indices = indices[9000:]

            train_dataset = CloudSegmentationDataset(taco_path, train_indices, selected_bands)
            val_dataset = CloudSegmentationDataset(taco_path, val_indices, selected_bands)
            test_dataset = CloudSegmentationDataset(taco_path, test_indices, selected_bands)

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

            model = CNN_KAN_Segmenter(in_channels=len(selected_bands), num_classes=4).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            best_miou = 0.0
            best_model_path = "results/CNN_KAN_Segmenter_best.pth"
            last_model_path = "results/CNN_KAN_Segmenter_last.pth"

            for epoch in range(100):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_miou = evaluate_val(model, val_loader)
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val mIoU = {val_miou:.4f}")

                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save(model.state_dict(), best_model_path)
                    print("Saved best model!")

            torch.save(model.state_dict(), last_model_path)

            print("\nEvaluating best model:")
            model.load_state_dict(torch.load(best_model_path))
            results_best = evaluate_test(model, test_loader)
            print("".join(results_best))
            log_file.write("\nEvaluation of Best Model:\n")
            log_file.writelines(results_best)
            log_file.write("\n")

            print("\nEvaluating last model:")
            model.load_state_dict(torch.load(last_model_path))
            results_last = evaluate_test(model, test_loader)
            print("".join(results_last))
            log_file.write("\nEvaluation of Last Model:\n")
            log_file.writelines(results_last)
            log_file.write("\n")
