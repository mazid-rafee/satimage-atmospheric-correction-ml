import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from tqdm import tqdm

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
        label = torch.from_numpy(label).long()
        return img, label

class SimpleEncoderDecoder(nn.Module):
    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.ReLU())
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.bottleneck(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.out(x)

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

if __name__ == '__main__':
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    indices = list(range(10000))
    band_sets = {
        "Bands_All_1_to_13": list(range(1, 14))
    }

    os.makedirs("results", exist_ok=True)
    log_path = "results/Simple_Encoder_Decoder.txt"

    with open(log_path, "a") as log_file:
        for name, selected_bands in band_sets.items():
            print(f"\n=== Training with {name} ===\n")
            log_file.write(f"\n=== Training with {name} ===\n")

            train_dataset = CloudSegmentationDataset(taco_path, indices[:8000], selected_bands)
            test_dataset = CloudSegmentationDataset(taco_path, indices[8000:], selected_bands)

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

            model = SimpleEncoderDecoder(in_channels=len(selected_bands), num_classes=4).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            for epoch in range(100):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")
                log_file.write(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}\n")

            results = evaluate_test(model, test_loader)
            log_file.writelines(results)
            log_file.write("\n" + "="*60 + "\n")
