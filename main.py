import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

class CloudSegmentationDataset(Dataset):
    def __init__(self, taco_path, indices):
        tqdm.write("Loading TACOs dataset...")
        self.dataset = tacoreader.load(taco_path)
        self.indices = indices
        tqdm.write(f"Dataset loaded with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        s2_l1c = self.dataset.read(sample_idx).read(0)
        s2_label = self.dataset.read(sample_idx).read(1)

        with rio.open(s2_l1c) as src, rio.open(s2_label) as dst:
            img = src.read(list(range(1, 14)), window=rio.windows.Window(0, 0, 512, 512)).astype(np.float32)
            label = dst.read(1, window=rio.windows.Window(0, 0, 512, 512)).astype(np.uint8)

        label = np.where(label == 0, 0, 1).astype(np.uint8)

        img = img / 3000.0
        img = np.transpose(img, (1, 2, 0))
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(label)

def main():
    taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
    all_indices = list(range(10000))
    train_idx, test_idx = all_indices[:8000], all_indices[8000:]

    tqdm.write("Preparing datasets and dataloaders...")
    train_dataset = CloudSegmentationDataset(taco_path, train_idx)
    test_dataset = CloudSegmentationDataset(taco_path, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

    tqdm.write("Initializing model...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=13, classes=1)
    model = model.to("cuda")
    print("Using device:", torch.cuda.current_device(), "-", torch.cuda.get_device_name())

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    sigmoid = nn.Sigmoid()
    threshold = 0.5

    tqdm.write("Starting training...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")):
            imgs = imgs.to("cuda")
            labels = labels.unsqueeze(1).float().to("cuda")
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 20 == 0:
                tqdm.write(f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    tqdm.write("Training complete. Starting evaluation...")

    model.eval()
    total_iou, total_f1, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs = imgs.to("cuda")
            labels = labels.unsqueeze(1).float().to("cuda")
            preds = sigmoid(model(imgs)) > threshold
            labels = labels.bool()

            intersection = (preds & labels).float().sum((1, 2, 3))
            union = (preds | labels).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-6)).mean().item()

            tp = (preds & labels).sum((1, 2, 3)).float()
            precision = tp / (preds.sum((1, 2, 3)) + 1e-6)
            recall = tp / (labels.sum((1, 2, 3)) + 1e-6)
            f1 = (2 * precision * recall / (precision + recall + 1e-6)).mean().item()

            total_iou += iou
            total_f1 += f1
            total += 1

            if (batch_idx + 1) % 10 == 0:
                tqdm.write(f"[Eval] Batch {batch_idx+1}/{len(test_loader)} - IoU: {iou:.4f}, F1: {f1:.4f}")

    tqdm.write(f"\nFinal Evaluation:")
    tqdm.write(f"Mean IoU: {total_iou / total:.4f}")
    tqdm.write(f"Mean F1 Score: {total_f1 / total:.4f}")

if __name__ == "__main__":
    main()
