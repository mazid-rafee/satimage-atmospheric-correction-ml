import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import tacoreader
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50
import rasterio as rio
import numpy as np
from tqdm import tqdm

class CloudSEN12Dataset(Dataset):
    def __init__(self, records, band_indexes):
        self.records = records
        self.band_indexes = band_indexes

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample = self.records.read(idx)
        image_path = sample.read(0)
        mask_path = sample.read(1)
        with rio.open(image_path) as img_src:
            image = img_src.read(self.band_indexes, window=rio.windows.Window(0, 0, 256, 256)).astype(np.float32)
            image = torch.from_numpy(image / 3000.0)
        with rio.open(mask_path) as lbl_src:
            mask = lbl_src.read(1, window=rio.windows.Window(0, 0, 256, 256)).astype(np.int64)
            mask = torch.from_numpy(mask)
        return image, mask

ds = tacoreader.load("tacofoundation:cloudsen12-l1c")
high_quality = ds[ds["label_type"] == "high"]
band_map = {'B2': 2, 'B3': 3, 'B4': 4, 'B8': 8, 'B11': 11, 'B12': 12}
bands = sorted(band_map.values())
dataset = CloudSEN12Dataset(high_quality, bands)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1)

model = deeplabv3_resnet50(weights=None, num_classes=6)
model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1)
model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1}")
    for images, masks in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
