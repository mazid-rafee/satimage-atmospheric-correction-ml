import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import numpy as np
import tacoreader
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
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

taco_path = "data/CloudSen12+/TACOs/mini-cloudsen12-l1c-high-512.taco"
indices = list(range(0, 10000))
train_indices = indices[:8000]
test_indices = indices[8000:10000]
selected_bands = [3, 4, 10]

train_ds = CloudSegmentationDataset(taco_path, train_indices, selected_bands)
test_ds = CloudSegmentationDataset(taco_path, test_indices, selected_bands)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4)

cfg = Config(dict(
    model=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 18, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            norm_cfg=dict(type='LN', requires_grad=True),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
            )
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=512,
            dropout_ratio=0.1,
            num_classes=4,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    )
))

model = build_segmentor(cfg.model)
model.init_weights()
model = model.to(device)

# To freeze the backbone
# for param in model.backbone.parameters():
#     param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=6e-5, weight_decay=0.01
)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        data_samples = []
        for i in range(labels.shape[0]):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=labels[i])
            data_samples.append(sample)
        losses = model(imgs, data_samples, mode='loss')
        loss = sum(v for v in losses.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def fast_confusion_matrix(preds, labels, num_classes=4):
    mask = (labels >= 0) & (labels < num_classes)
    return np.bincount(
        num_classes * labels[mask].astype(int) + preds[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def evaluate_test(model, loader):
    model.eval()
    num_classes = 4
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model.encode_decode(imgs, [dict(img_shape=(512, 512), ori_shape=(512, 512))])
            preds = logits.argmax(dim=1)
            conf_mat += fast_confusion_matrix(preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten(), num_classes)

    ious, f1s, lines = [], [], []
    for i in range(num_classes):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom else 0.0
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
    for epoch in range(5):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

    results = evaluate_test(model, test_loader)

    os.makedirs("results", exist_ok=True)
    with open("results/Segformer_mitb5_Cloudsen12.txt", "a") as f:
        f.writelines(results)
