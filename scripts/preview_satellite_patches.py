import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def load_and_normalize(path, bands=(1, 2, 3)):
    with rasterio.open(path) as src:
        img = src.read(bands).astype(np.float32)
        img = np.transpose(img, (1, 2, 0))
        if np.all(img == 0):
            raise ValueError(f"{path} contains only zeros.")
        for i in range(3):
            band = img[..., i]
            p2, p98 = np.percentile(band[band > 0], 2), np.percentile(band[band > 0], 98)
            if p98 - p2 > 1e-5:
                img[..., i] = (band - p2) / (p98 - p2 + 1e-6)
            else:
                min_val, max_val = band[band > 0].min(), band[band > 0].max()
                img[..., i] = (band - min_val) / (max_val - min_val + 1e-6)
        return np.clip(img, 0, 1)

script_dir = os.path.dirname(__file__)
patch_dir = os.path.abspath(os.path.join(script_dir, "..", "satellite_data", "patches"))
output_dir = os.path.abspath(os.path.join(script_dir, "..", "satellite_data", "visualizations"))
os.makedirs(output_dir, exist_ok=True)

files = {
    "Sentinel-2 TOA": "sentinel2_toa.tif",
    "Sentinel-2 BOA": "sentinel2_boa.tif",
    "Landsat 8 TOA": "landsat8_toa.tif",
    "Landsat 8 BOA": "landsat8_boa.tif"
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (title, filename) in zip(axes.ravel(), files.items()):
    path = os.path.join(patch_dir, filename)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        img = load_and_normalize(path)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    except Exception as e:
        ax.set_title(f"{title} (Error)")
        ax.axis("off")
        print(f"Error loading {path}: {e}")

plt.tight_layout()
out_path = os.path.join(output_dir, "satellite_patch_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"Saved preview: {out_path}")
