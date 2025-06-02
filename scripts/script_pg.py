import tacoreader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio as rio
import os

taco_path_l1c = os.path.join("..", "satellite_data", "CloudSen12+", "TACOs", "mini-cloudsen12-l1c-high-512.taco")
taco_path_l2a = os.path.join("..", "satellite_data", "CloudSen12+", "TACOs", "mini-cloudsen12-l2a-high-512.taco")

cloudsen12_l1c = tacoreader.load(taco_path_l1c)
cloudsen12_l2a = tacoreader.load(taco_path_l2a)

sample = 5426
s2l1c_str = cloudsen12_l1c.read(sample).read(0)
s2l2a_str = cloudsen12_l2a.read(sample).read(0)

with rio.open(s2l1c_str) as src, rio.open(s2l2a_str) as dst:
    s2l1c_data = src.read([4, 3, 2]).transpose(1, 2, 0) / 3000
    s2l2a_data = dst.read([4, 3, 2]).transpose(1, 2, 0) / 3000

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(s2l1c_data)
ax[0].set_title("Sentinel-2 L1C")
ax[0].axis("off")

ax[1].imshow(s2l2a_data)
ax[1].set_title("Sentinel-2 L2A")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("sentinel_l1c_l2a_comparison_sample100.png", dpi=200)
print("Saved image: sentinel_l1c_l2a_comparison_sample100.png")
