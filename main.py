import matplotlib
matplotlib.use("Agg")

import tacoreader
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

cloudsen12_l1c = tacoreader.load("tacofoundation:cloudsen12-l1c")
cloudsen12_l2a = tacoreader.load("tacofoundation:cloudsen12-l2a")

sample = 1000
s2l1c_str = cloudsen12_l1c.read(sample).read(0)
s2l2a_str = cloudsen12_l2a.read(sample).read(0)

with rio.open(s2l1c_str) as src:
    s2l1c_data = src.read([4, 3, 2]).transpose(1, 2, 0) / 3000
    transform_toa = src.transform
    crs_toa = src.crs
    height_toa, width_toa = src.height, src.width

with rio.open(s2l2a_str) as dst:
    s2l2a_data = dst.read([4, 3, 2]).transpose(1, 2, 0) / 3000
    transform_boa = dst.transform
    crs_boa = dst.crs
    height_boa, width_boa = dst.height, dst.width

s2l1c_data = np.clip(s2l1c_data, 0, 1)
s2l2a_data = np.clip(s2l2a_data, 0, 1)

left_toa, top_toa = transform_toa * (0, 0)
right_toa, bottom_toa = transform_toa * (width_toa, height_toa)
bbox_proj_toa = (left_toa, bottom_toa, right_toa, top_toa)
bbox_latlon_toa = transform_bounds(crs_toa, "EPSG:4326", *bbox_proj_toa)

left_boa, top_boa = transform_boa * (0, 0)
right_boa, bottom_boa = transform_boa * (width_boa, height_boa)
bbox_proj_boa = (left_boa, bottom_boa, right_boa, top_boa)
bbox_latlon_boa = transform_bounds(crs_boa, "EPSG:4326", *bbox_proj_boa)

print(f"TOA bounding box (lat/lon): {bbox_latlon_toa}")
print(f"BOA bounding box (lat/lon): {bbox_latlon_boa}")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(s2l1c_data)
ax[0].set_title("Sentinel-2 L1C")
ax[0].axis("off")

ax[1].imshow(s2l2a_data)
ax[1].set_title("Sentinel-2 L2A")
ax[1].axis("off")

# plt.tight_layout()
# plt.savefig("remote_cloudsen12_sample.png", dpi=150)
