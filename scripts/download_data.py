import ee
import geemap
import requests
from datetime import date, timedelta
import os

ee.Authenticate(auth_mode='notebook')
ee.Initialize()

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "satellite_data"))
patch_dir = os.path.join(base_dir, "patches")
os.makedirs(patch_dir, exist_ok=True)

bbox = [-81.5, 25.2, -80.5, 25.9]
region = ee.Geometry.Rectangle(bbox)
center = region.centroid()
center_coords = center.coordinates().getInfo()
lat = (bbox[1] + bbox[3]) / 2
lon = (bbox[0] + bbox[2]) / 2

max_attempts = 3
date_step_days = 30
pixel_size = 512

def create_patch(scale):
    size_m = pixel_size * scale
    half_deg = size_m / 111320 / 2
    cx, cy = center_coords
    return ee.Geometry.Rectangle([
        cx - half_deg, cy - half_deg,
        cx + half_deg, cy + half_deg
    ])

def get_valid_image(collection_id, name, bands, cloud_prop, scale):
    print(f"Searching for valid {name}...")
    today = date(2020, 6, 1)
    for attempt in range(max_attempts):
        start = today + timedelta(days=attempt * date_step_days)
        end = start + timedelta(days=date_step_days)
        start_str = str(start)
        end_str = str(end)
        print(f"Attempt {attempt + 1}: {start_str} → {end_str}")
        coll = (ee.ImageCollection(collection_id)
                .filterDate(start_str, end_str)
                .filterBounds(center)
                .sort(cloud_prop))
        try:
            img = coll.first()
            img.getInfo()
            img = img.select(bands).clip(region)
            patch = create_patch(scale)
            stats = img.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=patch,
                scale=scale,
                maxPixels=1e8
            ).getInfo()
            print("Band stats:", stats)
            band_valid = [
                stats.get(f"{b}_max", 0) > stats.get(f"{b}_min", 0)
                for b in bands
            ]
            if not all(band_valid):
                print("Patch is mostly zero or invalid. Trying next window...")
                continue
            return img, patch
        except Exception as e:
            print(f"No valid image found in this window: {e}")
    print(f"{name} not found after {max_attempts} attempts.")
    return None, None

def export_patch(image, patch, filename, scale):
    if image is None:
        return
    try:
        out_path = os.path.join(patch_dir, f"{filename}.tif")
        geemap.ee_export_image(
            image,
            filename=out_path,
            scale=scale,
            region=patch,
            file_per_band=False
        )
        print(f"Exported: {out_path}")
    except Exception as e:
        print(f"Export failed: {e}")

datasets = [
    ("COPERNICUS/S2_HARMONIZED", "sentinel2_toa", ["B4", "B3", "B2"], "CLOUDY_PIXEL_PERCENTAGE", 10),
    ("COPERNICUS/S2_SR_HARMONIZED", "sentinel2_boa", ["B4", "B3", "B2"], "CLOUDY_PIXEL_PERCENTAGE", 10),
    ("LANDSAT/LC08/C02/T1_TOA", "landsat8_toa", ["B4", "B3", "B2"], "CLOUD_COVER", 30),
    ("LANDSAT/LC08/C02/T1_L2", "landsat8_boa", ["SR_B4", "SR_B3", "SR_B2"], "CLOUD_COVER", 30)
]

for collection_id, filename, bands, cloud_prop, scale in datasets:
    image, patch = get_valid_image(collection_id, filename, bands, cloud_prop, scale)
    export_patch(image, patch, filename, scale)

def download_aeronet(lat, lon, radius_km=50):
    today = date.today()
    url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
    params = {
        "site": "",
        "year": today.year,
        "month": today.month,
        "day": today.day,
        "hour": 0,
        "average": 10,
        "if_no_html": 1,
        "level": 2,
        "location": f"{lat},{lon},{radius_km}"
    }
    r = requests.get(url, params=params)
    out_path = os.path.join(base_dir, "aeronet_observations.csv")
    with open(out_path, "w") as f:
        f.write(r.text)
    print("AERONET CSV saved:", out_path)

# download_aeronet(lat, lon)
