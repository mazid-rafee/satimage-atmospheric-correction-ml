from huggingface_hub import hf_hub_download
import os

taco_dir = os.path.join("..", "satellite_data", "CloudSen12+", "TACOs")
os.makedirs(taco_dir, exist_ok=True)

files_to_download = [
    "cloudsen12-l1c.0000.part.taco",
    "cloudsen12-l1c.0001.part.taco",
    "cloudsen12-l1c.0002.part.taco",
    "cloudsen12-l1c.0003.part.taco",
    "cloudsen12-l1c.0004.part.taco",
    "cloudsen12-l2a.0000.part.taco",
    "cloudsen12-l2a.0001.part.taco",
    "cloudsen12-l2a.0002.part.taco",
    "cloudsen12-l2a.0003.part.taco",
    "cloudsen12-l2a.0004.part.taco",
    "cloudsen12-l2a.0005.part.taco",
    "cloudsen12-extra.0000.part.taco",
    "cloudsen12-extra.0001.part.taco",
    "cloudsen12-extra.0002.part.taco"
]

for file in files_to_download:
    hf_hub_download(
        repo_id="tacofoundation/CloudSEN12",
        filename=file,
        repo_type="dataset",
        local_dir=taco_dir,
        local_dir_use_symlinks=False
    )
