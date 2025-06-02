import tacoreader
import os

taco_dir = os.path.join(".." , "satellite_data", "CloudSen12+", "TACOs")
output_path = os.path.join(taco_dir, "mini-cloudsen12-extra-high-512.taco")

dataset = tacoreader.load([
    os.path.join(taco_dir, "cloudsen12-extra.0000.part.taco"),
    os.path.join(taco_dir, "cloudsen12-extra.0001.part.taco"),
    os.path.join(taco_dir, "cloudsen12-extra.0002.part.taco")
])

subset_sp = dataset[(dataset["real_proj_shape"] == 509) & (dataset["label_type"] == "high")]
tacoreader.compile(dataframe=subset_sp, output=output_path, nworkers=5)