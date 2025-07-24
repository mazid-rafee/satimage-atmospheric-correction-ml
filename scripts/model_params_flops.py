import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.CNN_KAN_Segmenter import CNN_KAN_Segmenter
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    in_channels = 13
    num_classes = 4
    input_size = (in_channels, 512, 512)

    model = CNN_KAN_Segmenter(in_channels=in_channels, num_classes=num_classes)

    print("[INFO] Starting FLOPs and parameter count...")

    macs, params = get_model_complexity_info(
        model,
        input_size,
        as_strings=True,
        print_per_layer_stat=True,
        verbose=False
    )

    print(f"\n[Model Complexity for CNN_KAN_Segmenter]")
    print(f"FLOPs: {macs}")
    print(f"Parameters: {params}")
