import torch
import numpy as np
from main import UNet
from train import SegmentationDataset


ds = SegmentationDataset("data/images", "data/masks")
img, mask = ds[0]
print("=== 1. DATASET MASK CHECK ===")
print(f"Mask Min : {mask.min().item()}")
print(f"Mask Max : {mask.max().item()}")
print(f"Foreground pixels count: {(mask > 0).sum().item()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load("unet_weights.pth", map_location=device))
model.eval()

img_tensor = img.unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(img_tensor)

print("\n=== 2. RAW MODEL OUTPUT CHECK ===")
print(f"Min probability : {pred.min().item():.5f}")
print(f"Max probability : {pred.max().item():.5f}")
print(f"Mean probability: {pred.mean().item():.5f}")