import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from main import UNet  


def predict_image(image_path, model_path="unet_weights.pth", output_path="predicted_mask.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize the model and load trained weights
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load and preprocess input image
    raw_img = Image.open(image_path).convert("RGB")
    original_size = raw_img.size  # Save original (width, height) to restore later

    to_tensor = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor(),
    ])
    img_tensor = to_tensor(raw_img).unsqueeze(0).to(device)  # Shape: (1, 3, 572, 572)

    # 3. Forward pass through model
    with torch.no_grad():
        prediction = model(img_tensor)  # Shape: (1, 1, 388, 388)
    print("Min probability:", prediction.min().item())
    print("Max probability:", prediction.max().item())

    # 4. Post-processing
    # Threshold probabilities: > 0.5 is foreground (255), <= 0.5 is background (0)
    pred_mask = (prediction.squeeze().cpu().numpy() > 0.2).astype(np.uint8) * 255

    # 5. Convert to image, resize back to original size, and save
    mask_img = Image.fromarray(pred_mask)
    mask_img = mask_img.resize(original_size, resample=Image.NEAREST)
    
    mask_img.save(output_path)
    print(f"Prediction saved to {output_path}")
    mask_img.show()


if __name__ == "__main__":
    predict_image("img007.jpeg")

