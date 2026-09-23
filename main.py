import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class EncoderBlock(nn.Module):
    """Two 3x3 convolutions + ReLU, then a 2x2 max pool.
    Equivalent to the `encoder_block` function in the TensorFlow version.
    Returns the pooled output (used as input to the next encoder block
    AND as the skip connection into the matching decoder block, exactly
    like the original code)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x


class DecoderBlock(nn.Module):
    """Transposed-conv upsample, resize + concat the skip connection,
    then two 3x3 convolutions + ReLU. Equivalent to `decoder_block`."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)

    def forward(self, x, skip):
        x = self.up(x)
        # Resize the skip connection to match x's spatial size (H, W),
        # same role as the Keras `Resizing` layer.
        skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)  # concatenate along the CHANNEL dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # Contracting path (encoder)
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck1 = nn.Conv2d(512, 1024, kernel_size=3, padding=0)
        self.bottleneck2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=0)

        # Expansive path (decoder)
        self.dec1 = DecoderBlock(1024, 512, 512)
        self.dec2 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec4 = DecoderBlock(128, 64, 64)

        # Final 1x1 conv to map to `num_classes` channels
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = F.relu(self.bottleneck1(s4))
        b = F.relu(self.bottleneck2(b))

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        out = torch.sigmoid(self.out_conv(d4))
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, num_classes=2).to(device)
    model.eval()  # inference mode (matters once you add dropout/batchnorm)

    img = Image.open("doreamon.jpg").convert("RGB")
    img = img.resize((572, 572))

    # ToTensor() converts a PIL image (H, W, C) uint8 [0,255] into a
    # torch tensor (C, H, W) float32 [0,1] in one step.
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)  # add batch dim -> (1, 3, 572, 572)

    with torch.no_grad():  # disable gradient tracking, saves memory/time
        predictions = model(img_tensor)

    pred_mask = predictions.squeeze(0).cpu().numpy()          # (num_classes, H, W)
    pred_mask = np.argmax(pred_mask, axis=0).astype(np.uint8) * 255  # (H, W)

    pred_mask_img = Image.fromarray(pred_mask)
    pred_mask_img = pred_mask_img.resize(img.size)

    pred_mask_img.save("predicted_image.jpg")
    pred_mask_img.show()