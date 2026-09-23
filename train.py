import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from main import UNet


class SegmentationDataset(Dataset):
  

    def __init__(self, images_dir, masks_dir, size=(572, 572)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = sorted(os.listdir(images_dir))

        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # -> float32 [0,1], shape (3, H, W)
        ])
        # NEAREST interpolation for masks: resizing must never invent
        # in-between pixel values (e.g. 0.37) at object edges -- a mask
        # has to stay strictly 0 or 1.
        self.mask_transform = transforms.Compose([
            transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),  # -> float32 [0,1], shape (1, H, W)
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, fname)).convert("L")

        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask


def evaluate(model, dataloader, device, criterion):
    """Computes average loss on a dataset WITHOUT updating any weights.
    Used for the held-out test set, so we can see how the model does on
    images it never trained on."""
    model.eval()        # inference mode
    running_loss = 0.0

    with torch.no_grad():  # no need to track gradients when we're not training
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode="nearest")
            loss = criterion(outputs, masks_resized)
            running_loss += loss.item()

    return running_loss / len(dataloader)


def train(model, train_loader, test_loader, device, epochs=10, lr=1e-4):
    criterion = nn.BCELoss()                                 # measures per-pixel error
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # updates the weights using gradients

    for epoch in range(epochs):
        model.train()  # training mode (matters once you add dropout/batchnorm)
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()      # clear gradients from the previous step
            outputs = model(images)    # forward pass -> (B, 1, H', W')

            # This U-Net uses unpadded ('valid') convolutions, so the output
            # is spatially smaller than the input. Resize the ground-truth
            # mask down to match the model's actual output size before
            # comparing them -- shapes must match exactly for the loss.
            masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode="nearest")

            loss = criterion(outputs, masks_resized)  # how wrong was this batch?
            loss.backward()                            # backprop: compute every weight's gradient
            optimizer.step()                            # gradient descent step: update the weights

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        test_loss = evaluate(model, test_loader, device, criterion)
        print(f"Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.4f} - test loss: {test_loss:.4f}")

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Point these at your actual dataset folders (image files + matching mask files).
    full_dataset = SegmentationDataset(images_dir="data/images", masks_dir="data/masks")

    # 80/20 train-test split -- e.g. 20 images -> 16 train, 4 test.
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size  # remainder, so the two always add up correctly

    torch.manual_seed(42)  # fixes the random split so it's the same every run
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # no need to shuffle evaluation data

    model = UNet(in_channels=3, num_classes=1).to(device)  # num_classes=1: binary sigmoid output, pairs cleanly with BCELoss

    model = train(model, train_loader, test_loader, device, epochs=20, lr=1e-4)

    torch.save(model.state_dict(), "unet_weights.pth")
    print("Training complete -- weights saved to unet_weights.pth")