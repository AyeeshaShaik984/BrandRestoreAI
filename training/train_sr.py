
import os
import math
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

from model import BrandRestoreSR


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "datasets", "sr_pairs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


class SRDataset(Dataset):
    def __init__(self, split):
        self.lr_dir = os.path.join(DATA_DIR, split, "lr")
        self.hr_dir = os.path.join(DATA_DIR, split, "hr")

        self.files = sorted([
            f for f in os.listdir(self.lr_dir)
            if f.lower().endswith(".png")
            and os.path.exists(os.path.join(self.hr_dir, f))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]

        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        lr_tensor = to_tensor(lr_image)
        hr_tensor = to_tensor(hr_image)

        return lr_tensor, hr_tensor


train_dataset = SRDataset("train")
val_dataset = SRDataset("val")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = BrandRestoreSR().to(DEVICE)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def calculate_psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2)

    if mse.item() == 0:
        return 100.0

    return 10 * math.log10(1.0 / mse.item())


best_val_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for lr_images, hr_images in train_loader:

        lr_images = lr_images.to(DEVICE)
        hr_images = hr_images.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(lr_images)

        loss = criterion(predictions, hr_images)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():

        for lr_images, hr_images in val_loader:

            lr_images = lr_images.to(DEVICE)
            hr_images = hr_images.to(DEVICE)

            predictions = model(lr_images)

            loss = criterion(predictions, hr_images)

            val_loss += loss.item()
            val_psnr += calculate_psnr(
                predictions, hr_images
            )

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val PSNR: {val_psnr:.2f} dB"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "best_sr_model.pth")
        )

        print("Best model saved!")


torch.save(
    model.state_dict(),
    os.path.join(MODEL_DIR, "final_sr_model.pth")
)

print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
print("Models saved in:", MODEL_DIR)
