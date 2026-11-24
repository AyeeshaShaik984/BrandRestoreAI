
import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import BrandRestoreSR


BASE_DIR = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(BASE_DIR, "datasets", "sr_pairs", "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_sr_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

lr_dir = os.path.join(DATA_DIR, "lr")
hr_dir = os.path.join(DATA_DIR, "hr")

files = sorted([
    f for f in os.listdir(lr_dir)
    if f.lower().endswith(".png")
    and os.path.exists(os.path.join(hr_dir, f))
])

if not files:
    raise RuntimeError("No validation image pairs found.")

filename = files[0]

lr_path = os.path.join(lr_dir, filename)
hr_path = os.path.join(hr_dir, filename)

lr_image = Image.open(lr_path).convert("RGB")
hr_image = Image.open(hr_path).convert("RGB")

model = BrandRestoreSR().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

lr_tensor = to_tensor(lr_image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    sr_tensor = model(lr_tensor)

sr_tensor = sr_tensor.squeeze(0).cpu().clamp(0, 1)

sr_image = to_pil_image(sr_tensor)

bicubic_image = lr_image.resize(
    hr_image.size,
    Image.Resampling.BICUBIC
)

output_path = os.path.join(OUTPUT_DIR, "comparison.png")

plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(lr_image.resize(hr_image.size))
plt.title("Low Resolution")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(bicubic_image)
plt.title("Bicubic")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(sr_image)
plt.title("BrandRestore AI")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(hr_image)
plt.title("Original HR")
plt.axis("off")

plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()

print("Comparison saved to:", output_path)
print("Test image:", filename)
print("Device:", DEVICE)
