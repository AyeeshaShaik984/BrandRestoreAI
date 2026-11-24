
import os
import csv
import torch
import numpy as np

from PIL import Image
from torchvision.transforms.functional import to_tensor

from skimage.metrics import structural_similarity
from model import BrandRestoreSR


BASE_DIR = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(BASE_DIR, "datasets", "sr_pairs", "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_sr_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

LR_DIR = os.path.join(DATA_DIR, "lr")
HR_DIR = os.path.join(DATA_DIR, "hr")


model = BrandRestoreSR().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    if mse == 0:
        return 100.0

    return 10 * np.log10((255.0 ** 2) / mse)


def calculate_ssim(img1, img2):
    return structural_similarity(
        img1,
        img2,
        channel_axis=2,
        data_range=255
    )


files = sorted([
    f for f in os.listdir(LR_DIR)
    if f.lower().endswith(".png")
    and os.path.exists(os.path.join(HR_DIR, f))
])

results = []

print("Evaluating images:", len(files))
print("Device:", DEVICE)


for index, filename in enumerate(files, start=1):

    lr_image = Image.open(
        os.path.join(LR_DIR, filename)
    ).convert("RGB")

    hr_image = Image.open(
        os.path.join(HR_DIR, filename)
    ).convert("RGB")

    bicubic_image = lr_image.resize(
        hr_image.size,
        Image.Resampling.BICUBIC
    )

    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_image = (
        sr_tensor.squeeze(0)
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    sr_image = np.transpose(sr_image, (1, 2, 0))
    sr_image = (sr_image * 255).round().astype(np.uint8)

    hr_array = np.array(hr_image)
    bicubic_array = np.array(bicubic_image)

    sr_psnr = calculate_psnr(sr_image, hr_array)
    bicubic_psnr = calculate_psnr(bicubic_array, hr_array)

    sr_ssim = calculate_ssim(sr_image, hr_array)
    bicubic_ssim = calculate_ssim(bicubic_array, hr_array)

    results.append({
        "image": filename,
        "bicubic_psnr": bicubic_psnr,
        "sr_psnr": sr_psnr,
        "bicubic_ssim": bicubic_ssim,
        "sr_ssim": sr_ssim
    })

    if index % 25 == 0:
        print(f"Processed {index}/{len(files)}")


csv_path = os.path.join(OUTPUT_DIR, "evaluation.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "bicubic_psnr",
            "sr_psnr",
            "bicubic_ssim",
            "sr_ssim"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


avg_bicubic_psnr = np.mean(
    [r["bicubic_psnr"] for r in results]
)

avg_sr_psnr = np.mean(
    [r["sr_psnr"] for r in results]
)

avg_bicubic_ssim = np.mean(
    [r["bicubic_ssim"] for r in results]
)

avg_sr_ssim = np.mean(
    [r["sr_ssim"] for r in results]
)


print("\n===== FINAL RESULTS =====")

print(f"Bicubic PSNR: {avg_bicubic_psnr:.4f} dB")
print(f"Model PSNR:    {avg_sr_psnr:.4f} dB")

print(f"Bicubic SSIM: {avg_bicubic_ssim:.4f}")
print(f"Model SSIM:    {avg_sr_ssim:.4f}")

print("\nSaved evaluation results:", csv_path)
