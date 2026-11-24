import os
import csv
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "results", "evaluation.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CSV_PATH, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f)) if False else list(__import__("csv").DictReader(f))

bicubic_psnr = np.mean([float(r["bicubic_psnr"]) for r in rows])
sr_psnr = np.mean([float(r["sr_psnr"]) for r in rows])

bicubic_ssim = np.mean([float(r["bicubic_ssim"]) for r in rows])
sr_ssim = np.mean([float(r["sr_ssim"]) for r in rows])

# PSNR comparison
plt.figure(figsize=(7, 5))
plt.bar(["Bicubic", "BrandRestore AI"], [bicubic_psnr, sr_psnr])
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "psnr_comparison.png"), dpi=300)
plt.show()

# SSIM comparison
plt.figure(figsize=(7, 5))
plt.bar(["Bicubic", "BrandRestore AI"], [bicubic_ssim, sr_ssim])
plt.ylabel("SSIM")
plt.title("SSIM Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ssim_comparison.png"), dpi=300)
plt.show()

print("PSNR chart saved.")
print("SSIM chart saved.")
print("Results folder:", OUTPUT_DIR)