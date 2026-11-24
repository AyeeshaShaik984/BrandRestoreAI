
import csv
import os
import random

random.seed(42)

abo_train_path = "datasets/abo/subset/train.csv"
abo_val_path = "datasets/abo/subset/val.csv"

lld_root = "datasets/LLD/LLD-logo_sample"

output_dir = "datasets/combined"
os.makedirs(output_dir, exist_ok=True)

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

abo_train = read_csv(abo_train_path)
abo_val = read_csv(abo_val_path)

lld_images = []

for folder, _, files in os.walk(lld_root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, file)

            try:
                from PIL import Image

                with Image.open(path) as img:
                    width, height = img.size

                if width >= 160 and height >= 160:
                    lld_images.append({
                        "image_id": os.path.splitext(file)[0],
                        "height": height,
                        "width": width,
                        "path": os.path.relpath(path, lld_root)
                    })

            except Exception:
                continue

random.shuffle(lld_images)

lld_train = lld_images[:450]
lld_val = lld_images[450:]

combined_train = []
combined_val = []

for item in abo_train:
    combined_train.append({
        "source": "ABO",
        "image_id": item["image_id"],
        "height": item["height"],
        "width": item["width"],
        "path": item["path"]
    })

for item in abo_val:
    combined_val.append({
        "source": "ABO",
        "image_id": item["image_id"],
        "height": item["height"],
        "width": item["width"],
        "path": item["path"]
    })

for item in lld_train:
    combined_train.append({
        "source": "LLD",
        "image_id": item["image_id"],
        "height": item["height"],
        "width": item["width"],
        "path": item["path"]
    })

for item in lld_val:
    combined_val.append({
        "source": "LLD",
        "image_id": item["image_id"],
        "height": item["height"],
        "width": item["width"],
        "path": item["path"]
    })

random.shuffle(combined_train)
random.shuffle(combined_val)

fields = ["source", "image_id", "height", "width", "path"]

with open(os.path.join(output_dir, "train.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(combined_train)

with open(os.path.join(output_dir, "val.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(combined_val)

print("Combined dataset created!")
print("ABO training:", len(abo_train))
print("ABO validation:", len(abo_val))
print("LLD training:", len(lld_train))
print("LLD validation:", len(lld_val))
print("Combined training:", len(combined_train))
print("Combined validation:", len(combined_val))
print("Saved to:", output_dir)
