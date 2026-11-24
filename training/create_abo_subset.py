
import csv
import random
import os

input_path = "datasets/abo/filtered/selected_images.csv"
output_dir = "datasets/abo/subset"

os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    images = list(csv.DictReader(f))

random.seed(42)
selected = random.sample(images, min(2000, len(images)))

random.shuffle(selected)

train_images = selected[:1800]
val_images = selected[1800:]

fields = ["image_id", "height", "width", "path"]

with open(os.path.join(output_dir, "train.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(train_images)

with open(os.path.join(output_dir, "val.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(val_images)

print("Dataset subset created successfully!")
print("Training images:", len(train_images))
print("Validation images:", len(val_images))
print("Location:", output_dir)
