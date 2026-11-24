
import gzip
import csv
import os

metadata_path = "datasets/abo/abo-images-small/images/metadata/images.csv.gz"
image_root = "datasets/abo/abo-images-small/images/small"

selected = []
total = 0

with gzip.open(metadata_path, "rt", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total += 1

        height = int(row["height"])
        width = int(row["width"])

        if height < 160 or width < 160:
            continue

        ratio = width / height

        if ratio < 0.75 or ratio > 1.33:
            continue

        image_path = os.path.join(image_root, row["path"])

        if not os.path.isfile(image_path):
            continue

        selected.append({
            "image_id": row["image_id"],
            "height": height,
            "width": width,
            "path": row["path"]
        })

print("Total metadata images:", total)
print("Selected images:", len(selected))

print("\nFirst 20 selected images:\n")

for img in selected[:20]:
    print(img)

os.makedirs("datasets/abo/filtered", exist_ok=True)

output_path = "datasets/abo/filtered/selected_images.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_id", "height", "width", "path"]
    )

    writer.writeheader()
    writer.writerows(selected)

print("\nFiltered list saved to:", output_path)
