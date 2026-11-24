
import csv
import os
import random
from PIL import Image, ImageDraw

csv_path = "datasets/abo/filtered/selected_images.csv"
image_root = "datasets/abo/abo-images-small/images/small"

with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

random.seed(42)
samples = random.sample(rows, min(64, len(rows)))

thumb_size = 180
cols = 8
rows_count = 8

sheet = Image.new(
    "RGB",
    (cols * thumb_size, rows_count * thumb_size),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, item in enumerate(samples):
    path = os.path.join(image_root, item["path"])

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((150, 145))

        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size

        sheet.paste(img, (x + (thumb_size - img.width) // 2, y + 5))

        draw.text(
            (x + 5, y + 155),
            item["image_id"][:22],
            fill="black"
        )

    except Exception as e:
        print("Skipped:", path, e)

os.makedirs("datasets/abo/preview", exist_ok=True)

output = "datasets/abo/preview/contact_sheet.jpg"
sheet.save(output, quality=95)

print("Preview saved to:", output)
