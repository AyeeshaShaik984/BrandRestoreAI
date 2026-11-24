import csv
import os
from PIL import Image

BASE_DIR = os.getcwd()

ABO_ROOT = os.path.join(
    BASE_DIR, "datasets", "abo", "abo-images-small", "images", "small"
)

LLD_ROOT = os.path.join(
    BASE_DIR, "datasets", "LLD", "LLD-logo_sample"
)

INPUT_DIR = os.path.join(BASE_DIR, "datasets", "combined")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "sr_pairs")

HR_SIZE = 128
LR_SIZE = 32

def resolve_image_path(item):
    source = item["source"]
    path = item["path"]

    if os.path.isabs(path) and os.path.exists(path):
        return path

    if source == "ABO":
        candidates = [
            os.path.join(ABO_ROOT, path),
            os.path.join(BASE_DIR, path),
        ]
    else:
        candidates = [
            os.path.join(LLD_ROOT, path),
            os.path.join(BASE_DIR, path),
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def prepare_image(image):
    image = image.convert("RGB")

    width, height = image.size

    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2

    image = image.crop((
        left,
        top,
        left + side,
        top + side
    ))

    image = image.resize(
        (HR_SIZE, HR_SIZE),
        Image.Resampling.LANCZOS
    )

    return image


def create_pair(item, split):
    path = resolve_image_path(item)

    if path is None:
        print("Missing:", item["path"])
        return False

    filename = f'{item["source"]}_{item["image_id"]}.png'

    hr_dir = os.path.join(OUTPUT_DIR, split, "hr")
    lr_dir = os.path.join(OUTPUT_DIR, split, "lr")

    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    hr_path = os.path.join(hr_dir, filename)
    lr_path = os.path.join(lr_dir, filename)

    try:
        with Image.open(path) as img:
            hr_image = prepare_image(img)

        lr_image = hr_image.resize(
            (LR_SIZE, LR_SIZE),
            Image.Resampling.BICUBIC
        )

        hr_image.save(hr_path)
        lr_image.save(lr_path)

        return True

    except Exception as e:
        print("Error:", path, e)
        return False


def process_csv(csv_path, split):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    successful = 0

    for i, item in enumerate(rows, start=1):
        if create_pair(item, split):
            successful += 1

        if i % 100 == 0:
            print(f"{split}: processed {i}/{len(rows)}")

    print(f"{split} completed: {successful}/{len(rows)}")


def main():
    train_csv = os.path.join(INPUT_DIR, "train.csv")
    val_csv = os.path.join(INPUT_DIR, "val.csv")

    print("Creating training image pairs...")
    process_csv(train_csv, "train")

    print("\nCreating validation image pairs...")
    process_csv(val_csv, "val")

    print("\nHR-LR pair generation completed!")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()