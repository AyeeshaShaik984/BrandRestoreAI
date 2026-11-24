import os
from PIL import Image
from collections import Counter

ROOT = "datasets/abo/abo-images-small/images/small"

print("Scanning ABO images...\n")
print("Folder exists:", os.path.exists(ROOT))

total = 0
corrupted = 0
sizes = Counter()
extensions = Counter()

for root, dirs, files in os.walk(ROOT):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        extensions[ext] += 1

        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            continue

        path = os.path.join(root, file)

        try:
            with Image.open(path) as img:
                sizes[img.size] += 1
                total += 1

        except Exception:
            corrupted += 1

print("\nTotal readable images:", total)
print("Corrupted images:", corrupted)

print("\nFile extensions:")
for ext, count in extensions.most_common():
    print(ext, ":", count)

print("\nMost common dimensions:")
for size, count in sizes.most_common(20):
    print(size, ":", count)