
import os
from PIL import Image
from collections import Counter

root = "datasets/LLD/LLD-logo_sample"

total = 0
valid = 0
corrupted = 0
dimensions = Counter()
min_side = float("inf")
max_side = 0

for folder, _, files in os.walk(root):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            total += 1
            path = os.path.join(folder, file)

            try:
                with Image.open(path) as img:
                    width, height = img.size
                    dimensions[(width, height)] += 1
                    min_side = min(min_side, width, height)
                    max_side = max(max_side, width, height)
                    valid += 1

            except Exception:
                corrupted += 1

print("Total image files:", total)
print("Valid images:", valid)
print("Corrupted images:", corrupted)
print("Smallest side:", min_side)
print("Largest side:", max_side)

print("\nDimension distribution:\n")

for size, count in dimensions.most_common():
    print(size, ":", count)
