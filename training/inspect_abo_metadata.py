import gzip
import csv

path = "datasets\\abo\\abo-images-small\\images\\metadata\\images.csv.gz"

with gzip.open(path, "rt", encoding="utf-8") as f:
    reader = csv.reader(f)

    header = next(reader)
    print("COLUMNS:\n")
    print(header)

    print("\nFIRST 5 ROWS:\n")

    for i, row in enumerate(reader):
        print(row)

        if i == 4:
            break