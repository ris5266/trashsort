import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import hf_hub_download

from trashsort import config

REPO = "omasteam/waste-garbage-management-dataset"


# rebuild the eval images from labels.csv. we only ship our own labels, not the
# dataset images (they are scraped/stock photos), so we re-download them here.
def main():
    csv_path = os.path.join(config.EVAL_DIR, "labels.csv")
    img_dir = os.path.join(config.EVAL_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    n = 0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            name = r["file"]                # e.g. battery__battery_121.jpg
            orig = name.split("__", 1)[1]   # battery_121.jpg
            src = hf_hub_download(REPO, "%s/%s" % (r["material"], orig), repo_type="dataset")
            with open(src, "rb") as a, open(os.path.join(img_dir, name), "wb") as b:
                b.write(a.read())
            n += 1
    print("downloaded %d eval images -> %s" % (n, img_dir))


if __name__ == "__main__":
    main()
