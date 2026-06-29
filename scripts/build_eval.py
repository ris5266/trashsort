import os
import sys
import csv
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from huggingface_hub import HfApi, hf_hub_download

from trashsort import config

REPO = "omasteam/waste-garbage-management-dataset"
PER_CLASS = 10
SEED = 42

# gd material class -> draft german bin (just a starting guess, fix by hand)
DRAFT_BIN = {
    "battery": "sondermuell",
    "biological": "biomuell",
    "cardboard": "papier",
    "clothes": "restmuell",
    "glass": "altglas",
    "metal": "gelbe_tonne",
    "paper": "papier",
    "plastic": "gelbe_tonne",
    "shoes": "restmuell",
    "trash": "restmuell",
}


def main():
    random.seed(SEED)
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")

    # group image paths by class folder
    by_class = {}
    for f in files:
        if not f.lower().endswith(".jpg"):
            continue
        cls = f.split("/")[0]
        if cls not in DRAFT_BIN:
            continue
        by_class.setdefault(cls, []).append(f)

    img_dir = os.path.join(config.EVAL_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for cls, paths in by_class.items():
        pick = random.sample(paths, min(PER_CLASS, len(paths)))
        for p in pick:
            src = hf_hub_download(REPO, p, repo_type="dataset")
            name = "%s__%s" % (cls, os.path.basename(p))
            out = os.path.join(img_dir, name)
            with open(src, "rb") as a, open(out, "wb") as b:
                b.write(a.read())
            rows.append((name, cls, DRAFT_BIN[cls]))

    # write the draft label sheet
    csv_path = os.path.join(config.EVAL_DIR, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "material", "bin"])
        w.writerows(sorted(rows))

    print("saved %d images -> %s" % (len(rows), img_dir))
    print("draft labels -> %s  (review the 'bin' column by hand!)" % csv_path)


if __name__ == "__main__":
    main()
