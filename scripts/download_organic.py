import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from trashsort import config

PER_CLASS = 5

def main():
    print("downloading fruits-360 ...")
    ds = load_dataset("PedroSampaio/fruits-360", split="train")
    names = ds.features["label"].names

    by_label = {}
    for i, lbl in enumerate(ds["label"]):
        by_label.setdefault(lbl, []).append(i)

    out_dir = os.path.join(config.TRASHNET_DIR, "organic")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for lbl, idxs in by_label.items():
        step = max(1, len(idxs) // PER_CLASS)
        for i in idxs[::step][:PER_CLASS]:
            img = ds[i]["image"].convert("RGB")
            img.save(os.path.join(out_dir, "organic_%04d.jpg" % saved))
            saved += 1

    print("saved", saved, "organic images from", len(names), "fruit/veg types")
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
