import os
import sys
import shutil
import collections

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset

OUT = "data/trashnet/organic"
PER_CLASS = 22
MAXSIDE = 512


def main():
    ds = load_dataset("Nattakarn/fruit-and-vegetable-image-recognition", split="train")
    names = ds.features["label"].names

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    per = collections.Counter()
    saved = 0
    for i, ex in enumerate(ds):
        cls = names[ex["label"]]
        if per[cls] >= PER_CLASS:
            continue
        im = ex["image"].convert("RGB")
        if max(im.size) > MAXSIDE:
            s = MAXSIDE / max(im.size)
            im = im.resize((int(im.size[0] * s), int(im.size[1] * s)))
        im.save(os.path.join(OUT, "organic_%s_%04d.jpg" % (cls.replace(" ", "_"), i)))
        per[cls] += 1
        saved += 1
    print("saved %d real organic images (<=%d per type) -> %s" % (saved, PER_CLASS, OUT))


if __name__ == "__main__":
    main()
