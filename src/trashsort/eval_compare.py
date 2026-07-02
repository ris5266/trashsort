import os
import csv

import cv2

from . import config
from .bins import BINS
from .infer import Classifier
from .clip_recognizer import ClipRecognizer
from .frame import ObjectFramer

NAME_TO_KEY = {v["name"]: k for k, v in BINS.items()}

def load_labels():
    path = os.path.join(config.EVAL_DIR, "labels.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# compares the material classifier against the full pipeline
def main():
    rows = load_labels()
    clf = Classifier()
    framer = ObjectFramer()
    recognizer = ClipRecognizer()

    total = 0
    only_whole = 0
    only_crop = 0
    full = 0

    for r in rows:
        img_path = os.path.join(config.EVAL_DIR, "images", r["file"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        true_bin = r["bin"]
        total += 1

        # material classifier only, whole image
        _, _, bin_info = clf.predict(img)
        only_whole += NAME_TO_KEY.get(bin_info["name"], "?") == true_bin

        # cut out the main object
        bbox = framer.best_bbox(img)
        target = framer.crop(img, bbox) if bbox is not None else img

        # material classifier only, framed crop
        _, _, bin_info = clf.predict(target)
        only_crop += NAME_TO_KEY.get(bin_info["name"], "?") == true_bin

        # full pipeline: clip first, material classifier as fallback
        res = recognizer.recognize(target)
        if res["ok"]:
            bin_info = res["bin"]
        else:
            _, _, bin_info = clf.predict(target)
        full += NAME_TO_KEY.get(bin_info["name"], "?") == true_bin

    pct = lambda c: 100.0 * c / max(total, 1)
    print("n = %d" % total)
    print("material classifier (whole image): %d/%d = %.1f%%" % (only_whole, total, pct(only_whole)))
    print("material classifier (framed crop): %d/%d = %.1f%%" % (only_crop, total, pct(only_crop)))
    print("full pipeline (frame+clip+cnn):    %d/%d = %.1f%%" % (full, total, pct(full)))


if __name__ == "__main__":
    main()
