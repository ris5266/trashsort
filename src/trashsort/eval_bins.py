import os
import csv

import cv2

from . import config
from .bins import BINS
from .infer import Classifier
from .clip_recognizer import ClipRecognizer
from .frame import ObjectFramer

# reverse map: bin display name -> bin key
NAME_TO_KEY = {v["name"]: k for k, v in BINS.items()}


def load_labels():
    path = os.path.join(config.EVAL_DIR, "labels.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    rows = load_labels()
    clf = Classifier()
    framer = ObjectFramer()
    recognizer = ClipRecognizer()

    total = 0
    correct = 0
    framed = 0
    # per-stage tallies: [correct, total]
    stage = {"recognizer": [0, 0], "classifier": [0, 0]}
    misses = []

    for r in rows:
        img_path = os.path.join(config.EVAL_DIR, "images", r["file"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        true_bin = r["bin"]

        # same steps as the real pipeline, but we track which stage answered
        target = img
        bbox = framer.best_bbox(img)
        if bbox is not None:
            framed += 1
            target = framer.crop(img, bbox)

        res = recognizer.recognize(target)
        if res["ok"]:
            name, conf, bin_info = res["item"], res["conf"], res["bin"]
            which = "recognizer"
        else:
            name, conf, bin_info = clf.predict(target)
            which = "classifier"

        pred_bin = NAME_TO_KEY.get(bin_info["name"], "?")
        ok = pred_bin == true_bin
        total += 1
        correct += ok
        stage[which][1] += 1
        stage[which][0] += ok
        if not ok:
            misses.append((r["file"], true_bin, pred_bin, which, name))

    print("bin accuracy: %d/%d = %.1f%%" % (correct, total, 100.0 * correct / max(total, 1)))
    print("framer found object: %d/%d" % (framed, total))
    for k, (c, t) in stage.items():
        if t:
            print("  via %-10s %d/%d = %.1f%%" % (k, c, t, 100.0 * c / t))

    print("\nmisses (file | true -> pred | stage | named):")
    for m in misses:
        print("  %s | %s -> %s | %s | %s" % m)


if __name__ == "__main__":
    main()
