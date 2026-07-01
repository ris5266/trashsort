import os
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from . import config
from .bins import BINS
from .infer import Classifier
from .clip_recognizer import ClipRecognizer
from .frame import ObjectFramer

# reverse map: bin display name -> bin key
NAME_TO_KEY = {v["name"]: k for k, v in BINS.items()}


# confusion matrix over the whole pipeline (true bin vs predicted bin)
def save_confusion(y_true, y_pred):
    labels = list(BINS.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("predicted bin")
    plt.ylabel("true bin")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    out = os.path.join(config.CHECKPOINT_DIR, "bin_confusion.png")
    plt.savefig(out, dpi=120)
    print("saved", out)


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
    y_true, y_pred = [], []

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
        y_true.append(true_bin)
        y_pred.append(pred_bin)
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

    save_confusion(y_true, y_pred)


if __name__ == "__main__":
    main()
