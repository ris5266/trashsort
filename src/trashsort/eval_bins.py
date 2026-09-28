import os
import csv

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from . import config
from .bins import BINS, BIN_NAME_TO_KEY
from .pipeline import MaterialClassifier
from .clip_recognizer import ClipRecognizer
from .frame import ObjectFramer

# confusion matrix over the whole pipeline
def save_confusion(y_true, y_pred):
    labels = list(BINS.keys())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("predicted bin")
    plt.ylabel("true bin")
    for row in range(len(labels)):
        for column in range(len(labels)):
            if matrix[row, column]:
                plt.text(
                    column,
                    row,
                    matrix[row, column],
                    ha="center",
                    va="center",
                )
    plt.colorbar()
    plt.tight_layout()
    output_path = os.path.join(config.CHECKPOINT_DIR, "bin_confusion.png")
    plt.savefig(output_path, dpi=120)
    print("saved", output_path)


def load_labels():
    path = os.path.join(config.EVAL_DIR, "labels.csv")
    rows = []
    with open(path) as label_file:
        for row in csv.DictReader(label_file):
            rows.append(row)
    return rows


def main():
    rows = load_labels()
    material_classifier = MaterialClassifier()
    object_framer = ObjectFramer()
    item_recognizer = ClipRecognizer()

    total = 0
    correct = 0
    framed = 0
    # each stage stores its correct count and total count
    stage_counts = {"recognizer": [0, 0], "classifier": [0, 0]}
    misses = []
    y_true, y_pred = [], []

    for row in rows:
        image_path = os.path.join(config.EVAL_DIR, "images", row["file"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        true_bin = row["bin"]

        # mirror the app while recording which model answered
        target = image
        bbox = object_framer.best_bbox(image)
        if bbox is not None:
            framed += 1
            target = object_framer.crop(image, bbox)

        recognition = item_recognizer.recognize(target)
        if recognition["ok"]:
            name = recognition["item"]
            bin_info = recognition["bin"]
            stage_name = "recognizer"
        else:
            name, _, bin_info = material_classifier.predict(target)
            stage_name = "classifier"

        pred_bin = BIN_NAME_TO_KEY.get(bin_info["name"], "?")
        y_true.append(true_bin)
        y_pred.append(pred_bin)
        is_correct = pred_bin == true_bin
        total += 1
        correct += is_correct
        stage_counts[stage_name][1] += 1
        stage_counts[stage_name][0] += is_correct
        if not is_correct:
            misses.append((row["file"], true_bin, pred_bin, stage_name, name))

    print("bin accuracy: %d/%d = %.1f%%" % (correct, total, 100.0 * correct / max(total, 1)))
    print("framer found object: %d/%d" % (framed, total))
    for stage_name, (stage_correct, stage_total) in stage_counts.items():
        if stage_total:
            stage_accuracy = 100.0 * stage_correct / stage_total
            print(
                "  via %-10s %d/%d = %.1f%%"
                % (stage_name, stage_correct, stage_total, stage_accuracy)
            )

    print("\nmisses (file | true -> pred | stage | named):")
    for miss in misses:
        print("  %s | %s -> %s | %s | %s" % miss)

    save_confusion(y_true, y_pred)


if __name__ == "__main__":
    main()
