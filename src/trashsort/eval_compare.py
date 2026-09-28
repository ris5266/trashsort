import os
import csv

import cv2

from . import config
from .bins import BIN_NAME_TO_KEY
from .pipeline import MaterialClassifier
from .clip_recognizer import ClipRecognizer
from .frame import ObjectFramer

def load_labels():
    path = os.path.join(config.EVAL_DIR, "labels.csv")
    rows = []
    with open(path) as label_file:
        for row in csv.DictReader(label_file):
            rows.append(row)
    return rows

# compares the material classifier against the full pipeline
def main():
    rows = load_labels()
    material_classifier = MaterialClassifier()
    object_framer = ObjectFramer()
    item_recognizer = ClipRecognizer()

    total = 0
    whole_image_correct = 0
    cropped_image_correct = 0
    full_pipeline_correct = 0

    for row in rows:
        image_path = os.path.join(config.EVAL_DIR, "images", row["file"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        true_bin = row["bin"]
        total += 1

        # material classifier only
        _, _, bin_info = material_classifier.predict(image)
        predicted_bin = BIN_NAME_TO_KEY.get(bin_info["name"], "?")
        whole_image_correct += predicted_bin == true_bin

        # cut out the main object
        bbox = object_framer.best_bbox(image)
        target = object_framer.crop(image, bbox) if bbox is not None else image

        # material classifier only, framed crop
        _, _, bin_info = material_classifier.predict(target)
        predicted_bin = BIN_NAME_TO_KEY.get(bin_info["name"], "?")
        cropped_image_correct += predicted_bin == true_bin

        # full pipeline: clip first, material classifier as fallback
        recognition = item_recognizer.recognize(target)
        if recognition["ok"]:
            bin_info = recognition["bin"]
        else:
            _, _, bin_info = material_classifier.predict(target)
        predicted_bin = BIN_NAME_TO_KEY.get(bin_info["name"], "?")
        full_pipeline_correct += predicted_bin == true_bin

    def percentage(correct):
        return 100.0 * correct / max(total, 1)

    print("n = %d" % total)
    print("material classifier (whole image): %d/%d = %.1f%%" % (whole_image_correct, total, percentage(whole_image_correct)))
    print("material classifier (framed crop): %d/%d = %.1f%%" % (cropped_image_correct, total, percentage(cropped_image_correct)))
    print("full pipeline (frame+clip+cnn):    %d/%d = %.1f%%" % (full_pipeline_correct, total, percentage(full_pipeline_correct)))


if __name__ == "__main__":
    main()
