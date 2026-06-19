import os
import argparse

import cv2
import torch
import torch.nn.functional as F

from . import config
from .augmentation import eval_transform
from .bins import get_bin, coco_bin
from .model import build_model
from .preprocessing import normalize_lighting


class ObjectRecognizer:
    def __init__(self, model="yolov8s-seg.pt", conf=0.65):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf

    def best(self, img):
        r = self.model.predict(img, conf=0.25, verbose=False)[0]
        best = None
        for b in r.boxes:
            c = float(b.conf)
            if c < self.conf:
                continue
            name = self.model.names[int(b.cls)]
            info = coco_bin(name)
            if info is None:
                continue
            if best is None or c > best[1]:
                best = (name, c, info)
        return best


class Classifier:
    def __init__(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_best.pt")
        # load the model
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        self.classes = ckpt["classes"]
        self.use_norm = ckpt.get("use_lighting_norm", True)
        self.transform = eval_transform(ckpt["img_size"])
        self.model = build_model(ckpt["arch"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(config.DEVICE).eval()

    def predict(self, img):
        if self.use_norm:
            img = normalize_lighting(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.transform(image=rgb)["image"].unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        cls = self.classes[idx.item()]
        return cls, conf.item(), get_bin(cls)


def draw(frame, cls, conf, bin_info, bbox=None):
    w = frame.shape[1]
    color = bin_info["color"]
    cv2.rectangle(frame, (0, 0), (w, 86), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 86), color, 3)
    cv2.putText(frame, "%s (%d%%)" % (bin_info["name"], conf * 100), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "(erkannt: %s)" % cls, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
    return frame


# the full pipeline:
#   1. framer cuts out the main object
#   2. object recognizer tries to name it (e.g. electronics, fruit, bottles...)
#   3. if it cant, the material classifier judges by material
def classify(clf, frame, framer=None, recognizer=None):
    bbox = None
    target = frame
    if framer is not None:
        bbox = framer.best_bbox(frame)
        if bbox is not None:
            target = framer.crop(frame, bbox)

    # specific object recognition
    if recognizer is not None:
        hit = recognizer.best(target)
        if hit is not None:
            name, conf, bin_info = hit
            return name, conf, bin_info, bbox

    # material classifier fallback
    cls, conf, bin_info = clf.predict(target)
    return cls, conf, bin_info, bbox


def run_image(path, clf, framer=None, recognizer=None):
    img = cv2.imread(path)
    cls, conf, bin_info, bbox = classify(clf, img, framer, recognizer)
    print("%s -> %s (%.1f%%) [%s]  (erkannt: %s)" % (os.path.basename(path), bin_info["name"], conf * 100, bin_info["law"], cls))
    out = path.rsplit(".", 1)[0] + "_pred.jpg"
    cv2.imwrite(out, draw(img, cls, conf, bin_info, bbox))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--no-frame", action="store_true", help="dont auto-frame the object, classify the whole image")
    ap.add_argument("--no-detect", action="store_true", help="skip object recognition, only use the material classifier")
    args = ap.parse_args()

    clf = Classifier()
    framer = None
    if not args.no_frame:
        from .frame import ObjectFramer
        framer = ObjectFramer()
    recognizer = None
    if not args.no_detect:
        recognizer = ObjectRecognizer()

    run_image(args.image, clf, framer, recognizer)


if __name__ == "__main__":
    main()
