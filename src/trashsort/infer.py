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

    def detections(self, img, conf=0.3):
        r = self.model.predict(img, conf=conf, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((self.model.names[int(b.cls)], float(b.conf),
                        (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))
        return out


class Classifier:
    def __init__(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model.pt")
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


# the full pipeline:
#   1. framer cuts out the main object
#   2. clip recognizer tries to name the actual item and its bin
#   3. if its unsure, the material classifier judges by material
def classify(clf, frame, framer=None, recognizer=None):
    bbox = None
    target = frame
    if framer is not None:
        bbox = framer.best_bbox(frame)
        if bbox is not None:
            target = framer.crop(frame, bbox)

    # open-vocabulary item recognition
    if recognizer is not None:
        res = recognizer.recognize(target)
        if res["ok"]:
            return res["item"], res["conf"], res["bin"], bbox

    # material classifier fallback (clip was unsure)
    cls, conf, bin_info = clf.predict(target)
    return cls, conf, bin_info, bbox


def run_image(path, clf, framer=None, recognizer=None):
    img = cv2.imread(path)
    cls, conf, bin_info, _ = classify(clf, img, framer, recognizer)
    print("%s -> %s (%.1f%%) [%s]  (erkannt: %s)" % (os.path.basename(path), bin_info["name"], conf * 100, bin_info["law"], cls))


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
        from .clip_recognizer import ClipRecognizer
        recognizer = ClipRecognizer()

    run_image(args.image, clf, framer, recognizer)


if __name__ == "__main__":
    main()
