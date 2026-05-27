import os
import argparse

import cv2
import torch
import torch.nn.functional as F

from . import config
from .augmentation import eval_transform
from .bins import get_bin
from .model import build_model
from .preprocessing import normalize_lighting


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


def draw(frame, cls, conf, bin_info):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 86), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 86), bin_info["color"], 3)
    cv2.putText(frame, "%s (%d%%)" % (cls, conf * 100), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "-> " + bin_info["name"], (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bin_info["color"], 2)
    return frame


def run_image(path, clf):
    img = cv2.imread(path)
    cls, conf, bin_info = clf.predict(img)
    print("%s -> %s (%.1f%%) -> %s [%s]" % (os.path.basename(path), cls, conf * 100, bin_info["name"], bin_info["law"]))
    out = path.rsplit(".", 1)[0] + "_pred.jpg"
    cv2.imwrite(out, draw(img, cls, conf, bin_info))
    print("saved", out)


def run_webcam(clf, cam=0):
    cap = cv2.VideoCapture(cam)
    print("press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cls, conf, bin_info = clf.predict(frame)
        cv2.imshow("trashsort", draw(frame, cls, conf, bin_info))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    clf = Classifier()
    if args.image:
        run_image(args.image, clf)
    elif args.webcam:
        run_webcam(clf, args.cam)
    else:
        print("use --image PATH or --webcam")


if __name__ == "__main__":
    main()
