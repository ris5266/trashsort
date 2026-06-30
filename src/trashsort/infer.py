import os
import argparse

import cv2
import torch
import torch.nn.functional as F

from . import config
from .augmentation import eval_transform
from .bins import get_bin, BINS
from .items import CONDITIONS
from .model import build_model
from .preprocessing import normalize_lighting


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


# build the pfand/greasy/residue follow-up for an item, if it has one
def make_question(cond, default_bin):
    if not cond or cond not in CONDITIONS:
        return None
    c = CONDITIONS[cond]
    return {"text": c["q"], "yes_bin": BINS[c["yes"]], "no_bin": default_bin}


# the full pipeline, returns one structured result:
#   1. framer cuts out the main object (and counts the rest)
#   2. clip recognizer names the item + bin
#   3. if its unsure, the material classifier guesses by material
def analyze(clf, frame, framer=None, recognizer=None):
    bbox = None
    n_objects = 1
    target = frame
    if framer is not None:
        cands = framer.candidates(frame)
        n_objects = len(cands)
        if cands:
            bbox = cands[0]
            target = framer.crop(frame, bbox)

    if recognizer is not None:
        res = recognizer.recognize(target)
        if res["ok"]:
            return {"bbox": bbox, "n_objects": n_objects,
                    "item": res["item"], "bin": res["bin"], "conf": res["conf"],
                    "sure": True, "alts": res["top"],
                    "question": make_question(res["cond"], res["bin"])}
        # clip unsure -> material guess as tentative answer, keep clips top items
        cls, conf, bin_info = clf.predict(target)
        return {"bbox": bbox, "n_objects": n_objects,
                "item": cls, "bin": bin_info, "conf": conf,
                "sure": False, "alts": res["top"], "question": None}

    cls, conf, bin_info = clf.predict(target)
    return {"bbox": bbox, "n_objects": n_objects,
            "item": cls, "bin": bin_info, "conf": conf,
            "sure": True, "alts": [], "question": None}


def print_result(name, res):
    info = res["bin"]
    tag = "" if res["sure"] else "  (unsicher)"
    print("%s -> %s (%.0f%%) [%s]  (erkannt: %s)%s" % (
        name, info["name"], res["conf"] * 100, info["law"], res["item"], tag))
    if res["n_objects"] > 1:
        print("  hinweis: %d objekte gefunden, groesstes klassifiziert" % res["n_objects"])
    if not res["sure"] and res["alts"]:
        print("  meintest du eines davon?")
        for de, key, cond, p in res["alts"]:
            print("    - %s -> %s (%.0f%%)" % (de, BINS[key]["name"], p * 100))
    q = res["question"]
    if q:
        print("  rueckfrage: %s" % q["text"])
        print("    ja   -> %s" % q["yes_bin"]["name"])
        print("    nein -> %s" % q["no_bin"]["name"])


def run_image(path, clf, framer=None, recognizer=None):
    img = cv2.imread(path)
    res = analyze(clf, img, framer, recognizer)
    print_result(os.path.basename(path), res)


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
