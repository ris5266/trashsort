import os

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
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model.pt")
        # load the model
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        self.classes = ckpt["classes"]
        self.use_norm = ckpt.get("use_lighting_norm", True)
        self.transform = eval_transform(ckpt["img_size"])
        self.model = build_model(len(self.classes), pretrained=False)
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


# the full pipeline, returns one structured result:
#   1. framer cuts out the main object (and counts the rest)
#   2. clip recognizer names the item + bin
#   3. if its unsure, the material classifier guesses by material
def analyze(clf, frame, framer=None, recognizer=None, point=None):
    bbox = None
    n_objects = 1
    target = frame
    if framer is not None:
        cands = framer.candidates(frame)
        n_objects = len(cands)
        if cands:
            bbox = framer.pick(cands, point) if point is not None else cands[0]
            target = framer.crop(frame, bbox)

    if recognizer is not None:
        res = recognizer.recognize(target)
        if res["ok"]:
            return {"bbox": bbox, "n_objects": n_objects,
                    "item": res["item"], "bin": res["bin"], "conf": res["conf"],
                    "sure": True, "alts": res["top"]}
        # clip unsure -> material guess as tentative answer, keep clips top items
        cls, conf, bin_info = clf.predict(target)
        return {"bbox": bbox, "n_objects": n_objects,
                "item": cls, "bin": bin_info, "conf": conf,
                "sure": False, "alts": res["top"]}

    cls, conf, bin_info = clf.predict(target)
    return {"bbox": bbox, "n_objects": n_objects,
            "item": cls, "bin": bin_info, "conf": conf,
            "sure": True, "alts": []}
