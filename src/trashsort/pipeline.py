import os

import cv2
import torch
import torch.nn.functional as F

from . import config
from .augmentation import eval_transform
from .bins import get_bin_for_material
from .model import build_model
from .preprocessing import normalize_lighting


class MaterialClassifier:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model.pt")

        # restore the trained material model
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
        self.classes = checkpoint["classes"]
        self.use_lighting_norm = checkpoint.get("use_lighting_norm", True)
        self.transform = eval_transform(checkpoint["img_size"])
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(config.DEVICE).eval()

    def predict(self, image):
        if self.use_lighting_norm:
            image = normalize_lighting(image)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=rgb_image)["image"]
        tensor = tensor.unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            probabilities = F.softmax(self.model(tensor), dim=1)[0]

        confidence, class_index = probabilities.max(0)
        material = self.classes[class_index.item()]
        return material, confidence.item(), get_bin_for_material(material)


# full pipeline:
#   1. framer cuts out the main object
#   2. clip recognizer names the item + bin
#   3. if its unsure, the material classifier guesses by material
def classify_image(material_classifier, image, object_framer=None, item_recognizer=None, point=None):
    bbox = None
    n_objects = 1
    target = image
    if object_framer is not None:
        candidates = object_framer.candidates(image)
        n_objects = len(candidates)
        if candidates:
            bbox = object_framer.pick(candidates, point) if point is not None else candidates[0]
            target = object_framer.crop(image, bbox)

    # prefer a known item because it maps directly to a bin
    if item_recognizer is not None:
        recognition = item_recognizer.recognize(target)
        if recognition["ok"]:
            return {"bbox": bbox, "n_objects": n_objects, "item": recognition["item"], "bin": recognition["bin"], "conf": recognition["conf"], "sure": True, "alts": recognition["top"]}

        # fall back to material while keeping the closest item guesses
        material, confidence, bin_info = material_classifier.predict(target)
        return {"bbox": bbox, "n_objects": n_objects, "item": material, "bin": bin_info, "conf": confidence, "sure": False, "alts": recognition["top"]}

    material, confidence, bin_info = material_classifier.predict(target)
    return {"bbox": bbox, "n_objects": n_objects, "item": material, "bin": bin_info, "conf": confidence, "sure": True, "alts": []}
