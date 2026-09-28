import torch
import cv2
from PIL import Image
import open_clip

from . import config
from .items import ITEMS, DECOYS
from .bins import BINS

TEMPLATE = "a photo of %s."

# scores a crop against the item list
class ClipRecognizer:
    def __init__(self):
        model, _, preprocess = open_clip.create_model_and_transforms(config.CLIP_ARCH, pretrained=config.CLIP_PRETRAINED)
        self.model = model.to(config.DEVICE).eval()
        self.preprocess = preprocess
        tokenizer = open_clip.get_tokenizer(config.CLIP_ARCH)

        # item labels come first so their indexes match the item list
        self.items = ITEMS
        self.n_items = len(ITEMS)
        labels = [TEMPLATE % i["en"] for i in ITEMS] + [TEMPLATE % d for d in DECOYS]
        tokens = tokenizer(labels).to(config.DEVICE)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    def recognize(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.preprocess(Image.fromarray(image_rgb))
        image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = self.model.logit_scale.exp() * image_features @ self.text_features.T
            probabilities = scores.softmax(dim=-1)[0]

        ranked_indices = probabilities.argsort(descending=True)
        best_index = ranked_indices[0].item()
        is_decoy = best_index >= self.n_items

        # keep three item guesses for uncertain results
        top_matches = []
        for index in ranked_indices.tolist():
            if index >= self.n_items:
                continue
            item = self.items[index]
            top_matches.append(
                (item["de"], item["bin"], probabilities[index].item())
            )
            if len(top_matches) == 3:
                break

        confidence = probabilities[best_index].item()
        runner_up = probabilities[ranked_indices[1]].item()
        margin = confidence - runner_up
        is_confident = (
            not is_decoy
            and confidence >= config.CLIP_THRESH
            and margin >= config.CLIP_MARGIN
        )

        if is_decoy:
            # a background match means there may be no clear trash item
            return {"ok": False, "conf": confidence, "top": top_matches}

        item = self.items[best_index]
        return {
            "ok": is_confident,
            "item": item["de"],
            "bin": BINS[item["bin"]],
            "conf": confidence,
            "top": top_matches,
        }
