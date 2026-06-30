import torch
import cv2
from PIL import Image
import open_clip

from . import config
from .items import ITEMS, DECOYS
from .bins import BINS

TEMPLATE = "a photo of %s."


# open-vocabulary recognizer: scores a crop against our known item list
class ClipRecognizer:
    def __init__(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_ARCH, pretrained=config.CLIP_PRETRAINED)
        self.model = model.to(config.DEVICE).eval()
        self.preprocess = preprocess
        tokenizer = open_clip.get_tokenizer(config.CLIP_ARCH)

        # items first, decoys after -> index tells us which is which
        self.items = ITEMS
        self.n_items = len(ITEMS)
        labels = [TEMPLATE % i["en"] for i in ITEMS] + [TEMPLATE % d for d in DECOYS]
        tok = tokenizer(labels).to(config.DEVICE)
        with torch.no_grad():
            feats = self.model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self.text_feats = feats

    def recognize(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * f @ self.text_feats.T
            probs = logits.softmax(dim=-1)[0]

        order = probs.argsort(descending=True)
        best = order[0].item()
        is_decoy = best >= self.n_items

        # top item guesses (decoys skipped) for the unsure / pick-one case
        top = []
        for idx in order.tolist():
            if idx >= self.n_items:
                continue
            it = self.items[idx]
            top.append((it["de"], it["bin"], it.get("cond"), probs[idx].item()))
            if len(top) == 3:
                break

        conf = probs[best].item()
        margin = conf - probs[order[1]].item()
        sure = (not is_decoy) and conf >= config.CLIP_THRESH and margin >= config.CLIP_MARGIN

        if is_decoy:
            # best match was a desk/hand/wall -> probably not a trash item
            return {"ok": False, "reason": "no_item", "conf": conf, "top": top}

        it = self.items[best]
        return {
            "ok": sure,
            "reason": "ok" if sure else "unsure",
            "item": it["de"],
            "bin": BINS[it["bin"]],
            "cond": it.get("cond"),
            "conf": conf,
            "top": top,
        }
