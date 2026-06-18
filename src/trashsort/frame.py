import cv2
import numpy as np

from ultralytics import FastSAM


# class-agnostic object framing
class ObjectFramer:
    def __init__(self, model="FastSAM-s.pt", conf=0.4):
        self.model = FastSAM(model)
        self.conf = conf

    def best_bbox(self, frame):
        h, w = frame.shape[:2]
        r = self.model(frame, retina_masks=True, conf=self.conf, iou=0.9, verbose=False)[0]
        if r.masks is None:
            return None
        best = None
        best_score = 0
        for poly in r.masks.xy:
            if len(poly) < 3:
                continue
            x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
            frac = (bw * bh) / float(w * h)
            if frac > 0.95 or frac < 0.02:
                continue
            cx, cy = x + bw / 2.0, y + bh / 2.0
            central = 1 - (abs(cx - w / 2.0) / w + abs(cy - h / 2.0) / h)
            score = bw * bh * (0.5 + central)
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh)
        return best

    def crop(self, frame, bbox, pad=0.04):
        h, w = frame.shape[:2]
        x, y, bw, bh = bbox
        p = int(pad * max(bw, bh))
        return frame[max(0, y - p):min(h, y + bh + p),
                     max(0, x - p):min(w, x + bw + p)]
