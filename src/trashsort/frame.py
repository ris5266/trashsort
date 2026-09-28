import cv2
import numpy as np

from ultralytics import FastSAM

# class-agnostic object framing
class ObjectFramer:
    def __init__(self, model="FastSAM-s.pt", conf=0.4):
        self.model = FastSAM(model)
        self.conf = conf

    # find useful object boxes and rank large central ones first
    def candidates(self, frame):
        height, width = frame.shape[:2]
        result = self.model(
            frame,
            retina_masks=True,
            conf=self.conf,
            iou=0.9,
            verbose=False,
        )[0]
        if result.masks is None:
            return []

        scored = []
        for polygon in result.masks.xy:
            if len(polygon) < 3:
                continue

            x, y, box_width, box_height = cv2.boundingRect(
                polygon.astype(np.int32)
            )
            area_fraction = (box_width * box_height) / float(width * height)
            if area_fraction > 0.95 or area_fraction < 0.02:
                continue

            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            center_score = 1 - (
                abs(center_x - width / 2.0) / width
                + abs(center_y - height / 2.0) / height
            )
            score = box_width * box_height * (0.5 + center_score)
            scored.append((score, (x, y, box_width, box_height)))

        scored.sort(key=lambda candidate: candidate[0], reverse=True)
        return [box for _, box in scored]

    def best_bbox(self, frame):
        c = self.candidates(frame)
        return c[0] if c else None

    # choose the smallest box under the click, or the nearest box
    def pick(self, candidates, point):
        point_x, point_y = point
        boxes_under_point = [
            box for box in candidates
            if box[0] <= point_x <= box[0] + box[2]
            and box[1] <= point_y <= box[1] + box[3]
        ]
        if boxes_under_point:
            return min(boxes_under_point, key=lambda box: box[2] * box[3])

        def distance_to_center(box):
            center_x = box[0] + box[2] / 2
            center_y = box[1] + box[3] / 2
            return (center_x - point_x) ** 2 + (center_y - point_y) ** 2

        return min(candidates, key=distance_to_center)

    def crop(self, frame, bbox, pad=0.04):
        height, width = frame.shape[:2]
        x, y, box_width, box_height = bbox
        padding = int(pad * max(box_width, box_height))
        top = max(0, y - padding)
        bottom = min(height, y + box_height + padding)
        left = max(0, x - padding)
        right = min(width, x + box_width + padding)
        return frame[top:bottom, left:right]
