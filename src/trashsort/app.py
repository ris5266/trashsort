import cv2
import gradio as gr

from .infer import Classifier, ObjectRecognizer
from .frame import ObjectFramer
from .bins import coco_bin

# load the three models
CLF = Classifier()
FRAMER = ObjectFramer()
RECOG = ObjectRecognizer()

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def union_box(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2 - x1, y2 - y1)

# box around the object + tag
def annotate(bgr, bbox, text, color):
    out = bgr.copy()
    h, w = out.shape[:2]

    (bw0, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    fs = max(0.5, min((0.85 * w) / max(bw0, 1), 2.4))
    tk = max(2, int(round(fs)))                 # text thickness
    box_th = max(2, int(max(w, h) / 400))       # box thickness
    (tw, tht), baseln = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tk)
    pad = int(8 * fs)
    th = tht + baseln + 2 * pad                 # tag height
    margin = max(4, int(0.012 * h))
    if bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, box_th)
    else:
        x, y = margin, th + margin

    top = y - th - margin
    if top < margin:
        top = y + margin
    top = max(margin, min(top, h - th - margin))
    tx = max(margin, min(x, w - tw - 2 * pad))
    cv2.rectangle(out, (tx, top), (tx + tw + 2 * pad, top + th), color, -1)
    cv2.putText(out, text, (tx + pad, top + pad + tht), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), tk)
    return out

UNKNOWN = {"name": "unbekannt", "color": (150, 150, 150), "law": "-"}

# run the pipeline for the chosen mode
def sort(image, mode):
    if image is None:
        return None, "Please upload an image first."

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    use_object = mode in ("both", "object")
    use_material = mode in ("both", "material")

    # locate the object
    dets = RECOG.detections(bgr)
    if dets:
        top_cls = max(dets, key=lambda d: d[1])[0]
        bbox = union_box([d[2] for d in dets if d[0] == top_cls])
    else:
        bbox = FRAMER.best_bbox(bgr)
    crop = FRAMER.crop(bgr, bbox) if bbox is not None else bgr

    result = None
    if use_object:
        best = None
        for name, conf, _ in dets:
            if conf < RECOG.conf or coco_bin(name) is None:
                continue
            if best is None or conf > best[1]:
                best = (name, conf, coco_bin(name))
        if best is not None:
            result = ("Object recognized", best[0], best[1], best[2])

    if result is None and use_material:
        cls, conf, info = CLF.predict(crop)
        result = ("Material", cls, conf, info)

    if result is None:
        stage, label, conf, info = "Nothing recognized", "?", 0.0, UNKNOWN
    else:
        stage, label, conf, info = result

    annotated = annotate(bgr, bbox, "%s -> %s" % (label, info["name"]), info["color"])
    md = ("## → %s  (%.0f%%)\n\n**%s:** %s  \n**Law:** %s" %(info["name"], conf * 100, stage, label, info["law"]))
    return to_rgb(annotated), md


def build():
    modes = [
        ("Object & material recognition", "both"),
        ("Object recognition only", "object"),
        ("Material recognition only", "material"),
    ]
    with gr.Blocks(title="trashsort") as demo:
        gr.Markdown("# 🗑️ trashsort\n"
                    "Upload an image of an item. The object is cut out, recognized and assigned to the correct German bin.")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload image", type="numpy", sources=["upload"], height=460)
                mode = gr.Radio(choices=modes, value="both", label="Mode")
                btn = gr.Button("Sort", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Recognition", type="numpy", format="png", height=460)
                out_md = gr.Markdown()

        btn.click(sort, [img_in, mode], [out_img, out_md])
        # run automatically when an image is uploaded
        img_in.change(sort, [img_in, mode], [out_img, out_md])
    return demo


def main():
    build().launch()


if __name__ == "__main__":
    main()
