import cv2
import gradio as gr

from .infer import Classifier, analyze
from .frame import ObjectFramer
from .clip_recognizer import ClipRecognizer
from .bins import BINS

# load the models once
CLF = Classifier()
FRAMER = ObjectFramer()
RECOG = ClipRecognizer()

def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

# run the pipeline and build the answer text
def sort(image):
    if image is None:
        return None, "Please upload an image first."

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    res = analyze(CLF, bgr, FRAMER, RECOG)
    info = res["bin"]
    annotated = annotate(bgr, res["bbox"], "%s -> %s" % (res["item"], info["name"]), info["color"])

    parts = ["## → %s  (%.0f%%)\n\n**Erkannt:** %s  \n**Gesetz:** %s" % (
        info["name"], res["conf"] * 100, res["item"], info["law"])]
    if not res["sure"] and res["alts"]:
        lines = "\n".join("- %s → %s (%.0f%%)" % (de, BINS[k]["name"], p * 100)
                          for de, k, c, p in res["alts"])
        parts.append("\n\n_Unsicher – meintest du:_\n" + lines)
    if res["n_objects"] > 1:
        parts.append("\n\n_%d Objekte gefunden, größtes klassifiziert._" % res["n_objects"])
    q = res["question"]
    if q:
        parts.append("\n\n**Rückfrage:** %s  \n→ ja: %s  \n→ nein: %s" % (
            q["text"], q["yes_bin"]["name"], q["no_bin"]["name"]))
    return to_rgb(annotated), "".join(parts)


def build():
    with gr.Blocks(title="trashsort") as demo:
        gr.Markdown("# 🗑️ trashsort\n"
                    "Upload an image of an item. The object is cut out, recognized and assigned to the correct German bin.")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload image", type="numpy", sources=["upload"], height=460)
                btn = gr.Button("Sort", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Recognition", type="numpy", format="png", height=460)
                out_md = gr.Markdown()

        btn.click(sort, [img_in], [out_img, out_md])
        # run automatically when an image is uploaded
        img_in.change(sort, [img_in], [out_img, out_md])
    return demo


def main():
    build().launch()


if __name__ == "__main__":
    main()
