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
def sort(image, point=None, mode="full"):
    if image is None:
        return None, "Please upload an image first."

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    recog = RECOG if mode == "full" else None
    res = analyze(CLF, bgr, FRAMER, recog, point=point)
    info = res["bin"]
    annotated = annotate(bgr, res["bbox"], "%s -> %s" % (res["item"], info["name"]), info["color"])

    parts = ["## → %s  (%.0f%%)\n\n**Erkannt:** %s  \n**Gesetz:** %s" % (
        info["name"], res["conf"] * 100, res["item"], info["law"])]
    if not res["sure"] and res["alts"]:
        lines = "\n".join("- %s → %s (%.0f%%)" % (de, BINS[k]["name"], p * 100)
                          for de, k, p in res["alts"])
        parts.append("\n\n_Unsicher – meintest du:_\n" + lines)
    if res["n_objects"] > 1:
        if point is not None:
            parts.append("\n\n_%d Objekte gefunden, angeklicktes klassifiziert._" % res["n_objects"])
        else:
            parts.append("\n\n_%d Objekte gefunden, größtes klassifiziert. "
                         "Klicke ein Objekt im Bild an, um es auszuwählen._" % res["n_objects"])
    return to_rgb(annotated), "".join(parts)

def sort_at(image, mode, evt: gr.SelectData):
    point = (evt.index[0], evt.index[1])
    out_img, out_md = sort(image, point, mode)
    return point, out_img, out_md

def sort_fresh(image, mode):
    out_img, out_md = sort(image, None, mode)
    return None, out_img, out_md

CSS = """
#img_in, #out_img { height: 460px !important; }
#img_in img, #out_img img { height: 460px !important; object-fit: contain; }
#sort_btn { height: 48px !important; flex-grow: 0 !important; }
"""

def build():
    with gr.Blocks(title="trashsort") as demo:
        gr.HTML("<style>%s</style>" % CSS)
        gr.Markdown("# 🗑️ trashsort\n"
                    "Upload an image of an item. The object is cut out, recognized and assigned to the correct German bin.")
        point = gr.State(None)
        modes = [
            ("Object & material recognition", "full"),
            ("Material recognition only", "material"),
        ]
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload image", type="numpy", sources=["upload"], height=460, elem_id="img_in")
                mode = gr.Radio(choices=modes, value="full", label="Mode")
                btn = gr.Button("Sort", variant="primary", elem_id="sort_btn")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Recognition", type="numpy", format="png", height=460, elem_id="out_img")
                out_md = gr.Markdown()

        btn.click(sort, [img_in, point, mode], [out_img, out_md])
        img_in.select(sort_at, [img_in, mode], [point, out_img, out_md])
        img_in.upload(sort_fresh, [img_in, mode], [point, out_img, out_md])
        img_in.clear(lambda: (None, None, ""), None, [point, out_img, out_md])
        mode.change(sort, [img_in, point, mode], [out_img, out_md])
    return demo


def main():
    build().launch()


if __name__ == "__main__":
    main()
