import cv2
import gradio as gr

from .pipeline import MaterialClassifier, classify_image
from .frame import ObjectFramer
from .clip_recognizer import ClipRecognizer
from .bins import BINS

# load the models when the app starts
MATERIAL_CLASSIFIER = MaterialClassifier()
OBJECT_FRAMER = ObjectFramer()
ITEM_RECOGNIZER = ClipRecognizer()

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# draw a box + label around the chosen object
def annotate(image, bbox, text, color):
    output = image.copy()
    height, width = output.shape[:2]

    # shrink long labels until they fit inside the image
    (base_text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    font_scale = max(0.5, min((0.85 * width) / max(base_text_width, 1), 2.4))
    text_thickness = max(2, int(round(font_scale)))
    box_thickness = max(2, int(max(width, height) / 400))
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    padding = int(8 * font_scale)
    tag_height = text_height + baseline + 2 * padding
    margin = max(4, int(0.012 * height))

    if bbox is not None:
        x, y, box_width, box_height = bbox
        cv2.rectangle(
            output,
            (x, y),
            (x + box_width, y + box_height),
            color,
            box_thickness,
        )
    else:
        x, y = margin, tag_height + margin

    # place the label above the object, or below it when space is tight
    top = y - tag_height - margin
    if top < margin:
        top = y + margin
    top = max(margin, min(top, height - tag_height - margin))
    left = max(margin, min(x, width - text_width - 2 * padding))
    cv2.rectangle(
        output,
        (left, top),
        (left + text_width + 2 * padding, top + tag_height),
        color,
        -1,
    )
    cv2.putText(
        output,
        text,
        (left + padding, top + padding + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness,
    )
    return output

# run the pipeline
def sort_image(image, selected_point=None, mode="full"):
    if image is None:
        return None, "Please upload an image first."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    recognizer = ITEM_RECOGNIZER if mode == "full" else None
    result = classify_image(
        MATERIAL_CLASSIFIER,
        image_bgr,
        OBJECT_FRAMER,
        recognizer,
        point=selected_point,
    )
    bin_info = result["bin"]
    label = f'{result["item"]} -> {bin_info["name"]}'
    annotated = annotate(
        image_bgr,
        result["bbox"],
        label,
        bin_info["color"],
    )

    confidence = result["conf"] * 100
    parts = [f'## → {bin_info["name"]}  ({confidence:.0f}%)\n\n**Erkannt:** {result["item"]}  \n**Gesetz:** {bin_info["law"]}']
    if not result["sure"] and result["alts"]:
        alternatives = "\n".join(f'- {item} → {BINS[bin_key]["name"]} ({score * 100:.0f}%)' for item, bin_key, score in result["alts"])
        parts.append("\n\n_Unsicher – meintest du:_\n" + alternatives)

    if result["n_objects"] > 1:
        if selected_point is not None:
            parts.append(f'\n\n_{result["n_objects"]} Objekte gefunden, angeklicktes klassifiziert._')
        else:
            parts.append(f'\n\n_{result["n_objects"]} Objekte gefunden, größtes klassifiziert. Klicke ein Objekt im Bild an, um es auszuwählen._')

    return bgr_to_rgb(annotated), "".join(parts)

def sort_selected_object(image, mode, event: gr.SelectData):
    selected_point = (event.index[0], event.index[1])
    output_image, output_text = sort_image(image, selected_point, mode)
    return selected_point, output_image, output_text

def sort_uploaded_image(image, mode):
    output_image, output_text = sort_image(image, None, mode)
    return None, output_image, output_text

CSS = """
#img_in, #out_img { height: 460px !important; }
#img_in img, #out_img img { height: 460px !important; object-fit: contain; }
#sort_btn { height: 48px !important; flex-grow: 0 !important; }
"""

def build_app():
    with gr.Blocks(title="trashsort") as demo:
        gr.HTML("<style>%s</style>" % CSS)
        gr.Markdown("# 🗑️ trashsort\nUpload an image of an item. The object is cut out, recognized and assigned to the correct German bin.")
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

        btn.click(sort_image, [img_in, point, mode], [out_img, out_md])
        img_in.select(sort_selected_object, [img_in, mode], [point, out_img, out_md])
        img_in.upload(sort_uploaded_image, [img_in, mode], [point, out_img, out_md])
        img_in.clear(lambda: (None, None, ""), None, [point, out_img, out_md])
        mode.change(sort_image, [img_in, point, mode], [out_img, out_md])
    return demo


def main():
    build_app().launch()


if __name__ == "__main__":
    main()
