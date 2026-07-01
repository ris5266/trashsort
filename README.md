<div align="center">

  <img src="icon.png" alt="logo" width="400px" height="400px"/>

# trashsort
</div>

### A smart trash sorter that recognizes items and tells you which German recycling bin it belongs in: built with Python, [PyTorch](https://github.com/pytorch/pytorch), [Gradio](https://github.com/gradio-app/gradio), [Ultralytics FastSAM](https://github.com/ultralytics/ultralytics) and [OpenCLIP](https://github.com/mlfoundations/open_clip), with a hand-written knowledge base of common German household items and a self-trained [EfficientNet-B0](https://github.com/garythung/trashnet) fallback

<div align="center">

  ---
  [**Features**](#features) | [**Install**](#install) | [**How it works**](#how-it-works) | [**Evolution**](#how-the-project-evolved)

  ---

</div>

## Features

🗑️ **Sorts into 9 German bins**: maps items to Papier, Gelbe Tonne, Altglas, Biomüll, Restmüll, Pfand, Sondermüll, Elektroschrott and Altkleider following the *Mülltrennung* rules

🔍 **Open-vocabulary recognition**: a **CLIP (ViT-L/14)** recognizer scores the object against a curated list of **127 household items** (`items.py`), each mapped to its bin, so it knows *specific* things (battery → Sondermüll, Tetra Pak → Gelbe Tonne, shoes → Altkleider)

🧠 **Self trained CNN as fallback**: when CLIP isn't confident, a self-trained **EfficientNet-B0** material classifier (glass, plastic, paper, cardboard, metal, organic, trash) takes over

✂️ **FastSAM segmentation**: cuts the main object out of a cluttered photo first, so the background doesn't confuse the recognition

🖥️ **All in a Gradio app**: upload an image and see what bin the object belongs to

<div align="center">

  ![Sorting items into the right German bins](demo.gif)

</div>

## Confusion matrix

<div align="center">

  <img src="confusion_matrix.png" alt="confusion matrix on the held-out test set" width="520px"/>
</div>

The self trained **EfficientNet-B0 material classifier** was tested on **737 new images** from TrashNet, the real fruit & veg photos and Freiburg Groceries packaging. It reaches **94.8% accuracy** across the 7 material classes, with `organic` (99%) and `cardboard` (96%) being the strongest.

## Install

1. **Clone the repository**
```
git clone https://github.com/ris5266/trashsort.git
cd trashsort
```

2. **Install the dependencies** (the CLIP weights, ~1.7 GB, download automatically on first run)
```
pip install -r requirements.txt
pip install -e .
```

3. *(optional)* **Train the fallback classifier**: download the datasets and train the EfficientNet material model
```
python scripts/download_trashnet.py     # TrashNet dataset
python scripts/download_organic.py      # organic dataset
python scripts/download_groceries.py    # supermarket dataset
python scripts/add_groceries.py         # map packaging -> material classes
python -m trashsort.train
```

4. *(optional)* **Fetch the eval images** and run the benchmark
```
python scripts/build_eval.py
python -m trashsort.eval_bins
```

5. **Launch the gradio app**
```
python -m trashsort.app
```
or run a single image from the command line:
```
python -m trashsort.infer --image path/to/photo.jpg
```

## How it works

When you upload an image, it runs through a three-stage pipeline:

1. **Cut out the object:** the pretrained **FastSAM** segments the main object and crops it out of the background, so a messy phone photo (object on a desk, bad lighting) becomes a clean cut-out.

2. **Recognize the item:** **CLIP (ViT-L/14)** scores the crop against my **127-item knowledge base**. If it's confident, the named item maps straight to its bin. Ambiguous items raise a follow-up question (`e.g. deposit bottle → Hat es ein Pfand-Logo? → Pfand`) and when it's unsure it returns its top-3 guesses instead of forcing one.

3. **Classify the material:** if CLIP isn't confident, my trained **EfficientNet-B0** CNN, predicts the *material* of the crop `(cardboard, glass, metal, organic, paper, plastic, trash)`, which maps to a bin.

### German bins it maps to

| Bin (`bins.py`) | German               | Typical contents                      |
|-----------------|----------------------|---------------------------------------|
| `papier`        | Blaue Tonne          | paper, cardboard, boxes               |
| `gelbe_tonne`   | Gelbe Tonne / Sack   | plastic & metal packaging, composites |
| `altglas`       | Glascontainer        | glass bottles & jars                  |
| `biomuell`      | Braune Tonne         | food / organic waste                  |
| `restmuell`     | Schwarze Tonne       | residual waste                        |
| `pfand`         | Pfandrückgabe        | deposit bottles & cans                |
| `sondermuell`   | Schadstoffsammlung   | batteries, chemicals, paint, oil      |
| `elektroschrott`| —                    | electronics / e-waste                 |
| `altkleider`    | Altkleidercontainer  | wearable clothes & shoes              |

## How the project evolved

1. **TrashNet** had no food category, so organic waste landed in "trash" → added an `organic` class to the **EfficientNet-B0** classifier.

2. A classifier must pick one of its known classes for *everything* → added a pretrained **YOLOv8-seg (COCO)** object detector for things like electronics + a **confidence gate** for unknowns.

3. Fine-tuned **YOLOv8-seg** on the **TACO** litter dataset to output bins directly, but TACO has almost no food/glass (later dropped).

4. A whole-image classifier gets confused by the background → cut out the object first with **FastSAM**.

5. Dataset missing everyday items (e.g. chips bags) → added the **Freiburg Groceries** dataset, each product mapped to its material.

6. The bin choice was still hit-or-miss: **COCO** only knows 80 generic classes and rarely matched real trash (a bottle became a *"vase"* → Restmüll) → replaced the YOLO detector with an **open-vocabulary CLIP recognizer** scored against a curated list of German household items.

7. Built a **hand-verified 100-image eval set** to measure real accuracy → material-only **47%**, full CLIP pipeline **85%**.

8. Added top-3 when unsure, follow-up questions, multi-object and a new **Altkleider** bin for clothes & shoes.
