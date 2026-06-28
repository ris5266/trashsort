<div align="center">

  <img src="icon.png" alt="logo" width="400px" height="400px"/>

# trashsort
</div>

### A smart trash sorter that recognizes items and tells you which German recycling bin it belongs in: built with Python, [PyTorch](https://github.com/pytorch/pytorch), [Gradio](https://github.com/gradio-app/gradio) and [Ultralytics YOLO + FastSAM](https://github.com/ultralytics/ultralytics), trained on [TrashNet](https://github.com/garythung/trashnet), [real fruit & vegetable photos](https://huggingface.co/datasets/Nattakarn/fruit-and-vegetable-image-recognition) and the [Freiburg Groceries dataset](http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/)

<div align="center">

  ---
  [**Features**](#features) | [**Install**](#install) | [**How it works**](#how-it-works) | [**Evolution**](#how-the-project-evolved)

  ---

</div>

## Features

🗑️ **Sorts into German bins**: maps items to Papier, Gelbe Tonne, Altglas, Biomüll, Restmüll and Elektroschrott following the *Mülltrennung* rules

🔍 **Two-stage recognition**: a pretrained **YOLOv8-seg (COCO)** detector names things it knows (e.g. banana → Biomüll), and my own **EfficientNet-B0** classifier handles everything else by material (glass, plastic, paper, cardboard, metal, organic, trash)

🧠 **Train your own classifier**: retrain the **EfficientNet-B0** material model on the provided datasets (**TrashNet**, fruit & veg, **Freiburg Groceries**)

🖥️ **All in a Gradio app**: upload an image, pick a mode and see what bin the object belongs to

<div align="center">

  ![Sorting items into the right German bins](demo.gif)

</div>

## Confusion matrix

<div align="center">

  <img src="confusion_matrix.png" alt="confusion matrix on the held-out test set" width="520px"/>
</div>

The trained material classifier was tested on **737 new images** that the model never saw during training, drawn from **TrashNet**, the **real fruit & veg** photos and **Freiburg Groceries** packaging. It reaches **94.8% accuracy** across the 7 material classes, with `organic` (99%) and `cardboard` (96%) being the strongest.

Elektroschrott isn't here: electronics are caught by the **YOLOv8-seg (COCO)** detector, not the classifier.


## Install

1. **Clone the repository**
```
git clone https://github.com/ris5266/trashsort.git
cd trashsort
```

2. **Install the dependencies**
```
pip install -r requirements.txt
pip install -e .
```

3. **Train the classifier**: download the datasets and train the material model
```
python scripts/download_trashnet.py     # TrashNet dataset
python scripts/download_organic.py      # organic dataset
python scripts/download_groceries.py    # supermarket dataset
python scripts/add_groceries.py         # map packaging -> material classes
python -m trashsort.train
```

4. **Launch the app**
```
python -m trashsort.app
```

## How it works

When you upload an image, it runs through a three-stage process:

1. **Cut out the object:** the pretrained, off-the-shelf **YOLOv8-seg (COCO)** detector localizes objects with bounding boxes (good for transparent objects like bottles). For anything it doesn't recognize, the pretrained **FastSAM** frames the main object instead.

2. **Recognize the object:** if the **YOLO** detector names a waste object *(above a confidence gate)*, it maps straight to a bin (`e.g. banana → Biomüll`).

3. **Classify the material:** otherwise my trained **EfficientNet-B0** CNN, fine-tuned on **TrashNet**, **real fruit & vegetable** photos and **Freiburg Groceries**, predicts the *material* of the crop `(cardboard, glass, metal, organic, paper, plastic, trash)`, which maps to a bin.

### German bins it maps to

| Bin (`bins.py`) | German            | Typical contents                      |
|-----------------|-------------------|---------------------------------------|
| `papier`        | Blaue Tonne       | paper, cardboard, boxes               |
| `gelbe_tonne`   | Gelbe Tonne / Sack| plastic & metal packaging, composites |
| `altglas`       | Glascontainer     | glass bottles & jars                  |
| `biomuell`      | Braune Tonne      | food / organic waste                  |
| `restmuell`     | Schwarze Tonne    | residual waste                        |
| `elektroschrott`| —                 | electronics / e-waste                 |
| `pfand` / `sondermuell` | —         | deposit / hazardous                   |

## How the project evolved

1. **TrashNet** had no food category, so organic waste landed in "trash" → added an `organic` class to the **EfficientNet-B0** classifier.

2. A classifier must pick one of its known classes for *everything* → added a pretrained **YOLOv8-seg (COCO)** object detector for things like electronics + a **confidence gate** for unknowns.

3. Fine-tuned **YOLOv8-seg** on the **TACO** litter dataset to output bins directly, but TACO has almost no food/glass (later dropped).

4. The **EfficientNet-B0** classifier became the core, retrained on real **fruit & veg** photos.

5. Dataset missing everyday items (e.g. chips bags) → added the **Freiburg Groceries** dataset, each product mapped to its material (chips → plastic, cans → metal, jars → glass).

6. A whole-image classifier gets confused by the background → cut out the object first with **FastSAM**.

7. The two **cascade**: the **YOLO** detector names the object if it can, the **EfficientNet** classifier decides by material otherwise.

8. Glass bottles went to Gelbe Tonne → bottles defer to the **EfficientNet** classifier: **glass → Altglas**, **plastic → Gelbe Tonne**.
