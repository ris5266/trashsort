import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from . import config
from .augmentation import eval_transform
from .dataset import TrashDataset, worker_init
from .model import build_model


def main():
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "classifier_best.pt")
    split_path = os.path.join(config.CHECKPOINT_DIR, "test_split.json")
    if not os.path.exists(ckpt_path):
        print("no model found, train first")
        return

    # load the trained model
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    classes = ckpt["classes"]
    model = build_model(ckpt["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(config.DEVICE).eval()

    with open(split_path) as f:
        test_s = [(p, l) for p, l in json.load(f)]
    ds = TrashDataset(test_s, eval_transform(ckpt["img_size"]), ckpt.get("use_lighting_norm", True))
    loader = DataLoader(ds, batch_size=64, num_workers=config.NUM_WORKERS, worker_init_fn=worker_init)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.to(config.DEVICE))
            y_pred += out.argmax(1).cpu().tolist()
            y_true += labels.tolist()

    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("predicted")
    plt.ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    out = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
