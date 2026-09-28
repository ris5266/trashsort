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
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model.pt")
    split_path = os.path.join(config.CHECKPOINT_DIR, "test_split.json")
    if not os.path.exists(checkpoint_path):
        print("no model found, train first")
        return

    # load the trained model
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(config.DEVICE).eval()

    with open(split_path) as split_file:
        test_samples = [(path, label) for path, label in json.load(split_file)]
    dataset = TrashDataset(test_samples, eval_transform(checkpoint["img_size"]), checkpoint.get("use_lighting_norm", True))
    loader = DataLoader(dataset, batch_size=64, num_workers=config.NUM_WORKERS, worker_init_fn=worker_init)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(config.DEVICE))
            y_pred += predictions.argmax(1).cpu().tolist()
            y_true += labels.tolist()

    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    # confusion matrix
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("predicted")
    plt.ylabel("true")
    for row in range(len(classes)):
        for column in range(len(classes)):
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
            )
    plt.colorbar()
    plt.tight_layout()
    output_path = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(output_path, dpi=120)
    print("saved", output_path)


if __name__ == "__main__":
    main()
