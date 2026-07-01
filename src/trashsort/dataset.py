import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from . import config
from .preprocessing import normalize_lighting

EXTS = (".jpg", ".jpeg", ".png", ".bmp")

cv2.setNumThreads(0)


def worker_init(_):
    cv2.setNumThreads(0)


def list_images(root):
    # go through every class folder and collect
    samples = []
    for label, cls in enumerate(config.CLASSES):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.startswith(".") or not f.lower().endswith(EXTS):
                continue
            samples.append((os.path.join(folder, f), label))
    return samples


def make_splits(root):
    # split each class separately
    rng = random.Random(config.SEED)
    by_class = {i: [] for i in range(len(config.CLASSES))}
    for path, label in list_images(root):
        by_class[label].append(path)

    train, val, test = [], [], []
    for label, paths in by_class.items():
        rng.shuffle(paths)
        n = len(paths)
        n_test = int(round(n * config.TEST_SPLIT))
        n_val = int(round(n * config.VAL_SPLIT))
        test += [(p, label) for p in paths[:n_test]]
        val += [(p, label) for p in paths[n_test:n_test + n_val]]
        train += [(p, label) for p in paths[n_test + n_val:]]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class TrashDataset(Dataset):
    def __init__(self, samples, transform, use_lighting_norm=True):
        self.samples = samples
        self.transform = transform
        self.use_lighting_norm = use_lighting_norm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path)  # bgr
        if img is None:
            raise ValueError("could not read image: " + path)
        if self.use_lighting_norm:
            img = normalize_lighting(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, label


def class_weights(samples):
    n = len(config.CLASSES)
    counts = np.bincount([lbl for _, lbl in samples], minlength=n)
    counts = np.clip(counts, 1, None)
    return (counts.sum() / (n * counts)).astype(np.float32)
