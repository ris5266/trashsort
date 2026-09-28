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
    # collect each image with the number of its material class
    samples = []
    for label, class_name in enumerate(config.CLASSES):
        folder = os.path.join(root, class_name)
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.startswith(".") or not filename.lower().endswith(EXTS):
                continue
            samples.append((os.path.join(folder, filename), label))
    return samples


def make_splits(root):
    # split every class separately to keep the class balance
    randomizer = random.Random(config.SEED)
    by_class = {i: [] for i in range(len(config.CLASSES))}
    for path, label in list_images(root):
        by_class[label].append(path)

    train, val, test = [], [], []
    for label, paths in by_class.items():
        randomizer.shuffle(paths)
        count = len(paths)
        n_test = int(round(count * config.TEST_SPLIT))
        n_val = int(round(count * config.VAL_SPLIT))
        test += [(p, label) for p in paths[:n_test]]
        val += [(p, label) for p in paths[n_test:n_test + n_val]]
        train += [(p, label) for p in paths[n_test + n_val:]]

    randomizer.shuffle(train)
    randomizer.shuffle(val)
    randomizer.shuffle(test)
    return train, val, test


class TrashDataset(Dataset):
    def __init__(self, samples, transform, use_lighting_norm=True):
        self.samples = samples
        self.transform = transform
        self.use_lighting_norm = use_lighting_norm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path)
        if image is None:
            raise ValueError("could not read image: " + path)
        if self.use_lighting_norm:
            image = normalize_lighting(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image, label


def class_weights(samples):
    class_count = len(config.CLASSES)
    counts = np.bincount(
        [label for _, label in samples],
        minlength=class_count,
    )
    counts = np.clip(counts, 1, None)
    return (counts.sum() / (class_count * counts)).astype(np.float32)
