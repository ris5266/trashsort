import cv2
import numpy as np

def white_balance(img):
    img = img.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    img = img * (gray / (means + 1e-6))
    return np.clip(img, 0, 255).astype(np.uint8)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = c.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def normalize_lighting(img):
    return clahe(white_balance(img))
