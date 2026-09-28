import cv2
import numpy as np

def white_balance(image):
    # scale each color channel toward the same average brightness
    float_image = image.astype(np.float32)
    channel_means = float_image.reshape(-1, 3).mean(axis=0)
    target_mean = channel_means.mean()
    balanced = float_image * (target_mean / (channel_means + 1e-6))
    return np.clip(balanced, 0, 255).astype(np.uint8)

def improve_contrast(image):
    # adjust brightness without changing the color channels
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    contrast_filter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = contrast_filter.apply(lightness)
    improved = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(improved, cv2.COLOR_LAB2BGR)

def normalize_lighting(image):
    balanced = white_balance(image)
    return improve_contrast(balanced)
