import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config


# augmentations for training so the model sees lots of variations
def train_transform(size=config.IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0, fill=(114, 114, 114)),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=0.06, rotate=(-25, 25),
                 fill=(114, 114, 114), p=0.7),
        A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
        A.HueSaturationValue(15, 25, 15, p=0.4),
        A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=5)], p=0.2),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 24),
                        hole_width_range=(8, 24), fill=114, p=0.3),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])


def eval_transform(size=config.IMG_SIZE):
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=0, fill=(114, 114, 114)),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])
