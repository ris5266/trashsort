import os
import torch

# paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
TRASHNET_DIR = os.path.join(DATA_DIR, "trashnet")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

# the 6 trashnet classes
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# imagenet mean/std
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# hyperparams
ARCH = "efficientnet_b0"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_WORKERS = 8
SEED = 42
USE_LIGHTING_NORM = True
FREEZE_EPOCHS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
