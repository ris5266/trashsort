import os
import torch

# paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
TRASHNET_DIR = os.path.join(DATA_DIR, "trashnet")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

# trashnet classes + organic
CLASSES = ["cardboard", "glass", "metal", "organic", "paper", "plastic", "trash"]

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

# below this confidence the classifier says "unsure"
CONF_THRESH = 0.60

# clip recognizer
CLIP_ARCH = "ViT-L-14"
CLIP_PRETRAINED = "openai"
CLIP_THRESH = 0.28   # top prob must beat this or we say unsure (tuned on data/eval)
CLIP_MARGIN = 0.07   # top1 must beat top2 by this much (tuned on data/eval)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
