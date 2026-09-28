import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .augmentation import train_transform, eval_transform
from .dataset import TrashDataset, make_splits, class_weights, worker_init
from .model import build_model, freeze_backbone

def evaluate_model(model, loader, loss_function):
    model.eval()
    sample_count = 0
    correct_count = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            predictions = model(images)
            loss = loss_function(predictions, labels)
            total_loss += loss.item() * labels.size(0)
            correct_count += (predictions.argmax(1) == labels).sum().item()
            sample_count += labels.size(0)

    return total_loss / sample_count, correct_count / sample_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--no-lighting-norm", action="store_true")
    args = parser.parse_args()

    use_norm = not args.no_lighting_norm
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    print("device:", config.DEVICE)

    # prepare reproducible training, validation, and test groups
    train_samples, validation_samples, test_samples = make_splits(config.TRASHNET_DIR)
    if not train_samples:
        print("no images found, run scripts/download_trashnet.py first")
        return
    print("train", len(train_samples), "val", len(validation_samples), "test", len(test_samples))

    train_dataset = TrashDataset(train_samples, train_transform(), use_norm)
    validation_dataset = TrashDataset(validation_samples, eval_transform(), use_norm)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    # use class weights so small classes still matter during training
    model = build_model(len(config.CLASSES)).to(config.DEVICE)
    weights = torch.tensor(class_weights(train_samples)).to(config.DEVICE)
    loss_function = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.DEVICE.type == "cuda")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model.pt")
    best_acc = 0.0
    backbone_is_frozen = None

    for epoch in range(1, args.epochs + 1):
        should_freeze_backbone = epoch <= config.FREEZE_EPOCHS
        if should_freeze_backbone != backbone_is_frozen:
            freeze_backbone(model, should_freeze_backbone)
            backbone_is_frozen = should_freeze_backbone
            print("backbone frozen:", backbone_is_frozen)

        model.train()
        seen_count = 0
        correct_count = 0
        for images, labels in tqdm(
            train_loader,
            desc=f"epoch {epoch}",
            leave=False,
        ):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=config.DEVICE.type == "cuda"):
                predictions = model(images)
                loss = loss_function(predictions, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            correct_count += (predictions.argmax(1) == labels).sum().item()
            seen_count += labels.size(0)

        scheduler.step()
        _, validation_accuracy = evaluate_model(
            model,
            validation_loader,
            loss_function,
        )
        training_accuracy = correct_count / seen_count
        print(f"epoch {epoch} train_acc {training_accuracy:.3f} val_acc {validation_accuracy:.3f}")

        # keep only the checkpoint with the best validation score
        if validation_accuracy > best_acc:
            best_acc = validation_accuracy
            torch.save({
                "model_state": model.state_dict(),
                "classes": config.CLASSES,
                "img_size": config.IMG_SIZE,
                "use_lighting_norm": use_norm,
                "val_acc": validation_accuracy,
            }, ckpt_path)
            print("saved best", round(validation_accuracy, 3))

    split_path = os.path.join(config.CHECKPOINT_DIR, "test_split.json")
    with open(split_path, "w") as split_file:
        json.dump(test_samples, split_file)
    print("best val_acc", round(best_acc, 3))


if __name__ == "__main__":
    main()
