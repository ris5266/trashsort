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


def evaluate(model, loader, criterion):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=config.ARCH)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--no-lighting-norm", action="store_true")
    args = ap.parse_args()

    use_norm = not args.no_lighting_norm
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    print("device:", config.DEVICE, "arch:", args.arch)

    # data
    train_s, val_s, test_s = make_splits(config.TRASHNET_DIR)
    if len(train_s) == 0:
        print("no images found, run scripts/download_trashnet.py first")
        return
    print("train", len(train_s), "val", len(val_s), "test", len(test_s))

    train_ds = TrashDataset(train_s, train_transform(), use_norm)
    val_ds = TrashDataset(val_s, eval_transform(), use_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True, worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, worker_init_fn=worker_init)

    # model
    model = build_model(args.arch, len(config.CLASSES)).to(config.DEVICE)

    weights = torch.tensor(class_weights(train_s)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.DEVICE.type == "cuda")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "model.pt")
    best_acc = 0.0
    frozen = None

    for epoch in range(1, args.epochs + 1):
        want_frozen = epoch <= config.FREEZE_EPOCHS
        if want_frozen != frozen:
            freeze_backbone(model, args.arch, want_frozen)
            frozen = want_frozen
            print("backbone frozen:", frozen)

        model.train()
        seen = hits = 0
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc="epoch %d" % epoch, leave=False):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=config.DEVICE.type == "cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * labels.size(0)
            hits += (out.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print("epoch %d train_acc %.3f val_acc %.3f" % (epoch, hits / seen, val_acc))

        # save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "classes": config.CLASSES,
                "img_size": config.IMG_SIZE,
                "use_lighting_norm": use_norm,
                "val_acc": val_acc,
            }, ckpt_path)
            print("saved best", round(val_acc, 3))

    with open(os.path.join(config.CHECKPOINT_DIR, "test_split.json"), "w") as f:
        json.dump(test_s, f)
    print("best val_acc", round(best_acc, 3))


if __name__ == "__main__":
    main()
