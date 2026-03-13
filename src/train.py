from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.models import AVAILABLE_MODELS, create_model, get_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pill classifier with transfer learning")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing train/val folders")
    parser.add_argument("--model", type=str, default="resnet50", choices=AVAILABLE_MODELS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument("--out", type=str, default="models/best_model.pt")
    return parser.parse_args()


def build_dataloaders(data_root: str, input_size: int, batch_size: int, num_workers: int):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError("Dataset must include train/ and val/ directories.")

    train_ds = ImageFolder(train_dir, transform=get_transforms(input_size, is_train=True))
    val_ds = ImageFolder(val_dir, transform=get_transforms(input_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.class_to_idx


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, input_size = create_model(
        model_name=args.model,
        num_classes=2,
        pretrained=not args.no_pretrained,
    )

    train_loader, val_loader, class_to_idx = build_dataloaders(
        args.data_root,
        input_size=input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Recreate model with exact class count from dataset.
    model, input_size = create_model(
        model_name=args.model,
        num_classes=len(class_to_idx),
        pretrained=not args.no_pretrained,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "input_size": input_size,
            }
            torch.save(checkpoint, out_path)
            print(f"Saved best model to: {out_path}")

    metrics = {
        "best_val_acc": best_acc,
        "model": args.model,
        "data_root": args.data_root,
        "epochs": args.epochs,
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    train()
