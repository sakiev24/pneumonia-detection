import json
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import XRayDataset
from model import MODEL_PATH, OUTPUT_DIR, PneumoniaCNN, save_model

EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ("loss", "acc"), ("Loss", "Accuracy")):
        for split in ("train", "val"):
            ax.plot(epochs, history[f"{split}_{metric}"], marker="o", markersize=3, label=split)
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)


def main():
    torch.manual_seed(0)
    train_loader = DataLoader(XRayDataset("train", augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(XRayDataset("val"), batch_size=256)
    loss_fn = nn.CrossEntropyLoss()

    model = PneumoniaCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        results = (*run_epoch(model, train_loader, loss_fn, optimizer), *run_epoch(model, val_loader, loss_fn))
        for key, value in zip(history, results):
            history[key].append(value)
        train_loss, train_acc, val_loss, val_acc = results
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            save_model(model)
        print(
            f"epoch {epoch:2d}/{EPOCHS}  "
            f"train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}  "
            f"{time.time() - start:.0f}s{'  saved' if improved else ''}"
        )

    (OUTPUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history)
    print(f"Best validation accuracy {best_val_acc:.3f}, model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
