import math

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch import nn

from data import CLASSES, IMG_SIZE, XRayDataset
from model import OUTPUT_DIR, load_model

DIAGRAM_LAYERS = {
    nn.Conv2d: ("Conv 3x3", "#f0a35e"),
    nn.MaxPool2d: ("Max pool", "#f6cfa5"),
    nn.Flatten: ("Flatten", "#3b4a6b"),
    nn.Linear: ("Dense", "#3b4a6b"),
}


def layer_outputs(model):
    rows = []
    leaves = [module for module in model.modules() if not list(module.children())]
    hooks = [m.register_forward_hook(lambda m, _, out: rows.append((m, tuple(out.shape[1:])))) for m in leaves]
    with torch.no_grad():
        model(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE))
    for hook in hooks:
        hook.remove()
    return rows


def write_summary(rows):
    lines = [f"{'Layer':<14}{'Output shape':<20}{'Params':>10}"]
    for module, shape in rows:
        params = sum(p.numel() for p in module.parameters())
        lines.append(f"{type(module).__name__:<14}{' x '.join(map(str, shape)):<20}{params:>10,}")
    total = sum(sum(p.numel() for p in module.parameters()) for module, _ in rows)
    lines.append(f"Total parameters: {total:,}")
    summary = "\n".join(lines)
    print(summary)
    (OUTPUT_DIR / "model_summary.txt").write_text(summary)


def box_size(shape):
    if len(shape) == 3:
        return 0.25 + 0.08 * math.log2(shape[0]), 0.4 + 3.0 * shape[1] / IMG_SIZE
    return 0.2, 0.3 + 0.35 * math.log10(shape[0])


def plot_structure(rows):
    steps = [("Input", "#d9d2c5", (1, IMG_SIZE, IMG_SIZE))]
    steps += [(*DIAGRAM_LAYERS[type(m)], shape) for m, shape in rows if type(m) in DIAGRAM_LAYERS]
    steps.append(("Softmax", "#1f2a44", rows[-1][1]))

    fig, ax = plt.subplots(figsize=(1.25 * len(steps), 4.2))
    for i, (name, color, shape) in enumerate(steps):
        width, height = box_size(shape)
        ax.add_patch(Rectangle((i - width / 2, -height / 2), width, height, color=color))
        ax.text(i, -1.85, name, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(i, -2.2, "x".join(map(str, shape)), ha="center", va="top", fontsize=8, color="#555")
    ax.set(xlim=(-0.7, len(steps) - 0.3), ylim=(-2.6, 1.9), title="CNN structure (output shape of each step)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "structure_diagram.png", dpi=150)


def plot_feature_maps(model, image, name, channels=8):
    activations = []
    relus = [m for m in model.features.modules() if isinstance(m, nn.ReLU)]
    hooks = [m.register_forward_hook(lambda _, __, out: activations.append(out[0])) for m in relus]
    with torch.no_grad():
        probs = model(image.unsqueeze(0)).softmax(1)[0]
    for hook in hooks:
        hook.remove()

    fig, axes = plt.subplots(len(activations) + 1, channels, figsize=(channels * 1.4, (len(activations) + 1) * 1.5))
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(image[0], cmap="gray")
    axes[0, 0].set_title("input", fontsize=8)
    for row, fmap in zip(axes[1:], activations):
        for c, ax in enumerate(row):
            ax.imshow(fmap[c], cmap="viridis")
        row[0].set_title(f"conv {fmap.shape[0]}x{fmap.shape[1]}x{fmap.shape[2]}", fontsize=8, loc="left")
    prediction = CLASSES[probs.argmax().item()]
    fig.suptitle(f"Feature maps - true: {name}, predicted: {prediction} ({probs.max().item():.1%})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"feature_maps_{name}.png", dpi=150)


def main():
    model = load_model()
    rows = layer_outputs(model)
    write_summary(rows)
    plot_structure(rows)
    test_set = XRayDataset("test")
    for label, name in enumerate(CLASSES):
        index = int((test_set.labels == label).nonzero()[0])
        plot_feature_maps(model, test_set.images[index], name)


if __name__ == "__main__":
    main()
