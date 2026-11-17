"""Export everything the slide deck needs from the trained project.

Run from anywhere after train.py / evaluate.py:
    pneumonia_cnn/.venv/Scripts/python deck/build_assets.py
Writes deck/assets/results.js, deck/assets/weights.js and PNG images.
"""
import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn

DECK = Path(__file__).parent
PROJECT = DECK.parent / "pneumonia_cnn"
sys.path.insert(0, str(PROJECT))

from data import CLASSES, IMG_SIZE, XRayDataset, load_split, preprocess_image  # noqa: E402
from model import MODEL_PATH, OUTPUT_DIR, load_model  # noqa: E402

ASSETS = DECK / "assets"
ASSETS.mkdir(exist_ok=True)
(ASSETS / "samples").mkdir(exist_ok=True)
(ASSETS / "fmaps").mkdir(exist_ok=True)


def to_png(array, path, cmap=None, size=None):
    a = array.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    if cmap:
        rgb = (colormaps[cmap](a)[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
    else:
        img = Image.fromarray((a * 255).astype(np.uint8))
    if size:
        img = img.resize((size, size), Image.NEAREST)
    img.save(path)


def main():
    model = load_model()
    results = {"classes": CLASSES}

    # layer summary
    rows = []
    leaves = [m for m in model.modules() if not list(m.children())]
    hooks = [m.register_forward_hook(lambda m, _, out: rows.append((m, list(out.shape[1:])))) for m in leaves]
    with torch.no_grad():
        model(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE))
    for h in hooks:
        h.remove()
    results["layers"] = [
        {"name": type(m).__name__, "shape": s, "params": sum(p.numel() for p in m.parameters())} for m, s in rows
    ]
    results["total_params"] = sum(p.numel() for p in model.parameters())
    results["model_mb"] = round(MODEL_PATH.stat().st_size / 1e6, 2)

    results["history"] = json.loads((OUTPUT_DIR / "history.json").read_text())

    # test metrics
    test = XRayDataset("test")
    with torch.no_grad():
        logits = torch.cat([model(b) for b in test.images.split(256)])
    y_true, y_pred = test.labels.numpy(), logits.argmax(1).numpy()
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    results["test"] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist(),
        "matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # dataset counts + example images
    counts = {}
    for split in ("train", "val", "test"):
        _, labels = load_split(split)
        counts[split] = [int((labels == i).sum()) for i in range(len(CLASSES))]
    results["counts"] = counts
    images, labels = load_split("train")
    gallery = []
    for label, name in enumerate(CLASSES):
        for k, i in enumerate(np.flatnonzero(labels == label)[:4]):
            fname = f"train_{name}_{k}.png"
            Image.fromarray(images[i]).save(ASSETS / fname)
            gallery.append({"file": f"assets/{fname}", "label": name})
    results["gallery"] = gallery

    # demo samples with real predictions
    samples = []
    for path in sorted((PROJECT / "samples").glob("*.png")):
        shutil.copy(path, ASSETS / "samples" / path.name)
        with torch.no_grad():
            probs = model(preprocess_image(path).unsqueeze(0)).softmax(1)[0].tolist()
        uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        samples.append({
            "file": f"assets/samples/{path.name}", "name": path.stem, "data": uri,
            "truth": path.stem.split("_")[-1], "probs": probs,
        })
    results["samples"] = samples

    # feature maps of first test image of each class
    relus = [m for m in model.features.modules() if isinstance(m, nn.ReLU)]
    fmaps = {}
    for label, name in enumerate(CLASSES):
        index = int((test.labels == label).nonzero()[0])
        image = test.images[index]
        acts = []
        hs = [m.register_forward_hook(lambda _, __, out: acts.append(out[0])) for m in relus]
        with torch.no_grad():
            probs = model(image.unsqueeze(0)).softmax(1)[0].tolist()
        for h in hs:
            h.remove()
        to_png(image[0].numpy(), ASSETS / "fmaps" / f"{name}_input.png")
        layers = []
        for li, a in enumerate(acts):
            files = []
            for c in range(6):
                fn = f"{name}_l{li}_c{c}.png"
                to_png(a[c].numpy(), ASSETS / "fmaps" / fn, cmap="cividis", size=128)
                files.append(f"assets/fmaps/{fn}")
            layers.append({"shape": list(a.shape), "files": files})
        fmaps[name] = {"input": f"assets/fmaps/{name}_input.png", "layers": layers, "probs": probs}
    results["fmaps"] = fmaps

    (ASSETS / "results.js").write_text("window.RESULTS = " + json.dumps(results, indent=1) + ";\n")

    # weights for in-browser inference, BatchNorm folded into each conv
    tensors = []
    for block in model.features:
        conv, bn = block[0], block[1]
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * scale[:, None, None, None]
        b = (conv.bias - bn.running_mean) * scale + bn.bias
        tensors += [w, b]
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            tensors += [layer.weight, layer.bias]
    flat = torch.cat([t.detach().flatten() for t in tensors]).numpy().astype("<f4")
    shapes = [list(t.shape) for t in tensors]
    b64 = base64.b64encode(flat.tobytes()).decode()
    (ASSETS / "weights.js").write_text(
        f"window.WEIGHTS = {{shapes: {json.dumps(shapes)}, data: \"{b64}\"}};\n"
    )
    print(json.dumps(results["test"], indent=1))
    print("samples", [(s["file"], [round(x, 4) for x in s["probs"]]) for s in samples])


if __name__ == "__main__":
    main()
