import argparse

import torch

from data import CLASSES, preprocess_image
from model import load_model


def main():
    parser = argparse.ArgumentParser(description="Classify chest X-ray images as normal or pneumonia")
    parser.add_argument("images", nargs="+", help="paths to X-ray images")
    args = parser.parse_args()

    model = load_model()
    for path in args.images:
        with torch.no_grad():
            probs = model(preprocess_image(path).unsqueeze(0)).softmax(1)[0]
        best = probs.argmax().item()
        details = "  ".join(f"{name} {p:.1%}" for name, p in zip(CLASSES, probs.tolist()))
        print(f"{path}: {CLASSES[best].upper()} ({probs[best].item():.1%})   [{details}]")


if __name__ == "__main__":
    main()
