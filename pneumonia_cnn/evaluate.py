import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from data import CLASSES, XRayDataset
from model import OUTPUT_DIR, load_model


def main():
    model = load_model()
    test_set = XRayDataset("test")
    with torch.no_grad():
        logits = torch.cat([model(batch) for batch in test_set.images.split(256)])
    y_true = test_set.labels.numpy()
    y_pred = logits.argmax(1).numpy()

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=3)
    summary = f"Test accuracy: {accuracy_score(y_true, y_pred):.3f}\n\n{report}"
    print(summary)
    (OUTPUT_DIR / "test_metrics.txt").write_text(summary)

    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASSES, cmap="Blues", colorbar=False
    )
    display.ax_.set(title="Confusion matrix (test set)", xlabel="Predicted", ylabel="True")
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)


if __name__ == "__main__":
    main()
