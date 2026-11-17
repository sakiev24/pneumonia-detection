import torch
from torch import nn

from data import CLASSES, IMG_SIZE, ROOT

OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model.pt"


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * (IMG_SIZE // 16) ** 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def save_model(model):
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)


def load_model():
    model = PneumoniaCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    return model.eval()
