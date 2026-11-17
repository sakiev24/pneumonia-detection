import io
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import openpyxl
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).parent
IMG_SIZE = 128
CLASSES = ["normal", "pneumonia"]
SOURCE_FOLDERS = ["Normal", "Lung_Opacity"]
SEED = 0

ARCHIVE = ROOT / "data" / "raw" / "covid19-radiography-database.zip"
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database"
ARCHIVE_ROOT = "COVID-19_Radiography_Dataset"
RSNA_URL = "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data"
DATA_FILE = ROOT / "data" / f"rsna_pneumonia_{IMG_SIZE}.npz"
SAMPLES_DIR = ROOT / "samples"

AUGMENT = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))


def download():
    if ARCHIVE.exists():
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".part")
    print(f"Downloading {ARCHIVE.name} (816 MB) ...")
    urllib.request.urlretrieve(ARCHIVE_URL, partial)
    partial.rename(ARCHIVE)


def fit_square(image):
    return ImageOps.fit(image.convert("L"), (IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def rsna_images(archive, folder):
    prefix = f"{ARCHIVE_ROOT}/{folder}/images/"
    files = {PurePosixPath(name).stem.lower(): name for name in archive.namelist() if name.startswith(prefix)}
    metadata = io.BytesIO(archive.read(f"{ARCHIVE_ROOT}/{folder}.metadata.xlsx"))
    sheet = openpyxl.load_workbook(metadata, read_only=True).active
    rows = sheet.iter_rows(min_row=2, values_only=True)
    return sorted(files[name.lower()] for name, *_, url in rows if url == RSNA_URL)


def build():
    download()
    rng = np.random.default_rng(SEED)
    with zipfile.ZipFile(ARCHIVE) as archive:
        normal, pneumonia = (rsna_images(archive, folder) for folder in SOURCE_FOLDERS)
        normal = sorted(rng.choice(normal, size=len(pneumonia), replace=False).tolist())
        paths = normal + pneumonia
        print(f"Resizing {len(paths)} images to {IMG_SIZE}x{IMG_SIZE} ...")
        images = np.stack([np.asarray(fit_square(Image.open(archive.open(path)))) for path in paths])
    labels = np.array([0] * len(normal) + [1] * len(pneumonia), dtype=np.uint8)

    indices = np.arange(len(labels))
    train, rest = train_test_split(indices, test_size=0.3, stratify=labels, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=SEED)
    splits = {"train": train, "val": val, "test": test}
    np.savez(
        DATA_FILE,
        **{f"{split}_images": images[i] for split, i in splits.items()},
        **{f"{split}_labels": labels[i] for split, i in splits.items()},
    )


def load_split(split):
    if not DATA_FILE.exists():
        build()
    with np.load(DATA_FILE) as data:
        return data[f"{split}_images"], data[f"{split}_labels"]


class XRayDataset(Dataset):
    def __init__(self, split, augment=False):
        images, labels = load_split(split)
        self.images = torch.from_numpy(images).unsqueeze(1).float() / 255
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.augment:
            image = AUGMENT(image)
        return image, self.labels[i]


def preprocess_image(path):
    image = fit_square(Image.open(path))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).unsqueeze(0)


def export_samples(per_class=5):
    images, labels = load_split("test")
    SAMPLES_DIR.mkdir(exist_ok=True)
    for label, name in enumerate(CLASSES):
        for i in np.flatnonzero(labels == label)[:per_class]:
            Image.fromarray(images[i]).save(SAMPLES_DIR / f"test{i:04d}_{name}.png")


def describe():
    for split in ("train", "val", "test"):
        _, labels = load_split(split)
        counts = ", ".join(f"{name} {np.sum(labels == i)}" for i, name in enumerate(CLASSES))
        print(f"{split:<6}{len(labels):>6} images  ({counts})")


if __name__ == "__main__":
    build()
    describe()
    export_samples()
