"""
Dataset loader for the NIH ChestX-ray sample set.

Expects:
  data/nih_sample/images/*.png (or .jpg)
  data/nih_sample/labels.csv with columns:
      Image Index, Finding Labels
  where Finding Labels is a pipe-separated string like
  "Cardiomegaly|Effusion" or "No Finding".

Adjust CONDITIONS below to match the classes you decide to target.
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Selected based on per-class support in the NIH sample set: everything here
# has >=140 positive examples except Pneumonia (62), which is kept despite
# being rarer because it's central to the project's clinical narrative --
# pos_weight-based class balancing (see train.py) helps compensate.
# Dropped for insufficient data: Emphysema, Edema, Fibrosis, Hernia.
CONDITIONS = [
    "No Finding",
    "Infiltration",
    "Effusion",
    "Atelectasis",
    "Nodule",
    "Mass",
    "Pneumothorax",
    "Consolidation",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Pneumonia",
]

IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),  # chest x-rays are grayscale
    transforms.RandomHorizontalFlip(p=0.0),  # anatomy is not left/right symmetric-safe by default; keep off unless you verify labeling convention
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, conditions=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform or eval_transform
        self.conditions = conditions or CONDITIONS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        labels_str = str(row["Finding Labels"])
        present = set(l.strip() for l in labels_str.split("|"))
        target = torch.zeros(len(self.conditions), dtype=torch.float32)
        for i, c in enumerate(self.conditions):
            if c in present:
                target[i] = 1.0

        return image, target


def sanity_check(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    missing = 0
    for name in df["Image Index"].head(200):
        if not os.path.exists(os.path.join(image_dir, name)):
            missing += 1
    print(f"Checked 200 rows: {missing} images missing from {image_dir}")
    print("Label distribution (first 200 rows):")
    print(df["Finding Labels"].head(200).value_counts().head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv", default=r"D:\Datasets\nih\sample_labels.csv")
    parser.add_argument("--images", default=r"D:\Datasets\nih\sample\images")
    args = parser.parse_args()

    if args.check:
        sanity_check(args.csv, args.images)