"""
Full evaluation of the trained classifier: per-class precision/recall/F1
(not just the macro-average printed during training), plus a simple
confusion summary. Run this once train.py has finished.

Usage:
    python evaluate.py --checkpoint models/chest_classifier.pt
"""

import argparse

import pandas as pd
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, eval_transform
from model import build_model


def evaluate(
    checkpoint="models/chest_classifier.pt",
    csv_path=r"D:\Datasets\nih\sample_labels.csv",
    image_dir=r"D:\Datasets\nih\sample\images",
    threshold=0.5,
    val_split=0.15,
    out_csv="outputs/evaluation_report.csv",
):
    ckpt = torch.load(checkpoint, map_location="cpu")
    conditions = ckpt["conditions"]
    model = build_model(num_classes=len(conditions))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    full_ds = ChestXrayDataset(csv_path, image_dir, transform=eval_transform, conditions=conditions)

    # Re-derive the same validation split used during training (same seed)
    # so this evaluates on held-out data, not training data.
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(full_ds)), [n_train, n_val], generator=generator
    )
    val_ds = torch.utils.data.Subset(full_ds, val_idx.indices)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            probs = torch.sigmoid(model(images))
            all_probs.append(probs)
            all_targets.append(targets)

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= threshold).astype(int)

    print(f"Best val F1 recorded during training: {ckpt.get('best_val_f1', 'n/a')}")
    print(f"Evaluating on {len(val_ds)} held-out images\n")

    report = classification_report(
        targets, preds, target_names=conditions, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    print(report_df.round(3).to_string())

    report_df.to_csv(out_csv)
    print(f"\nSaved detailed report to {out_csv}")

    # Per-class confusion matrices (TN, FP, FN, TP) -- useful for spotting
    # whether a class is being predicted too often (false positives) or
    # missed entirely (false negatives), which the macro F1 alone hides.
    print("\nPer-class confusion (TN, FP, FN, TP):")
    cms = multilabel_confusion_matrix(targets, preds)
    for cond, cm in zip(conditions, cms):
        tn, fp, fn, tp = cm.ravel()
        print(f"  {cond:20s} TN={tn:4d} FP={fp:4d} FN={fn:4d} TP={tp:4d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/chest_classifier.pt")
    parser.add_argument("--csv", default=r"D:\Datasets\nih\sample_labels.csv")
    parser.add_argument("--images", default=r"D:\Datasets\nih\sample\images")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        csv_path=args.csv,
        image_dir=args.images,
        threshold=args.threshold,
    )