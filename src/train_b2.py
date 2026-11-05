"""Baseline B2: pretrained ResNet18 on MIPs from 3 views, trained with the shared 5-fold CV.

Owner: Can

Needs the cached volumes from src/data.py and a GPU, so it runs on Kaggle (see kaggle/train_b2_notebook.py):
    python -m src.train_b2 --volumes <folder with the volumes> --train-csv <train.csv> --out /kaggle/working

Output in --out:
    oof_b2.csv   out-of-fold predictions (SeriesInstanceUID, fold, 14 labels), same format as results/oof_b0.csv
    b2_log.csv   training loss and validation score after every epoch, per fold

Every fold is trained for a fixed number of epochs and the model after the last epoch predicts the
validation fold. We don't pick the best epoch on the validation fold: that would make the CV score
look better than it is.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data import load_volume, make_mips, volume_paths
from src.models import MIPNet
from src.utils import ID_COL, LABEL_COLS, LABEL_WEIGHTS, competition_score, cv_report, load_folds, run_cv, save_oof


def _mirror(label):
    return label.replace("Left", "@").replace("Right", "Left").replace("@", "Right")


# Label order after a left-right flip: "Left X" and "Right X" swap places, the others stay
LR_SWAP = [LABEL_COLS.index(_mirror(label)) for label in LABEL_COLS]


def flip_views(axial, coronal, sagittal):
    """Mirror the head left-right. x is the last axis of the axial and coronal views; the sagittal view
    looks along x, so it stays the same."""
    return axial.flip(-1), coronal.flip(-1), sagittal


def load_mips(train, paths):
    """MIPs of every series in train, in the same row order. paths: {SeriesInstanceUID: .npz path}.
    Series without a cached volume get zeros and ok = False."""
    n = len(train)
    axial = np.zeros((n, 3, 192, 192), np.uint8)
    coronal = np.zeros((n, 3, 160, 192), np.uint8)
    sagittal = np.zeros((n, 3, 160, 192), np.uint8)
    ok = np.zeros(n, bool)
    ct_bright = {}
    start = time.time()
    for i, (uid, modality) in enumerate(zip(train[ID_COL], train["Modality"])):
        if uid in paths:
            volume = load_volume(paths[uid])
            if modality == "CTA":
                ct_bright[uid] = float((volume == 255).mean())
            axial[i], coronal[i], sagittal[i] = make_mips(volume, is_ct=modality == "CTA")
            ok[i] = True
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"MIPs: {i + 1}/{n}, {time.time() - start:.0f} s")
    print(f"{(~ok).sum()} series have no cached volume, they get the training mean as prediction")

    # Check of the CT rescale rule: in a correct CT only bone is >= 700 HU (about 4-7% of the box).
    # If HU were shifted up by 1024, the whole brain would be white too (share far above 20%).
    # Almost no bone (below 1%) means the crop missed the head or the values are on another scale.
    if ct_bright:
        bright = pd.Series(ct_bright)
        print(f"\nCT check, share of voxels >= 700 HU: {bright.describe().round(3).to_dict()}")
        print(f"{(bright > 0.2).sum()} CTs above 20% (suspicious): {list(bright[bright > 0.2].index[:5])}")
        low = bright[bright < 0.01].sort_values()
        print(f"{len(low)} CTs below 1% (suspicious): {low.round(4).to_dict()}")
    return (axial, coronal, sagittal), ok


class MIPDataset(Dataset):
    """Rows `rows` of the MIP arrays. With augment=True: random left-right flip (labels swapped too)
    and a random shift of up to max_shift pixels."""

    def __init__(self, views, rows, labels=None, augment=False, max_shift=8):
        self.views, self.rows, self.labels = views, rows, labels
        self.augment, self.max_shift = augment, max_shift

    def __len__(self):
        return len(self.rows)

    def _shift(self, img):
        # torch random numbers (not numpy): every DataLoader worker gets its own torch seed
        dy, dx = torch.randint(-self.max_shift, self.max_shift + 1, (2,)).tolist()
        return torch.roll(img, shifts=(dy, dx), dims=(1, 2))

    def __getitem__(self, i):
        views = tuple(torch.from_numpy(v[self.rows[i]]) for v in self.views)
        y = torch.zeros(len(LABEL_COLS)) if self.labels is None else torch.from_numpy(self.labels[i])
        if self.augment:
            if torch.rand(1).item() < 0.5:
                views, y = flip_views(*views), y[LR_SWAP]
            views = tuple(self._shift(v) for v in views)
        return views, y


@torch.no_grad()
def predict(model, loader, device):
    """Probabilities averaged over the original and the left-right flipped image (flip back the labels)."""
    model.eval()
    out = []
    for views, _ in loader:
        views = [v.to(device) for v in views]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(*views).float())
            p_flip = torch.sigmoid(model(*flip_views(*views)).float())[:, LR_SWAP]
        out.append(((p + p_flip) / 2).cpu().numpy())
    return np.concatenate(out)


def make_fit_predict(views, ok, row_of, args, log):
    """fit_predict(train_part, valid_part) for src.utils.run_cv: trains one MIPNet per fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_weights = torch.tensor(LABEL_WEIGHTS[LABEL_COLS].to_numpy() / LABEL_WEIGHTS.mean(), dtype=torch.float32, device=device)

    def fit_predict(train_part, valid_part):
        fold = int(valid_part["fold"].iloc[0])
        torch.manual_seed(args.seed + fold)
        train_part = train_part[ok[train_part[ID_COL].map(row_of).to_numpy()]]
        valid_ok = ok[valid_part[ID_COL].map(row_of).to_numpy()]

        loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
        train_loader = DataLoader(
            MIPDataset(views, train_part[ID_COL].map(row_of).to_numpy(),
                       train_part[LABEL_COLS].to_numpy(np.float32), augment=True),
            shuffle=True, drop_last=True, **loader_args)
        valid_loader = DataLoader(MIPDataset(views, valid_part[ID_COL][valid_ok].map(row_of).to_numpy()), **loader_args)

        model = MIPNet(pretrained=not args.no_pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Learning rate: grows linearly over the first 10% of the steps (warmup), then cosine decay to 0
        total = args.epochs * len(train_loader)
        warmup = max(1, total // 10)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
            (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))))
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        # Series without a volume: training mean, like B0
        pred = np.tile(train_part[LABEL_COLS].mean().to_numpy(), (len(valid_part), 1))
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for views_batch, y in train_loader:
                views_batch = [v.to(device, non_blocking=True) for v in views_batch]
                y = y.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(*views_batch)
                # Same weights as the metric: Aneurysm Present counts as much as all 13 locations
                loss = F.binary_cross_entropy_with_logits(logits.float(), y, weight=loss_weights)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(loss.item())

            # Validation score after every epoch is only for the learning curve, not for choosing an epoch
            if valid_ok.any():
                pred[valid_ok] = predict(model, valid_loader, device)
            score = competition_score(valid_part, pd.DataFrame(pred, columns=LABEL_COLS))
            log.append({"fold": fold, "epoch": epoch, "train_loss": np.mean(losses), "valid_score": score,
                        "seconds": time.time() - start})
            print(f"fold {fold} epoch {epoch:2d}: train loss {np.mean(losses):.4f}, valid score {score:.4f}, "
                  f"{time.time() - start:.0f} s")
        return pred

    return fit_predict


def main():
    parser = argparse.ArgumentParser(description="Train baseline B2 (ResNet18 on MIPs) with 5-fold CV.")
    parser.add_argument("--volumes", required=True, help="folder with the cached volumes (or their zip archives)")
    parser.add_argument("--extract-to", help="where to unzip packed volumes (default: next to them)")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="only the first N series (for a test run)")
    parser.add_argument("--no-pretrained", action="store_true", help="random weights instead of ImageNet")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    folds = load_folds()
    if args.limit:
        train = train.head(args.limit)
        folds = folds[folds[ID_COL].isin(train[ID_COL])]
    views, ok = load_mips(train, volume_paths(args.volumes, args.extract_to))
    row_of = pd.Series(np.arange(len(train)), index=train[ID_COL])

    log = []
    start = time.time()
    oof = run_cv(train, make_fit_predict(views, ok, row_of, args, log), folds=folds)
    print(f"Training done in {(time.time() - start) / 60:.1f} min")

    out = Path(args.out)
    path = save_oof(oof, "b2", results_dir=out)
    pd.DataFrame(log).round(5).to_csv(out / "b2_log.csv", index=False)
    print(f"Saved {path} and {out / 'b2_log.csv'}")

    report = cv_report(train, oof)
    for name, table in report.items():
        print(f"\n{name}\n{table.round(4).to_string()}")


if __name__ == "__main__":
    main()
