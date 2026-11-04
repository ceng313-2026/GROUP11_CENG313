"""Competition metric, CV split and helpers.

Owner: Taha

    python -m src.utils check-metric   # sanity check of the metric on random labels
    python -m src.utils make-folds     # write splits/folds.csv from data/raw/train.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ID_COL = "SeriesInstanceUID"
FOLDS_PATH = Path(__file__).resolve().parent.parent / "splits" / "folds.csv"

LOCATION_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]
TARGET_COL = "Aneurysm Present"
LABEL_COLS = LOCATION_COLS + [TARGET_COL]

# Aneurysm Present weighs as much as all 13 locations together
LABEL_WEIGHTS = pd.Series({**{col: 1.0 for col in LOCATION_COLS}, TARGET_COL: 13.0})


def _align(y_true, y_pred):
    """Put y_pred rows in the same order as y_true.

    If both tables have a SeriesInstanceUID column, rows are matched by it.
    Otherwise both tables must already be in the same order.
    """
    if ID_COL in y_true.columns and ID_COL in y_pred.columns:
        y_pred = y_pred.set_index(ID_COL).reindex(y_true[ID_COL])
        if y_pred[LABEL_COLS].isna().any().any():
            raise ValueError("y_pred is missing some series from y_true")
        y_true = y_true.set_index(ID_COL)
    elif len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}")
    return y_true[LABEL_COLS].to_numpy(), y_pred[LABEL_COLS].to_numpy(dtype=float)


def label_aucs(y_true, y_pred):
    """AUC of each of the 14 labels, as a Series indexed by label name.

    y_true holds 0/1 labels, y_pred holds probabilities, both with the LABEL_COLS columns.
    A label with only one class in y_true gets NaN, because AUC is undefined for it.
    """
    t, p = _align(y_true, y_pred)
    aucs = {}
    for i, col in enumerate(LABEL_COLS):
        if len(np.unique(t[:, i])) < 2:
            aucs[col] = np.nan
        else:
            aucs[col] = roc_auc_score(t[:, i], p[:, i])
    return pd.Series(aucs)


def competition_score(y_true, y_pred):
    """Weighted mean of the 14 AUCs: Aneurysm Present weight 13, each location weight 1.

    Labels with NaN AUC are left out of the mean.
    """
    aucs = label_aucs(y_true, y_pred)
    valid = aucs.notna()
    weights = LABEL_WEIGHTS[aucs.index][valid]
    return float((aucs[valid] * weights).sum() / weights.sum())


def make_folds(train, n_splits=5, seed=42):
    """Assign every series of train.csv to one of n_splits CV folds.

    Folds are stratified by modality and by the rarest aneurysm location of each series ('none' for
    negatives), so every fold gets about the same positive rate, modality mix and location counts.
    Every series has its own PatientID (EDA 2), so no patient grouping is needed.
    Strata with fewer than n_splits series (rare locations in a small modality, 24 series in total)
    are merged into one 'other' stratum.
    """
    location_counts = train[LOCATION_COLS].sum()

    def rarest_location(row):
        present = [col for col in LOCATION_COLS if row[col] == 1]
        return min(present, key=location_counts.get) if present else "none"

    strata = train["Modality"] + " | " + train.apply(rarest_location, axis=1)
    strata[strata.map(strata.value_counts()) < n_splits] = "other"

    folds = np.zeros(len(train), dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(splitter.split(train, strata)):
        folds[val_idx] = fold
    return pd.DataFrame({ID_COL: train[ID_COL].to_numpy(), "fold": folds})


def load_folds(path=FOLDS_PATH):
    """The shared CV split: one row per series with its fold number (0-4)."""
    return pd.read_csv(path)


def _check_metric():
    # Sanity check on random labels: perfect -> 1.0, random -> about 0.5, reversed -> 0.0
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.integers(0, 2, size=(1000, 14)), columns=LABEL_COLS)
    noise = pd.DataFrame(rng.random((1000, 14)), columns=LABEL_COLS)
    print(f"perfect : {competition_score(y, y):.4f}")
    print(f"random  : {competition_score(y, noise):.4f}")
    print(f"reversed: {competition_score(y, 1 - y):.4f}")


def _write_folds(train_csv, out):
    train = pd.read_csv(train_csv)
    folds = make_folds(train)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False)

    # Summary: every fold should look the same
    by_fold = train.merge(folds, on=ID_COL).groupby("fold")
    summary = pd.DataFrame({
        "series": by_fold.size(),
        "positive_rate": by_fold[TARGET_COL].mean().round(3),
        "min_location_positives": by_fold[LOCATION_COLS].sum().min(axis=1),
    })
    print(f"Saved {len(folds)} series to {out}")
    print(summary.to_string())
    print(pd.crosstab(by_fold.obj["fold"], by_fold.obj["Modality"]).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric sanity check and CV folds.")
    parser.add_argument("command", choices=["check-metric", "make-folds"])
    parser.add_argument("--train", default="data/raw/train.csv", help="train.csv for make-folds")
    parser.add_argument("--out", type=Path, default=FOLDS_PATH, help="output CSV for make-folds")
    args = parser.parse_args()
    if args.command == "check-metric":
        _check_metric()
    else:
        _write_folds(args.train, args.out)
