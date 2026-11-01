"""Competition metric, CV split and helpers.

Owner: Taha

Planned:
- 5-fold CV split, saved to a file
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ID_COL = "SeriesInstanceUID"

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


if __name__ == "__main__":
    # Sanity check on random labels: perfect -> 1.0, random -> about 0.5, reversed -> 0.0
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.integers(0, 2, size=(1000, 14)), columns=LABEL_COLS)
    noise = pd.DataFrame(rng.random((1000, 14)), columns=LABEL_COLS)
    print(f"perfect : {competition_score(y, y):.4f}")
    print(f"random  : {competition_score(y, noise):.4f}")
    print(f"reversed: {competition_score(y, 1 - y):.4f}")
