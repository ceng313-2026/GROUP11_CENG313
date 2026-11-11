"""Kaggle (late) submission of B1-DICOM: LightGBM on DICOM header features.

Owner: Can

This is a code competition: Kaggle runs the notebook again on the hidden test series and calls
predict() once per series. There is no train.csv row for a test series, so B1 (age, sex, modality
from train.csv) cannot be used as it is. B1-DICOM reads the same information from the DICOM header
(src.data.header_features) and is trained on the header features of all 4,348 training series.

Setup:
1. Upload the current code: python kaggle/push_code.py --user <kaggle username> -m "Add B1-DICOM"
2. New Notebook -> Add Input: the competition data and the group11-code dataset.
   Accelerator: None (CPU). Internet: OFF (required for submissions).
3. Paste this file into one cell and use Save Version -> Save & Run All.
   The run reads the training headers, prints the 5-fold CV score, trains on all series and runs the
   local test gateway.
4. Competition page -> Submit Prediction -> choose this notebook and version -> Submit.
   The public and private score appear under My Submissions (screenshot -> submissions/).

Output in /kaggle/working:
    oof_b1_dicom.csv            out-of-fold predictions, copy to results/
    train_header_features.csv   header features of the training series (data, not committed)

We wrote this ourselves. Only the inference server calls at the end follow the competition's
Evaluation API (kaggle_evaluation package that comes with the competition data).
"""
import glob
import os
import shutil
import sys
import time
import traceback

import pandas as pd

OUT = "/kaggle/working"
CODE = "/tmp/code"
IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))  # set by Kaggle when it scores the notebook


def find_one(patterns):
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"nothing matches {patterns}, is the dataset added as input?")


def find_competition():
    """Folder of the competition data: train_localizers.csv next to a series/ folder."""
    for pattern in ["/kaggle/input/*/train_localizers.csv", "/kaggle/input/*/*/train_localizers.csv",
                    "/kaggle/input/*/*/*/train_localizers.csv"]:
        for hit in sorted(glob.glob(pattern)):
            if os.path.isdir(os.path.join(os.path.dirname(hit), "series")):
                return os.path.dirname(hit)
    print("Inputs found:", *sorted(glob.glob("/kaggle/input/*/*")), sep="\n  ")
    raise FileNotFoundError("competition data not found: Add Input -> Competitions -> RSNA Intracranial Aneurysm Detection")


def setup_code():
    """Copy the flat files of the group11-code dataset back into src/ and splits/."""
    folder = os.path.dirname(find_one(["/kaggle/input/*/data.py", "/kaggle/input/*/*/data.py",
                                       "/kaggle/input/datasets/*/*/data.py"]))
    os.makedirs(f"{CODE}/src", exist_ok=True)
    os.makedirs(f"{CODE}/splits", exist_ok=True)
    for f in glob.glob(f"{folder}/*.py"):
        shutil.copy(f, f"{CODE}/src/")
    if os.path.exists(f"{folder}/folds.csv"):
        shutil.copy(f"{folder}/folds.csv", f"{CODE}/splits/")
    sys.path.insert(0, CODE)
    print(f"Code from {folder}")


setup_code()
from src.data import HEADER_FEATURES, header_features, header_table  # noqa: E402
from src.models import fit_b1_dicom, predict_b1_dicom  # noqa: E402
from src.utils import ID_COL, LABEL_COLS, cv_report, load_folds, run_cv  # noqa: E402

ROOT = find_competition()
train = pd.read_csv(f"{ROOT}/train.csv")

# 1. Header features of every training series (first file of each series only)
start = time.time()
features = header_table([f"{ROOT}/series/{uid}" for uid in train[ID_COL]])
print(f"Header features of {len(features)} series in {time.time() - start:.0f} s")
print("Missing values per feature:", features[HEADER_FEATURES].isna().mean().round(2).to_dict())
data = train.merge(features, on=ID_COL, how="left", validate="one_to_one")

# 2. CV score with the shared folds (only in our own run, not when Kaggle scores the notebook)
if not IS_RERUN:
    features.to_csv(f"{OUT}/train_header_features.csv", index=False)

    def fit_predict(train_part, valid_part):
        models = fit_b1_dicom(train_part[HEADER_FEATURES], train_part)
        return predict_b1_dicom(models, valid_part[HEADER_FEATURES]).to_numpy()

    oof = run_cv(data, fit_predict, load_folds())
    report = cv_report(data, oof)
    print("B1-DICOM CV:", report["summary"].round(4).to_dict())
    print("Score per fold:", report["per_fold"].round(4).tolist())
    print("Score per modality:", report["per_modality"].round(4).to_dict())
    oof.round(6).to_csv(f"{OUT}/oof_b1_dicom.csv", index=False)

# 3. Final model on all training series
MODELS = fit_b1_dicom(data[HEADER_FEATURES], data)
FALLBACK = data[LABEL_COLS].mean()  # used only if a test series cannot be read


def predict(series_path):
    """Called by the gateway once per test series. Returns one row with the 14 LABEL_COLS probabilities."""
    try:
        row = pd.DataFrame([header_features(series_path)])[HEADER_FEATURES]
        probs = predict_b1_dicom(MODELS, row).iloc[0]
    except Exception:  # one unreadable series should not fail the whole submission
        traceback.print_exc()
        probs = FALLBACK
    # The gateway copies each test series to /kaggle/shared; it has to be removed after the prediction
    shutil.rmtree("/kaggle/shared", ignore_errors=True)
    return pd.DataFrame({col: [float(probs[col])] for col in LABEL_COLS})


# 4. Inference server (Evaluation API of the competition)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # the kaggle_evaluation package is in the competition data folder
import kaggle_evaluation.rsna_inference_server  # noqa: E402

inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
if IS_RERUN:
    inference_server.serve()
else:
    try:
        inference_server.run_local_gateway()
        print("Local gateway finished:", os.listdir(OUT))
    except Exception:
        # Only a local check; the scoring run uses serve() above
        traceback.print_exc()
