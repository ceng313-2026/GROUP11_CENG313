"""Make the cached volumes of all 4,348 series (src/data.py) in a Kaggle Notebook.

Setup:
1. Upload our code first: python kaggle/push_code.py (see that file).
2. New Notebook -> Add Input: the competition data and the group11-code dataset. Accelerator: None (CPU).
3. Paste this file into one cell. Run it once with LIMIT = 20 to test.
4. Set LIMIT = None and use Save Version -> Save & Run All (runs in the background, up to 12 hours).
5. When it is done: open the version -> Output -> New Dataset, name it group11-volumes.

Kaggle saves only about 500 output files per notebook, so the volumes are written to /kaggle/temp first
and then packed into 8 zip files in /kaggle/working (src.data.pack_volumes). Read them with
src.data.volume_paths(), which also unzips them if Kaggle has not.

Output in /kaggle/working (about 8 GB):
    volumes_0.zip ... volumes_7.zip   volumes/<SeriesInstanceUID>.npz inside
    volumes_meta.csv                  one row per series: source spacing, crop box, raw_min, rescaled, errors
    localizers_vox.csv                localizer points in volume coordinates (z, y, x)
"""
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

LIMIT = 20  # None = all series
SHARD = "0/1"  # "0/2" and "1/2" split the work over two notebooks

OUT = "/kaggle/working"
CODE = "/tmp/code"



def find_one(patterns):
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"nothing matches {patterns}, is the dataset added as input?")


def find_competition():
    """Folder of the competition data. Kaggle has used several layouts, so look for train_localizers.csv
    next to a series/ folder instead of trusting one path."""
    for pattern in ["/kaggle/input/*/train_localizers.csv", "/kaggle/input/*/*/train_localizers.csv",
                    "/kaggle/input/*/*/*/train_localizers.csv"]:
        for hit in sorted(glob.glob(pattern)):
            if os.path.isdir(os.path.join(os.path.dirname(hit), "series")):
                return os.path.dirname(hit)
    print("Inputs found:", *sorted(glob.glob("/kaggle/input/*/*")), sep="\n  ")
    raise FileNotFoundError("competition data not found: Add Input -> Competitions -> RSNA Intracranial Aneurysm Detection")


def scratch_dir(name):
    """A folder that is not saved with the notebook: /kaggle/temp, or /tmp if that is not writable."""
    for base in ["/kaggle/temp", "/tmp"]:
        try:
            os.makedirs(f"{base}/{name}", exist_ok=True)
            return f"{base}/{name}"
        except OSError:
            pass
    raise OSError("no writable scratch folder")


def setup_code():
    """Copy the flat files of the group11-code dataset back into src/ and splits/."""
    folder = os.path.dirname(find_one(["/kaggle/input/*/data.py", "/kaggle/input/datasets/*/*/data.py"]))
    os.makedirs(f"{CODE}/src", exist_ok=True)
    os.makedirs(f"{CODE}/splits", exist_ok=True)
    for f in glob.glob(f"{folder}/*.py"):
        shutil.copy(f, f"{CODE}/src/")
    if os.path.exists(f"{folder}/folds.csv"):
        shutil.copy(f"{folder}/folds.csv", f"{CODE}/splits/")
    sys.path.insert(0, CODE)
    print(f"Code from {folder}")


def run(cmd):
    """Run a command and show its output while it runs."""
    proc = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}")


setup_code()
WORK = scratch_dir("processed")  # no file limit here, but not saved
root = find_competition()
cmd = [sys.executable, "-m", "src.data", "--root", root, "--out", WORK, "--shard", SHARD]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
run(cmd)

from src.data import pack_volumes  # noqa: E402  (importable only after setup_code)

pack_volumes(WORK, OUT)

# Quick check of the run
meta = pd.read_csv(f"{OUT}/volumes_meta.csv")
errors = meta[meta["error"].notna()] if "error" in meta else meta.iloc[:0]
print(f"\n{len(meta)} series, {len(errors)} errors")
if len(errors):
    print(errors[["SeriesInstanceUID", "error"]].to_string())
print("\nRescale applied? (False = pixels were already in HU, header intercept ignored)")
print(pd.crosstab(meta["modality"], meta["rescaled"]))
print(f"\nSeconds per series: {meta['seconds'].describe().round(1).to_dict()}")
points = pd.read_csv(f"{OUT}/localizers_vox.csv")
print(f"\n{len(points)} localizer points, {points['inside'].mean():.1%} inside the cropped box")
