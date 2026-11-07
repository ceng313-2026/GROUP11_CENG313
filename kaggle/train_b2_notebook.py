"""Train baseline B2 (src/train_b2.py) with the shared 5-fold CV in a Kaggle Notebook.

Owner: Can

Setup:
1. New Notebook -> Add Input: the competition data (for train.csv), group11-code and group11-volumes.
2. Accelerator: GPU (T4 or P100). Internet: on (downloads the ImageNet weights of ResNet18).
3. Paste this file into one cell. Run it once with LIMIT = 300 and EPOCHS = 1 to test.
4. Set LIMIT = None, EPOCHS = 15 and use Save Version -> Save & Run All.
5. Download oof_b2.csv and b2_log.csv from the Output tab and put them in results/ in the repo.
"""
import glob
import os
import shutil
import subprocess
import sys

LIMIT = 300  # None = all series
EPOCHS = 1  # 15 for the real run

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
    shutil.copy(f"{folder}/folds.csv", f"{CODE}/splits/")
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
train_csv = f"{find_competition()}/train.csv"
volumes = os.path.dirname(find_one(["/kaggle/input/*/volumes_meta.csv", "/kaggle/input/datasets/*/*/volumes_meta.csv"]))
print(f"Volumes from {volumes}")

cmd = [sys.executable, "-m", "src.train_b2", "--volumes", volumes, "--extract-to", scratch_dir("volumes"),
       "--train-csv", train_csv, "--out", OUT, "--epochs", str(EPOCHS)]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
run(cmd)
