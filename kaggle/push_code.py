"""Upload our code (src/*.py and splits/folds.csv) to Kaggle as a private dataset.

Owner: Can

The notebooks (kaggle/preprocess_notebook.py, kaggle/train_b2_notebook.py) take the code from this
dataset instead of cloning GitHub, so every Kaggle run uses a fixed, known version of the code.

Run from the repo root, with the Kaggle CLI set up (README, step 2):
    python kaggle/push_code.py --user ahmetcan0 --new          # first time: creates <user>/group11-code
    python kaggle/push_code.py --user ahmetcan0 -m "Add B2"    # later: uploads a new version

The files are uploaded flat (no folders); the notebooks put them back into src/ and splits/.
To let the other team member use it: Kaggle -> the dataset page -> Settings -> Sharing.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build" / "kaggle_code"
SLUG = "group11-code"


def main():
    parser = argparse.ArgumentParser(description="Upload src/ and splits/folds.csv as a private Kaggle dataset.")
    parser.add_argument("--user", required=True, help="Kaggle username")
    parser.add_argument("--new", action="store_true", help="create the dataset (first upload)")
    parser.add_argument("-m", "--message", default="Update code", help="version note for later uploads")
    args = parser.parse_args()

    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    files = sorted((REPO / "src").glob("*.py")) + [REPO / "splits" / "folds.csv"]
    for f in files:
        shutil.copy(f, BUILD / f.name)
    metadata = {"title": SLUG, "id": f"{args.user}/{SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (BUILD / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Uploading:", ", ".join(f.name for f in files))

    if args.new:
        cmd = ["kaggle", "datasets", "create", "-p", str(BUILD)]  # private unless --public is added
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(BUILD), "-m", args.message]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
