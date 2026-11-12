# GROUP11_CENG313 — RSNA Intracranial Aneurysm Detection

Term project for **CENG313 Introduction to Data Science**, Gazi University, Fall 2026.

We work on the Kaggle competition [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection).

## Team

| Name | Student ID | GitHub |
| --- | --- | --- |
| Ahmet Can Gümüş | 23118080007 | [@Ahmet-45](https://github.com/Ahmet-45) |
| Muhammed Taha Eren | 23118080020 | [@xMte0665](https://github.com/xMte0665) |

## Competition

Each sample is a 3D brain scan (a series of slices) from CTA, MRA or MRI. For each series, the model predicts 14 probabilities:

- **Aneurysm Present**: is there an aneurysm anywhere?
- **13 locations**: in which artery region is the aneurysm?

**Metric:** AUC is computed for each of the 14 labels. *Aneurysm Present* has weight 13, each location has weight 1. In other words, half of the score is the *Aneurysm Present* AUC and the other half is the mean AUC of the 13 locations.

## Dataset

| | |
| --- | --- |
| Total size | 311.25 GB, about 1.0M DICOM files |
| Series | 4,348 (one row each in `train.csv`) |
| Modality | CTA 1,808 · MRA 1,252 · MRI T2 983 · MRI T1post 305 |
| Aneurysm Present | 1,863 positive series (42.8%) |
| Localizers | 2,254 rows in `train_localizers.csv` |

| Patients | 4,348: every series has its own DICOM `PatientID` (`train.csv` has no patient ID) |
| Slices per series | 15 to 1,441 (median: CTA 334, MRA 164, MRI T1post 166, MRI T2 30) |
| Distance between slices | 0.25 to 10 mm (median: MRA 0.55, CTA 0.62, MRI T1post 1.0, MRI T2 5.0) |
| File layout | One DICOM file per slice, except 322 series stored as a single multi-frame file |

Series with an aneurysm per location (249 positive series have more than one location):

| Location | Series | Location | Series |
| --- | --- | --- | --- |
| Anterior Communicating Artery | 363 | Basilar Tip | 110 |
| Left Supraclinoid Internal Carotid Artery | 330 | Right Posterior Communicating Artery | 101 |
| Right Middle Cerebral Artery | 294 | Right Infraclinoid Internal Carotid Artery | 98 |
| Right Supraclinoid Internal Carotid Artery | 278 | Left Posterior Communicating Artery | 86 |
| Left Middle Cerebral Artery | 219 | Left Infraclinoid Internal Carotid Artery | 78 |
| Other Posterior Circulation | 113 | Right Anterior Cerebral Artery | 56 |
| | | Left Anterior Cerebral Artery | 46 |

It is a code competition: submissions run as a Kaggle Notebook. Late submissions are open.

The data is not stored in this repository. Download it from the competition's Data page and place it under `data/raw/`.

## Approach

### Assignment 1 — EDA + Baselines

The full write-up is [`assignment1_eda_baseline.ipynb`](assignment1_eda_baseline.ipynb).

**EDA findings**
- *Aneurysm Present* is fairly balanced (42.8%), but each location is rare (1–8%), and 13% of the positive series have more than one location.
- The positive rate depends strongly on modality (CTA 54%, MRA 44%, MRI 25%), sex (women 47%, men 34%) and age (29% under 40, about 46% over 50).
- The scans differ a lot: number of slices, slice spacing, image size (164 sizes) and length (808 CTAs also include the neck and chest).
- Header traps: `SliceThickness` is often not the real slice spacing, and many CT headers have a `RescaleIntercept` that must not be applied (pixels are already in HU).
- All aneurysms are around the circle of Willis in the middle of the head. On MRI T2 and T1post, a maximum intensity projection does not show the vessels.
- 36 CTAs are bone-subtracted (no skull), and 31 of them are positive: a possible shortcut for a model.

**Preprocessing** (`src/data.py`): every series becomes one uint8 volume of 160 × 192 × 192 voxels of 1 mm, with the same axes for all scans, cut from the top of the head. CT uses a −100 to 700 HU window, MR the 0.5–99.9 percentiles of the head. Localizer points are converted to volume coordinates.

**Evaluation**: the same 5-fold CV (`splits/folds.csv`, stratified by modality and rarest location) and the competition metric for every model, each fold scored on its own and averaged.

| ID | Model | Input |
| --- | --- | --- |
| B0 | Training-set mean | — |
| B1 | Logistic regression (spline for age) / LightGBM, one model per label | Age, sex, modality |
| B2 | ResNet18 (ImageNet), bone removed on CT | Maximum intensity projections (3 views) |
| B1 + B2 | Mean of the per-fold ranks of B1-LGBM and B2, no tuning | Both |
| B1-DICOM | LightGBM per label (B1-LGBM settings), our Kaggle submission | Age, sex and DICOM tags that tell the scan types apart, all read from the header (see How to Run, step 8) |

### Assignment 2 — Proposed Solution

> TODO: describe the final model.

## Results

All results use the same 5-fold CV and the competition metric. Each fold is scored on its own and the 5 scores are averaged (`src.utils.cv_report`). Out-of-fold predictions are in `results/oof_<model>.csv`.

| Model | CV score | Aneurysm Present AUC | Mean location AUC | Kaggle score |
| --- | --- | --- | --- | --- |
| B0 | 0.500 ± 0.000 | 0.500 | 0.500 | — |
| B1-LR | 0.659 ± 0.007 | 0.675 | 0.644 | — |
| B1-LGBM | 0.666 ± 0.010 | 0.676 | 0.655 | — |
| B2 | 0.679 ± 0.012 | 0.704 | 0.654 | — |
| B1 + B2 | 0.708 ± 0.005 | 0.727 | 0.690 | — |
| B1-DICOM | 0.694 ± 0.008 | 0.727 | 0.660 | |
| Ours | | | | |

Score inside each modality (same metric, computed on the series of one modality in each fold):

| Model | CTA | MRA | MRI T2 | MRI T1post |
| --- | --- | --- | --- | --- |
| B1-LR | 0.583 | 0.602 | 0.596 | 0.595 |
| B1-LGBM | 0.590 | 0.619 | 0.668 | 0.650 |
| B2 | 0.668 | 0.647 | 0.570 | 0.624 |
| B1 + B2 | 0.676 | 0.678 | 0.652 | 0.667 |
| B1-DICOM | 0.661 | 0.646 | 0.657 | 0.623 |

B1 gets most of its score from the link between modality and positive rate. Inside CTA and MRA the image model B2 is clearly better than B1, so it learns from the vessels; on MRI T2 and T1post it is worse, because their projections do not show the vessels. Image and patient data are complementary: together they reach 0.708.

## Repository Structure

```
GROUP11_CENG313/
├── README.md
├── requirements.txt
├── .gitignore                         # excludes data/
├── assignment1_eda_baseline.ipynb     # Assignment 1 write-up: EDA summary, baselines, comparison
├── assignment2_solution.ipynb         # Assignment 2 (in progress)
├── src/
│   ├── __init__.py
│   ├── data.py                        # DICOM loading, preprocessing, MIPs
│   ├── models.py                      # model definitions
│   ├── train_b2.py                    # baseline B2 training with 5-fold CV
│   └── utils.py                       # competition metric, CV split, helpers
├── notebooks/                         # detailed analyses that the assignment notebooks summarize
│   ├── 01_eda_labels.ipynb            # labels, modality, age, sex, localizers
│   ├── 02_eda_dicom_meta.ipynb        # DICOM metadata of all series
│   ├── 03_baselines.ipynb             # B0 and B1
│   └── 04_eda_images.ipynb            # cached volumes, MIPs, localizer positions
├── kaggle/                            # scripts that run on Kaggle (one notebook cell each) and the code upload
├── splits/                            # folds.csv, the shared CV split
├── results/                           # out-of-fold predictions of every model
├── submissions/                       # Kaggle submission score screenshots
├── data/                              # not tracked by git
└── ROADMAP.md
```

Open the notebooks from the repository root (or from `notebooks/` for the detailed ones); both work, because each notebook finds the root itself.

## How to Run

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

On Windows, `pip install -r requirements.txt` installs the CPU-only build of PyTorch. To train on an NVIDIA GPU, reinstall it with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2. Kaggle API token

Accept the competition rules on Kaggle first. Then create a token in Kaggle → Settings → API and save it to `~/.kaggle/access_token` (or `~/.kaggle/kaggle.json` for the older file format). Never commit it. Alternatively, run `kaggle auth login` to sign in through the browser.

Use the Kaggle API only from the command line. `import kaggle` in Python authenticates on import and raises an error when no token is found, so don't import it in notebooks or in `src/`.

### 3. Data

The CSV files are small:

```bash
kaggle competitions download -c rsna-intracranial-aneurysm-detection -f train.csv -p data/raw
kaggle competitions download -c rsna-intracranial-aneurysm-detection -f train_localizers.csv -p data/raw
```

The full image data (311 GB) is too large for most local disks. For local development, make a small subset in a Kaggle Notebook with the competition data attached, then download `subset.zip` and extract it into `data/raw/`:

```python
import shutil, pandas as pd
root = "/kaggle/input/competitions/rsna-intracranial-aneurysm-detection"
df = pd.read_csv(f"{root}/train.csv")
# 3 positive + 3 negative series from each modality -> 24 series, about 1.1 GB
sample = df.groupby(["Modality", "Aneurysm Present"]).head(3)["SeriesInstanceUID"]
for uid in sample:
    shutil.copytree(f"{root}/series/{uid}", f"/kaggle/working/subset/series/{uid}")
shutil.make_archive("/kaggle/working/subset", "zip", "/kaggle/working/subset")
```

Expected layout:

```
data/raw/
├── train.csv
├── train_localizers.csv
└── series/<SeriesInstanceUID>/*.dcm
```

### 4. Preprocessing

`src/data.py` turns every series into one uint8 volume of 160 × 192 × 192 voxels (1 mm), cut from the top of the head:

```bash
python -m src.data --root data/raw --out data/processed
```

Use `--limit 20` for a test run. Use `--shard 0/2` and `--shard 1/2` to split the work over two sessions. Series that are already saved are skipped, so a stopped run can be started again.

For all 4,348 series we run it on Kaggle (CPU, no accelerator):

1. The Kaggle Notebooks take our code from a private Kaggle dataset, so every run uses a fixed version of the code. Upload it: `python kaggle/push_code.py --user <kaggle username> --new` the first time, `python kaggle/push_code.py --user <kaggle username> -m "<note>"` after every change.
2. New Notebook with the competition data and `group11-code` as inputs. Paste `kaggle/preprocess_notebook.py` into one cell, test with `LIMIT = 20`, then set `LIMIT = None` and use Save Version → Save & Run All.
3. Make a dataset (`group11-volumes`) from the notebook's output.

Kaggle saves only about 500 output files per notebook, so on Kaggle the volumes are packed into 8 zip files (`volumes_0.zip` … `volumes_7.zip`, about 8 GB). `src.data.volume_paths()` finds the volumes in both layouts and unzips them if needed:

```python
from src.data import volume_paths, load_volume
paths = volume_paths("/kaggle/input/group11-volumes", extract_to="/kaggle/temp/volumes")  # {SeriesInstanceUID: path}
volume = load_volume(paths[uid])  # uint8, (160, 192, 192)
```

Output:

```
data/processed/
├── volumes/<SeriesInstanceUID>.npz   # read with src.data.load_volume()
├── volumes_meta.csv                  # source spacing, crop box, errors
└── localizers_vox.csv                # localizer points in volume coordinates (z, y, x)
```

### 5. CV folds

Every model uses the same 5-fold split, saved in `splits/folds.csv` (one row per series: `SeriesInstanceUID`, `fold`). It is stratified by modality and by the rarest aneurysm location of each series. To make it again from `train.csv`:

```bash
python -m src.utils make-folds
```

In code, use `src.utils.load_folds()` for the split and `src.utils.competition_score()` for the metric.

### 6. Baseline B2

`src/train_b2.py` trains one ResNet18 per fold (ImageNet weights) on maximum intensity projections from 3 views (`src.data.make_mips`). On CT, bone (above about 590 HU) is removed before the projection, otherwise the skull hides the vessels. It uses `src.utils.run_cv`, so folds and scoring are the same as for B0 and B1.

It needs a GPU, so it runs on Kaggle: a notebook with the competition data, `group11-code` and `group11-volumes` as inputs, GPU on and Internet on. Paste `kaggle/train_b2_notebook.py` into one cell. It writes `oof_b2.csv` and `b2_log.csv` (loss and validation score per epoch) to the output; copy them to `results/`.

Every fold is trained for a fixed 15 epochs and the last model is used. We don't pick the best epoch on the validation fold, because that would make the CV score too optimistic.

### 7. Notebooks

The notebooks are saved with their outputs, so they can be read without running them. To run them again, they need these local files (all under `data/`, not tracked by git):

| File | Used by | How to get it |
| --- | --- | --- |
| `data/raw/train.csv`, `data/raw/train_localizers.csv` | all | step 3 |
| `data/interim/dicom_meta.csv` | `02`, `assignment1` | Run `kaggle/extract_dicom_meta.py` in a Kaggle Notebook with the competition data (set `LIMIT = None`, about 51 min), then `kaggle kernels output <user>/<notebook> -p data/interim` (ours: `tahaeren/group11-dicom-metadata`) |
| `data/volumes_sample/` | `04`, `assignment1` | A few volumes and the two tables of `group11-volumes`, see below |

```bash
kaggle datasets download ahmetcan0/group11-volumes -f volumes_meta.csv -p data/volumes_sample
kaggle datasets download ahmetcan0/group11-volumes -f localizers_vox.csv -p data/volumes_sample
# one volume: the archive number is the position of the series in the sorted list of all series, modulo 8
kaggle datasets download ahmetcan0/group11-volumes -f volumes_<i>/volumes/<SeriesInstanceUID>.npz -p data/volumes_sample/volumes
```

`03_baselines.ipynb` trains B0 and B1 (a few seconds on CPU) and writes `results/oof_b0.csv`, `oof_b1_lr.csv` and `oof_b1_lgbm.csv`. `assignment1_eda_baseline.ipynb` reads all OOF files from `results/`.

### 8. Kaggle submission (B1-DICOM)

This is a code competition: Kaggle runs our notebook again on the hidden test series and calls `predict()` once per series. A test series has no row in `train.csv`, so B1 can't be used as it is. **B1-DICOM** reads age, sex and modality from the DICOM header instead (`src.data.header_features`). The `Modality` tag only says CT or MR, so it also uses tags that tell the MR types apart (3D/2D acquisition, TR, TE, flip angle, contrast agent, keywords in the series description). The model is LightGBM per label with the B1-LGBM settings (`src.models.fit_b1_dicom`).

1. Upload the current code: `python kaggle/push_code.py --user <kaggle username> -m "Add B1-DICOM"`.
2. New Notebook with the competition data and `group11-code` as inputs, accelerator None, **Internet off** (required for submissions). Paste `kaggle/submit_b1_notebook.py` into one cell, then Save Version → Save & Run All. The run prints the 5-fold CV score and writes `oof_b1_dicom.csv` (copy it to `results/`).
3. Competition page → Submit Prediction → choose the notebook version. The score screenshot goes to `submissions/`.

## Progress Log

| Date | Who | What |
| --- | --- | --- |
| 2026-10-29 | Both | Selected competition, made project plan |
| 2026-11-01 | Can | Setup: accepted rules, checked dataset (311 GB, 4,348 series), confirmed late submissions are open, tested DICOM reading on a 24-series subset, added repo skeleton |
| 2026-11-01 | Taha | Competition metric (`src/utils.py`), tested on random and real labels |
| 2026-11-02 | Taha | Read the DICOM headers of all 4,348 series on Kaggle (`kaggle/extract_dicom_meta.py`, 51 min, no errors). EDA of labels, modality, age, sex and localizers (`notebooks/01_eda_labels.ipynb`) and of the DICOM metadata (`notebooks/02_eda_dicom_meta.ipynb`). Every series has its own PatientID, so no patient overlap |
| 2026-11-04 | Taha | 5-fold CV split (`splits/folds.csv`), stratified by modality and rarest location |
| 2026-11-04 | Taha | Shared CV code (`run_cv`, `cv_report`, `save_oof` in `src/utils.py`) and baseline B0 (`notebooks/03_baselines.ipynb`): 0.500 on every fold, as expected |
| 2026-11-04 | Taha | Baseline B1 on age, sex and modality only: logistic regression with a spline for age (0.659) and LightGBM (0.666). An image model has to clearly beat about 0.66 |
| 2026-11-04 | Can | Preprocessing (`src/data.py`): DICOM loader for single- and multi-frame series, same orientation for all scans, head crop, 1 mm resampling, CT/MR normalization, localizers in volume coordinates. Tested on the 24-series subset. Found that many CT headers have a wrong `RescaleIntercept` (pixels are already in HU) and that the localizer frame `f` starts at 0 |
| 2026-11-04 | Can | Data format agreed with Taha. B2 code (`src/train_b2.py`, `MIPNet` in `src/models.py`): ResNet18 on 3-view MIPs with bone removed on CT, left-right flip with swapped labels, flip TTA. Tested end to end on the subset (CPU). Kaggle scripts: code upload, full preprocessing (volumes packed into 8 zips because of Kaggle's output file limit), B2 training. Full preprocessing run started on Kaggle |
| 2026-11-05 | Can | Full preprocessing on Kaggle: 4,348 series in 240 min, 0 errors, all 2,254 localizers inside the cropped box, rescale skipped for all 1,808 CTs (pixels already in HU). Shared as the Kaggle dataset `group11-volumes`. 36 CTAs have almost no bone above 700 HU: the scanner already removed the skull (bone-subtracted CTA), not a data error |
| 2026-11-05 | Can | Baseline B2 (`results/oof_b2.csv`, 18 min on a T4): 0.679, Aneurysm Present AUC 0.704. Only slightly above B1 overall, but clearly better inside CTA (0.668 vs 0.590) and MRA (0.647 vs 0.619), so it learns from the vessels. Worse than B1 on MRI T2 (0.570), where MIPs show bright fluid instead of vessels. Train loss keeps falling, validation score flat after about epoch 9: overfitting. A fixed 50/50 rank mean of B1-LGBM and B2 reaches 0.708, so image and patient data are complementary |
| 2026-11-05 | Taha | EDA of the images (`notebooks/04_eda_images.ipynb`) on 5 sample volumes from `group11-volumes`: slices and MIPs per modality with the localizer point, bone removal on CT, positions of all 2,254 localizers in the box. Left/right labels match the x position, so the reorientation is correct. MRA scans start lower than the top of the head, so aneurysms sit about 45 mm higher in the box than in other modalities. MIPs of MRI T2 and T1post do not show the vessels, which explains the weak B2 scores there. Added an example of the 36 bone-subtracted CTAs: 31 of them are positive (86% vs 54% for all CTAs), a possible shortcut for a model |
| 2026-11-07 | Taha | Assignment 1 write-up (`notebooks/assignment1_eda_baseline.ipynb`): EDA summary, preprocessing, evaluation, baselines and a comparison table per modality. `src.utils.rank_average` for the B1 + B2 combination (0.708). README: dataset facts, EDA summary, results per modality, how to run the notebooks |
| 2026-11-11 | Can | Repo layout changed to the course format: `assignment1_eda_baseline.ipynb` moved to the root, `assignment2_solution.ipynb` skeleton and `submissions/` added, detailed notebooks stay in `notebooks/` |
| 2026-11-11 | Can | B1-DICOM for the Kaggle late submission: B1 with inputs read from the DICOM header (`src.data.header_features`, `src.models.fit_b1_dicom`, `kaggle/submit_b1_notebook.py`), because test series have no `train.csv` row. Headers of all 4,348 series in 43 s. CV 0.694 ± 0.008, above B1-LGBM (0.666), mostly inside CTA (0.661 vs 0.590): header tags identify hospital and protocol. Submitted as a late submission |
| 2026-11-11 | Can | Assignment 1 notebook ordered by the 7 graded sections: data overview with types, `describe()`, missing values and duplicates (found 3 duplicate rows in `train_localizers.csv`), missing values in the DICOM headers, preprocessing of the tabular inputs, B1-DICOM in the results, Kaggle submission section |

## Kaggle Submission History

| Date | Model | Public score | Private score | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-11 | B1-DICOM | – | – | Late submission, notebook `group11-b1-dicom-submission` version 1, CV 0.694. Failed: the scoring run went over the 12-hour limit |
| 2026-11-12 | B1-DICOM | pending | pending | Version 2: training header features from the code dataset, one LightGBM thread, removes only the current series from `/kaggle/shared` |

## References

- [Kaggle competition page](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- RSNA Intracranial Aneurysm (RSNA-ICA) Dataset, July 2025. https://doi.org/10.1148/dataset.ica.2025