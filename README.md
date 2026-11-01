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

> TODO: label distribution per location (from EDA).

Notes from a first look at a 24-series subset:
- Most series have one DICOM file per slice, but some are a single multi-frame file.
- Slice count ranges from about 24 (MRI T2) to over 800 (CTA). Image size ranges from 256×208 to 1008×1008.
- Slice thickness ranges from 0.5 mm to 5 mm.
- `train.csv` has no patient ID, but the DICOM `PatientID` tag does.

It is a code competition: submissions run as a Kaggle Notebook. Late submissions are open.

The data is not stored in this repository. Download it from the competition's Data page and place it under `data/raw/`.

## Approach

### Assignment 1 — EDA + Baselines

> TODO: short summary of EDA findings and preprocessing choices.

| ID | Model | Input |
| --- | --- | --- |
| B0 | Training-set mean | — |
| B1 | Logistic regression / LightGBM | Age, sex, modality |
| B2 | Pretrained 2D CNN | Maximum intensity projections (3 views) |

### Assignment 2 — Proposed Solution

> TODO: describe the final model.

## Results

All results use the same 5-fold CV and the competition metric.

| Model | CV score | Aneurysm Present AUC | Mean location AUC | Kaggle score |
| --- | --- | --- | --- | --- |
| B0 | | | | |
| B1 | | | | |
| B2 | | | | |
| Ours | | | | |

## Repository Structure

```
GROUP11_CENG313/
├── notebooks/
│   ├── assignment1_eda_baseline.ipynb
│   └── assignment2_solution.ipynb
├── src/
│   ├── data.py       # DICOM loading, preprocessing, datasets
│   ├── models.py     # model definitions
│   └── utils.py      # competition metric, CV split, helpers
├── data/             # not tracked by git
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## How to Run

> TODO: preprocessing command, training command, how to run each notebook.

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Kaggle API token

Accept the competition rules on Kaggle first. Then create a token in Kaggle → Settings → API and save it to `~/.kaggle/access_token` (or `~/.kaggle/kaggle.json` for the older file format). Never commit it.

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

## Progress Log

| Date | Who | What |
| --- | --- | --- |
| 2026-10-29 | Both | Selected competition, made project plan |
| 2026-11-01 | Can | Setup: accepted rules, checked dataset (311 GB, 4,348 series), confirmed late submissions are open, tested DICOM reading on a 24-series subset, added repo skeleton |

## Kaggle Submission History

| Date | Model | Public LB | Notes |
| --- | --- | --- | --- |
| | | | |

## References

- [Kaggle competition page](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- RSNA Intracranial Aneurysm (RSNA-ICA) Dataset, July 2025. https://doi.org/10.1148/dataset.ica.2025