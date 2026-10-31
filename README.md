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

> TODO: total size, number of series, modality counts, label distribution (from EDA).

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
├── ROADMAP.md
└── README.md
```

## How to Run

> TODO: environment setup, preprocessing command, training command, how to run each notebook.

```bash
pip install -r requirements.txt
```

## Progress Log

| Date | Who | What |
| --- | --- | --- |
| 2026-10-29 | Both | Selected competition, made project plan |
| | | |

## Kaggle Submission History

| Date | Model | Public LB | Notes |
| --- | --- | --- | --- |
| | | | |

## References

- [Kaggle competition page](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- RSNA Intracranial Aneurysm (RSNA-ICA) Dataset, July 2025. https://doi.org/10.1148/dataset.ica.2025