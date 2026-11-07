# Roadmap — GROUP11_CENG313

Project plan for the CENG313 Introduction to Data Science term project, based on the Kaggle competition
[RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection).

**Team:** Ahmet Can Gümüş (Can) · Muhammed Taha Eren (Taha)

**Goal in one sentence:** In Assignment 1 we understand the data and set a baseline score with simple models. In Assignment 2 we build a stronger model and beat that baseline.

---

## Timeline

| Period | Work | Milestone |
| --- | --- | --- |
| Oct 29 – Nov 1 | Setup: data access, repo, project skeleton | |
| Nov 2 – Nov 6 | EDA + preprocessing pipeline (in parallel) | |
| Nov 6 – Nov 10 | Baseline models (B0, B1, B2) | |
| Nov 10 – Nov 11 | Assignment 1 notebook + README update | **Assignment 1 due (Nov, week 2)** |
| Nov 16 – Nov 20 | Midterm week — no project work | |
| Nov 23 – Dec 6 | Main model: setup and training | |
| Dec 7 – Dec 13 | Experiments, comparison, Kaggle submission | |
| Dec 14 – Dec 18 | Assignment 2 notebook + README update | **Assignment 2 due (Dec, week 3)** |

---

## Ownership

Each part has one owner who is responsible for finishing it. Both of us read each other's code, so we both understand the whole project.

The owners below are a guide, not a strict rule. Either of us can pick up or help with the other's tasks when needed. We only tell each other first, so we don't do the same work twice.

| Area | Owner |
| --- | --- |
| Repo setup, project skeleton | Can |
| DICOM loading and preprocessing (`src/data.py`) | Can |
| EDA | Taha |
| Competition metric and CV split (`src/utils.py`) | Taha |
| Baselines B0 and B1 | Taha |
| Baseline B2 — image model (`src/models.py`) | Can |
| Main model — setup and training | Can |
| Experiments, comparison table, per-modality analysis | Taha |
| Kaggle submission notebook | Taha |
| Assignment notebooks and README | Both |

### Shared contracts

These three things affect both of us. We agree on them in the first week, and nobody changes them alone:

1. **Preprocessed data format:** shape, normalization, and file naming of the cached `.npz` volumes.
2. **CV split:** which series is in which fold (saved to a file, patient-level grouping if needed).
3. **Metric function:** one implementation of the competition metric, used by every model.

---

## Phase 1 — EDA + Baseline (Assignment 1)

### Step 0 — Setup
- [x] Accept competition rules on Kaggle, check total dataset size
- [x] Check whether late submissions are open
- [x] Download a small subset first and test the pipeline on it
- [x] Create repo structure (`src/`, `notebooks/`) and README skeleton

### Step 1 — Exploratory Data Analysis
- [x] **Labels:** number of positive series, count per location, series with multiple locations
- [x] **Modality:** CTA / MRA / MRI counts, positive rate per modality
- [x] **Demographics:** age and sex vs. aneurysm presence
- [x] **DICOM metadata:** number of slices, slice thickness, pixel spacing, image size
- [x] **Visuals:** sample slices per modality, localizer points drawn on images
- [x] **Leakage check:** do some patients have more than one series? (No: every series has its own PatientID)

### Step 2 — Preprocessing (`src/data.py`)
- [x] Read a DICOM series (one file per slice or multi-frame), sort slices by position, build a 3D volume
- [x] Turn every volume to the same orientation (top → bottom, front → back, right → left)
- [x] Crop a fixed box from the top of the head and resample to 1 mm voxels (fixed shape 160 × 192 × 192)
- [x] Normalize intensity (windowing for CT, percentile scaling for MR)
- [x] Convert localizer points to volume coordinates
- [x] Test on the 24-series subset
- [x] Agree on the data format together (shared contract 1)
- [x] Cache all 4,348 series on Kaggle and share them as a Kaggle dataset

### Step 3 — Baselines
All baselines use the same 5-fold CV and the competition metric.

| ID | Model | Input |
| --- | --- | --- |
| B0 | Predict training-set mean for every series | Nothing (sanity floor, AUC ≈ 0.5) |
| B1 | Logistic regression / LightGBM | Age, sex, modality |
| B2 | Pretrained 2D CNN (e.g. ResNet18), 14 outputs | Maximum intensity projections (MIP) from 3 views |

- [x] Implement the competition metric
- [x] Create and save CV folds
- [x] B0
- [x] B1
- [x] B2

### Step 4 — Write-up
- [x] `assignment1_eda_baseline.ipynb`: data → findings → baseline results table
- [x] Update README progress log

---

## Phase 2 — Proposed Solution (Assignment 2)

We start with Option A. If time allows, we move to Option B.

| Option | Idea | Difficulty |
| --- | --- | --- |
| A | 3D CNN on the downsampled volume, with an auxiliary task that predicts localizer points, so the model learns where to look | Medium |
| B | Two stages: first find and crop the vessel region, then classify the crop | High |

### Tasks
- [ ] 3D data pipeline and first version of the model
- [ ] Training and first results
- [ ] Comparison table: B0, B1, B2 and our model with the same CV and metric
- [ ] Per-modality AUC (CTA / MRA / MRI)
- [ ] Ablation: with vs. without the auxiliary localizer loss
- [ ] At least one Kaggle submission, result added to README
- [ ] `assignment2_solution.ipynb` and README update

---

## Working rules

- Each person works on their own branch (`taha`, `ahmetcan`) and merges it into `main` when a piece of work is done. We don't use pull requests.
- After merging, we tell each other what changed, and pull `main` into our own branch.
- One commit per piece of work, with a clear message (e.g. `Add DICOM loader`, `Add EDA label analysis`). No large last-day uploads.
- Every experiment result goes into the README progress log.
- Baselines are written by us, not copied from public Kaggle notebooks.