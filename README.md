# ChargePoint - AI/ML Intern (NOC Services Automation)
## Take-Home Technical Exercise — EV Charging Log Anomaly Detection

This repository contains my submission for the **ChargePoint AI/ML Intern (NOC Services Automation) Take-Home Technical Exercise**.

The goal is to detect anomalous EV charging station events from a synthetic event-level charging log dataset using an **unsupervised anomaly detection pipeline**.

---

## 1) Problem Summary

Each row in `charging_logs.csv` represents an **event** within a charging session (not a full session summary).  
The task is to identify anomalous events using an unsupervised or semi-supervised ML approach.

This submission implements:

- data loading and preprocessing
- exploratory data analysis (EDA)
- feature engineering (telemetry + temporal + session + station-relative features)
- anomaly detection model training (**Isolation Forest**)
- threshold-based anomaly labeling
- a lightweight inference script (`predict.py`) that outputs `is_anomaly` (0/1)

---

## 2) Repository Contents

### Required deliverables
- `README.md` — setup and usage instructions (this file)
- `REPORT.md` — technical report (problem understanding, EDA, modeling, evaluation, results, tradeoffs)
- `AI_USAGE.md` — documentation of AI tool usage and validation
- `predict.py` — lightweight inference script (CLI)
- `src/` — source code for data loading, feature engineering, training
- `artifacts/` — saved model and preprocessing artifacts for inference

### Additional files (supporting)
- `outputs/` — training outputs / summaries / sample predictions (generated during development)
- `notebooks/` — optional EDA / prototyping notebook(s), if included
- `data/charging_logs.csv` or `charging_logs.csv` — provided synthetic dataset (if included in repo)

---

## 3) Approach Overview

### Model
- **Isolation Forest** (unsupervised anomaly detection)

### Feature engineering (high level)
- Core telemetry features (`voltage`, `current`, `power_kw`, `temperature_c`, `duration_sec`, `energy_kwh`)
- Time features (`hour`, `day_of_week`, `is_weekend`)
- Physics consistency checks (e.g., `voltage * current` vs `power_kw`)
- Session-sequence features (event index, elapsed time, within-session deltas)
- Station-relative baseline deviation features

### Thresholding
- Anomaly scores are converted to `is_anomaly` using a saved score threshold (selected via training-time percentile configuration)

---

## 4) Environment Setup

### Option A: Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: Use an existing environment

Install the dependencies listed in `requirements.txt`, then run the commands below from the project root.

---

## 5) Training the Model

Run training from the project root:

```bash
python -m src.train --input data/charging_logs.csv --contamination 0.01 --threshold-percentile 99.5
```

### What training does
- Loads and parses the input CSV
- Builds engineered features
- Fits the anomaly detection model
- Computes anomaly scores on training data
- Selects/saves an anomaly threshold
- Saves inference artifacts in `artifacts/`

---

## 6) Running Inference (predict.py)

The required lightweight inference script is:

```bash
python predict.py --input data/charging_logs.csv --output outputs/predictions_test.csv
```

### Inference behavior
- Loads the input CSV
- Loads saved model + preprocessing artifacts from `artifacts/`
- Rebuilds preprocessing + feature engineering consistently
- Scores anomaly likelihood
- Writes output CSV with all original columns plus:
  - `anomaly_score`
  - `is_anomaly` (0/1)

### Expected output format

The output CSV preserves the input rows and appends the anomaly columns.

---

## 7) Reproducibility / Smoke Test (used before submission)

Example validation commands used to verify final outputs:

```bash
python -m src.train --input data/charging_logs.csv --contamination 0.01 --threshold-percentile 99.5
python predict.py --input data/charging_logs.csv --output outputs/predictions_test.csv
```

Optional quick output check:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/predictions_test.csv")
print("rows:", len(df))
print("has anomaly_score:", "anomaly_score" in df.columns)
print("has is_anomaly:", "is_anomaly" in df.columns)
print(df["is_anomaly"].value_counts(dropna=False).to_dict())
PY
```

---

## 8) Evaluation Notes

This is an unsupervised anomaly detection task. In this submission:
- `error_code` / `message` are treated as proxy signals for sanity-check analysis only
- They are not used as supervised labels for model training
- Threshold selection and result interpretation are discussed in detail in `REPORT.md`

The focus is on engineering judgment, feature design, anomaly triage usefulness, and production practicality for NOC-style monitoring.

---

## 9) Optional Notebook (EDA / Prototyping)

If a notebook is included in `notebooks/`, it contains exploratory analysis and prototyping work used to inform feature engineering and model decisions.

The production/reproducible pipeline used for final inference is the Python code in `src/` + `predict.py`.

---

## 10) Notes on AI Tool Usage

AI tools were used as part of the development workflow (brainstorming, code-review assistance, report/doc polish, edge-case checks), while implementation, debugging, validation, and final decisions were manually executed and verified.

See `AI_USAGE.md` for full details.

---

## 11) Submission Checklist Mapping (ChargePoint Requirements)

- Source code (preprocessing, EDA support, feature engineering, model training/evaluation, inference pipeline)
- `REPORT.md` (technical report)
- `AI_USAGE.md` (AI tool usage + validation)
- Lightweight inference script (`predict.py`)
- Optional notebook (if included)

---

## 12) Context

This repository is submitted solely for the ChargePoint take-home exercise review process.

---
