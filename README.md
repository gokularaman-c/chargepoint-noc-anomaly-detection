# EV Charging Log Anomaly Detection
## End-to-End Unsupervised Anomaly Detection Pipeline for EV Charging Event Logs

This repository contains an end-to-end machine learning pipeline for detecting anomalous EV charging station events from a synthetic event-level charging log dataset using an **unsupervised anomaly detection approach**.

The repository covers data loading, exploratory data analysis (EDA), feature engineering, anomaly model training, artifact management, and CLI-based inference.

---

## 1) Project Summary

Each row in `charging_logs.csv` represents an **event** within a charging session, not a full session summary.  
The goal is to identify anomalous events using an unsupervised or semi-supervised machine learning approach.

This repository implements:

- data loading and preprocessing
- exploratory data analysis (EDA)
- feature engineering (telemetry + temporal + session + station-relative features)
- anomaly detection model training using **Isolation Forest**
- threshold-based anomaly labeling
- a lightweight inference script (`predict.py`) that outputs `is_anomaly` (0/1)

---

## 2) Repository Contents

### Core files
- `README.md` — setup and usage instructions
- `REPORT.md` — technical report covering problem understanding, EDA, modeling, evaluation, results, and tradeoffs
- `AI_USAGE.md` — documentation of AI tool usage and validation
- `predict.py` — lightweight inference script (CLI)
- `src/` — source code for data loading, feature engineering, and training
- `artifacts/` — saved model and preprocessing artifacts for inference

### Supporting files
- `outputs/` — training outputs, summaries, and sample predictions
- `notebooks/` — optional EDA / prototyping notebook(s)
- `data/charging_logs.csv` or `charging_logs.csv` — synthetic event-level charging log dataset

---

## 3) Approach Overview

### Model
- **Isolation Forest** (unsupervised anomaly detection)

### Feature engineering (high level)
- Core telemetry features (`voltage`, `current`, `power_kw`, `temperature_c`, `duration_sec`, `energy_kwh`)
- Time features (`hour`, `day_of_week`, `is_weekend`)
- Physics-consistency checks (for example, `voltage * current` versus `power_kw`)
- Session-sequence features (event index, elapsed time, within-session deltas)
- Station-relative baseline deviation features

### Final anomaly logic
The pipeline produces:
- `is_explicit_fault = (error_code != 0)`
- `is_model_anomaly = (anomaly_score >= threshold)`
- `is_anomaly = is_explicit_fault OR is_model_anomaly`

This hybrid logic is useful in anomaly detection settings because explicit fault codes are always flagged, while the model surfaces additional silent anomalies among events where no explicit fault code is raised.

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
python -m src.train --input data/charging_logs.csv --artifacts-dir artifacts --outputs-dir outputs --contamination 0.01 --threshold-percentile 99.5
```

### What training does

* Loads and parses the input CSV
* Builds engineered features
* Fits the anomaly detection model
* Computes anomaly scores on the full dataset
* Selects and saves an anomaly threshold
* Saves inference artifacts in `artifacts/`
* Saves metrics and inspection files in `outputs/`

---

## 6) Running Inference (`predict.py`)

Run inference with:

```bash
python predict.py --input data/charging_logs.csv --output outputs/predictions_test.csv --artifacts-dir artifacts --include-flags
```

### Inference behavior

* Loads the input CSV
* Loads saved model and preprocessing artifacts from `artifacts/`
* Rebuilds preprocessing and feature engineering consistently
* Computes `anomaly_score`
* Computes:

  * `is_explicit_fault = (error_code != 0)`
  * `is_model_anomaly = (anomaly_score >= threshold)`
  * `is_anomaly = is_explicit_fault OR is_model_anomaly`
* Writes output CSV with all original columns plus:

  * `anomaly_score`
  * `is_anomaly` (0/1)
  * optional debug columns: `is_model_anomaly`, `is_explicit_fault`

### Expected output format

The output CSV preserves the input rows and appends the anomaly-related columns.

---

## 7) Reproducibility / Smoke Test

Example commands used to validate final outputs:

```bash
python -m src.train --input data/charging_logs.csv --artifacts-dir artifacts --outputs-dir outputs --contamination 0.01 --threshold-percentile 99.5
python predict.py --input data/charging_logs.csv --output outputs/predictions_test.csv --artifacts-dir artifacts --include-flags
```

Optional quick output check:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("outputs/predictions_test.csv")

print("rows:", len(df))
print("has anomaly_score:", "anomaly_score" in df.columns)
print("has is_anomaly:", "is_anomaly" in df.columns)

# Optional debug flags if --include-flags was used
print("has is_model_anomaly:", "is_model_anomaly" in df.columns)
print("has is_explicit_fault:", "is_explicit_fault" in df.columns)

print(df["is_anomaly"].value_counts(dropna=False).to_dict())

faults = (df["error_code"].fillna(0).astype(int) != 0)
print("proxy faults (error_code!=0):", int(faults.sum()))
print("faults flagged:", int(((faults) & (df["is_anomaly"] == 1)).sum()))
PY
```

---

## 8) Evaluation Notes

This is an unsupervised anomaly detection pipeline. In this implementation:

* `error_code != 0` is treated as an explicit fault indicator and is always flagged in the final `is_anomaly`
* the Isolation Forest model flags additional anomalies based on telemetry behavior via `anomaly_score`
* `message` is not used as a supervised training label; it is used only for interpretation and EDA
* detailed results, proxy sanity checks, and tradeoffs are documented in `REPORT.md`

The focus is on:

* engineering judgment
* feature design
* anomaly triage usefulness
* reproducibility
* practical ML pipeline design

---

## 9) Optional Notebook

An optional notebook is included in `notebooks/` and contains exploratory analysis and prototyping work that informed feature engineering and modeling decisions.

The reproducible pipeline used for final inference is the Python code in `src/` together with `predict.py`.

---

## 10) Notes on AI Tool Usage

AI tools were used as part of the development workflow for brainstorming, code-review assistance, documentation refinement, and edge-case checks, while implementation, debugging, validation, and final decisions were manually executed and verified.

See `AI_USAGE.md` for full details.

---

## 11) Project Structure Summary

This repository includes:

* source code for preprocessing, EDA support, feature engineering, model training/evaluation, and inference
* `REPORT.md` for technical documentation
* `AI_USAGE.md` for AI usage and validation notes
* a lightweight inference script (`predict.py`)
* optional notebook(s) for exploratory work

---

## 12) Scope

This repository focuses on anomaly detection for EV charging event logs, with emphasis on reproducibility, structured engineering, and practical inference design.
