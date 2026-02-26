# ChargePoint Take-Home Exercise — Anomaly Detection for EV Charging Logs

## 1) Problem Understanding

The task is to build an anomaly detection pipeline for EV charging station logs and provide a reproducible inference script (`predict.py`) that labels anomalies in a CSV file.

I implemented an unsupervised anomaly detection approach using **Isolation Forest** on engineered tabular features derived from charging telemetry (voltage, current, power, temperature, duration, energy) and temporal/session context features.

The solution is designed to be:
- reproducible,
- deployable as a CLI script,
- practical for NOC-style monitoring workflows.

---

## 2) Dataset Overview

Input dataset fields (provided):
- `station_id`
- `timestamp`
- `session_id`
- `voltage`
- `current`
- `power_kw`
- `temperature_c`
- `duration_sec`
- `energy_kwh`
- `error_code`
- `message`

### Observed dataset summary
- Rows: **199,566**
- Columns: **11**
- Stations: **20**
- Sessions: **4,000**
- Missing values: **0**
- `error_code != 0` rows (proxy faults): **1,109** (**0.556%**)

### Notes
`error_code` and `message` were treated as **proxy fault indicators for sanity-check evaluation only**, not as training targets for the anomaly detector.

---

## 3) EDA Highlights

### Data quality
- No missing values in the provided CSV.
- Timestamps parsed successfully.
- Error codes are highly imbalanced (majority = `0`).

### Fault/message patterns
Most rows are `OK`, but there are repeated non-OK messages such as:
- Inconsistent metering data observed
- Unexpected reboot during active session
- Repeated handshake failures across multiple modules
- Unknown hardware fault code: 0x7F3A
- Severe voltage instability detected

### Distribution and telemetry observations
- `power_kw`, `temperature_c`, and `voltage` have realistic central ranges.
- There are some extreme values/outliers (e.g., very low/high voltage, temperature spikes, negative power values) that are relevant for anomaly detection.
- Session lengths vary, with median session event count around ~50 and upper tail around ~80 events/session.

---

## 4) Feature Engineering

I built a reusable feature pipeline with the following feature groups.

### A. Core numeric telemetry
- `voltage`
- `current`
- `power_kw`
- `temperature_c`
- `duration_sec`
- `energy_kwh`

### B. Temporal features
- `hour`
- `day_of_week`
- `is_weekend`

### C. Physics consistency features
These help detect sensor inconsistencies or hardware behavior deviations:
- `calc_power_kw = voltage * current / 1000`
- `power_residual = power_kw - calc_power_kw`
- `power_ratio_safe` (ratio computed only when expected power is sufficiently > 0)
- `energy_rate_kw = energy_kwh / duration_hours`
- `energy_power_gap = energy_rate_kw - power_kw`

### D. Session-sequence features (within `session_id`)
Sorted by timestamp and computed deltas:
- event index in session
- elapsed seconds from session start
- deltas for voltage/current/power/temperature/energy
- absolute deltas (magnitude of change)

### E. Station-relative normalization features
Station-level baseline statistics are fitted and used to derive station-relative deviations (to detect “abnormal for this station” behavior).

### Final feature matrix
- Feature frame shape: **(199,566, 56)**
- Model feature count used for training: **34**

---

## 5) Model Choice and Rationale

### Selected model: Isolation Forest (unsupervised)

I selected **Isolation Forest** as the primary model because it is:
- fast and scalable for tabular telemetry,
- robust as a baseline for rare-event detection,
- easy to operationalize in a CLI inference script,
- practical for a NOC setting where reproducibility and speed matter.

### Why not use a heavier model first?
For a take-home assignment with production-style expectations (`predict.py`, reproducibility, clear reasoning), a strong and well-engineered Isolation Forest baseline is a better tradeoff than a complex deep learning model with higher implementation/debug risk.

I also considered lightweight tabular anomaly detectors such as **ECOD** and **COPOD** as potential baselines; however, I prioritized a strong end-to-end **Isolation Forest** pipeline for this submission due to its speed, operational simplicity, and reproducible deployment fit for the take-home requirements.

---

## 6) Training Setup

### Primary run (used for final pipeline)
- Model: Isolation Forest
- Contamination: `0.01`
- Thresholding: anomaly score percentile = **99.5**
- Training rows: **all rows (199,566)**

### Alternative run (comparison / ablation)
I also trained a variant using `--fit-on-normal-only` (proxy-normal rows only) to test whether excluding proxy-fault rows improves separation.

Observation: the proxy-evaluation outcome remained effectively unchanged (no overlap at the selected threshold), suggesting the proxy fault labels are not strongly separable in the current unsupervised feature space at this operating threshold.

---

## 7) Results

### Definitions (important)
This pipeline produces three related flags:

- **Explicit faults**: `error_code != 0` (treated as actionable anomalies in a NOC setting)
- **Model anomalies**: `anomaly_score >= threshold` from Isolation Forest
- **Final anomalies**: `is_anomaly = explicit_fault OR model_anomaly`

This makes the system practical for operations: known fault codes are always flagged, and the ML model surfaces additional “silent anomalies” when no explicit fault code is raised.

### Primary training summary
- Rows scored: **199,566**
- Model features: **34**
- Threshold (p99.5): **0.641829**
- Explicit faults (`error_code != 0`): **1,109** (**0.556%**)
- Model anomalies (score ≥ threshold): **998** (**0.500%**)
- Final anomalies (OR): **2,107** (**1.056%**)

### Proxy evaluation (sanity check only)
Since ground-truth anomaly labels are not provided, I use `error_code != 0` only as a proxy sanity check.

**Model-only vs proxy faults (`error_code != 0`):**
- Precision: **0.0000**
- Recall: **0.0000**
- F1: **0.0000**
- Precision@K (K = #proxy_faults): **0.0000**
- Confusion Matrix `[[tn, fp], [fn, tp]]`: `[[197459, 998], [1109, 0]]`

**Hybrid (explicit faults OR model) vs proxy faults:**
- Precision: **0.5263**
- Recall: **1.0000**
- F1: **0.6897**
- Confusion Matrix `[[tn, fp], [fn, tp]]`: `[[197459, 998], [0, 1109]]`

### Interpretation
The Isolation Forest **model-only** anomalies do not overlap with proxy faults at the strict p99.5 operating point, indicating the model is prioritizing a different anomaly family (e.g., telemetry inconsistencies, session-dynamics irregularities, station-relative deviations) than explicit fault-code events.

The **hybrid** rule + ML approach reflects a realistic NOC workflow:
- explicit fault codes are always actionable and should be flagged deterministically,
- ML is used to surface additional “silent anomalies” among `error_code == 0` events for triage and investigation.

---

## 8) Inference Pipeline (`predict.py`)

I implemented a CLI inference script:

```bash
python predict.py --input <input_csv> --output <output_csv> --artifacts-dir artifacts --include-flags
```

### Behavior
- Loads saved model + artifacts
- Rebuilds inference features with the same preprocessing pipeline
- Computes `anomaly_score`
- Computes:
  - `is_explicit_fault = (error_code != 0)`
  - `is_model_anomaly = (anomaly_score >= threshold)`
  - `is_anomaly = is_explicit_fault OR is_model_anomaly`
- Writes output CSV with:
  - `anomaly_score`
  - `is_anomaly`
  - (optional debug) `is_model_anomaly`, `is_explicit_fault`

This script was tested successfully on the provided dataset and produces consistent counts between training and inference.

---

## 9) Production Considerations (NOC-Focused)

### Strengths of current approach
- Fast scoring for large telemetry logs
- Reproducible pipeline with saved artifacts
- Easy deployment as batch CLI inference
- Feature engineering captures physics and session dynamics (not just raw values)
- Hybrid decisioning is operationally realistic: explicit fault codes are always flagged, and ML surfaces additional silent anomalies

### Practical limitations
- Threshold is global (one threshold may not be optimal for every station)
- No text semantics from message field are used
- Proxy labels are weak and may not represent all anomaly types
- Concept drift (firmware updates, hardware replacements, weather/season effects) can change distributions over time

### Recommended next improvements
1. Station-specific thresholds (or station-aware calibration) to reduce false positives
2. Time-windowed retraining / drift monitoring with alert-volume tracking
3. Hybrid approach (implemented): explicit fault codes (`error_code != 0`) are flagged deterministically; ML flags additional silent anomalies. Future extension: add curated message-pattern rules and severity bands based on anomaly_score
4. Text feature extraction from message (keyword features or embeddings) to better capture protocol/software-level faults
5. Ranking/triage outputs (severity bands + “why flagged” signals such as power mismatch, temperature delta, station deviation)

---

## 10) What I Would Improve With More Time
	•	Benchmark multiple unsupervised detectors (e.g., LOF, COPOD, ECOD) under the same feature pipeline and threshold-selection protocol for a fair comparison
	•	Add per-station evaluation slices and threshold tuning
	•	Add more explainability outputs (top contributing features per anomaly)
	•	Add unit tests for feature engineering and inference consistency
	•	Package the project with a lightweight config system for production runs

---

## 11) Conclusion

This submission provides a practical, reproducible anomaly detection baseline for EV charging log monitoring with:
	•	a robust feature engineering pipeline,
	•	an unsupervised Isolation Forest model,
	•	a production-friendly predict.py inference script.

The proxy-label mismatch highlights an important operational reality: telemetry anomalies and explicit fault codes may represent different failure modes. This makes the solution a good baseline for NOC workflows, with clear extension paths for hybrid rule + ML monitoring.

