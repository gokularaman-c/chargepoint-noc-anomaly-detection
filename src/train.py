from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.data import load_charging_logs
from src.features import (
    build_feature_frame,
    get_default_model_feature_columns,
    prepare_model_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly detection model for ChargePoint logs.")
    parser.add_argument("--input", type=str, default="data/charging_logs.csv", help="Path to input CSV")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save model artifacts")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Directory to save outputs/metrics")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Expected anomaly fraction for IsolationForest (used during fit)",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.5,
        help="Percentile on anomaly_score used for binary anomaly flag",
    )
    parser.add_argument(
        "--fit-on-normal-only",
        action="store_true",
        help="If set, train only on rows where error_code == 0 (semi-supervised style).",
    )
    return parser.parse_args()


def safe_float(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    artifacts_dir = Path(args.artifacts_dir)
    outputs_dir = Path(args.outputs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading data...")
    df_raw = load_charging_logs(input_path)

    print("[2/7] Building features + fitting station baselines...")
    df_feat, station_baselines = build_feature_frame(
        df_raw,
        fit_station_stats=True,
        include_proxy_cols=True,
    )

    feature_columns = get_default_model_feature_columns(df_feat)

    print("[3/7] Preparing model matrix...")
    X_all, fill_values = prepare_model_matrix(
        df_feat,
        feature_columns,
        fit_fill_values=True,
    )

    # Optional semi-supervised training on proxy-normal rows only
    if args.fit_on_normal_only:
        train_mask = df_feat["proxy_fault_error"] == 0
        X_train = X_all.loc[train_mask].copy()
        print(f"[4/7] Training on proxy-normal rows only: {X_train.shape[0]} rows")
    else:
        train_mask = pd.Series(True, index=df_feat.index)
        X_train = X_all
        print(f"[4/7] Training on all rows: {X_train.shape[0]} rows")

    model = IsolationForest(
        n_estimators=300,
        contamination=args.contamination,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train)

    print("[5/7] Scoring rows...")
    # sklearn score_samples: higher = more normal, lower = more anomalous
    normal_score = model.score_samples(X_all)
    anomaly_score = -normal_score  # higher = more anomalous

    df_scored = df_feat.copy()
    df_scored["normal_score"] = normal_score
    df_scored["anomaly_score"] = anomaly_score

    threshold = float(np.percentile(df_scored["anomaly_score"], args.threshold_percentile))
    df_scored["is_anomaly"] = (df_scored["anomaly_score"] >= threshold).astype(int)

    # Proxy evaluation (for analysis only)
    y_true = df_scored["proxy_fault_error"].astype(int).values
    y_pred = df_scored["is_anomaly"].astype(int).values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Precision@K where K = number of proxy faults (common unsupervised sanity check)
    k = int(df_scored["proxy_fault_error"].sum())
    topk_idx = df_scored["anomaly_score"].nlargest(k).index if k > 0 else []
    topk_precision = (
        float(df_scored.loc[topk_idx, "proxy_fault_error"].mean()) if k > 0 else 0.0
    )

    # Save artifacts
    print("[6/7] Saving artifacts...")
    joblib.dump(model, artifacts_dir / "model.pkl")

    with open(artifacts_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(artifacts_dir / "fill_values.json", "w") as f:
        json.dump({k: safe_float(v) for k, v in fill_values.items()}, f, indent=2)

    station_baselines.to_csv(artifacts_dir / "station_baselines.csv", index=False)

    threshold_payload = {
        "threshold_percentile": float(args.threshold_percentile),
        "anomaly_score_threshold": threshold,
        "score_definition": "anomaly_score = -IsolationForest.score_samples(X); higher means more anomalous",
        "contamination_fit_param": float(args.contamination),
        "fit_on_normal_only": bool(args.fit_on_normal_only),
    }
    with open(artifacts_dir / "threshold.json", "w") as f:
        json.dump(threshold_payload, f, indent=2)

    metrics = {
        "n_rows": int(len(df_scored)),
        "n_features": int(len(feature_columns)),
        "proxy_fault_count": int(df_scored["proxy_fault_error"].sum()),
        "proxy_fault_rate": float(df_scored["proxy_fault_error"].mean()),
        "predicted_anomaly_count": int(df_scored["is_anomaly"].sum()),
        "predicted_anomaly_rate": float(df_scored["is_anomaly"].mean()),
        "threshold_percentile": float(args.threshold_percentile),
        "anomaly_score_threshold": threshold,
        "proxy_eval_confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "proxy_eval_precision": float(precision),
        "proxy_eval_recall": float(recall),
        "proxy_eval_f1": float(f1),
        "precision_at_k_where_k_equals_proxy_fault_count": float(topk_precision),
    }

    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save top anomalies preview for report/manual inspection
    preview_cols = [
        "station_id", "timestamp", "session_id",
        "voltage", "current", "power_kw", "temperature_c", "duration_sec", "energy_kwh",
        "error_code", "message_clean", "proxy_fault_error",
        "anomaly_score", "is_anomaly",
        "power_residual", "power_ratio_safe", "energy_power_gap",
    ]
    preview_cols = [c for c in preview_cols if c in df_scored.columns]

    df_scored.sort_values("anomaly_score", ascending=False).head(500)[preview_cols].to_csv(
        outputs_dir / "top_anomalies.csv", index=False
    )

    # Full prediction dump (can be large but useful during development)
    df_scored[preview_cols].to_csv(outputs_dir / "train_scored_preview.csv", index=False)

    # Save baselines/summary stats for report
    summary = {
        "feature_columns_sample": feature_columns[:10],
        "feature_columns_count": len(feature_columns),
        "station_baselines_shape": list(station_baselines.shape),
        "raw_shape": list(df_raw.shape),
        "feature_frame_shape": list(df_feat.shape),
    }
    with open(outputs_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[7/7] Done.")
    print("\n=== Training Summary ===")
    print(f"Rows: {len(df_scored)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Threshold (p{args.threshold_percentile}): {threshold:.6f}")
    print(f"Predicted anomalies: {df_scored['is_anomaly'].sum()} ({df_scored['is_anomaly'].mean()*100:.3f}%)")
    print(f"Proxy faults: {df_scored['proxy_fault_error'].sum()} ({df_scored['proxy_fault_error'].mean()*100:.3f}%)")
    print("\n=== Proxy Evaluation (for sanity check only) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision@K (K=#proxy_faults): {topk_precision:.4f}")
    print(f"Confusion Matrix [ [tn, fp], [fn, tp] ]:\n{cm}")


if __name__ == "__main__":
    main()