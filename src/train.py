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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "cm": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


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

    # Model-only anomaly flag (this is the ML detector output)
    df_scored["is_model_anomaly"] = (df_scored["anomaly_score"] >= threshold).astype(int)

    # Explicit faults (rule layer) - safe and consistent with predict.py
    df_scored["is_explicit_fault"] = (df_scored["proxy_fault_error"].astype(int) != 0).astype(int)

    # Final hybrid anomaly flag (what NOC would page on)
    df_scored["is_anomaly"] = (
        (df_scored["is_model_anomaly"] == 1) | (df_scored["is_explicit_fault"] == 1)
    ).astype(int)

    # Proxy labels (for sanity check only)
    y_true = df_scored["proxy_fault_error"].astype(int).values
    y_pred_model = df_scored["is_model_anomaly"].astype(int).values
    y_pred_hybrid = df_scored["is_anomaly"].astype(int).values

    model_metrics = _compute_metrics(y_true, y_pred_model)
    hybrid_metrics = _compute_metrics(y_true, y_pred_hybrid)

    # Precision@K where K = number of proxy faults (common unsupervised sanity check)
    k = int(df_scored["proxy_fault_error"].sum())
    topk_idx = df_scored["anomaly_score"].nlargest(k).index if k > 0 else []
    topk_precision = float(df_scored.loc[topk_idx, "proxy_fault_error"].mean()) if k > 0 else 0.0

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
        "threshold_percentile": float(args.threshold_percentile),
        "anomaly_score_threshold": threshold,
        "contamination_fit_param": float(args.contamination),
        "fit_on_normal_only": bool(args.fit_on_normal_only),
        "proxy_fault_count": int(df_scored["proxy_fault_error"].sum()),
        "proxy_fault_rate": float(df_scored["proxy_fault_error"].mean()),
        "explicit_fault_count": int(df_scored["is_explicit_fault"].sum()),
        "explicit_fault_rate": float(df_scored["is_explicit_fault"].mean()),
        "model_anomaly_count": int(df_scored["is_model_anomaly"].sum()),
        "model_anomaly_rate": float(df_scored["is_model_anomaly"].mean()),
        "final_anomaly_count": int(df_scored["is_anomaly"].sum()),
        "final_anomaly_rate": float(df_scored["is_anomaly"].mean()),
        "proxy_eval_model_only": {
            "tn": model_metrics["tn"],
            "fp": model_metrics["fp"],
            "fn": model_metrics["fn"],
            "tp": model_metrics["tp"],
            "precision": model_metrics["precision"],
            "recall": model_metrics["recall"],
            "f1": model_metrics["f1"],
            "confusion_matrix": model_metrics["cm"].tolist(),
            "precision_at_k_where_k_equals_proxy_fault_count": float(topk_precision),
        },
        "proxy_eval_hybrid": {
            "tn": hybrid_metrics["tn"],
            "fp": hybrid_metrics["fp"],
            "fn": hybrid_metrics["fn"],
            "tp": hybrid_metrics["tp"],
            "precision": hybrid_metrics["precision"],
            "recall": hybrid_metrics["recall"],
            "f1": hybrid_metrics["f1"],
            "confusion_matrix": hybrid_metrics["cm"].tolist(),
        },
    }

    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    preview_cols = [
        "station_id", "timestamp", "session_id",
        "voltage", "current", "power_kw", "temperature_c", "duration_sec", "energy_kwh",
        "error_code", "message_clean", "proxy_fault_error",
        "anomaly_score",
        "is_model_anomaly", "is_explicit_fault", "is_anomaly",
        "power_residual", "power_ratio_safe", "energy_power_gap",
    ]
    preview_cols = [c for c in preview_cols if c in df_scored.columns]

    df_scored.sort_values("anomaly_score", ascending=False).head(500)[preview_cols].to_csv(
        outputs_dir / "top_anomalies.csv", index=False
    )
    df_scored[preview_cols].to_csv(outputs_dir / "train_scored_preview.csv", index=False)

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
    print(f"Explicit faults (error_code!=0): {int(df_scored['is_explicit_fault'].sum())} ({df_scored['is_explicit_fault'].mean()*100:.3f}%)")
    print(f"Model anomalies (score>=threshold): {int(df_scored['is_model_anomaly'].sum())} ({df_scored['is_model_anomaly'].mean()*100:.3f}%)")
    print(f"Final anomalies (OR): {int(df_scored['is_anomaly'].sum())} ({df_scored['is_anomaly'].mean()*100:.3f}%)")

    print("\n=== Proxy Evaluation (sanity check only) ===")
    print("Model-only vs proxy faults (error_code!=0):")
    print(f"Precision: {model_metrics['precision']:.4f}")
    print(f"Recall:    {model_metrics['recall']:.4f}")
    print(f"F1:        {model_metrics['f1']:.4f}")
    print(f"Precision@K (K=#proxy_faults): {topk_precision:.4f}")
    print(f"Confusion Matrix [ [tn, fp], [fn, tp] ]:\n{model_metrics['cm']}")

    print("\nHybrid (explicit faults OR model) vs proxy faults:")
    print(f"Precision: {hybrid_metrics['precision']:.4f}")
    print(f"Recall:    {hybrid_metrics['recall']:.4f}")
    print(f"F1:        {hybrid_metrics['f1']:.4f}")
    print(f"Confusion Matrix [ [tn, fp], [fn, tp] ]:\n{hybrid_metrics['cm']}")


if __name__ == "__main__":
    main()