import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_charging_logs
from src.features import build_feature_frame, prepare_model_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly inference on charging logs CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory with saved model artifacts")
    parser.add_argument(
        "--include-score",
        action="store_true",
        help="Include anomaly_score column in output CSV (default: included anyway)",
    )
    parser.add_argument(
        "--include-flags",
        action="store_true",
        help="Include is_model_anomaly and is_explicit_fault columns in output CSV",
    )
    return parser.parse_args()


def _call_build_feature_frame_inference(df_raw, station_baselines):
    tried = []
    try:
        return build_feature_frame(
            df_raw,
            fit_station_stats=False,
            station_baselines=station_baselines,
            include_proxy_cols=False,
        )
    except TypeError as e:
        tried.append(str(e))

    try:
        return build_feature_frame(
            df_raw,
            fit_station_stats=False,
            station_stats_df=station_baselines,
            include_proxy_cols=False,
        )
    except TypeError as e:
        tried.append(str(e))

    try:
        return build_feature_frame(
            df_raw,
            fit_station_stats=False,
            station_baselines_df=station_baselines,
            include_proxy_cols=False,
        )
    except TypeError as e:
        tried.append(str(e))

    try:
        return build_feature_frame(
            df_raw,
            fit_station_stats=False,
            include_proxy_cols=False,
        )
    except TypeError as e:
        tried.append(str(e))
        raise RuntimeError(
            "Could not call build_feature_frame for inference. "
            "Please check parameter names in src/features.py.\n"
            + "\n".join(tried)
        )


def _prepare_model_matrix_inference(df_feat, feature_columns, fill_values):
    try:
        out = prepare_model_matrix(
            df_feat,
            feature_columns,
            fit_fill_values=False,
            fill_values=fill_values,
        )
        if isinstance(out, tuple):
            return out[0]
        return out
    except TypeError:
        pass

    out = prepare_model_matrix(
        df_feat,
        feature_columns,
        fill_values=fill_values,
    )
    if isinstance(out, tuple):
        return out[0]
    return out


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    artifacts_dir = Path(args.artifacts_dir)

    required_files = [
        artifacts_dir / "model.pkl",
        artifacts_dir / "feature_columns.json",
        artifacts_dir / "fill_values.json",
        artifacts_dir / "station_baselines.csv",
        artifacts_dir / "threshold.json",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifact files:\n" + "\n".join(missing))

    print("[1/6] Loading input CSV...")
    df_raw = load_charging_logs(input_path)

    print("[2/6] Loading artifacts...")
    model = joblib.load(artifacts_dir / "model.pkl")
    with open(artifacts_dir / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    with open(artifacts_dir / "fill_values.json", "r") as f:
        fill_values = json.load(f)
    with open(artifacts_dir / "threshold.json", "r") as f:
        threshold_info = json.load(f)
    station_baselines = pd.read_csv(artifacts_dir / "station_baselines.csv")

    anomaly_threshold = float(threshold_info["anomaly_score_threshold"])

    print("[3/6] Building inference features...")
    feat_out = _call_build_feature_frame_inference(df_raw, station_baselines)
    df_feat = feat_out[0] if isinstance(feat_out, tuple) else feat_out

    print("[4/6] Preparing model matrix...")
    X = _prepare_model_matrix_inference(df_feat, feature_columns, fill_values)

    print("[5/6] Scoring anomalies...")
    normal_score = model.score_samples(X)
    anomaly_score = -normal_score

    # Model-based anomaly flag
    is_model_anomaly = (anomaly_score >= anomaly_threshold)

    # Explicit faults (rule layer) - SAFE: only error_code!=0
    # (Avoid message-based logic to prevent over-flagging on unseen datasets.)
    error_code_series = df_raw.get("error_code", pd.Series([0] * len(df_raw)))
    is_explicit_fault = error_code_series.fillna(0).astype(int).ne(0)

    # Final anomaly flag = explicit faults OR model anomaly
    is_anomaly = (is_explicit_fault | is_model_anomaly).astype(int)

    print("[6/6] Writing output CSV...")
    df_out = df_raw.copy()

    # Always include anomaly_score + is_anomaly (required output)
    df_out["anomaly_score"] = anomaly_score
    df_out["is_anomaly"] = is_anomaly

    # Optional debug flags
    if args.include_flags:
        df_out["is_model_anomaly"] = is_model_anomaly.astype(int)
        df_out["is_explicit_fault"] = is_explicit_fault.astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print("\n=== Inference Summary ===")
    print(f"Input rows: {len(df_raw)}")
    print(f"Output rows: {len(df_out)}")
    print(f"Anomaly threshold: {anomaly_threshold:.6f}")
    print(f"Explicit faults (error_code!=0): {int(is_explicit_fault.sum())} ({is_explicit_fault.mean()*100:.3f}%)")
    print(f"Model anomalies (score>=threshold): {int(is_model_anomaly.sum())} ({is_model_anomaly.mean()*100:.3f}%)")
    print(f"Final anomalies (OR): {int(df_out['is_anomaly'].sum())} ({df_out['is_anomaly'].mean()*100:.3f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()