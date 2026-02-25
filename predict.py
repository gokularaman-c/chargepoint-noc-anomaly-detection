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
    return parser.parse_args()


def _call_build_feature_frame_inference(df_raw, station_baselines):
    """
    Handles minor signature differences depending on how your src/features.py was written.
    """
    # Try common signatures in order
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

    # Last fallback if your function returns (df_feat, station_baselines) and ignores passed baselines
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
    """
    Handles minor return/signature differences.
    """
    # Common pattern: returns X only
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

    # Fallback if function expects no fit_fill_values flag in inference
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

    # Validate artifact files
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
    if isinstance(feat_out, tuple):
        df_feat = feat_out[0]
    else:
        df_feat = feat_out

    print("[4/6] Preparing model matrix...")
    X = _prepare_model_matrix_inference(df_feat, feature_columns, fill_values)

    print("[5/6] Scoring anomalies...")
    normal_score = model.score_samples(X)
    anomaly_score = -normal_score
    is_anomaly = (anomaly_score >= anomaly_threshold).astype(int)

    print("[6/6] Writing output CSV...")
    df_out = df_raw.copy()
    df_out["anomaly_score"] = anomaly_score
    df_out["is_anomaly"] = is_anomaly

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print("\n=== Inference Summary ===")
    print(f"Input rows: {len(df_raw)}")
    print(f"Output rows: {len(df_out)}")
    print(f"Anomaly threshold: {anomaly_threshold:.6f}")
    print(f"Predicted anomalies: {int(df_out['is_anomaly'].sum())} ({df_out['is_anomaly'].mean()*100:.3f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()