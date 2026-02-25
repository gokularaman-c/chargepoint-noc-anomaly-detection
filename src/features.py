from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASE_NUMERIC_COLS = [
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
    "duration_sec",
    "energy_kwh",
]

SESSION_DELTA_SOURCE_COLS = [
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
    "energy_kwh",
]

# We will compute station-relative z-scores for these
STATION_BASELINE_COLS = [
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
    "power_residual",
    "power_ratio_safe",
    "energy_power_gap",
]

OK_MESSAGE_PATTERN = r"^OK(\s*\(ref=.*\))?$"


def clean_message_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["message_clean"] = out["message"].astype(str).str.strip()
    return out


def add_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add proxy fault indicators for EDA/evaluation only.
    Do NOT use as model training targets in unsupervised training.
    """
    out = df.copy()

    out["proxy_fault_error"] = (out["error_code"] != 0).astype(int)

    if "message_clean" not in out.columns:
        out = clean_message_column(out)

    out["proxy_fault_message_clean"] = (
        ~out["message_clean"].str.match(OK_MESSAGE_PATTERN, case=False, na=False)
    ).astype(int)

    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("'timestamp' column is required for time features.")
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    return out


def add_physics_features(df: pd.DataFrame, min_expected_power: float = 0.1) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["calc_power_kw"] = (out["voltage"] * out["current"]) / 1000.0
    out["power_residual"] = out["power_kw"] - out["calc_power_kw"]

    out["power_ratio_safe"] = np.where(
        out["calc_power_kw"] > min_expected_power,
        out["power_kw"] / (out["calc_power_kw"] + eps),
        np.nan,
    )

    out["duration_hours"] = out["duration_sec"] / 3600.0
    out["energy_rate_kw"] = np.where(
        out["duration_hours"] > 0,
        out["energy_kwh"] / (out["duration_hours"] + eps),
        np.nan,
    )

    out["energy_power_gap"] = out["energy_rate_kw"] - out["power_kw"]

    return out


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sequential features within each session_id:
    - event index
    - elapsed seconds from session start
    - deltas and absolute deltas for key sensor columns
    """
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("'timestamp' column is required for session features.")
    if not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # Sort session-wise for meaningful diffs
    out = out.sort_values(["session_id", "timestamp"], kind="mergesort").copy()

    out["event_idx_in_session"] = out.groupby("session_id").cumcount()

    session_start_ts = out.groupby("session_id")["timestamp"].transform("min")
    out["elapsed_sec_from_session_start"] = (out["timestamp"] - session_start_ts).dt.total_seconds()

    for col in SESSION_DELTA_SOURCE_COLS:
        dcol = f"delta_{col}"
        out[dcol] = out.groupby("session_id")[col].diff()
        out[f"abs_delta_{col}"] = out[dcol].abs()

    return out


def fit_station_baselines(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute station-level mean/std baselines for selected columns.
    Also stores a __GLOBAL__ row for fallback during inference on unseen stations.
    """
    cols = list(columns) if columns is not None else list(STATION_BASELINE_COLS)

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for station baselines: {missing_cols}")

    grouped_mean = df.groupby("station_id")[cols].mean()
    grouped_std = df.groupby("station_id")[cols].std(ddof=0).replace(0, np.nan)

    mean_df = grouped_mean.add_suffix("__station_mean")
    std_df = grouped_std.add_suffix("__station_std")

    baselines = pd.concat([mean_df, std_df], axis=1).reset_index()

    # Add global fallback row
    global_mean = df[cols].mean().to_dict()
    global_std = df[cols].std(ddof=0).replace(0, np.nan).to_dict()

    global_row = {"station_id": "__GLOBAL__"}
    for c in cols:
        global_row[f"{c}__station_mean"] = global_mean.get(c, np.nan)
        global_row[f"{c}__station_std"] = global_std.get(c, np.nan)

    baselines = pd.concat([baselines, pd.DataFrame([global_row])], ignore_index=True)
    return baselines


def apply_station_baselines(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Add station-relative z-score features:
      <col>__station_z
    using station-level mean/std, with global fallback for unseen stations.
    """
    out = df.copy()
    cols = list(columns) if columns is not None else list(STATION_BASELINE_COLS)

    required_baseline_cols = ["station_id"]
    for c in cols:
        required_baseline_cols.extend([f"{c}__station_mean", f"{c}__station_std"])

    missing_b = [c for c in required_baseline_cols if c not in baselines.columns]
    if missing_b:
        raise ValueError(f"Baselines dataframe missing columns: {missing_b}")

    # Separate global fallback
    global_row = baselines.loc[baselines["station_id"] == "__GLOBAL__"]
    if global_row.empty:
        raise ValueError("Baselines must contain a '__GLOBAL__' fallback row.")

    global_row = global_row.iloc[0]
    station_only = baselines.loc[baselines["station_id"] != "__GLOBAL__"].copy()

    out = out.merge(station_only, on="station_id", how="left")

    eps = 1e-6
    for c in cols:
        mean_col = f"{c}__station_mean"
        std_col = f"{c}__station_std"

        # Fill unseen station baselines with global values
        out[mean_col] = out[mean_col].fillna(global_row[mean_col])
        out[std_col] = out[std_col].fillna(global_row[std_col])

        # avoid divide-by-zero / nan std
        out[std_col] = out[std_col].replace(0, np.nan)
        out[f"{c}__station_z"] = (out[c] - out[mean_col]) / (out[std_col] + eps)

    return out


def build_feature_frame(
    df: pd.DataFrame,
    fit_station_stats: bool = False,
    station_baselines: Optional[pd.DataFrame] = None,
    include_proxy_cols: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    End-to-end feature engineering pipeline.

    Returns
    -------
    (df_features, station_baselines_used_or_fitted)
    """
    out = df.copy()

    # Core feature blocks
    out = clean_message_column(out)
    if include_proxy_cols:
        out = add_proxy_columns(out)

    out = add_time_features(out)
    out = add_physics_features(out)
    out = add_session_features(out)

    fitted_baselines = None

    if fit_station_stats:
        fitted_baselines = fit_station_baselines(out, columns=STATION_BASELINE_COLS)
        out = apply_station_baselines(out, fitted_baselines, columns=STATION_BASELINE_COLS)
    elif station_baselines is not None:
        out = apply_station_baselines(out, station_baselines, columns=STATION_BASELINE_COLS)

    return out, fitted_baselines


def get_default_model_feature_columns(df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Default numeric feature set for anomaly detection model.
    If df is provided, returns only columns that exist.
    """
    cols = [
        # raw sensors
        "voltage",
        "current",
        "power_kw",
        "temperature_c",
        "duration_sec",
        "energy_kwh",
        # time context
        "hour",
        "day_of_week",
        "is_weekend",
        # physics consistency
        "calc_power_kw",
        "power_residual",
        "power_ratio_safe",
        "duration_hours",
        "energy_rate_kw",
        "energy_power_gap",
        # session dynamics
        "event_idx_in_session",
        "elapsed_sec_from_session_start",
        "delta_voltage",
        "delta_current",
        "delta_power_kw",
        "delta_temperature_c",
        "delta_energy_kwh",
        "abs_delta_voltage",
        "abs_delta_current",
        "abs_delta_power_kw",
        "abs_delta_temperature_c",
        "abs_delta_energy_kwh",
        # station-relative z features (if available)
        "voltage__station_z",
        "current__station_z",
        "power_kw__station_z",
        "temperature_c__station_z",
        "power_residual__station_z",
        "power_ratio_safe__station_z",
        "energy_power_gap__station_z",
    ]

    if df is None:
        return cols

    return [c for c in cols if c in df.columns]


def prepare_model_matrix(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    fit_fill_values: bool = False,
    fill_values: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Build model-ready numeric matrix:
    - selects feature columns
    - replaces inf/-inf with NaN
    - fills NaN using median values (fit mode) or provided fill_values (inference mode)

    Returns
    -------
    (X, fill_values_dict)
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    X = df[list(feature_columns)].copy()

    # Force numeric (safety)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    if fit_fill_values:
        fill_values = X.median(numeric_only=True).to_dict()
    elif fill_values is None:
        raise ValueError("fill_values must be provided when fit_fill_values=False")

    X = X.fillna(fill_values)

    # Final safety: if any column still fully NaN, fill with 0
    X = X.fillna(0.0)

    return X, dict(fill_values)