from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = [
    "station_id",
    "timestamp",
    "session_id",
    "voltage",
    "current",
    "power_kw",
    "temperature_c",
    "duration_sec",
    "energy_kwh",
    "error_code",
    "message",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_charging_logs(
    csv_path: Union[str, Path],
    parse_timestamp: bool = True,
    sort_rows: bool = True,
) -> pd.DataFrame:
    """
    Load ChargePoint charging logs CSV and perform basic validation.

    Parameters
    ----------
    csv_path : str | Path
        Path to input CSV.
    parse_timestamp : bool
        Whether to parse 'timestamp' column to datetime.
    sort_rows : bool
        Whether to sort by ['session_id', 'timestamp'] after parsing.

    Returns
    -------
    pd.DataFrame
        Loaded and validated dataframe.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_required_columns(df)

    if parse_timestamp:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        null_ts = int(df["timestamp"].isna().sum())
        if null_ts > 0:
            raise ValueError(f"Failed to parse {null_ts} timestamp values in 'timestamp' column.")

    if sort_rows:
        # Stable sorting for reproducibility
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.sort_values(["session_id", "timestamp"], kind="mergesort").reset_index(drop=True)
        else:
            df = df.sort_values(["session_id"], kind="mergesort").reset_index(drop=True)

    return df