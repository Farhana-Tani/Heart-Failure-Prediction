"""Input validation and dataset loading."""
from pathlib import Path
import pandas as pd
from src.config import FEATURES, TARGET


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame[TARGET].isna().any() or not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0 and 1.")
    return frame


def validate_patient(record: dict) -> pd.DataFrame:
    missing = set(FEATURES) - set(record)
    if missing:
        raise ValueError(f"Patient record is missing: {sorted(missing)}")
    patient = pd.DataFrame([{name: record[name] for name in FEATURES}])
    if patient.isna().any().any():
        raise ValueError("Patient inputs cannot be empty.")
    return patient
