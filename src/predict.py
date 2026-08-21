"""Reusable prediction API for the trained model."""
import joblib
import pandas as pd
from src.config import MODEL_PATH
from src.data import validate_patient


def predict_risk(record: dict, model_path=MODEL_PATH) -> dict:
    artifact = joblib.load(model_path)
    probability = float(artifact["model"].predict_proba(validate_patient(record))[:, 1][0])
    threshold = artifact["threshold"]
    return {
        "mortality_risk_probability": probability,
        "risk_band": "High" if probability >= threshold else "Lower",
        "screen_positive": probability >= threshold,
        "decision_threshold": threshold,
    }


def explain_prediction(record: dict, model_path=MODEL_PATH) -> pd.DataFrame:
    """Show local, counterfactual feature influence against training-set medians.

    Positive values mean the submitted feature value raises estimated risk compared
    with replacing that one feature by its training-set median. This is an
    explanatory aid, not a causal interpretation.
    """
    artifact = joblib.load(model_path)
    patient = validate_patient(record)
    original_probability = float(artifact["model"].predict_proba(patient)[:, 1][0])
    rows = []
    for feature in artifact["features"]:
        counterfactual = patient.copy()
        counterfactual.loc[0, feature] = artifact["reference_values"][feature]
        counterfactual_probability = float(artifact["model"].predict_proba(counterfactual)[:, 1][0])
        rows.append({"feature": feature, "risk_change": original_probability - counterfactual_probability})
    return pd.DataFrame(rows).sort_values("risk_change", key=lambda s: s.abs(), ascending=False)
