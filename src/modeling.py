"""Model construction, calibration, and threshold selection."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def candidate_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("classifier", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", random_state=random_state, n_jobs=-1))]),
    }


def calibrated_model(base_model):
    return CalibratedClassifierCV(base_model, method="sigmoid", cv=5)


def choose_threshold(y_true, probabilities, minimum_recall: float = 0.80) -> float:
    from sklearn.metrics import f1_score, recall_score
    candidates = np.arange(0.10, 0.91, 0.01)
    feasible = [(f1_score(y_true, probabilities >= t), t) for t in candidates if recall_score(y_true, probabilities >= t, zero_division=0) >= minimum_recall]
    return float(max(feasible)[1]) if feasible else float(max((f1_score(y_true, probabilities >= t), t) for t in candidates)[1])
