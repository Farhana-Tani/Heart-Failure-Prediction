"""Train and evaluate a calibrated heart-failure mortality risk model."""
import argparse
import json
from datetime import datetime, timezone
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from src.config import DATA_PATH, FEATURES, FIGURE_PATH, METRICS_PATH, MODEL_PATH
from src.data import load_dataset
from src.modeling import calibrated_model, candidate_models, choose_threshold


def main(data_path=DATA_PATH):
    frame = load_dataset(data_path)
    X, y = frame[FEATURES], frame["DEATH_EVENT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores, models = {}, candidate_models()
    for name, model in models.items():
        probabilities = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
        scores[name] = {"roc_auc": float(roc_auc_score(y_train, probabilities)), "average_precision": float(average_precision_score(y_train, probabilities))}
    winner_name = max(scores, key=lambda n: scores[n]["roc_auc"])
    model = calibrated_model(models[winner_name]); model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    predictions = probabilities >= threshold
    metrics = {"selected_model": winner_name, "decision_threshold": threshold, "test_roc_auc": float(roc_auc_score(y_test, probabilities)), "test_average_precision": float(average_precision_score(y_test, probabilities)), "test_precision": float(precision_score(y_test, predictions, zero_division=0)), "test_recall": float(recall_score(y_test, predictions, zero_division=0)), "test_f1": float(f1_score(y_test, predictions, zero_division=0)), "confusion_matrix": confusion_matrix(y_test, predictions).tolist(), "training_rows": int(len(X_train)), "test_rows": int(len(X_test)), "features": FEATURES, "trained_at_utc": datetime.now(timezone.utc).isoformat(), "candidate_cv_scores": scores}
    MODEL_PATH.parent.mkdir(exist_ok=True); METRICS_PATH.parent.mkdir(exist_ok=True)
    artifact = {"model": model, "threshold": threshold, "features": FEATURES, "metrics": metrics,
                "reference_values": X_train.median().to_dict()}
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[0]); axes[0].set_title("Holdout ROC curve")
    CalibrationDisplay.from_predictions(y_test, probabilities, n_bins=6, ax=axes[1]); axes[1].set_title("Holdout calibration")
    fig.tight_layout(); fig.savefig(FIGURE_PATH, dpi=160); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", default=str(DATA_PATH))
    main(parser.parse_args().data)
