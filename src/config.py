from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_failure_clinical_records_dataset.csv"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIGURE_PATH = ROOT / "reports" / "evaluation.png"

TARGET = "DEATH_EVENT"

# `time` is deliberately excluded: it is follow-up duration, only known after baseline.
FEATURES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]

BOOLEAN_FEATURES = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
DISPLAY_NAMES = {
    "age": "Age (years)",
    "anaemia": "Anaemia", "creatinine_phosphokinase": "Creatinine phosphokinase (mcg/L)",
    "diabetes": "Diabetes", "ejection_fraction": "Ejection fraction (%)",
    "high_blood_pressure": "High blood pressure", "platelets": "Platelets (kiloplatelets/mL)",
    "serum_creatinine": "Serum creatinine (mg/dL)", "serum_sodium": "Serum sodium (mEq/L)",
    "sex": "Sex assigned male", "smoking": "Smoking",
}
