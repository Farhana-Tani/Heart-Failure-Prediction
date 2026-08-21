import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.config import DISPLAY_NAMES, FEATURES, METRICS_PATH, MODEL_PATH
from src.predict import explain_prediction, predict_risk

st.set_page_config(page_title="Heart Failure Risk Screening", page_icon="❤", layout="wide")
st.title("Heart Failure Mortality Risk Screening")
st.warning("Educational decision-support prototype only. It does not diagnose disease, recommend treatment, or replace a qualified clinician.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Add the dataset to data/ and run: python -m src.train")
    st.stop()

with st.form("patient_inputs"):
    left, right = st.columns(2)
    with left:
        age = st.number_input(DISPLAY_NAMES["age"], 18, 110, 60)
        anaemia = st.selectbox(DISPLAY_NAMES["anaemia"], [0, 1], format_func=lambda v: "Yes" if v else "No")
        cpk = st.number_input(DISPLAY_NAMES["creatinine_phosphokinase"], 1, 10000, 250)
        diabetes = st.selectbox(DISPLAY_NAMES["diabetes"], [0, 1], format_func=lambda v: "Yes" if v else "No")
        ef = st.number_input(DISPLAY_NAMES["ejection_fraction"], 1, 100, 38)
        hbp = st.selectbox(DISPLAY_NAMES["high_blood_pressure"], [0, 1], format_func=lambda v: "Yes" if v else "No")
    with right:
        platelets = st.number_input(DISPLAY_NAMES["platelets"], 1000.0, 1000000.0, 263000.0)
        creatinine = st.number_input(DISPLAY_NAMES["serum_creatinine"], 0.1, 20.0, 1.1, step=0.1)
        sodium = st.number_input(DISPLAY_NAMES["serum_sodium"], 100, 180, 137)
        sex = st.selectbox(DISPLAY_NAMES["sex"], [0, 1], format_func=lambda v: "Male" if v else "Female")
        smoking = st.selectbox(DISPLAY_NAMES["smoking"], [0, 1], format_func=lambda v: "Yes" if v else "No")
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    record = dict(zip(FEATURES, [age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking]))
    result = predict_risk(record)
    st.metric("Estimated mortality risk", f"{result['mortality_risk_probability']:.1%}")
    st.subheader(f"Screening result: {result['risk_band']} risk")
    st.caption(f"This uses a preselected screening threshold of {result['decision_threshold']:.0%}. A positive screen requires clinical review; it is not a diagnosis.")
    st.subheader("What influenced this estimate?")
    explanation = explain_prediction(record).head(6)
    explanation["feature"] = explanation["feature"].map(DISPLAY_NAMES)
    st.bar_chart(explanation.set_index("feature")["risk_change"])
    st.caption("Each bar estimates the change in risk when that input is compared with its training-set median, holding the other submitted inputs unchanged. It is not a causal explanation.")
    with st.expander("View submitted inputs"):
        st.dataframe(pd.DataFrame([record]), use_container_width=True)

if METRICS_PATH.exists():
    with st.expander("Model card summary"):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json({key: metrics[key] for key in ["selected_model", "test_roc_auc", "test_average_precision", "test_precision", "test_recall", "test_f1", "training_rows", "test_rows"]})
        st.caption("The model excludes `time` because it represents follow-up duration and would leak future information into a baseline prediction.")
