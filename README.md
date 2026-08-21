# Heart Failure Mortality Risk Screening Assistant

An explainable, reproducible machine-learning prototype for estimating **mortality risk during follow-up in patients with heart failure**. It is an educational clinical decision-support demonstration, not a diagnostic, treatment, or triage system.

## Why this project is portfolio-ready

- Prevents temporal leakage by excluding `time`, a post-baseline follow-up variable.
- Compares logistic regression and random forest using stratified cross-validation.
- Calibrates the selected model's probabilities and evaluates discrimination, precision/recall, F1, and calibration.
- Uses an explicitly documented risk threshold rather than reporting accuracy alone.
- Includes a simple, safety-framed Streamlit interface, local counterfactual explanations, tests, Docker support, and a model-card summary.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Download the public UCI dataset, train, and start the app:

```bash
python scripts/download_data.py
python -m src.train
streamlit run app.py
```

Visit `http://localhost:8501`.

## Evaluation approach

A 75/25 stratified split is held out for final evaluation. Candidate models are compared with 5-fold stratified cross-validation on the training partition. The winning model is probability-calibrated, then evaluated on the untouched holdout set. Training writes the following reproducible artifacts:

- `models/risk_model.joblib` — the complete fitted pipeline and threshold
- `reports/metrics.json` — model-selection and holdout metrics
- `reports/evaluation.png` — ROC and calibration charts

## Responsible-AI limitations

- The dataset is small and may not represent local clinical populations.
- External validation is required before any real-world use.
- A prediction reflects associations in historical data, not causation.
- Thresholds and performance must be reviewed with clinicians for the intended setting.
- Never enter protected health information into a public deployment.

## Run tests and Docker

```bash
pytest
docker build -t heart-failure-risk-assistant .
docker run --rm -p 8501:8501 heart-failure-risk-assistant
```
