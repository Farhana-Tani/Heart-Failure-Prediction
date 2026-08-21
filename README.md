# Heart Failure Mortality Risk Screening Assistant

An explainable and reproducible machine-learning application for estimating **mortality risk during follow-up in patients with heart failure**.

> **Important:** This project is an educational machine-learning and clinical decision-support prototype. It is **not a medical diagnostic, treatment, or triage system** and must not be used to make real clinical decisions.

## 📌 Project Overview

Heart failure is a major cardiovascular condition associated with significant morbidity and mortality. Early identification of patients who may have a higher mortality risk can potentially support clinical research and risk-stratification studies.

This project develops a machine-learning pipeline that uses routinely available clinical features to estimate the probability of mortality during follow-up.

The project focuses on:

- Machine-learning-based risk prediction
- Explainable predictions
- Probability calibration
- Reproducible model training
- Proper train/validation/test methodology
- Evaluation using clinically relevant metrics
- A user-friendly Streamlit interface
- Docker support
- Automated testing

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a reproducible machine-learning pipeline for heart-failure mortality-risk prediction.
2. Compare multiple candidate classification models.
3. Select the best-performing model using cross-validation.
4. Calibrate predicted probabilities.
5. Evaluate the final model on an untouched holdout dataset.
6. Provide an interpretable risk estimate through a Streamlit application.
7. Package the application for reproducible deployment using Docker.
8. Demonstrate responsible-AI considerations for healthcare machine learning.

---

## 🧠 Machine Learning Approach

The project uses a structured machine-learning workflow.

### Data preprocessing

The pipeline handles:

- Numerical features
- Categorical/binary features
- Missing-value processing where required
- Feature preprocessing
- Model training and validation

All preprocessing steps are included in the machine-learning pipeline to reduce the risk of data leakage.

### Dataset split

The dataset is divided using a **75/25 stratified split**:

- **75%** → model development
- **25%** → final holdout evaluation

The holdout set is kept untouched during model selection.

### Cross-validation

Candidate models are compared using:

**5-fold stratified cross-validation**

This provides a more reliable estimate of model performance during the model-selection stage.

### Probability calibration

The selected model is probability-calibrated so that the predicted risk probabilities are more meaningful and interpretable.

The final calibrated model is then evaluated on the untouched holdout dataset.

---

## 📊 Model Evaluation

The project evaluates model performance using several metrics, including:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Calibration performance

ROC and calibration plots are generated as part of the evaluation process.

The project stores evaluation results in:

```text
reports/metrics.json
reports/evaluation.png
```

---

## 🔍 Explainability

Because this is a healthcare-related machine-learning project, interpretability is important.

The application is designed to present the model's estimated risk in an understandable way rather than simply returning a binary prediction.

The project emphasizes:

- Risk probability
- Model transparency
- Reproducibility
- Appropriate interpretation of predictions
- Clear communication of limitations

A model prediction should be interpreted as a statistical estimate based on the training data, **not as a diagnosis or medical conclusion**.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application.

Users can provide the required patient features through the web interface and receive a model-generated estimated mortality risk.

The application is intended for:

- Machine-learning demonstrations
- Academic projects
- Research prototyping
- Learning healthcare AI
- Demonstrating model deployment

It is **not intended for clinical use**.

---

## 📁 Project Structure

```text
Heart-Failure-Risk-Assistant/
│
├── app.py
├── Dockerfile
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── ...
│
├── models/
│   └── risk_model.joblib
│
├── reports/
│   ├── metrics.json
│   └── evaluation.png
│
├── scripts/
│   └── download_data.py
│
├── src/
│   ├── ...
│   └── train.py
│
├── tests/
│   └── ...
│
└── Heart-Failure-Prediction/
    └── ...
```

> The exact contents of some directories may change as the project evolves.

---

## ⚙️ Requirements

The project requires:

- Python 3.10+
- scikit-learn
- pandas
- NumPy
- Streamlit
- joblib
- matplotlib
- pytest

The complete dependency list is provided in:

```text
requirements.txt
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Farhana-Tani/Heart-Failure-Prediction.git
```

Move into the project directory:

```bash
cd Heart-Failure-Prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

If activation is successful, your terminal should show something similar to:

```text
(.venv)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download the Dataset

Run:

```bash
python scripts/download_data.py
```

The project uses a publicly available heart-failure dataset for educational and research purposes.

Before using the dataset, review its original documentation and licensing/usage conditions.

---

## 🏋️ Train the Model

Run:

```bash
python -m src.train
```

The training process will:

1. Load the dataset.
2. Prepare the features and target.
3. Split the data into development and holdout sets.
4. Perform stratified cross-validation.
5. Compare candidate models.
6. Select the best-performing model.
7. Calibrate the model probabilities.
8. Evaluate the final model.
9. Save the trained model.
10. Generate evaluation reports.

The trained model is saved as:

```text
models/risk_model.joblib
```

---

## ▶️ Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

in your web browser.

---

## 🧪 Run Tests

Run the test suite with:

```bash
pytest
```

Tests help verify that the project's core functionality works as expected.

---

## 🐳 Docker

The project also includes a Dockerfile.

### Build the Docker image

```bash
docker build -t heart-failure-risk-assistant .
```

### Run the container

```bash
docker run --rm -p 8501:8501 heart-failure-risk-assistant
```

Then open:

```text
http://localhost:8501
```

---

## 📈 Reproducibility

Reproducibility is an important part of this project.

The project aims to make the training and evaluation process reproducible by:

- Keeping preprocessing inside the ML pipeline
- Using stratified dataset splitting
- Using cross-validation for model selection
- Keeping a separate holdout set
- Saving the trained model
- Saving evaluation metrics
- Saving evaluation visualizations
- Defining dependencies in `requirements.txt`

Generated artifacts include:

```text
models/risk_model.joblib
reports/metrics.json
reports/evaluation.png
```

---

## ⚠️ Responsible AI & Limitations

Healthcare machine learning requires particular care.

This project has several important limitations.

### Dataset limitations

The dataset is relatively small and may not represent the diversity of real-world patient populations.

Therefore, model performance on this dataset does not guarantee performance on other populations or healthcare systems.

### External validation

The model has not been established as clinically validated.

Before any real-world application, it would require:

- External validation
- Larger datasets
- Evaluation across diverse populations
- Clinical expert review
- Prospective validation
- Appropriate regulatory and ethical review

### Correlation does not imply causation

Machine-learning predictions are based on statistical relationships in historical data.

A feature associated with higher predicted risk does not necessarily cause the outcome.

### Probability ≠ diagnosis

A predicted probability should not be interpreted as:

- A diagnosis
- A treatment recommendation
- A clinical decision
- A guarantee of an outcome

### Privacy

Do not enter real patient-identifiable information into a public or demonstration deployment.

---

## 🔐 Data Privacy

This application is intended for educational and research demonstration.

**Never upload or enter protected health information (PHI), personally identifiable information (PII), or other confidential patient information into a public deployment.**

For real healthcare applications, appropriate privacy, security, consent, governance, and regulatory requirements must be addressed.

---

## 🧪 Example Workflow

The complete workflow can be summarized as:

```text
Public Dataset
      │
      ▼
Data Loading
      │
      ▼
Data Preprocessing
      │
      ▼
75/25 Stratified Split
      │
      ├──────────────► Holdout Test Set
      │
      ▼
5-Fold Cross-Validation
      │
      ▼
Model Selection
      │
      ▼
Probability Calibration
      │
      ▼
Final Holdout Evaluation
      │
      ▼
Saved Model
      │
      ▼
Streamlit Application
      │
      ▼
Estimated Mortality Risk
```

---

## 🛠️ Technologies Used

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Programming language  |
| Pandas       | Data processing       |
| NumPy        | Numerical computation |
| Scikit-learn | Machine learning      |
| Joblib       | Model serialization   |
| Matplotlib   | Visualization         |
| Streamlit    | Web application       |
| Pytest       | Testing               |
| Docker       | Containerization      |
| Git          | Version control       |
| GitHub       | Source-code hosting   |

---

## 📚 Potential Future Improvements

Several improvements could make the project more robust.

### Data

- Use a larger and more diverse dataset.
- Perform external validation.
- Investigate dataset shift.
- Evaluate performance across demographic subgroups.

### Machine Learning

- Compare additional algorithms.
- Perform systematic hyperparameter optimization.
- Investigate ensemble methods.
- Improve probability calibration.
- Evaluate decision-curve analysis.
- Perform uncertainty estimation.

### Explainability

Future versions could include:

- SHAP explanations
- Feature importance
- Local explanations
- Global model explanations
- Individual prediction explanations

### Application

Potential application improvements include:

- Better UI/UX
- Model-confidence information
- Interactive visualizations
- Prediction history
- Model monitoring
- Authentication
- Secure deployment

---

## 🎓 Academic Purpose

This project demonstrates an end-to-end machine-learning workflow covering:

```text
Data
 ↓
Preprocessing
 ↓
Model Selection
 ↓
Cross-Validation
 ↓
Calibration
 ↓
Evaluation
 ↓
Explainability
 ↓
Deployment
 ↓
Testing
 ↓
Containerization
```

It is designed as an academic and portfolio project demonstrating practical skills in **machine learning, healthcare AI, model evaluation, responsible AI, software engineering, and deployment**.

---

## 👩‍💻 Author

**Farhana** 

B.Sc. (Hons) in Physics

Interested in:

- Machine Learning
- Artificial Intelligence
- Computer Vision
- Healthcare AI
- Computational Physics
- AI Engineering

---

## 📄 Disclaimer

This software is provided for **educational and research purposes only**.

It is not a medical device and has not been clinically validated.

The predictions generated by this application should **not** be used to diagnose, treat, triage, or make medical decisions about any patient.

Always consult qualified healthcare professionals for medical decisions.
