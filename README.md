# PMSM Temperature Prediction System

## Project Overview

This project predicts Permanent Magnet Synchronous Motor (PMSM) temperature using Machine Learning techniques.

The system uses electrical, thermal, and operational motor sensor data to predict PM temperature in real time.

The project includes:
- Feature Engineering
- Multiple Model Training
- Random Forest Regression
- Flask API
- Streamlit Dashboard
- Real-Time Prediction Pipeline

---

## Dataset Information

The dataset contains sensor measurements from Permanent Magnet Synchronous Motors.

Features include:
- Voltage measurements
- Current measurements
- Motor speed
- Coolant temperature
- Ambient temperature
- Operational profiles

## Dataset Availability

The original PMSM dataset is not included in this repository due to GitHub file size limitations.

The project is designed to work with PMSM sensor datasets containing:
- voltage measurements
- current measurements
- motor speed
- coolant temperature
- ambient temperature

Users can use their own PMSM datasets or publicly available motor sensor datasets.

### Target Variable
- PM Temperature

---

## Feature Engineering

Custom engineered features were created to improve model performance:

- Current Magnitude
- Voltage Magnitude
- Temperature Difference

These features help capture electrical and thermal relationships in the motor system.

---

## Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost
- LightGBM
- CatBoost

---

## Best Model

Random Forest Regressor achieved the best performance for PMSM temperature prediction.

### Evaluation Metrics
- R² Score
- RMSE
- MAE

---

## Project Architecture

```text
pmsm-temperature-prediction/
│
├── app.py
├── streamlit_app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   ├── feature_importance.csv
│   └── model_results.csv
│
├── data/
│
├── notebooks/
│
├── src/
│   ├── components/
│   └── pipeline/

## Live Demo

Streamlit App:
https://pmsm-temperature-prediction-6tfz6xweynuwvg2mz2tmaw.streamlit.app/