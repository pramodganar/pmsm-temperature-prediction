"""Streamlit front end for the magnet-temperature model.

Single and batch scoring both run through predict.predict, so the app applies
the same preprocessing as training and the command-line tools.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from predict import predict
from src.config import FEATURES, MODELS_DIR

MODEL_PATH = MODELS_DIR / "model.joblib"

# slider bounds and defaults from the observed ranges in the EDA
FEATURE_RANGES = {
    "u_q": (-30.0, 135.0, 47.0),
    "u_d": (-135.0, 135.0, -8.0),
    "i_d": (-280.0, 5.0, -51.0),
    "i_q": (-295.0, 305.0, 30.0),
    "motor_speed": (-300.0, 6100.0, 2000.0),
    "coolant": (13.0, 105.0, 19.0),
    "ambient": (14.0, 31.0, 24.0),
}

# static, from the session-holdout evaluation; update if the model is retrained
HOLDOUT_METRICS = {"RMSE": "10.84 C", "MAE": "8.45 C", "R2": "0.67"}

# illustrative operating points to seed the sliders
PRESETS = {
    "Cold start": {"u_q": 5, "u_d": 0, "i_d": 0, "i_q": 0,
                   "motor_speed": 0, "coolant": 18, "ambient": 22},
    "Steady cruise": {"u_q": 50, "u_d": -10, "i_d": -50, "i_q": 30,
                      "motor_speed": 2500, "coolant": 25, "ambient": 24},
    "High load": {"u_q": 100, "u_d": -80, "i_d": -150, "i_q": 150,
                  "motor_speed": 5000, "coolant": 60, "ambient": 28},
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def single_prediction(model):
    st.subheader("Single record")
    st.caption("Set the sensor readings, or load an operating point, then "
               "estimate the magnet temperature.")

    for name, (_, _, default) in FEATURE_RANGES.items():
        st.session_state.setdefault(name, default)

    preset_cols = st.columns(len(PRESETS))
    for col, (label, point) in zip(preset_cols, PRESETS.items()):
        if col.button(label):
            for name, value in point.items():
                st.session_state[name] = float(value)
            st.rerun()

    slider_cols = st.columns(2)
    for i, name in enumerate(FEATURES):
        low, high, _ = FEATURE_RANGES[name]
        slider_cols[i % 2].slider(name, low, high, key=name)

    if st.button("Predict"):
        record = pd.DataFrame([{name: st.session_state[name] for name in FEATURES}])
        pm = predict(record, model)["pm_pred"].iloc[0]
        st.metric("Predicted pm", f"{pm:.1f} C")


def batch_prediction(model):
    st.subheader("Batch scoring")
    st.caption(f"Upload a CSV with the columns: {', '.join(FEATURES)}. "
               "Missing values are handled by the model.")

    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return

    df = pd.read_csv(upload)
    if df.empty:
        st.error("the uploaded file has no rows")
        return

    try:
        scored = predict(df, model)
    except ValueError as err:
        st.error(str(err))
        return
    except Exception as err:
        st.error(f"could not score the file: {err}")
        return

    imputed = int(df[FEATURES].isna().any(axis=1).sum())
    st.write(f"Scored {len(scored)} records."
             + (f" {imputed} had a missing value filled by the model."
                if imputed else ""))

    pm = scored["pm_pred"]
    stats = st.columns(3)
    stats[0].metric("min pm", f"{pm.min():.1f} C")
    stats[1].metric("mean pm", f"{pm.mean():.1f} C")
    stats[2].metric("max pm", f"{pm.max():.1f} C")

    st.dataframe(scored.head(50))
    st.download_button("Download predictions",
                       scored.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="PMSM magnet temperature", layout="wide")
    st.title("PMSM permanent magnet temperature")
    st.write("Estimate the permanent magnet tooth temperature of a PMSM motor "
             "from its electrical and thermal sensor readings.")

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run `python train.py` first.")
        return

    model = load_model()

    with st.sidebar:
        st.header("Model")
        st.write("Gradient boosting, evaluated on held-out motor runs.")
        for name, value in HOLDOUT_METRICS.items():
            st.metric(name, value)
        importance = Path("reports/figures/permutation_importance.png")
        if importance.exists():
            st.image(str(importance), caption="Permutation importance")

    single_tab, batch_tab = st.tabs(["Single", "Batch"])
    with single_tab:
        single_prediction(model)
    with batch_tab:
        batch_prediction(model)


if __name__ == "__main__":
    main()
