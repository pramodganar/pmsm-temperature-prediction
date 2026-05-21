import streamlit as st
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline


st.title(
    "PMSM Temperature Prediction"
)

pipeline = PredictionPipeline()


profile_id = st.number_input(
    "Profile ID",
    value=1
)

u_q = st.number_input(
    "u_q",
    value=12.0
)

coolant = st.number_input(
    "Coolant Temperature",
    value=40.0
)

u_d = st.number_input(
    "u_d",
    value=5.0
)

motor_speed = st.number_input(
    "Motor Speed",
    value=1200.0
)

i_d = st.number_input(
    "i_d",
    value=2.0
)

i_q = st.number_input(
    "i_q",
    value=3.0
)

ambient = st.number_input(
    "Ambient Temperature",
    value=30.0
)


current_magnitude = (
    i_d**2 + i_q**2
) ** 0.5

voltage_magnitude = (
    u_d**2 + u_q**2
) ** 0.5

temp_difference = (
    coolant - ambient
)


if st.button("Predict"):

    data = pd.DataFrame([{

        "profile_id": profile_id,
        "u_q": u_q,
        "coolant": coolant,
        "u_d": u_d,
        "motor_speed": motor_speed,
        "i_d": i_d,
        "i_q": i_q,
        "ambient": ambient,
        "current_magnitude":
        current_magnitude,
        "voltage_magnitude":
        voltage_magnitude,
        "temp_difference":
        temp_difference
    }])

    prediction = pipeline.predict(data)

    st.success(

        f"Predicted PM Temperature: {prediction[0]:.2f}"
    )