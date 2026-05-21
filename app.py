from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__)

pipeline = PredictionPipeline()


@app.route("/")
def home():

    return "PMSM Temperature Prediction API"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)

    return jsonify({

        "predicted_pm_temperature":
        float(prediction[0])
    })


if __name__ == "__main__":

    app.run(debug=True)