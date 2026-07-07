"""Flask JSON API for the magnet-temperature model.

Scores through predict.predict, the same function behind the CLI and the
Streamlit app, so preprocessing is identical across all three surfaces.

POST /predict accepts one record object or a list of them and returns the
predicted pm for each. GET /health is a readiness check.
"""

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from predict import predict
from src.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "model.joblib"


def create_app(model=None):
    app = Flask(__name__)
    if model is None:
        model = joblib.load(MODEL_PATH)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/predict")
    def predict_endpoint():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="request body must be JSON"), 400
        records = payload if isinstance(payload, list) else [payload]
        try:
            scored = predict(pd.DataFrame(records), model)
        except ValueError as err:
            return jsonify(error=str(err)), 400
        return jsonify(predictions=scored["pm_pred"].tolist())

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000)
