"""Tests for the Flask endpoint. It must go through the same predict.predict as
the CLI and app, so its output matches for the same records. Synthetic data.
"""

import numpy as np
import pandas as pd

from src.config import FEATURES, GROUP_COL, TARGET
from src.data_prep import split_by_profile
from src.model import build_model
from predict import predict
from api import create_app


def make_frame(n_groups=10, per_group=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x = rng.normal(size=(per_group, len(FEATURES)))
        frame = pd.DataFrame(x, columns=FEATURES)
        frame[GROUP_COL] = g
        frame[TARGET] = 65 + 12 * x[:, 0] + 6 * x[:, 4] - 5 * x[:, 2]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def fitted_model(df):
    X_train, _, y_train, _, _ = split_by_profile(df)
    return build_model().fit(X_train, y_train)


def client_and_model():
    df = make_frame()
    model = fitted_model(df)
    return create_app(model).test_client(), model, df


def test_health():
    client, _, _ = client_and_model()
    assert client.get("/health").json == {"status": "ok"}


def test_predict_matches_predict_function():
    """The endpoint returns the same numbers as a direct predict.predict call."""
    client, model, df = client_and_model()
    records = df[FEATURES].head(5).to_dict(orient="records")

    resp = client.post("/predict", json=records)
    assert resp.status_code == 200

    expected = predict(pd.DataFrame(records), model)["pm_pred"].tolist()
    np.testing.assert_allclose(resp.json["predictions"], expected)


def test_predict_accepts_single_object():
    client, _, df = client_and_model()
    resp = client.post("/predict", json=df[FEATURES].iloc[0].to_dict())
    assert resp.status_code == 200
    assert len(resp.json["predictions"]) == 1


def test_predict_rejects_missing_columns():
    client, _, df = client_and_model()
    bad = df[FEATURES].drop(columns=["ambient"]).iloc[0].to_dict()
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 400
    assert "missing required columns" in resp.json["error"]


def test_predict_rejects_non_json():
    client, _, _ = client_and_model()
    assert client.post("/predict", data="not json").status_code == 400
