"""Tests for the scoring path (predict.predict), the function behind both the
CLI and the Streamlit app. Synthetic data, no raw dataset or saved model needed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.config import FEATURES, GROUP_COL, TARGET
from src.data_prep import split_by_profile
from src.model import build_model
from predict import predict


def make_frame(n_groups=10, per_group=60, seed=0):
    """Synthetic sessions with a pm-like target in a realistic temperature band."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x = rng.normal(size=(per_group, len(FEATURES)))
        frame = pd.DataFrame(x, columns=FEATURES)
        frame[GROUP_COL] = g
        # roughly 20-110 C so predictions land in a plausible pm range
        frame[TARGET] = 65 + 12 * x[:, 0] + 6 * x[:, 4] - 5 * x[:, 2]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def fitted_model(df):
    X_train, _, y_train, _, _ = split_by_profile(df)
    return build_model().fit(X_train, y_train)


def test_predict_output_shape_and_column():
    """One prediction per input row, original columns preserved, pm_pred added."""
    df = make_frame()
    model = fitted_model(df)
    out = predict(df[FEATURES], model)

    assert len(out) == len(df)
    assert "pm_pred" in out.columns
    assert list(out.index) == list(df.index)  # row order preserved


def test_predict_rejects_missing_columns():
    df = make_frame()
    model = fitted_model(df)
    with pytest.raises(ValueError, match="missing required columns"):
        predict(df[FEATURES].drop(columns=["ambient"]), model)


def test_predictions_are_plausible_temperatures():
    """No NaNs, and values stay in a physically sane band for magnet temperature."""
    df = make_frame()
    model = fitted_model(df)
    pm = predict(df[FEATURES], model)["pm_pred"].to_numpy()

    assert np.isfinite(pm).all()
    assert pm.min() > -20 and pm.max() < 200


def test_predict_uses_only_declared_features():
    """Extra columns in the input must not change the scored result."""
    df = make_frame()
    model = fitted_model(df)

    base = predict(df[FEATURES], model)["pm_pred"].to_numpy()
    noisy = df[FEATURES].copy()
    noisy["unused_sensor"] = np.random.default_rng(1).normal(size=len(noisy))
    with_extra = predict(noisy, model)["pm_pred"].to_numpy()

    np.testing.assert_allclose(base, with_extra)


def test_grouped_cv_folds_never_share_a_session():
    """The validation contract: no profile_id appears in both sides of any fold."""
    df = make_frame(n_groups=12)
    groups = df[GROUP_COL].to_numpy()
    for train_idx, val_idx in GroupKFold(n_splits=4).split(df, groups=groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
