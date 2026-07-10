"""Smoke tests for the pipeline. Synthetic data so they run without the raw
dataset present.
"""

import numpy as np
import pandas as pd

from src.config import FEATURES, GROUP_COL, TARGET
from src.data_prep import split_by_profile
from src.features import build_features
from src.model import build_model


def make_frame(n_groups=10, per_group=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x = rng.normal(size=(per_group, len(FEATURES)))
        frame = pd.DataFrame(x, columns=FEATURES)
        frame[GROUP_COL] = g
        frame[TARGET] = x[:, 0] * 2 + x[:, 4] - x[:, 2] + rng.normal(scale=0.1, size=per_group)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def test_split_groups_are_disjoint():
    df = make_frame()
    X_train, X_test, _, _, _ = split_by_profile(df, test_size=0.3)
    train_g = set(df.loc[X_train.index, GROUP_COL])
    test_g = set(df.loc[X_test.index, GROUP_COL])
    assert train_g and test_g
    assert train_g.isdisjoint(test_g)


def test_build_features_returns_feature_columns():
    df = make_frame(n_groups=2, per_group=5)
    assert list(build_features(df).columns) == FEATURES


def test_predict_handles_missing_values():
    df = make_frame()
    X_train, X_test, y_train, _, _ = split_by_profile(df)
    model = build_model().fit(X_train, y_train)

    X_missing = X_test.copy()
    X_missing.iloc[0, 0] = np.nan
    preds = model.predict(X_missing)
    assert not np.isnan(preds).any()
