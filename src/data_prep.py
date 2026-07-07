"""Loading and the leakage-safe train/test split.

The split holds out whole `profile_id` sessions. Within a session the readings
are a dense time series with near-perfect step-to-step autocorrelation, so a
random split would leak almost-identical neighbours into the test set and report
falsely optimistic scores.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import RAW_DATA, TARGET, GROUP_COL, FEATURES, RANDOM_STATE
from .features import build_features


def load_raw():
    return pd.read_csv(RAW_DATA)


def clean(df):
    """Drop rows without a profile_id; they can't be assigned to a session for
    the grouped split (and in this dataset they also lack i_q)."""
    return df.dropna(subset=[GROUP_COL]).reset_index(drop=True)


def split_by_profile(df, test_size=0.2, random_state=RANDOM_STATE):
    """Return X_train, X_test, y_train, y_test, groups_train.

    groups_train carries the session id of each training row for grouped
    cross-validation later. No profile_id appears in both train and test.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    X_train, y_train = build_features(train), train[TARGET]
    X_test, y_test = build_features(test), test[TARGET]
    groups_train = train[GROUP_COL].to_numpy()
    return X_train, X_test, y_train, y_test, groups_train
