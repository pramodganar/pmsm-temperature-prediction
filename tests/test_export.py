"""Test the raw-data export against a synthetic SQLite database, since the
real Regression.db is not distributed with the repo.
"""

import sqlite3

import pandas as pd
import pytest

from src.config import FEATURES, GROUP_COL, TARGET
from scripts.export_raw import export, TABLE


def make_db(path, n_rows=20):
    df = pd.DataFrame({c: range(n_rows) for c in
                       [GROUP_COL, *FEATURES, TARGET]}, dtype=float)
    with sqlite3.connect(path) as con:
        df.to_sql(TABLE, con, index=False)
    return df


def test_export_round_trips_the_table(tmp_path):
    db = tmp_path / "Regression.db"
    out = tmp_path / "raw.csv"
    written = make_db(db)

    returned = export(db, out)

    read_back = pd.read_csv(out)
    pd.testing.assert_frame_equal(read_back, written)
    pd.testing.assert_frame_equal(returned, written)


def test_export_fails_clearly_on_missing_table(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    with pytest.raises(ValueError, match="not found"):
        export(db, tmp_path / "raw.csv")
