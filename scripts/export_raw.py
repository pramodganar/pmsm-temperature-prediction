"""Export the raw training data from the provided SQLite database.

Writes the `electric_motor_temperature` table to data/raw/ as CSV, straight
from SQLite so no spreadsheet row limit can silently truncate it. The metrics
reported in this repo come from the provided extract; retraining from any
other copy of the data (e.g. the public Kaggle release) will not reproduce
them, so this script is the supported path to the raw CSV.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DATA

TABLE = "electric_motor_temperature"


def export(db_path, output=RAW_DATA, table=TABLE):
    with sqlite3.connect(db_path) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        if table not in names:
            raise ValueError(
                f"table {table!r} not found in {db_path}; tables: {names}")
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description=f"Export the {TABLE} table from Regression.db to CSV.")
    parser.add_argument("--db", required=True, help="path to Regression.db")
    parser.add_argument("--output", default=str(RAW_DATA),
                        help="destination CSV (default: data/raw/)")
    args = parser.parse_args()

    df = export(args.db, args.output)
    print(f"wrote {len(df)} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
