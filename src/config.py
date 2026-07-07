"""Shared constants so training and prediction stay in sync."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "electric_motor_temperature.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42

TARGET = "pm"
GROUP_COL = "profile_id"
FEATURES = ["u_q", "u_d", "i_d", "i_q", "motor_speed", "coolant", "ambient"]
