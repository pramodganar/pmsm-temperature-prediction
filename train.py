"""Train the magnet-temperature model and persist it to models/.

Reports held-out metrics on a leakage-safe session split, then refits the same
pipeline on all available data for the saved artifact.
"""

import joblib

from src.config import MODELS_DIR, TARGET
from src.data_prep import clean, load_raw, split_by_profile
from src.evaluate import regression_metrics
from src.features import build_features
from src.model import build_model

MODEL_PATH = MODELS_DIR / "model.joblib"


def main():
    df = clean(load_raw())

    X_train, X_test, y_train, y_test, _ = split_by_profile(df)
    model = build_model().fit(X_train, y_train)

    metrics = regression_metrics(y_test, model.predict(X_test))
    print("held-out session metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")

    # refit on everything for the deployed artifact, now that the split has
    # served its purpose of giving an honest estimate
    final = build_model().fit(build_features(df), df[TARGET])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
