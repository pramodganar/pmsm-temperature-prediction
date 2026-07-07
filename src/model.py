"""Model definitions.

`candidate_models` is the field compared in 04_model_comparison. `build_model`
is the one that won that comparison and is what train.py fits and persists.
"""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import make_preprocessor


def candidate_models():
    """The shortlist, each as a self-contained pipeline. Linear models get
    scaling; the tree ensembles don't need it."""
    return {
        "ridge": Pipeline([
            ("pre", make_preprocessor(scale=True)),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "lasso": Pipeline([
            ("pre", make_preprocessor(scale=True)),
            ("model", Lasso(alpha=0.001, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("pre", make_preprocessor(scale=False)),
            ("model", RandomForestRegressor(
                n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "hist_gbm": Pipeline([
            ("pre", make_preprocessor(scale=False)),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]),
    }


def build_model():
    """Selected model: tuned histogram gradient boosting. Best grouped-CV and
    holdout RMSE of the shortlist, fits in seconds, and handles missing values
    natively. Settings come from the grid search in 04_model_comparison."""
    return Pipeline([
        ("pre", make_preprocessor(scale=False)),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=200, max_depth=None,
            random_state=RANDOM_STATE)),
    ])
