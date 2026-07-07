"""Metrics and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .config import RANDOM_STATE


def regression_metrics(y_true, y_pred):
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def score_table(model, X_train, y_train, X_test, y_test):
    """Train and test metrics for a fitted model, side by side to expose
    any overfitting."""
    train = regression_metrics(y_train, model.predict(X_train))
    test = regression_metrics(y_test, model.predict(X_test))
    return pd.DataFrame({"train": train, "test": test}).T.round(3)


def per_session_rmse(y_true, y_pred, groups):
    """RMSE within each held-out session, sorted. Aggregate metrics can hide
    runs the model handles badly; this surfaces them."""
    frame = pd.DataFrame({"g": np.asarray(groups),
                          "y": np.asarray(y_true),
                          "p": np.asarray(y_pred)})
    rmse = frame.groupby("g").apply(
        lambda d: root_mean_squared_error(d.y, d.p), include_groups=False)
    return rmse.sort_values()


def plot_diagnostics(y_true, y_pred, sample=20000, save_path=None):
    """Predicted vs actual, residuals vs predicted, residual distribution.

    Scatters are drawn on a fixed-seed subsample so the figure stays light on
    a large test set without changing the picture.
    """
    y_true = np.asarray(y_true)
    resid = y_true - y_pred

    idx = np.arange(len(y_true))
    if sample and len(idx) > sample:
        idx = np.random.default_rng(RANDOM_STATE).choice(idx, sample, replace=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax[0].scatter(y_pred[idx], y_true[idx], s=3, alpha=0.15, color="steelblue")
    ax[0].plot(lims, lims, "k--", lw=1)
    ax[0].set(xlabel="predicted pm (C)", ylabel="actual pm (C)",
              title="Predicted vs actual")

    ax[1].scatter(y_pred[idx], resid[idx], s=3, alpha=0.15, color="indianred")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel="predicted pm (C)", ylabel="residual (C)",
              title="Residuals vs predicted")

    ax[2].hist(resid, bins=60, color="slategray")
    ax[2].set(xlabel="residual (C)", title="Residual distribution")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
