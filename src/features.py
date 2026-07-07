"""Feature construction and preprocessing shared by train and predict.

Keeping the feature list and the preprocessing in one place is what guarantees
training and prediction transform inputs identically.
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES


def build_features(df):
    """Select the model inputs. Single definition so train and predict agree."""
    return df[FEATURES].copy()


def make_preprocessor(scale=True):
    """Median imputation, optionally followed by standardisation.

    Fit on training data only (inside a model pipeline), so the imputed values
    and scaling statistics never see the test set. Tree models can pass
    scale=False since they don't need it.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
