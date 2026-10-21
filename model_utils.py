"""Everything that touches the trained model."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import add_calendar_features

MODEL_PATH = Path(__file__).parent / "sales_forecasting_model.pkl"
DEFAULT_FEATURES = ["Day", "Month", "Year", "DayOfWeek"]


@st.cache_resource(show_spinner="Loading forecasting model...")
def load_model(path: str = str(MODEL_PATH)):
    """Load the pickled model once and reuse it.

    SECURITY NOTE: only unpickle files you created or trust.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def required_features(model) -> list:
    """The exact column names (and order) the model was trained on."""
    names = getattr(model, "feature_names_in_", None)
    return [str(n) for n in names] if names is not None else DEFAULT_FEATURES


def check_features(model) -> list:
    """Return features the model needs that this app can NOT build from a date."""
    return [f for f in required_features(model) if f not in DEFAULT_FEATURES]


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Predict sales for rows that already contain the feature columns."""
    features = required_features(model)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    return model.predict(df[features])


def forecast_future(model, last_date, horizon_days: int) -> pd.DataFrame:
    """Forecast the next `horizon_days` days after `last_date`.

    Lower/Upper = 10th and 90th percentile of the individual trees' predictions
    (a rough measure of how much the trees disagree, not a statistical guarantee).
    """
    dates = pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon_days)
    future = add_calendar_features(pd.DataFrame({"Date": dates}))
    features = required_features(model)
    future["Forecast"] = model.predict(future[features])

    if hasattr(model, "estimators_"):
        x = future[features].to_numpy(dtype=float)
        per_tree = np.vstack([tree.predict(x) for tree in model.estimators_])
        future["Lower"] = np.percentile(per_tree, 10, axis=0)
        future["Upper"] = np.percentile(per_tree, 90, axis=0)
    else:
        future["Lower"] = np.nan
        future["Upper"] = np.nan
    return future


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "MAPE": mape,
    }


def feature_importances(model):
    """Return a DataFrame of feature importances, or None if unavailable."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    return (
        pd.DataFrame({"Feature": required_features(model), "Importance": imp})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
