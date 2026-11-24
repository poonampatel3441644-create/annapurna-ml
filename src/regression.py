"""Regression models and the cost function for the Annapurna project."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from src.preprocessing import CAT_COLS, NUM_COLS


def build_simple():
    """Simple linear regression: students_registered -> plates_eaten."""
    prep = ColumnTransformer([("num", "passthrough", ["students_registered"])])
    return Pipeline([("prep", prep), ("reg", LinearRegression())])


def build_multiple():
    """Multiple linear regression on all inputs (text columns one-hot encoded)."""
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("reg", LinearRegression())])


def build_polynomial(degree=2):
    """Polynomial regression on the numeric features plus encoded categories."""
    num_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree, include_bias=False)),
    ])
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
         ("num", num_pipe, NUM_COLS)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("reg", LinearRegression())])


def evaluate(y_true, y_pred):
    """MAE, MSE and R2 for one set of predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(models, X_train, y_train, X_val, y_val):
    """Fit each model on train data and score it on the validation set."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = evaluate(y_val, model.predict(X_val))
        rows.append({"Model": name, **scores})
    return pd.DataFrame(rows).round(3)


def final_test_score(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train on train+validation, then score ONCE on the untouched test set."""
    final = clone(model)
    final.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    return evaluate(y_test, final.predict(X_test))


# ---------------- Cost function (Practical 11) ----------------
def mse_cost(w, b, xs, y):
    """Cost J(w, b) = half of the mean squared error for y = w*x + b."""
    return np.mean((y - (w * xs + b)) ** 2) / 2


def gradient_descent(xs, y, lr=0.1, iterations=100):
    """Fit w and b by gradient descent; returns w, b and the cost history."""
    w, b, history = 0.0, 0.0, []
    for _ in range(iterations):
        err = (w * xs + b) - y
        w -= lr * np.mean(err * xs)
        b -= lr * np.mean(err)
        history.append(mse_cost(w, b, xs, y))
    return w, b, history
