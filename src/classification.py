"""Classification models for predicting high-waste meals."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.preprocessing import CAT_COLS, CLF_TARGET, FLAG_COLS, NUM_COLS

PLAN_NUM_COLS = NUM_COLS + ["plates_cooked", "cook_ratio"]


def add_cook_ratio(df):
    """Planned plates per registered student (known before the meal)."""
    df = df.copy()
    df["cook_ratio"] = df["plates_cooked"] / df["students_registered"]
    return df


def get_clf_data(df, include_plan=True):
    """Inputs and target for the high-waste classifier.

    include_plan=False  -> demand-side inputs only (first attempt)
    include_plan=True   -> also uses the planned quantity (final version)
    leftover_kg is never used: the label is built from it (leakage).
    """
    df = add_cook_ratio(df)
    num_cols = PLAN_NUM_COLS if include_plan else NUM_COLS
    features = CAT_COLS + FLAG_COLS + num_cols
    return df[features], df[CLF_TARGET], num_cols


def make_preprocessor(num_cols):
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
         ("num", StandardScaler(), num_cols)],
        remainder="passthrough",
    )


def get_models():
    """The four classifiers plus an 'always No' baseline."""
    return {
        "Baseline (always No)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(class_weight="balanced"),
    }


def get_cv(n_splits=5, seed=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def compare_classifiers(X, y, num_cols, models=None):
    """5-fold cross-validated accuracy, precision, recall and F1 for each model."""
    models = models or get_models()
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
    }
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("prep", make_preprocessor(num_cols)), ("clf", clf)])
        res = cross_validate(pipe, X, y, cv=get_cv(), scoring=scoring)
        rows.append({
            "Model": name,
            "Accuracy": res["test_accuracy"].mean(),
            "Precision": res["test_precision"].mean(),
            "Recall": res["test_recall"].mean(),
            "F1": res["test_f1"].mean(),
        })
    return pd.DataFrame(rows).round(3)


def cv_predictions(X, y, num_cols, clf):
    """Cross-validated predictions (used for the confusion matrix)."""
    pipe = Pipeline([("prep", make_preprocessor(num_cols)), ("clf", clf)])
    return cross_val_predict(pipe, X, y, cv=get_cv())
