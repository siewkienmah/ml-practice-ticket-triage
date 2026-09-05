"""Supplied practice scaffold - do not edit.

This mirrors the shape of the preprocessing/baseline/evaluation helpers you
will be given in the Final Assessment's own ML scaffold. Getting comfortable
calling functions like these, rather than hand-rolling preprocessing, is the
whole point of this practice set.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Impute + scale numeric columns; impute + one-hot encode categorical columns."""
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_steps, numeric_columns),
        ("categorical", categorical_steps, categorical_columns),
    ])


def build_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    """Wrap any scikit-learn classifier behind the same preprocessing steps."""
    return Pipeline([("prepare", preprocessor), ("model", classifier)])


def build_baseline(preprocessor: ColumnTransformer) -> Pipeline:
    """A DummyClassifier baseline: always predicts the most frequent class."""
    return build_pipeline(preprocessor, DummyClassifier(strategy="most_frequent"))


def evaluate_predictions(model: Pipeline, x_test, y_test) -> dict:
    """Return accuracy, macro-F1 and a confusion matrix for a fitted model."""
    predicted = model.predict(x_test)
    labels = sorted(set(y_test) | set(predicted))
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "predicted": predicted.tolist(),
    }
