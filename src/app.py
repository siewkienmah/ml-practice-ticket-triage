"""ML PRACTICE: Help-Desk Ticket Triage
======================================

Trains the exact tool combination the Final Assessment's ML task (Task 4)
will check: load and audit a CSV, clean it with pandas, hold out a fixed
test set, compare a DummyClassifier baseline against one trained classifier
through the supplied preprocessing scaffold, and report accuracy, macro-F1
and a confusion matrix.

Complete every TODO below. Do not rename the functions, their arguments, or
TARGET_COLUMN / IDENTIFIER_COLUMN - the tests and the CI workflow call these
functions directly by name.

Run it with:
    python -m src.app --data data/helpdesk_tickets.csv --output outputs/result.json

Check your work with:
    pytest -v
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.pipeline_tools import build_baseline, build_pipeline, build_preprocessor, evaluate_predictions

TARGET_COLUMN = "response_tier"
IDENTIFIER_COLUMN = "ticket_id"


def load_data(path: str) -> pd.DataFrame:
    """TODO (Task 4a): load the CSV at `path` with pandas and return it."""
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """TODO (Task 4b): return (numeric_columns, categorical_columns).

    Excludes IDENTIFIER_COLUMN ("ticket_id") and TARGET_COLUMN ("response_tier").
    """
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for col in df.columns:
        if col in (IDENTIFIER_COLUMN, TARGET_COLUMN):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    return numeric_columns, categorical_columns


def make_split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """TODO (Task 4c): build X (features) and y (target), then call
    sklearn's train_test_split with a FIXED random_state and stratify=y.

    Return exactly: x_train, x_test, y_train, y_test
    """
    numeric_columns, categorical_columns = get_feature_columns(df)
    feature_columns = numeric_columns + categorical_columns

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_baseline(x_train, y_train, numeric_columns, categorical_columns):
    """TODO (Task 4c): build a preprocessor with build_preprocessor(), wrap it
    in a DummyClassifier baseline with build_baseline(), fit it on the
    training data, and return the fitted pipeline.
    """
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    baseline_pipeline = build_baseline(preprocessor)
    baseline_pipeline.fit(x_train, y_train)
    return baseline_pipeline


def train_classifier(x_train, y_train, numeric_columns, categorical_columns):
    """TODO (Task 4c): build a preprocessor, wrap ONE real scikit-learn
    classifier of your choice in it with build_pipeline(), fit it, and return
    the fitted pipeline.
    """
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    model = LogisticRegression(random_state=42, max_iter=1000)
    classifier_pipeline = build_pipeline(preprocessor, model)
    classifier_pipeline.fit(x_train, y_train)
    return classifier_pipeline


def run(data_path: str = "data/helpdesk_tickets.csv", output_path: str = "outputs/result.json") -> dict:
    """Provided - wires your functions together and saves outputs/result.json."""
    df = load_data(data_path)
    numeric_columns, categorical_columns = get_feature_columns(df)
    x_train, x_test, y_train, y_test = make_split(df)

    baseline = train_baseline(x_train, y_train, numeric_columns, categorical_columns)
    classifier = train_classifier(x_train, y_train, numeric_columns, categorical_columns)

    result = {
        "baseline": evaluate_predictions(baseline, x_test, y_test),
        "classifier": evaluate_predictions(classifier, x_test, y_test),
    }

    result["test_index"] = [int(index) for index in x_test.index]
    result["actual"] = y_test.tolist()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Baseline   accuracy={result['baseline']['accuracy']:.3f}  macro_f1={result['baseline']['macro_f1']:.3f}")
    print(f"Classifier accuracy={result['classifier']['accuracy']:.3f}  macro_f1={result['classifier']['macro_f1']:.3f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/helpdesk_tickets.csv")
    parser.add_argument("--output", default="outputs/result.json")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()