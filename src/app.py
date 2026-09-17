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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline_tools import build_baseline, build_pipeline, build_preprocessor, evaluate_predictions

TARGET_COLUMN = "response_tier"
IDENTIFIER_COLUMN = "ticket_id"


def load_data(path: str) -> pd.DataFrame:
    """TODO (Task 4a): load the CSV with pandas."""
    return pd.read_csv(path)


def get_feature_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """TODO (Task 4b): return numeric and categorical feature columns."""

    feature_df = df.drop(
        columns=[IDENTIFIER_COLUMN, TARGET_COLUMN],
        errors="ignore",
    )

    numeric_columns = feature_df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = feature_df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    return numeric_columns, categorical_columns


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """TODO (Task 4c): create a reproducible stratified train/test split."""

    numeric_columns, categorical_columns = get_feature_columns(df)
    feature_columns = numeric_columns + categorical_columns

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return x_train, x_test, y_train, y_test


def train_baseline(
    x_train,
    y_train,
    numeric_columns,
    categorical_columns,
):
    """TODO (Task 4c): train the baseline model."""
def train_baseline(
    x_train,
    y_train,
    numeric_columns,
    categorical_columns,
):
    """TODO (Task 4c): train the baseline model."""

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    classifier = DummyClassifier(strategy="most_frequent")

    pipeline = build_pipeline(
        preprocessor,
        classifier
    )

    pipeline.fit(x_train, y_train)

    return pipeline
    """TODO (Task 4c): create a reproducible stratified train/test split."""

    numeric_columns, categorical_columns = get_feature_columns(df)
    feature_columns = numeric_columns + categorical_columns

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return x_train, x_test, y_train, y_test
def train_classifier(x_train, y_train, numeric_columns, categorical_columns):
    """TODO (Task 4c): build a preprocessor, wrap ONE real scikit-learn
    classifier of your choice in it with build_pipeline() (for example
    DecisionTreeClassifier, LogisticRegression or KNeighborsClassifier),
    fit it, and return the fitted pipeline.

    Import your chosen classifier at the top of this file.
    """
def train_classifier(
    x_train,
    y_train,
    numeric_columns,
    categorical_columns,
):
    """TODO (Task 4c): train a real classifier."""

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    pipeline = build_pipeline(
        preprocessor,
        classifier
    )

    pipeline.fit(x_train, y_train)

    return pipeline

def run(data_path: str = "data/helpdesk_tickets.csv", output_path: str = "outputs/result.json") -> dict:
    """Provided - wires your functions together and saves outputs/result.json.
    You should not need to edit this, but read it so you know what it expects
    from the functions above.
    """
    df = load_data(data_path)
    numeric_columns, categorical_columns = get_feature_columns(df)
    x_train, x_test, y_train, y_test = make_split(df)

    baseline = train_baseline(x_train, y_train, numeric_columns, categorical_columns)
    classifier = train_classifier(x_train, y_train, numeric_columns, categorical_columns)

    result = {
        "baseline": evaluate_predictions(baseline, x_test, y_test),
        "classifier": evaluate_predictions(classifier, x_test, y_test),
    }

    # Keep the held-out row references and true labels beside the predictions.
    # Students need these fields to identify and explain specific errors in
    # REFLECTION.md without reconstructing the split in a separate notebook.
    result["test_index"] = [int(index) for index in x_test.index]
    result["actual"] = y_test.tolist()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Baseline   accuracy={result['baseline']['accuracy']:.3f}  macro_f1={result['baseline']['macro_f1']:.3f}")
    print(f"Classifier accuracy={result['classifier']['accuracy']:.3f}  macro_f1={result['classifier']['macro_f1']:.3f}")

    # TODO (Task 4e): inspect at least three rows your classifier got wrong.
    # Compare result['classifier']['predicted'] with result['actual']; use
    # result['test_index'] to locate each original ticket in the CSV.
    # Write one defensible feature/preprocessing improvement, and one
    # bias/fairness/deployment limitation, into REFLECTION.md.

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/helpdesk_tickets.csv")
    parser.add_argument("--output", default="outputs/result.json")
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
