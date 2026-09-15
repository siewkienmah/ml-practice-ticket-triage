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

from src.pipeline_tools import (
    build_baseline,
    build_pipeline,
    build_preprocessor,
    evaluate_predictions,
)


TARGET_COLUMN = "response_tier"
IDENTIFIER_COLUMN = "ticket_id"


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV at `path` with pandas and return it."""

    df = pd.read_csv(path)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns).

    The identifier and target columns are excluded because they are not
    predictors.
    """

    # Start with all columns except the identifier and target.
    feature_columns = [
        column
        for column in df.columns
        if column not in {IDENTIFIER_COLUMN, TARGET_COLUMN}
    ]

    # Numeric predictors.
    numeric_columns = df[feature_columns].select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Categorical predictors.
    categorical_columns = df[feature_columns].select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
):
    """Build X and y, then make a fixed stratified train/test split.

    Returns:
        x_train, x_test, y_train, y_test
    """

    numeric_columns, categorical_columns = get_feature_columns(df)

    feature_columns = numeric_columns + categorical_columns

    # X = predictor/features
    X = df[feature_columns]

    # y = target/class to predict
    y = df[TARGET_COLUMN]

    # Fixed random_state makes the split reproducible.
    # stratify=y keeps the class proportions similar in train/test.
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
    categorical_columns
):
    """Build and train the DummyClassifier baseline."""

    # Build the same preprocessing step used by the real classifier.
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    # Build the DummyClassifier baseline.
    baseline = build_baseline(preprocessor)

    # Train the baseline using the training data only.
    baseline.fit(x_train, y_train)

    return baseline


def train_classifier(
    x_train,
    y_train,
    numeric_columns,
    categorical_columns
):
    """Build and train one real scikit-learn classifier.

    LogisticRegression is used as the real classifier.
    """

    # Build the preprocessing step.
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    # Build the real ML pipeline.
    classifier = build_pipeline(
        preprocessor,
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

    # Train only on the training data.
    classifier.fit(x_train, y_train)

    return classifier


def run(
    data_path: str = "data/helpdesk_tickets.csv",
    output_path: str = "outputs/result.json"
) -> dict:
    """Run the complete ML pipeline and save the results."""

    df = load_data(data_path)

    numeric_columns, categorical_columns = get_feature_columns(df)

    x_train, x_test, y_train, y_test = make_split(df)

    baseline = train_baseline(
        x_train,
        y_train,
        numeric_columns,
        categorical_columns
    )

    classifier = train_classifier(
        x_train,
        y_train,
        numeric_columns,
        categorical_columns
    )

    result = {
        "baseline": evaluate_predictions(
            baseline,
            x_test,
            y_test
        ),
        "classifier": evaluate_predictions(
            classifier,
            x_test,
            y_test
        ),
    }

    # Keep the held-out row references and true labels beside predictions.
    result["test_index"] = [
        int(index) for index in x_test.index
    ]

    result["actual"] = y_test.tolist()

    # Create output directory if it does not already exist.
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save results as JSON.
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Baseline   "
        f"accuracy={result['baseline']['accuracy']:.3f}  "
        f"macro_f1={result['baseline']['macro_f1']:.3f}"
    )

    print(
        f"Classifier "
        f"accuracy={result['classifier']['accuracy']:.3f}  "
        f"macro_f1={result['classifier']['macro_f1']:.3f}"
    )

    # TODO (Task 4e):
    #
    # Inspect at least three rows that the classifier predicted incorrectly.
    #
    # Compare:
    #     result["classifier"]["predicted"]
    #     result["actual"]
    #
    # Use:
    #     result["test_index"]
    #
    # to locate the original rows in the CSV.
    #
    # Then write the findings in REFLECTION.md.

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/helpdesk_tickets.csv"
    )

    parser.add_argument(
        "--output",
        default="outputs/result.json"
    )

    args = parser.parse_args()

    run(
        args.data,
        args.output
    )


if __name__ == "__main__":
    main()