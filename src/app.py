"""ML PRACTICE: Help-Desk Ticket Triage
======================================

Trains the exact tool combination the Final Assessment's ML task (Task 4)
will check: load and audit a CSV, clean it with pandas, hold out a fixed
test set, compare a DummyClassifier baseline against one trained classifier
through the supplied preprocessing scaffold, and report accuracy, macro-F1
and a confusion matrix.
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
    """Load the CSV at path with pandas and return it."""
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical predictor columns.

    The identifier and target columns are excluded because:
    - ticket_id is only an identifier, not a useful predictor.
    - response_tier is the target that the model must predict.
    """

    feature_df = df.drop(
        columns=[IDENTIFIER_COLUMN, TARGET_COLUMN],
        errors="ignore",
    )

    numeric_columns = feature_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = feature_df.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return numeric_columns, categorical_columns


def make_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Build X/y and create a fixed stratified train/test split."""

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
    """Build and fit the DummyClassifier baseline."""

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    baseline = build_baseline(preprocessor)

    baseline.fit(x_train, y_train)

    return baseline


def train_classifier(
    x_train,
    y_train,
    numeric_columns,
    categorical_columns,
):
    """Build and fit one real scikit-learn classifier."""

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    pipeline = build_pipeline(
        preprocessor,
        classifier,
    )

    pipeline.fit(x_train, y_train)

    return pipeline


def run(
    data_path: str = "data/helpdesk_tickets.csv",
    output_path: str = "outputs/result.json",
) -> dict:
    """Run the complete ML pipeline and save the evaluation results."""

    df = load_data(data_path)

    numeric_columns, categorical_columns = get_feature_columns(df)

    x_train, x_test, y_train, y_test = make_split(df)

    baseline = train_baseline(
        x_train,
        y_train,
        numeric_columns,
        categorical_columns,
    )

    classifier = train_classifier(
        x_train,
        y_train,
        numeric_columns,
        categorical_columns,
    )

    result = {
        "baseline": evaluate_predictions(
            baseline,
            x_test,
            y_test,
        ),
        "classifier": evaluate_predictions(
            classifier,
            x_test,
            y_test,
        ),
    }

    # Keep the held-out row references and true labels beside the predictions.
    result["test_index"] = [
        int(index) for index in x_test.index
    ]

    result["actual"] = y_test.tolist()

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/helpdesk_tickets.csv",
    )

    parser.add_argument(
        "--output",
        default="outputs/result.json",
    )

    args = parser.parse_args()

    run(
        args.data,
        args.output,
    )


if __name__ == "__main__":
    main()