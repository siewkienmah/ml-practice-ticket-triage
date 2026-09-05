"""Automated practice checks.

These run locally (`pytest -v`) and again in GitHub Actions once your pull
request's workflow run is approved. They check that your pipeline is built
correctly, not that you chose the "best" classifier - there is no single
correct answer here, only a correctly built one. This is exactly the kind
of technical check the Final Assessment's own workflow will run on your
submission, so treat a red test here as a preview of what a marker's CI
would flag.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app import (
    TARGET_COLUMN,
    IDENTIFIER_COLUMN,
    get_feature_columns,
    load_data,
    make_split,
    run,
    train_baseline,
    train_classifier,
)

DATA_PATH = "data/helpdesk_tickets.csv"


@pytest.fixture(scope="module")
def df():
    return load_data(DATA_PATH)


def test_data_loads_with_expected_columns(df):
    expected = {
        "ticket_id", "device_type", "error_code", "wait_time_minutes",
        "reported_severity", "reopened_count", "resolution_notes_length",
        "response_tier",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) > 100


def test_identifier_and_target_excluded_from_features(df):
    numeric_columns, categorical_columns = get_feature_columns(df)
    feature_columns = set(numeric_columns) | set(categorical_columns)
    assert IDENTIFIER_COLUMN not in feature_columns, (
        "ticket_id is a row identifier, not a predictor - exclude it (Task 4b)."
    )
    assert TARGET_COLUMN not in feature_columns, "response_tier is the label, not a feature."
    assert len(feature_columns) >= 4, "use more than one or two columns - most fields carry signal."


def test_split_is_reproducible_and_stratified(df):
    x_train_a, x_test_a, y_train_a, y_test_a = make_split(df, random_state=42)
    x_train_b, x_test_b, y_train_b, y_test_b = make_split(df, random_state=42)
    assert list(y_test_a) == list(y_test_b), (
        "the same random_state should always produce the same split - "
        "the Final Assessment expects a fixed, reproducible split too."
    )
    # stratified: every class present in the training data must survive into the test set
    assert set(y_test_a.unique()) == set(df[TARGET_COLUMN].unique())


def test_baseline_is_a_dummy_classifier(df):
    numeric_columns, categorical_columns = get_feature_columns(df)
    x_train, x_test, y_train, y_test = make_split(df)
    baseline = train_baseline(x_train, y_train, numeric_columns, categorical_columns)
    predictions = set(baseline.predict(x_test))
    assert len(predictions) == 1, (
        "a DummyClassifier baseline should predict exactly one class for every row - "
        "if you see more than one, you have swapped in a real model here."
    )


def test_classifier_trains_and_beats_random_guessing(df):
    numeric_columns, categorical_columns = get_feature_columns(df)
    x_train, x_test, y_train, y_test = make_split(df)
    baseline = train_baseline(x_train, y_train, numeric_columns, categorical_columns)
    classifier = train_classifier(x_train, y_train, numeric_columns, categorical_columns)

    from src.pipeline_tools import evaluate_predictions
    baseline_metrics = evaluate_predictions(baseline, x_test, y_test)
    classifier_metrics = evaluate_predictions(classifier, x_test, y_test)

    for metrics in (baseline_metrics, classifier_metrics):
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        n_labels = len(metrics["labels"])
        assert len(metrics["confusion_matrix"]) == n_labels
        assert all(len(row) == n_labels for row in metrics["confusion_matrix"])

    assert classifier_metrics["macro_f1"] >= baseline_metrics["macro_f1"] + 0.05, (
        "your trained classifier's macro-F1 should clearly beat the DummyClassifier "
        "baseline's. If it does not, revisit your feature choices or preprocessing "
        "before assuming the classifier itself is the problem (Task 4c-4d)."
    )


def test_run_writes_output_file(tmp_path):
    output_path = tmp_path / "result.json"
    result = run(data_path=DATA_PATH, output_path=str(output_path))

    assert output_path.exists(), "run() must write its result to --output"
    with open(output_path) as f:
        saved = json.load(f)

    for section in ("baseline", "classifier"):
        assert section in saved
        for key in ("accuracy", "macro_f1", "labels", "confusion_matrix"):
            assert key in saved[section]
    assert len(saved["test_index"]) == len(saved["actual"])
    assert len(saved["actual"]) == len(saved["classifier"]["predicted"])
    assert saved == result
