# ML Practice: Help-Desk Ticket Triage

Ungraded practice for BCS2143 / BIT3203, built to train the exact tool
combination the **Final Assessment's Task 4 (ML pipeline and evaluation,
25 marks)** will check: loading and auditing a CSV, cleaning it with pandas,
holding out a fixed test set, comparing a `DummyClassifier` baseline against
one trained classifier through a shared preprocessing scaffold, and reporting
accuracy, macro-F1 and a confusion matrix.

The scenario (triaging IT help-desk tickets) and the dataset are different
from the Final Assessment's own case. The tools, workflow and checks are
deliberately the same.

## What you will do

1. Fork this repository and clone your fork.
2. Complete the five `TODO`s in `src/app.py` (see the table below).
3. Run `pytest -v` locally until all checks pass.
4. Push your branch to your fork and open a pull request back to this
   repository. See `HOW_TO_SUBMIT.md` for the exact steps and what happens
   next (a maintainer-approved GitHub Actions run, exactly like the Final
   Assessment's own CI).

## Recommended self-study order

1. Read the theory slides to understand the complete ML pipeline.
2. Use the student guide for setup, reasoning prompts and troubleshooting.
3. Complete the five `TODO`s in `src/app.py` from top to bottom.
4. Re-run `pytest -v` after each completed function.
5. Run the full pipeline and complete `REFLECTION.md` using the saved
   `test_index`, `actual` and `classifier.predicted` lists.
6. Use the teaching slides as a checkpoint and self-test before submitting.

| TODO | Mirrors Final Assessment | What to do |
|---|---|---|
| `load_data` | Task 4(a) | Load the CSV with pandas |
| `get_feature_columns` | Task 4(b) | Choose predictors; exclude the identifier |
| `make_split` | Task 4(c) | Fixed, stratified train/test split |
| `train_baseline` / `train_classifier` | Task 4(c)-(d) | Baseline vs. one real classifier, through the supplied scaffold |
| (in `run`, after training) | Task 4(e) | Inspect 3 errors; write `REFLECTION.md` |

## Install

```text
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```text
python -m src.app --data data/helpdesk_tickets.csv --output outputs/result.json
```

## Check your work

```text
pytest -v
```

Every failing test tells you what is wrong and which Task 4 sub-part it
maps to. Do not edit `tests/test_pipeline.py` or `src/pipeline_tools.py` -
only `src/app.py` and, once your pipeline runs, `REFLECTION.md`.

When the pipeline runs successfully, `outputs/result.json` contains the
held-out row indices (`test_index`), true labels (`actual`) and model
predictions (`classifier.predicted`) in matching order. Use them to find the
three errors required by `REFLECTION.md`.

## Why this matters for the Final Assessment

Task 4 is worth more marks than any other single task. The pattern you
practice here - baseline vs. classifier, accuracy *and* macro-F1 *and* a
confusion matrix, features chosen deliberately rather than left to
guesswork - is exactly what the marking rubric rewards. A green check on
this practice repository does not predict your Final Assessment mark; it
tells you that you can currently build the pipeline the Final Assessment
requires.
