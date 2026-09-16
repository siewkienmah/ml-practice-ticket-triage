# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index **66** (ticket `TCK-0015`): predicted `log_only`, actually
   `assign_technician`. The model likely relied on `reported_severity=low`
   and the moderate `wait_time_minutes=18.0` to down-triage the ticket, and
   it underweighted the stronger escalation signals: `reopened_count=3` and
   `resolution_notes_length=129`. A recurring ticket with a long resolution
   note is more likely to need a technician, so the model is missing an
   interaction between `reopened_count` and `resolution_notes_length`.

2. CSV row index **196** (ticket `TCK-0138`): predicted `assign_technician`,
   actually `escalate_now`. This is a severity-underestimation error: the
   ticket has `reported_severity=high`, `error_code=E47` and
   `device_type=printer`, yet the model picked the middle tier. Because a
   DecisionTree splits one feature at a time, it can fail to combine
   "high severity" with "printer error E47" into a single escalation rule,
   so genuinely urgent tickets get sent to the wrong queue.

3. CSV row index **2** (ticket `TCK-0004`): predicted `log_only`, actually
   `assign_technician`. The key problem here is `error_code` is missing
   (NaN). The preprocessor imputes categorical NaNs with the most frequent
   value, which erases the fact that the code is unknown. For a projector
   with `wait_time_minutes=33.4` and `reported_severity=high`, the missing
   error code probably hides the real signal that would have justified
   assigning a technician.

## One defensible improvement

Replace the single most-frequent imputation for `error_code` with an
explicit `"missing"` category (e.g. `SimpleImputer(strategy="constant",
fill_value="missing")` for the categorical branch, followed by the existing
`OneHotEncoder`). This turns "unknown error code" into a signal the tree
can learn from instead of silently hiding it behind `E10`/`E52`. As a
second, complementary change, add an engineered numeric feature
`reopened_x_notes = reopened_count * resolution_notes_length` to give the
model a single column that directly captures the "recurring, hard-to-close"
pattern behind error #1. Both changes are small, testable, and do not
require a bigger or more complex model.

## One limitation

This classifier routes real help-desk tickets, and because it is trained on
historical ticket data it will inherit any historical bias in that data. If
certain `device_type`s (for example `printer` or `projector`) were
historically under-triaged - because of which team owned them, or which
departments filed them - the model will learn to keep under-triaging them,
even when severity and wait time say otherwise. The people most affected are
those whose tickets are systematically pushed toward `log_only`, because
their issues will sit in a low-priority queue and could take much longer to
resolve. A deployment safeguard would be to (a) monitor prediction
distribution per `device_type` and `reported_severity`, and (b) keep a human
review step for any ticket the model puts in `log_only` but which has
`reported_severity=high`.