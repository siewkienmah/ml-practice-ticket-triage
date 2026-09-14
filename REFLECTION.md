# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index` `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index **66**: predicted `log_only`, actually `assign_technician`.
   The ticket has low reported severity and a wait time of 18.0 minutes, which
   may have led the model towards `log_only`. However, it was reopened 3 times,
   which may indicate that technician attention was needed.

2. CSV row index **81**: predicted `log_only`, actually `assign_technician`.
   The ticket has medium reported severity, a wait time of 26.8 minutes, and
   zero reopened cases. These feature values may have made the ticket look less
   urgent and led the model towards `log_only`.

3. CSV row index **172**: predicted `log_only`, actually `assign_technician`.
   The ticket has high reported severity, a wait time of 21.4 minutes, and was
   reopened once. Although the severity is high, the other feature values may
   have caused the model to predict `log_only`.

## One defensible improvement

One specific improvement would be to add or engineer a feature that combines
reported severity with wait time and reopened count. This could help the model
better identify tickets that need technician assignment when several signals
indicate higher priority.

## One limitation

A limitation is that the classifier may incorrectly route some tickets to a
lower response tier. This could affect users with urgent technical problems
because their tickets may be logged instead of being assigned to a technician.
Human staff should review important or high-severity tickets before relying on
the prediction.
