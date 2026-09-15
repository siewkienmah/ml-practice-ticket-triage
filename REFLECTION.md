# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index **66**: predicted `log_only`, actually `assign_technician`.
   The `reported_severity` was **low**, which may have led the model towards
   `log_only`. However, the ticket had a `reopened_count` of **3** and
   `resolution_notes_length` of **129**, which may indicate that the ticket
   needed technician support.

2. CSV row index **81**: predicted `log_only`, actually `assign_technician`.
   The `reported_severity` was **medium** and the `wait_time_minutes` was
   **26.8**. The ticket had a `reopened_count` of **0**, which may have made
   the model predict `log_only` even though the actual class was
   `assign_technician`.

3. CSV row index **172**: predicted `log_only`, actually `assign_technician`.
   The ticket had **high** `reported_severity`, a `wait_time_minutes` of
   **21.4**, and a `reopened_count` of **1**. These features may suggest that
   the ticket needed technician support, but the model still predicted
   `log_only`.

## One defensible improvement

One specific improvement would be to add an **urgency score** that combines
`reported_severity`, `wait_time_minutes`, and `reopened_count`. This could
help the classifier identify tickets that need technician support more
accurately.

## One limitation

One limitation is that the classifier may incorrectly route urgent tickets to
`log_only`. Customers with serious technical problems could be affected
because their tickets may not be sent to a technician quickly enough. Human
staff should review important or high-severity tickets before making the final
routing decision
