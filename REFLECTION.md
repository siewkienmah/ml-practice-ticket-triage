# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index 66: predicted `log_only`, actually `assign_technician`. The ticket has a low reported severity, which may have influenced the classifier to predict log_only. However, it has been reopened three times, suggesting that the issue may require technician attention. The model may not have learned to give sufficient importance to reopened_count.

2. CSV row index 175: predicted `log_only`, actually `assign_technician`. The ticket has medium reported severity and a wait time of 20.4 minutes, but the classifier predicted log_only. The model may not have learned how to combine the error code, device type, and wait time effectively when deciding whether technician intervention is required.

3. CSV row index 196: predicted `assign_technician`, actually `escalate_now`. The ticket has high reported severity and a wait time of 14.5 minutes. The classifier selected assign_technician instead of escalate_now, suggesting that it may not distinguish urgent high-severity cases effectively.

## One defensible improvement

One improvement would be to evaluate class-specific performance and adjust the model to better identify underrepresented or difficult-to-classify response tiers. In particular, the confusion matrix shows that only two of the six actual escalate_now cases were correctly classified. I would investigate class weighting or tune the Decision Tree's parameters, then compare macro-F1 and the confusion matrix against the original model using the same train/test split.

## One limitation

The classifier may make incorrect triage decisions because it relies on the available features and patterns in the training data. For example, a high-severity ticket may be assigned to a technician instead of being escalated immediately. If deployed without human oversight, such errors could delay responses to urgent incidents. The model should therefore be evaluated on additional representative tickets, particularly urgent cases, and its recommendations should be reviewed by help-desk staff before being used in real operations.
