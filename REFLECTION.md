# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index _66__: predicted `_log_only__`, actually `_assign_technician__`. Which feature values
   may have led the model towards the wrong class?
    This ticket is a `desktop` issue with error code `E21`. The wait time was 18.0 minutes, the reported severity was `low`, the reopened count was 3, and the resolution notes length was 129.

   The model may have predicted `log_only` because the wait time was not very long and the severity was low. However, the ticket had already been reopened 3 times, which suggests that the problem still needed technician attention.

2. CSV row index _196__: predicted `_assign_technician__`, actually `_escalate_now__`. Which feature values
   may have led the model towards the wrong class?
   This ticket is a `printer` issue with error code `E47`. The wait time was 14.5 minutes, the reported severity was `high`, the reopened count was 1, and the resolution notes length was 74.

   The model may have predicted `assign_technician` because the wait time was short and the reopened count was low. However, the high severity meant that the ticket should actually be escalated.

3. CSV row index _43__: predicted `_escalate_now__`, actually `_assign_technician__`. Which feature values
   may have led the model towards the wrong class?
    This ticket is a `desktop` issue with error code `E10`. The wait time was 45.7 minutes, the reported severity was `medium`, the reopened count was 0, and the resolution notes length was 11.

   The model may have predicted `escalate_now` because the waiting time was quite long. However, the correct class was only `assign_technician`.

## One defensible improvement

Describe one specific, defensible change to your features or preprocessing
that might reduce these errors (not "use a bigger model" - something you
could implement and test).

One possible improvement is to create a new feature that combines waiting time and reopened count.
For example:
`wait_reopen_risk = wait_time_minutes * (reopened_count + 1)`
A ticket that has waited for a long time and has also been reopened several times may need more attention.
After adding this feature, I would retrain the model and compare the macro-F1 score and confusion matrix to see if the model improves.


## One limitation

State one bias, fairness, or deployment limitation of this classifier if it
were actually used to route real help-desk tickets. Who could be affected,
and how?

One limitation is that the classes are imbalanced.
The `escalate_now` class is much smaller than the other classes. In the test set, the classifier correctly predicted only 2 out of 6 actual `escalate_now` tickets.
This can be risky in a real help desk because an urgent ticket may not be escalated correctly.
Therefore, the model should be used to support staff decisions, not completely replace human judgement.