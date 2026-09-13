# Reflection (Task 4e practice)

## Three errors

1. **CSV row index 196 (TCK-0138)**: predicted `assign_technician`, actually `escalate_now`.
   The ticket has `severity=high` and `reopened_count=1`, which are escalation signals, but
   `wait_time_minutes=14.5` and `notes_len=74` are both moderate. The model appears to have
   weighted the numeric features (short wait, mid-length notes) toward a less urgent class,
   overriding the categorical high-severity signal. The combination of E47 + high severity +
   at least one reopen was not enough on its own to push the prediction to escalate_now.

2. **CSV row index 66 (TCK-0015)**: predicted `escalate_now`, actually `assign_technician`.
   This is the mirror problem: `reopened_count=3` and `notes_len=129` are high numeric values
   that the model associated with escalation, even though `severity=low`. The model
   over-weighted the numeric signals and produced a false escalation — the ticket needed a
   technician, not emergency routing.

3. **CSV row index 209 (TCK-0192)**: predicted `assign_technician`, actually `escalate_now`.
   `wait_time_minutes=7.7` and `notes_len=44` are both small, which likely looked like a minor
   ticket to the model despite `severity=high` and `reopened_count=2`. Short wait time can
   mean the ticket was filed quickly on a serious issue, not that the issue is trivial — the
   model has no way to distinguish these cases from the current features alone.

## One defensible improvement

The model misses six `escalate_now` tickets in the test set, predicting all of them as
`assign_technician`. A concrete, testable fix is to pass `class_weight='balanced'` to
`RandomForestClassifier`. This makes the model penalise misses on the minority class
(`escalate_now`, 10% of rows) proportionally more during training, without adding or
changing any features. It is one parameter change and its effect can be measured directly
by comparing the `escalate_now` row of the confusion matrix before and after.

A complementary change would be to add an explicit interaction feature:
`reopened_x_severity` (encoding severity as 0/1/2 and multiplying by `reopened_count`).
Rows like TCK-0192 (high × 2 = 4) would then carry a distinct numeric signal that the tree
cannot currently derive by itself from two separate columns.

## One limitation

`escalate_now` accounts for only 10% of tickets in this dataset, and a model that
misclassifies an escalation as `assign_technician` causes real harm — a critical issue goes
unattended. Standard accuracy treats every misclassification equally, but the real-world
cost of a missed escalation is far higher than the cost of a false alarm. If this classifier
were deployed to route actual help-desk tickets, it would systematically under-serve users
whose urgent tickets arrive quickly (short `wait_time_minutes`) or whose `reported_severity`
field is missing — groups that the model has learned to associate with lower urgency. Without
a human review step for borderline predictions or a cost-sensitive threshold, those users
would face delayed responses for genuinely critical issues, with no visibility into why.
