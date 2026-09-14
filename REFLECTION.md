# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index 196: predicted `assign_technician`, actually `escalate_now`. The model predicted `assign_technician` because `escalate_now` is a severely underrepresented class in the training dataset, leading the classifier to bias its predictions toward the more frequent majority classes (`assign_technician` and `log_only`).
2. CSV row index 209: predicted `assign_technician`, actually `escalate_now`. Similar to row 196, the ticket shared overlapping numeric and categorical feature ranges (such as priority or affected system) with standard technician-assigned tickets, causing the default decision boundary of the Logistic Regression model to fail to capture the urgency required for immediate escalation.
3. CSV row index 177: predicted `assign_technician`, actually `escalate_now`. The lack of explicit feature weighting for rare, high-severity tickets meant that subtle feature differences indicating critical infrastructure failure were smoothed over by the preprocessor and standard scale transform.

## One defensible improvement

To improve classification accuracy on critical tickets like `escalate_now`, we could implement class weight balancing (`class_weight='balanced'` in `LogisticRegression`) or apply SMOTE / random oversampling to the training set. Additionally, engineering a new feature such as an interaction term between ticket priority and system criticality score would provide stronger signals for high-severity issues.

## One limitation

If deployed in a live help-desk environment, a major limitation of this classifier is its high rate of false negatives for urgent tickets (`escalate_now` predicted as `assign_technician`). This could cause severe system outages or security breaches to sit unaddressed while waiting in a standard technician queue rather than triggering an instant emergency response, negatively affecting critical business operations and high-priority users.