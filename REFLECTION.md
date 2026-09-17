# Reflection

## Model Performance

The baseline model achieved an accuracy of 0.483 and a macro F1-score of 0.217.
The trained classifier performed better, achieving an accuracy of 0.683 and a
macro F1-score of 0.478.

This shows that the trained classifier learned useful patterns from the ticket
features and performed substantially better than the DummyClassifier.

The DummyClassifier predicted every test example as `log_only`, which is the
most frequent class. Therefore, it provides a useful baseline but does not
actually learn relationships between the ticket features and response tier.

## Three Misclassified Examples

### Error 1 — Test Index 66

- Actual class: `assign_technician`
- Predicted class: `log_only`

The model classified this ticket as a simple logging case even though a
technician was actually required. One possible reason is that some of the
ticket's features may look similar to normal low-priority tickets. The
available features may not provide enough information to clearly distinguish
between these two classes.

### Error 2 — Test Index 196

- Actual class: `escalate_now`
- Predicted class: `assign_technician`

This is an important error because an urgent escalation was predicted only as
a technician assignment. The confusion matrix shows that all six
`escalate_now` examples in the test set were predicted as
`assign_technician`. This suggests that the model has difficulty learning the
`escalate_now` class.

A likely reason is that `escalate_now` has fewer examples than the other
classes, making it harder for the classifier to learn its pattern.

### Error 3 — Test Index 161

- Actual class: `log_only`
- Predicted class: `assign_technician`

In this case, the model predicted that technician intervention was necessary
when the ticket only needed to be logged. This may happen when the feature
values of a `log_only` ticket overlap with tickets that normally require
technical assistance.

## What I Learned

The trained classifier performs much better than the baseline, but accuracy
alone does not tell the complete story. Although the classifier achieved
68.3% accuracy, it failed to correctly identify any `escalate_now` cases.

The macro F1-score of 0.478 reflects this weakness because macro F1 gives
importance to the performance of every class, including less frequent
classes.

## Possible Improvements

The model could be improved by collecting more examples of the
`escalate_now` class, adding more informative ticket features, and addressing
class imbalance during training. A classifier with class weighting could also
be tested so that mistakes on the less frequent class receive more
importance.

Additional text-based features from ticket descriptions may also help the
model recognise urgent language and distinguish escalation cases from normal
technician assignments.