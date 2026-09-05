# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index ___: predicted `___`, actually `___`. Which feature values
   may have led the model towards the wrong class?
2. CSV row index ___: predicted `___`, actually `___`. Which feature values
   may have led the model towards the wrong class?
3. CSV row index ___: predicted `___`, actually `___`. Which feature values
   may have led the model towards the wrong class?

## One defensible improvement

Describe one specific, defensible change to your features or preprocessing
that might reduce these errors (not "use a bigger model" - something you
could implement and test).

## One limitation

State one bias, fairness, or deployment limitation of this classifier if it
were actually used to route real help-desk tickets. Who could be affected,
and how?
