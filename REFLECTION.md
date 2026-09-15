# Reflection (Task 4e practice)

Fill this in after `python -m src.app` runs successfully.

## Three errors

Open `outputs/result.json`. The lists `test_index`, `actual` and
`classifier.predicted` use the same order. Compare each prediction with its
true label. When they differ, use the matching `test_index` value to locate
the original ticket in `data/helpdesk_tickets.csv`.

1. CSV row index ___: predicted `___`, actually `___`. Which feature values
   may have led the model towards the wrong class?
   - The ticket was a network device with error code E33, a wait time of only 1.0 minute, low reported severity, 1 reopened case, and a resolution notes length of 45. The network device, E33 error code, or reopened count may have matched patterns from technician-assignment tickets in the training data, causing the model to predict `assign_technician` even though the severity and waiting time were low.

2. CSV row index ___: predicted `___`, actually `___`. Which feature values
   may have led the model towards the wrong class?
   - The ticket involved a printer with error code E47, a wait time of 14.5 minutes, high reported severity, 1 reopening, and a resolution notes length of 74. The model may have relied too much on the device type, error code, or moderate waiting time and did not give enough importance to the high severity, causing an urgent ticket to be predicted as `log_only`.

3. CSV row index ___: predicted `___`, actually `___`. Which feature values
   may have led the model towards the wrong class?
   - The ticket involved a printer with error code E21, a long wait time of 52.8 minutes, high reported severity, 0 reopenings, and a resolution notes length of 85. The high severity and long waiting time suggest that escalation was appropriate, but the model may have been influenced by the zero reopened count or patterns associated with printer and E21 tickets, leading it to predict `assign_technician` instead.
## One defensible improvement

Describe one specific, defensible change to your features or preprocessing
that might reduce these errors (not "use a bigger model" - something you
could implement and test).

- I would add an urgency feature using `reported_severity` and `wait_time_minutes`. I could change severity into numbers, such as low = 1, medium = 2, and high = 3. Then I could combine this value with the waiting time. This may help the model notice tickets that have high severity and long waiting times, especially tickets that should be `escalate_now`.

## One limitation

State one bias, fairness, or deployment limitation of this classifier if it
were actually used to route real help-desk tickets. Who could be affected,
and how?

- One limitation is that the model can classify an urgent `escalate_now` ticket as `log_only` or `assign_technician`. If this happens in a real help-desk system, users with serious technical problems may need to wait longer for support. A critical system problem may also take longer to reach the correct support team.