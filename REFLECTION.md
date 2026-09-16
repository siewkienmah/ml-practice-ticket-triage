## Three errors

1. CSV row index 66: predicted `escalate_now`, actually `assign_technician`. The model probably got confused by the `reopened_count` being 3. It might have given too much weight to the ticket being reopened multiple times, even though the `reported_severity` was only "low".
2. CSV row index 196: predicted `assign_technician`, actually `escalate_now`. The `reported_severity` was "high", but the `resolution_notes_length` was very short (only 74 characters). The model might incorrectly think that short notes mean it is not a serious issue yet.
3. CSV row index 217: predicted `assign_technician`, actually `log_only`. The `error_code` was missing (NaN). Our preprocessor probably filled this empty spot with the most common error code from the dataset, which tricked the model into thinking there was a real error when it was just a log.

## One defensible improvement

Instead of filling missing categorical values (like the empty `error_code` in row 217) with the most frequent item, I would change the preprocessor to fill them with a specific word like "Missing". This way, the model learns that an empty error code is its own distinct clue, rather than getting confused by fake, assumed data that we injected during preprocessing.

## One limitation

If we used this in a real IT helpdesk, it could unfairly delay help for brand-new, urgent problems. As seen in row 196, a severe ticket might not get escalated just because it hasn't been worked on long enough to have long notes. This means the model penalizes tickets that haven't been touched yet, meaning a genuinely urgent ticket might sit in the queue simply because an IT worker hasn't had time to type out a long paragraph about it.
