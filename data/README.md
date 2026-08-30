# Practice dataset: help-desk ticket triage

`helpdesk_tickets.csv` is a synthetic dataset of 240 IT help-desk tickets.
Each row is one ticket; the label to predict is `response_tier`.

| Column | Type | Notes |
|---|---|---|
| `ticket_id` | identifier | unique per row - not a predictor |
| `device_type` | categorical | laptop / desktop / printer / projector / network |
| `error_code` | categorical | E10-E52; some values are genuinely missing |
| `wait_time_minutes` | numeric | minutes the ticket has been open; a few missing |
| `reported_severity` | categorical | low / medium / high; some missing |
| `reopened_count` | numeric | how many times the ticket was reopened |
| `resolution_notes_length` | numeric | length of the technician's notes so far |
| `response_tier` | **target** | `log_only`, `assign_technician`, or `escalate_now` |

This is shared, ungraded practice data - everyone works from the same file.
The missing values and class imbalance are deliberate: the Final
Assessment's own data will have both, and handling them correctly is part
of what Task 4 checks.
