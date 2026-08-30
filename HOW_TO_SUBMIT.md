# How to submit this practice for an automated check

This repository uses the same fork-and-pull-request pattern many open-source
projects use, so that everyone can get an automated GitHub Actions check
without needing their own private repository or write access to this one.

## 1. Fork

Click **Fork** at the top of this repository's GitHub page. This creates
your own personal copy under your GitHub account.

## 2. Clone your fork and branch

```text
git clone https://github.com/YOUR-USERNAME/ml-practice-ticket-triage.git
cd ml-practice-ticket-triage
git checkout -b practice
```

## 3. Do the work

Complete the `TODO`s in `src/app.py`. Run `pytest -v` until every check
passes locally before you push - this saves you a round trip waiting for a
workflow approval.

## 4. Push to your fork

```text
git add src/app.py REFLECTION.md
git commit -m "Complete the ticket triage ML pipeline"
git push origin practice
```

## 5. Open a pull request back to this repository

On GitHub, open a pull request from `YOUR-USERNAME:practice` into
`this-repository:main`.

## 6. Wait for the workflow to be approved

Because you are not a collaborator on this repository, GitHub will show
your pull request's check as **"Workflow awaiting approval."** This is
expected - it is the same "workflow approval" gate that keeps GitHub
Actions from running arbitrary code from anyone's fork automatically.
The lecturer will see your pull request in the repository's **Actions**
tab and click **Approve and run**. You do not need to do anything else;
you will see the check turn green or red directly on your pull request
once it runs.

## 7. Read the result

- **All checks passed**: your pipeline is built the way the Final
  Assessment expects. Keep it as a reference for your own repository later.
- **Some checks failed**: open the failed check's log. Each test explains
  in plain language what is wrong and which Final Assessment sub-task it
  mirrors. Fix `src/app.py`, push again, and the same pull request will
  re-run automatically once approved again.

This practice is **not graded** and your pull request will not be merged.
Once you have a green check (or you have learned what you needed to from a
red one), you can close your pull request.
