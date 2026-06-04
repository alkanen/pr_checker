---
name: get-review-feedback
description: Retrieves unresolved github PR feedback comments from the provided PR number. Use when user asks you to fix, analyse or correct feedback for a pull request, add the PR number as an argument.
allowed-tools: Bash(scripts/gh-review-feedback.sh *), Bash(scripts/gh-bot-reviews.sh *), Bash(scripts/gh-review-summaries.sh *)
---

# Get Review Feedback

Use all three scripts with the PR number as argument to fetch all pending feedback:

- [gh-review-feedback.sh](scripts/gh-review-feedback.sh) — unresolved Copilot/human review threads (inline comments)
- [gh-bot-reviews.sh](scripts/gh-bot-reviews.sh) — pr-checker bot reviews posted after the most recent commit (issue-level comments)
- [gh-review-summaries.sh](scripts/gh-review-summaries.sh) — review-level summary bodies posted after the most recent commit (top-level review comments)

Assess the comments and determine which issues are valid, consult the user if
any design decisions must be made. Then fix the relevant issues.

Run any relevant tests and linters when the fixes are in place to ensure there
are no regressions and that the code is up to spec.

Finally report back a short summary to the user.
