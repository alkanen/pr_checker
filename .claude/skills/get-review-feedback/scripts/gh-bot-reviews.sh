#!/usr/bin/env bash
# Fetch pr-checker bot review comments posted after the most recent commit on a PR.
# Comments older than the latest commit have already been acted on (a push was made
# in response), so only newer ones are surfaced.
#
# Usage:
#   ./gh-bot-reviews.sh <pr-number>
#
# Output: JSON array of comment body strings, empty array if nothing pending.

set -euo pipefail

PR="${1:?Usage: $0 <pr-number>}"
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

LATEST_COMMIT_TS=$(gh api --paginate "repos/${REPO}/pulls/${PR}/commits" | \
  jq -rs '[.[] | .[].commit.committer.date] | sort | last')

gh api --paginate "repos/${REPO}/issues/${PR}/comments" | \
  jq -rs --arg since "$LATEST_COMMIT_TS" '
    [.[] | .[] | select(
      (.body | contains("*Posted by pr-checker*")) and
      .created_at > $since
    ) | .body]
  '
