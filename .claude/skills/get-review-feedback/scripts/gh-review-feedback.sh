# Source - https://stackoverflow.com/a/66072198

# Posted by Glenn 'devalias' Grant, modified by community. See post 'Timeline' for change history

# Retrieved 2026-03-08, License - CC BY-SA 4.0

# Usage:
#   Retrieve unresolved review comments
#     ./gh-review-feedback myorganisation myrepo 1337
#
#   Count of unresolved review comment threads
#     ./gh-review-feedback myorganisation myrepo 1337 | jq length

#gh api graphql -f owner="$1" -f repo="$2" -F pr="$3" -f query='
gh api graphql -f owner="alkanen" -f repo="pr_checker" -F pr="$1" -f query='
  query FetchReviewComments($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        url
        reviewDecision
        reviewThreads(last: 100) {
          edges {
            node {
              isResolved
              isOutdated
              isCollapsed
              comments(last: 100) {
                totalCount
                nodes {
                  author {
                    login
                  }
                  body
                  url
                }
              }
            }
          }
        }
      }
    }
  }
' | jq -c '.data.repository.pullRequest.reviewThreads.edges | map(select(.node.isResolved == false)) | map(.node | del(.isResolved, .isOutdated, .isCollapsed)) | map(.comments.nodes |= map(del(.url) | .author = .author.login))'
