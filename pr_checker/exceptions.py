class StalePRError(Exception):
    """Raised when the PR head SHA no longer matches the job's expected SHA."""
