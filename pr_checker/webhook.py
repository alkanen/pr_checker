import hashlib
import hmac
from typing import Any

from fastapi import HTTPException

from pr_checker.models import PRJob, ReviewTrigger

_TRIGGER_MAP: dict[str, ReviewTrigger] = {
    "opened": ReviewTrigger.OPENED,
    "synchronize": ReviewTrigger.UPDATED,
    "review_requested": ReviewTrigger.REVIEW_REQUESTED,
}


def validate_signature(payload: bytes, signature_header: str | None, secret: str) -> None:
    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing X-Hub-Signature-256 header")
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Invalid signature format")
    expected = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Invalid signature")


def parse_pr_event(payload: dict[str, Any]) -> PRJob | None:
    action: str = payload.get("action", "")
    trigger = _TRIGGER_MAP.get(action)
    if trigger is None:
        return None

    try:
        pr: dict[str, Any] = payload["pull_request"]
        repo: dict[str, Any] = payload["repository"]
        return PRJob(
            repo_full_name=repo["full_name"],
            pr_number=pr["number"],
            pr_title=pr["title"],
            head_sha=pr["head"]["sha"],
            base_sha=pr["base"]["sha"],
            head_branch=pr["head"]["ref"],
            base_branch=pr["base"]["ref"],
            trigger=trigger,
        )
    except (KeyError, TypeError):
        return None
