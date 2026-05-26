import hashlib
import hmac
from typing import Any

import pytest
from fastapi import HTTPException

from pr_checker.models import ReviewTrigger
from pr_checker.webhook import parse_pr_event, validate_signature

_SECRET = "test-secret"  # noqa: S105


def _sig(payload: bytes, secret: str = _SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def _pr_payload(action: str = "opened") -> dict[str, Any]:
    return {
        "action": action,
        "number": 42,
        "pull_request": {
            "number": 42,
            "title": "Add feature",
            "head": {"sha": "abc123", "ref": "feature-branch"},
            "base": {"sha": "def456", "ref": "main"},
        },
        "repository": {"full_name": "owner/repo"},
    }


# --- validate_signature ---


def test_valid_signature_does_not_raise() -> None:
    payload = b'{"action":"opened"}'
    validate_signature(payload, _sig(payload, _SECRET), _SECRET)


def test_invalid_signature_raises_401() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_signature(b"payload", "sha256=badhex", _SECRET)
    assert exc_info.value.status_code == 401


def test_missing_signature_raises_401() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_signature(b"payload", None, _SECRET)
    assert exc_info.value.status_code == 401


def test_wrong_format_raises_401() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_signature(b"payload", "md5=something", _SECRET)
    assert exc_info.value.status_code == 401


def test_tampered_payload_raises_401() -> None:
    payload = b'{"action":"opened"}'
    sig = _sig(payload, _SECRET)
    with pytest.raises(HTTPException) as exc_info:
        validate_signature(b'{"action":"closed"}', sig, _SECRET)
    assert exc_info.value.status_code == 401


def test_empty_secret_raises_500() -> None:
    with pytest.raises(HTTPException) as exc_info:
        validate_signature(b"payload", "sha256=anything", "")
    assert exc_info.value.status_code == 500


# --- parse_pr_event ---


def test_parse_opened() -> None:
    job = parse_pr_event(_pr_payload("opened"))
    assert job is not None
    assert job.trigger == ReviewTrigger.OPENED
    assert job.pr_number == 42
    assert job.pr_title == "Add feature"
    assert job.repo_full_name == "owner/repo"
    assert job.head_sha == "abc123"
    assert job.base_sha == "def456"
    assert job.head_branch == "feature-branch"
    assert job.base_branch == "main"


def test_parse_synchronize_maps_to_updated() -> None:
    job = parse_pr_event(_pr_payload("synchronize"))
    assert job is not None
    assert job.trigger == ReviewTrigger.UPDATED


def test_parse_review_requested() -> None:
    job = parse_pr_event(_pr_payload("review_requested"))
    assert job is not None
    assert job.trigger == ReviewTrigger.REVIEW_REQUESTED


def test_parse_closed_returns_none() -> None:
    assert parse_pr_event(_pr_payload("closed")) is None


def test_parse_labeled_returns_none() -> None:
    assert parse_pr_event(_pr_payload("labeled")) is None


def test_parse_unknown_action_returns_none() -> None:
    assert parse_pr_event(_pr_payload("assigned")) is None


def test_parse_missing_pull_request_key_returns_none() -> None:
    assert parse_pr_event({"action": "opened", "repository": {"full_name": "o/r"}}) is None


def test_parse_missing_nested_key_returns_none() -> None:
    payload: dict[str, Any] = {
        "action": "opened",
        "pull_request": {"number": 1},  # missing title, head, base
        "repository": {"full_name": "o/r"},
    }
    assert parse_pr_event(payload) is None
