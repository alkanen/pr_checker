"""Tests for logging_setup: TRACE level, configure_logging, job_id threading."""

import json
import logging
from collections.abc import Iterator

import pytest

from pr_checker.logging_setup import (
    _HANDLER_MARKER,
    TRACE,
    configure_logging,
    reset_job_id,
    set_job_id,
)


def _our_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, _HANDLER_MARKER, False)]


@pytest.fixture(autouse=True)
def _reset_loggers() -> Iterator[None]:
    """Remove only handlers we installed to avoid disturbing pytest's capture."""
    yield
    for logger in (logging.getLogger(), logging.getLogger("pr_checker")):
        for h in _our_handlers(logger):
            logger.removeHandler(h)
    logging.getLogger("pr_checker").propagate = True


def test_trace_level_is_registered() -> None:
    assert logging.getLevelName(TRACE) == "TRACE"
    assert TRACE == 5


def test_trace_level_below_debug() -> None:
    assert TRACE < logging.DEBUG


def test_configure_logging_sets_level() -> None:
    configure_logging("DEBUG", "text")
    assert logging.getLogger("pr_checker").level == logging.DEBUG


def test_configure_logging_info_default() -> None:
    configure_logging()
    assert logging.getLogger("pr_checker").level == logging.INFO


def test_configure_logging_unknown_level_falls_back_to_info() -> None:
    configure_logging("NOTAREALLEVEL", "text")
    assert logging.getLogger("pr_checker").level == logging.INFO


def test_configure_logging_propagate_disabled() -> None:
    configure_logging("INFO", "text")
    assert logging.getLogger("pr_checker").propagate is False


def test_configure_logging_installs_root_handler() -> None:
    configure_logging("INFO", "text")
    assert len(_our_handlers(logging.getLogger())) == 1


def test_json_format_produces_valid_json(capfd: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    logging.getLogger("pr_checker.test").info("hello world")
    out = capfd.readouterr().err
    obj = json.loads(out.strip())
    assert obj["msg"] == "hello world"
    assert obj["level"] == "INFO"
    assert "ts" in obj
    assert "logger" in obj


def test_json_format_includes_job_id(capfd: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    token = set_job_id("job-abc123")
    logging.getLogger("pr_checker.test").info("with job")
    reset_job_id(token)
    out = capfd.readouterr().err
    obj = json.loads(out.strip())
    assert obj.get("job_id") == "job-abc123"


def test_json_format_omits_job_id_when_empty(capfd: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    set_job_id("")
    logging.getLogger("pr_checker.test").info("no job")
    out = capfd.readouterr().err
    obj = json.loads(out.strip())
    assert "job_id" not in obj


def test_text_format_does_not_raise(capfd: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "text")
    logging.getLogger("pr_checker.test").info("text log")
    out = capfd.readouterr().err
    assert "text log" in out


def test_configure_logging_idempotent() -> None:
    configure_logging("INFO", "text")
    configure_logging("DEBUG", "text")
    assert len(_our_handlers(logging.getLogger("pr_checker"))) == 1
    assert len(_our_handlers(logging.getLogger())) == 1


def test_second_call_updates_formatter(capfd: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "text")
    configure_logging("INFO", "json")
    logging.getLogger("pr_checker.test").info("switched")
    out = capfd.readouterr().err
    obj = json.loads(out.strip())
    assert obj["msg"] == "switched"
