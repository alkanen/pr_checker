from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_job_id: ContextVar[str] = ContextVar("job_id", default="")

# Marker attribute set on every handler we install, so configure_logging can
# remove its own handlers on a subsequent call without disturbing handlers added
# by pytest, uvicorn, or other frameworks.
_HANDLER_MARKER = "_pr_checker_handler"


def set_job_id(value: str) -> Token[str]:
    return _job_id.set(value)


def reset_job_id(token: Token[str]) -> None:
    _job_id.reset(token)


class _JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__["job_id"] = _job_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        job_id = getattr(record, "job_id", "")
        if job_id:
            obj["job_id"] = job_id
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        asctime = self.formatTime(record, self._DATEFMT)
        job_id = getattr(record, "job_id", "")
        job_part = f"[{job_id}] " if job_id else ""
        s = f"{asctime} {record.levelname:<8} {job_part}{record.name}: {record.message}"
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s += "\n"
            s += record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s += "\n"
            s += self.formatStack(record.stack_info)
        return s


def _resolve_level(level_name: str) -> int:
    level = logging.getLevelName(level_name.upper())
    return level if isinstance(level, int) else logging.INFO


def _make_handler(level: int, formatter: logging.Formatter) -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(level)
    h.addFilter(_JobIdFilter())
    h.setFormatter(formatter)
    setattr(h, _HANDLER_MARKER, True)
    return h


def _remove_our_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if getattr(h, _HANDLER_MARKER, False):
            logger.removeHandler(h)


def configure_logging(level_name: str = "INFO", fmt: str = "text") -> None:
    level = _resolve_level(level_name)
    formatter: logging.Formatter = _JsonFormatter() if fmt.lower() == "json" else _TextFormatter()

    # Root logger: covers third-party libraries (httpx, asyncio, etc.).
    # We only remove handlers we previously installed to avoid disturbing
    # pytest's capture handler or uvicorn's own handlers.
    root = logging.getLogger()
    _remove_our_handlers(root)
    root.setLevel(level)
    root.addHandler(_make_handler(level, formatter))

    # pr_checker logger: propagate=False so its records don't double-emit
    # through the root handler we just installed.
    pr_logger = logging.getLogger("pr_checker")
    _remove_our_handlers(pr_logger)
    pr_logger.setLevel(level)
    pr_logger.propagate = False
    pr_logger.addHandler(_make_handler(level, formatter))

    # Redirect uvicorn's loggers through root so startup/access/error lines
    # share our formatter and level rather than using uvicorn's default text handler.
    # uvicorn sets propagate=False on these loggers in its dictConfig; we undo that
    # here so they fall through to the root handler we just installed.
    # Guard on propagate=False so we only strip uvicorn's own handlers and don't
    # touch loggers that were already reconfigured by a prior configure_logging call.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        if not uv.propagate:
            for h in list(uv.handlers):
                uv.removeHandler(h)
            uv.propagate = True
