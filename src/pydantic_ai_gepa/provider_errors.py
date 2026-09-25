"""Classify provider failures that require operator intervention."""

from __future__ import annotations

import builtins
import re
from collections.abc import Iterator

from pydantic_ai.exceptions import ModelHTTPError


PROVIDER_STOP_ERROR_CODES = frozenset(
    {
        "billing_hard_limit_reached",
        "credit_balance_exhausted",
        "insufficient_quota",
        # OpenAI hard spend limits: the project's or the organization's monthly cap.
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
    }
)
"""Provider error codes that mean billing, not the request, is the problem."""
_CREDENTIAL_ERROR_STATUS_CODES = frozenset({401, 403})
# Exception groups are builtins from Python 3.11; on 3.10 pydantic-ai raises the
# `exceptiongroup` backport's.
_BaseExceptionGroup: type[BaseException] | None = getattr(
    builtins, "BaseExceptionGroup", None
)
if _BaseExceptionGroup is None:  # pragma: no cover - Python 3.10
    try:
        from exceptiongroup import BaseExceptionGroup as _BaseExceptionGroup
    except ImportError:
        _BaseExceptionGroup = None
PROVIDER_STOP_REASON = "Provider billing or credential failure"
"""The run's ``stop_reason`` when a provider stop ends optimization early."""
# A credential failure as `str(ModelHTTPError)` ("status_code: 401") or an OpenAI
# SDK `APIStatusError` ("Error code: 401 - ...") renders it, lowercased.
_CREDENTIAL_STATUS_TEXT = re.compile(r"\b(?:status_code|error code): (?:401|403)\b")


class ProviderStopError(RuntimeError):
    """A provider failure an operator must fix before evaluation can continue.

    Raise it from an evaluate callable or adapter when such a failure reaches it
    without the original ``ModelHTTPError``, for example when a case ran in a
    child process and only its error message came back. Evaluation stops on it
    the same way it stops on the provider's own error.
    """


def _exception_chain(exc: BaseException, seen: set[int]) -> Iterator[BaseException]:
    """Yield an exception and its explicit/implicit causes, skipping any seen."""

    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_provider_stop_error(exc: BaseException) -> bool:
    """Return whether a provider failure cannot recover without an operator.

    Authentication/authorization failures and exhausted billing credit should
    stop an evaluation run. Other provider failures, including ordinary rate
    limiting, retain the existing per-case failure behavior.
    """

    return _is_stop(exc, set())


def _is_stop(exc: BaseException, seen: set[int]) -> bool:
    # One `seen` set spans the chain and any exception groups in it: a group's
    # child can have the group itself as its context.
    for error in _exception_chain(exc, seen):
        if isinstance(error, ProviderStopError):
            return True
        if _BaseExceptionGroup is not None and isinstance(error, _BaseExceptionGroup):
            # e.g. a FallbackModel whose every provider failed: stop only when
            # each child would stop on its own.
            children = error.exceptions
            if children and all(_is_stop(child, seen) for child in children):
                return True
            continue
        if not isinstance(error, ModelHTTPError):
            continue
        if error.status_code in _CREDENTIAL_ERROR_STATUS_CODES:
            return True
        if _mentions_stop_code(str(error.body)):
            return True
    return False


def _mentions_stop_code(text: str) -> bool:
    lowered = text.lower()
    return any(code in lowered for code in PROVIDER_STOP_ERROR_CODES)


def is_provider_stop_message(text: str) -> bool:
    """Return whether a provider error's text describes a stop failure.

    For runners that only see a failed case's message, such as a child process's
    saved ``str(exc)``. It matches the billing codes and a 401/403 status as
    ``ModelHTTPError`` renders it. Pass only messages of provider HTTP errors:
    a code echoed in case content would match too.
    """

    return bool(_CREDENTIAL_STATUS_TEXT.search(text.lower())) or _mentions_stop_code(
        text
    )


__all__ = [
    "PROVIDER_STOP_ERROR_CODES",
    "PROVIDER_STOP_REASON",
    "ProviderStopError",
    "is_provider_stop_error",
    "is_provider_stop_message",
]
