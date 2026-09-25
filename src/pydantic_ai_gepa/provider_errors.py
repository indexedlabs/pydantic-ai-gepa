"""Classify provider failures that require operator intervention."""

from __future__ import annotations

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
# How `str(ModelHTTPError)` renders a credential failure.
_CREDENTIAL_STATUS_TEXT = re.compile(r"\bstatus_code: (?:401|403)\b")


class ProviderStopError(RuntimeError):
    """A provider failure an operator must fix before evaluation can continue.

    Raise it from an evaluate callable or adapter when such a failure reaches it
    without the original ``ModelHTTPError``, for example when a case ran in a
    child process and only its error message came back. Evaluation stops on it
    the same way it stops on the provider's own error.
    """


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its explicit/implicit causes without looping."""

    seen: set[int] = set()
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

    for error in _exception_chain(exc):
        if isinstance(error, ProviderStopError):
            return True
        if not isinstance(error, ModelHTTPError):
            continue
        if error.status_code in _CREDENTIAL_ERROR_STATUS_CODES:
            return True
        body = str(error.body).lower()
        if any(code in body for code in PROVIDER_STOP_ERROR_CODES):
            return True
    return False


def is_provider_stop_message(text: str) -> bool:
    """Return whether a provider error's text describes a stop failure.

    For runners that only see a failed case's message, such as a child process's
    saved ``str(exc)``. It matches the billing codes and a 401/403 status as
    ``ModelHTTPError`` renders it. Pass only messages of provider HTTP errors:
    a code echoed in case content would match too.
    """

    lowered = text.lower()
    return bool(_CREDENTIAL_STATUS_TEXT.search(lowered)) or any(
        code in lowered for code in PROVIDER_STOP_ERROR_CODES
    )


__all__ = [
    "PROVIDER_STOP_ERROR_CODES",
    "ProviderStopError",
    "is_provider_stop_error",
    "is_provider_stop_message",
]
