"""Classify provider failures that require operator intervention."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic_ai.exceptions import ModelHTTPError


PROVIDER_STOP_ERROR_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "insufficient_quota",
        # OpenAI hard spend limits: the project's or the organization's monthly cap.
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
    }
)
"""Provider error codes that mean billing, not the request, is the problem."""
_CREDENTIAL_ERROR_STATUS_CODES = frozenset({401, 403})


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
        if mentions_provider_stop_code(str(error.body)):
            return True
    return False


def mentions_provider_stop_code(text: str) -> bool:
    """Return whether an error message or body names a billing stop code."""

    lowered = text.lower()
    return any(code in lowered for code in PROVIDER_STOP_ERROR_CODES)


__all__ = [
    "PROVIDER_STOP_ERROR_CODES",
    "ProviderStopError",
    "is_provider_stop_error",
    "mentions_provider_stop_code",
]
