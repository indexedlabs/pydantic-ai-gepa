"""Classify provider failures that require operator intervention."""

from __future__ import annotations

from collections.abc import Iterator

from pydantic_ai.exceptions import ModelHTTPError


_BILLING_ERROR_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "insufficient_quota",
    }
)
_CREDENTIAL_ERROR_STATUS_CODES = frozenset({401, 403})


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
        if not isinstance(error, ModelHTTPError):
            continue
        if error.status_code in _CREDENTIAL_ERROR_STATUS_CODES:
            return True
        body = str(error.body).lower()
        if any(code in body for code in _BILLING_ERROR_CODES):
            return True
    return False


__all__ = ["is_provider_stop_error"]
