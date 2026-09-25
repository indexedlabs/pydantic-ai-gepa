"""Concurrency helpers shared by evaluation paths."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from .provider_errors import is_provider_stop_error
from ._validation import validation_active

T = TypeVar("T")


async def gather_cancelling_on_provider_stop(*awaitables: Awaitable[T]) -> list[T]:
    """Run like ``asyncio.gather``; on a provider stop, cancel and drain the rest.

    After a billing or credential stop every other case fails the same way, so
    cases still in flight are cancelled rather than calling the provider again,
    possibly after their caller has left the candidate context. Other failures
    keep ``gather``'s behavior: the remaining cases run to completion, so paid
    rollouts still record usage and cache their results. Validation always
    drains pending tasks before its evidence context closes. Cancellation reaches
    coroutines only; work already handed to a thread keeps running.
    """

    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException as error:
        if is_provider_stop_error(error) or validation_active():
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        raise
