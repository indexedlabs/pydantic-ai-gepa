"""Concurrency helpers shared by evaluation paths."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def gather_cancelling_siblings(*awaitables: Awaitable[T]) -> list[T]:
    """Run like ``asyncio.gather``; on the first failure, cancel and drain the rest.

    Plain ``gather`` propagates the first exception but leaves the other tasks
    running, so cases already in flight keep calling a provider after a stop
    error, possibly outside the candidate context their caller has since left.
    """

    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if unfinished:
            await asyncio.gather(*unfinished, return_exceptions=True)
