"""Response-level dollar accounting shared by all model calls in a GEPA run."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.messages import ModelResponse
from pydantic_ai.run import AgentRunResult

from .exceptions import UsageBudgetExceeded

SpendCategory = Literal["reflection", "rollout"]
RolloutKind = Literal["training", "validation"]
PriceFn = Callable[[ModelResponse], float | None]
COST_STOP_REASON = "Cost budget reached"

_rollout_kind: ContextVar[RolloutKind] = ContextVar(
    "gepa_rollout_kind", default="training"
)


@contextmanager
def use_rollout_kind(kind: RolloutKind) -> Iterator[None]:
    """Tag rollouts started in this context as ``kind`` for cost projections.

    The step declares the kind of the dataset its cases come from: reflect
    minibatches evaluate training-set cases (``"training"``, the default),
    while full validation and merge subsamples evaluate validation-set cases
    (``"validation"``). The kind is a cost bucket, declared by the step rather
    than inferred from the evidence-withholding context.
    """
    token = _rollout_kind.set(kind)
    try:
        yield
    finally:
        _rollout_kind.reset(token)


class CostBudgetExceeded(UsageBudgetExceeded):
    """A dollar budget or unknown price stopped further paid work."""

    def __init__(self, reason: str = COST_STOP_REASON) -> None:
        self.stop_reason = reason
        super().__init__(reason)


class ModelSpend(BaseModel):
    """Aggregate usage; dollars include only responses with a known price."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    dollars: float = 0.0
    unpriced_requests: int = 0
    unpriced_input_tokens: int = 0
    unpriced_output_tokens: int = 0


class SpendReport(BaseModel):
    """An aggregate snapshot containing no case-level information."""

    total_dollars: float = 0.0
    reflection_dollars: float = 0.0
    rollout_dollars: float = 0.0
    by_category: dict[str, dict[str, ModelSpend]] = Field(default_factory=dict)
    by_model: dict[str, ModelSpend] = Field(default_factory=dict)
    unpriced_usage: dict[str, ModelSpend] = Field(default_factory=dict)
    max_token_cost: float | None = None
    stopped_by_cost: bool = False
    stop_reason: str | None = None


@dataclass
class _StepUsage:
    category: SpendCategory
    requests: int = 0
    dollars: float = 0.0
    kind: RolloutKind = "training"


class SpendMeter:
    """Thread-safe response ledger and per-kind running-mean step projections.

    ``price_fn`` can override a response's price in US dollars. Returning None
    falls back to the bundled genai-prices catalog via ``ModelResponse.cost``.
    No background catalog updates are enabled here.

    Rollout observations, dollars, and the highest single-rollout cost are
    tracked separately for training and validation rollouts so a batch is
    projected at its own kind's mean instead of a blended one. With both
    ``max_token_cost`` and ``max_concurrent`` set, ``admit_rollout`` gates
    every rollout start: a capped run ends within one rollout of the cap, at
    the highest cost observed for that rollout's kind. A rollout costing more
    than any earlier rollout of its kind (a new price high) can overshoot by
    its own excess, and the first observed rollout of a kind starts without a
    projection. Reflection overshoot stays bounded by the reflection
    projection plus the response backstop.
    """

    def __init__(
        self,
        max_token_cost: float | None = None,
        price_fn: PriceFn | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        if max_token_cost is not None and (
            not math.isfinite(max_token_cost) or max_token_cost <= 0
        ):
            raise ValueError("max_token_cost must be finite and > 0")
        if max_concurrent is not None and max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        self.max_token_cost = max_token_cost
        self.price_fn = price_fn
        self.max_concurrent = max_concurrent
        self._lock = RLock()
        self._usage: dict[SpendCategory, dict[str, ModelSpend]] = {
            "reflection": {},
            "rollout": {},
        }
        self._observations: dict[SpendCategory, int] = {"reflection": 0, "rollout": 0}
        self._kind_observations: dict[RolloutKind, int] = {
            "training": 0,
            "validation": 0,
        }
        self._kind_dollars: dict[RolloutKind, float] = {
            "training": 0.0,
            "validation": 0.0,
        }
        self._kind_highest: dict[RolloutKind, float] = {
            "training": 0.0,
            "validation": 0.0,
        }
        self._active: dict[RolloutKind, int] = {"training": 0, "validation": 0}
        self._reserved = 0.0
        self._condition = asyncio.Condition()
        self._in_admission: ContextVar[bool] = ContextVar(
            "gepa_in_admission", default=False
        )
        self.stop_reason: str | None = None
        self._step_usage: ContextVar[_StepUsage | None] = ContextVar(
            "gepa_step_usage", default=None
        )

    def check(self) -> None:
        """Prevent queued work or another tool round after a cost stop."""
        with self._lock:
            if self.max_token_cost is not None and self._total() >= self.max_token_cost:
                self.stop_reason = self.stop_reason or COST_STOP_REASON
            if self.stop_reason is not None:
                raise CostBudgetExceeded(self.stop_reason)

    def record(self, category: SpendCategory, response: ModelResponse) -> None:
        """Record exactly one new response, including an over-budget response."""
        price_error: Exception | None = None
        try:
            dollars = self.price_fn(response) if self.price_fn is not None else None
            if dollars is None:
                dollars = float(response.cost().total_price)
            dollars = float(dollars)
            if not math.isfinite(dollars) or dollars < 0:
                raise ValueError(
                    "price_fn must return finite nonnegative US dollars or None"
                )
        except Exception as error:
            # Pricing failures must not become ordinary failed cases/proposals:
            # retain the paid response and use the graceful cost-stop path.
            price_error = error
            dollars = None
        name = response.model_name or "<unknown>"
        with self._lock:
            usage = self._usage[category].setdefault(name, ModelSpend())
            observation = self._step_usage.get()
            if observation is not None and observation.category == category:
                observation.requests += 1
                if dollars is not None:
                    observation.dollars += dollars
            usage.requests += 1
            usage.input_tokens += response.usage.input_tokens
            usage.output_tokens += response.usage.output_tokens
            if dollars is None:
                usage.unpriced_requests += 1
                usage.unpriced_input_tokens += response.usage.input_tokens
                usage.unpriced_output_tokens += response.usage.output_tokens
                if self.max_token_cost is not None:
                    self.stop_reason = f"Cannot price model {name!r} with a cost budget"
                    if price_error is not None:
                        self.stop_reason += (
                            f": {type(price_error).__name__}: {price_error}"
                        )
            else:
                usage.dollars += dollars
                if (
                    self.max_token_cost is not None
                    and self._total() > self.max_token_cost
                ):
                    self.stop_reason = self.stop_reason or COST_STOP_REASON
            if self.stop_reason is not None:
                raise CostBudgetExceeded(self.stop_reason) from price_error

    def _total(self, category: SpendCategory | None = None) -> float:
        categories = [category] if category else self._usage
        return sum(u.dollars for c in categories for u in self._usage[c].values())

    def can_start(
        self,
        category: SpendCategory,
        count: int = 1,
        *,
        following_rollouts: int = 0,
        rollout_kind: RolloutKind | None = None,
    ) -> bool:
        """Allow the first observation; otherwise require the projected spend to fit.

        ``rollout_kind`` projects a rollout batch at that kind's own mean so an
        expensive validation batch is not underestimated by cheap training
        rollouts; an unobserved kind projects zero, as before. ``following_rollouts``
        are training rollouts and project at the training mean.
        """
        with self._lock:
            if self.stop_reason is not None:
                return False
            observations = self._observations[category]
            projection = (
                self._total(category) / observations * count if observations else 0
            )
            if category == "rollout" and rollout_kind is not None:
                kind_observations = self._kind_observations[rollout_kind]
                projection = (
                    self._kind_dollars[rollout_kind] / kind_observations * count
                    if kind_observations
                    else 0
                )
            training_observations = self._kind_observations["training"]
            if following_rollouts and training_observations:
                projection += (
                    self._kind_dollars["training"]
                    / training_observations
                    * following_rollouts
                )
            if self.max_token_cost is not None and (
                self._total() >= self.max_token_cost
                or self._total() + projection > self.max_token_cost
            ):
                self.stop_reason = COST_STOP_REASON
                return False
            return True

    @contextmanager
    def step(
        self, category: SpendCategory, *, kind: RolloutKind = "training"
    ) -> Iterator[None]:
        """Observe a complete rollout or reflection step, including failures."""
        observation = _StepUsage(category, kind=kind)
        token = self._step_usage.set(observation)
        try:
            yield
        finally:
            self._step_usage.reset(token)
            with self._lock:
                # Concurrent runs must not count another run's responses.
                if observation.requests:
                    self._observations[category] += 1
                    if category == "rollout":
                        # Steps with no response (cache hits, setup failures)
                        # stay free and never dilute a kind's projections.
                        self._kind_observations[observation.kind] += 1
                        self._kind_dollars[observation.kind] += observation.dollars
                        self._kind_highest[observation.kind] = max(
                            self._kind_highest[observation.kind], observation.dollars
                        )

    def _try_admit(self, kind: RolloutKind) -> float | None:
        """Return a rollout's reservation, None to wait, or raise on a cost stop."""
        with self._lock:
            if self.stop_reason is not None:
                raise CostBudgetExceeded(self.stop_reason)
            assert self.max_token_cost is not None and self.max_concurrent is not None
            observations = self._kind_observations[kind]
            highest = self._kind_highest[kind]
            remaining = self.max_token_cost - self._total() - self._reserved
            concurrency = max(1, self.max_concurrent)
            # An unobserved kind, or headroom that cannot cover a full
            # concurrent batch at the observed high, starts one at a time.
            limit = (
                concurrency
                if observations and remaining >= concurrency * highest
                else 1
            )
            if sum(self._active.values()) >= limit:
                return None
            # Reserve at the observed high so in-flight rollouts cannot each
            # spend the same headroom on an underestimated mean.
            projection = highest if observations else 0.0
            if remaining <= 0 or projection > remaining:
                if any(self._active.values()):
                    # In-flight rollouts settle first; a failed or cheap one
                    # can free enough headroom to admit this rollout.
                    return None
                self.stop_reason = COST_STOP_REASON
                raise CostBudgetExceeded(self.stop_reason)
            return projection

    @asynccontextmanager
    async def admit_rollout(self, kind: RolloutKind) -> AsyncIterator[None]:
        """Gate a capped rollout start on the dollars it is projected to cost.

        Uncapped meters (or meters without ``max_concurrent``) admit
        immediately with no waiting or serialization, as before.
        """
        if (
            self.max_token_cost is None
            or self.max_concurrent is None
            or self._in_admission.get()
        ):
            # A nested agent run shares the outer rollout's reservation.
            yield
            return
        reservation: float | None = None
        async with self._condition:
            while reservation is None:
                try:
                    reservation = self._try_admit(kind)
                except CostBudgetExceeded:
                    # Wake waiting rollouts so they observe the stop and exit.
                    self._condition.notify_all()
                    raise
                if reservation is None:
                    await self._condition.wait()
            self._active[kind] += 1
            self._reserved += reservation
            token = self._in_admission.set(True)
        try:
            yield
        finally:
            self._in_admission.reset(token)
            async with self._condition:
                self._active[kind] -= 1
                self._reserved -= reservation
                self._condition.notify_all()

    def report(self) -> SpendReport:
        with self._lock:
            by_model: dict[str, ModelSpend] = {}
            for models in self._usage.values():
                for name, usage in models.items():
                    combined = by_model.setdefault(name, ModelSpend())
                    for key, value in usage.model_dump().items():
                        setattr(combined, key, getattr(combined, key) + value)
            return SpendReport(
                total_dollars=self._total(),
                reflection_dollars=self._total("reflection"),
                rollout_dollars=self._total("rollout"),
                by_category={
                    c: {n: u.model_copy() for n, u in m.items()}
                    for c, m in self._usage.items()
                },
                by_model=by_model,
                unpriced_usage={
                    n: ModelSpend(
                        requests=u.unpriced_requests,
                        input_tokens=u.unpriced_input_tokens,
                        output_tokens=u.unpriced_output_tokens,
                    )
                    for n, u in by_model.items()
                    if u.unpriced_requests
                },
                max_token_cost=self.max_token_cost,
                stopped_by_cost=self.stop_reason is not None,
                stop_reason=self.stop_reason,
            )


@dataclass
class SpendCapability(AbstractCapability[Any]):
    """Meter every response, including tool rounds and failed output retries."""

    meter: SpendMeter
    category: SpendCategory

    @classmethod
    def get_serialization_name(cls) -> None:
        return None

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.meter.check()
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self.meter.record(self.category, response)
        return response

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        self.meter.check()
        if self.category == "rollout":
            kind = _rollout_kind.get()
            async with self.meter.admit_rollout(kind):
                with self.meter.step("rollout", kind=kind):
                    return await handler()
        return await handler()


_active_rollout: ContextVar[SpendCapability | None] = ContextVar(
    "gepa_active_rollout", default=None
)


def current_rollout_capability() -> SpendCapability | None:
    """Capability for agents called by the current CLI evaluate/metric callback.

    Attach this to student and judge agents (including nested agents) to account
    for their responses. Outside a CLI evaluation this returns ``None``.
    """
    return _active_rollout.get()


def report_cached_rollout() -> None:
    """Declare that the current CLI callable served a cached result.

    A declared hit with no model responses costs zero and is excluded from cost
    projections. Any fresh responses are still charged normally. Repeated calls
    in one rollout count once; outside a CLI rollout this has no effect.
    """
    capability = current_rollout_capability()
    if capability is not None and hasattr(capability.meter, "declare_cached_rollout"):
        capability.meter.declare_cached_rollout()


@contextmanager
def rollout_spend(meter: SpendMeter) -> Iterator[None]:
    """Expose an evaluation's meter to caller-owned agents."""
    token = _active_rollout.set(SpendCapability(meter, "rollout"))
    try:
        yield
    finally:
        _active_rollout.reset(token)
