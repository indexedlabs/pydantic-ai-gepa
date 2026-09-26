"""Response-level dollar accounting shared by all model calls in a GEPA run."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
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


def admission_limit(
    max_concurrent: int, observations: int, remaining: float, highest: float
) -> int:
    """In-flight rollout slots a kind may fill under a cap.

    The limit ramps with the kind's observation count: at most
    ``min(max_concurrent, observations)`` and at least 1, so a cheap first
    observation cannot start a full concurrent batch reserved at an
    underestimated cost. Above 1 the limit holds only while the remaining
    headroom covers the whole ramped batch at the kind's highest observed
    cost; otherwise rollouts start one at a time and recheck before each
    start. An unobserved kind always starts one at a time.
    """
    limit = max(1, min(max_concurrent, observations))
    if observations and limit > 1 and remaining < limit * highest:
        return 1
    return limit


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
    every rollout start. Admission keeps spent + in-flight reservations + the
    new reservation within the cap, each reservation at the kind's highest
    observed cost ``h`` at that rollout's admission time, and a kind's
    in-flight slots ramp with its observation count ``n``: at most
    ``min(max_concurrent, n)`` (at least 1), and above 1 only while the
    remaining headroom covers the ramped slots at ``h``. The bound on a
    capped run, stated plainly:

    - After in-flight rollouts settle, overshoot is at most the sum over the
      rollouts in flight at the stop of ``max(0, c_i - h_i)``: each rollout's
      actual cost above the high it was reserved at. With ``M`` the kind's
      true highest rollout cost, that is at most ``min(max_concurrent, n) *
      (M - h)``; a cheap first case (n = 1) ends the run within one rollout
      of the cap. An adversarial order (many cheap cases, then expensive
      ones) can still reach ``max_concurrent * (M - h)``; a declared
      per-rollout cost ceiling would close that and is a tracked follow-up,
      not implemented here.
    - The first rollout of a kind runs alone with no projection and can
      overshoot by its own cost.
    - Reflection overshoot stays bounded by the reflection projection plus
      the response backstop.

    A child meter forwards each priced response and each step observation to
    its parent, and must pass both rollout gates. Parent price overrides take
    precedence; each response is priced once and charged identically to both
    ledgers. A local cap stops only that child. The parent's report is aggregate
    across all children and direct evaluations. Composed reflections run one
    at a time, without competing rollout reservations, to retain the reflection
    projection/response-backstop caveat above.

    Near the cap the gate is conservative: with nothing in flight it stops
    once the kind's highest observed rollout no longer fits, so one outlier
    rollout can end a run with headroom left.

    ``reserve`` withholds a live, callable-computed dollar amount from this
    meter's own local cap so work running *outside* this meter (a composed
    helper's follow-up fair comparison) stays fundable while this meter's
    work is still in progress. The reserve is re-evaluated on every check,
    so it tracks the parent's observed rollout costs as they arrive, and
    projects at the per-rollout admission bound (see ``rollout_projection``).
    It only ever stops this child earlier; the run's cap and its documented
    margin are unchanged.
    """

    def __init__(
        self,
        max_token_cost: float | None = None,
        price_fn: PriceFn | None = None,
        max_concurrent: int | None = None,
        *,
        parent: SpendMeter | None = None,
        reserve: Callable[[], float] | None = None,
    ) -> None:
        if max_token_cost is not None and (
            not math.isfinite(max_token_cost) or max_token_cost <= 0
        ):
            raise ValueError("max_token_cost must be finite and > 0")
        if max_concurrent is not None and max_concurrent <= 0:
            raise ValueError("max_concurrent must be > 0")
        if reserve is not None and max_token_cost is None:
            raise ValueError("reserve requires a local max_token_cost")
        self.parent = parent
        self.max_token_cost = max_token_cost
        self.price_fn = price_fn
        self.max_concurrent = max_concurrent
        self._reserve = reserve
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
        self._reflection_active = False
        self._condition = asyncio.Condition()
        self._in_admission: ContextVar[bool] = ContextVar(
            "gepa_in_admission", default=False
        )
        self.stop_reason: str | None = None
        self._step_usage: ContextVar[_StepUsage | None] = ContextVar(
            "gepa_step_usage", default=None
        )

    def _local_cap(self) -> float | None:
        """This meter's own cap net of its live reserve; None when uncapped."""
        if self.max_token_cost is None:
            return None
        reserve = self._reserve() if self._reserve is not None else 0.0
        return self.max_token_cost - max(0.0, reserve)

    def current_spend(self) -> float:
        """This meter's total metered dollars so far (aggregate, no case data)."""
        with self._lock:
            return self._total()

    def rollout_projection(self, count: int, *, kind: RolloutKind) -> float:
        """Project ``count`` rollouts of ``kind`` against this meter's history.

        Every rollout projects at the kind's highest observed cost: that is
        the bound ``_try_admit`` enforces, since each admission's remaining
        headroom must cover the observed high when the rollout runs alone
        near the cap. Projecting cheaper rollouts at the running mean would
        let a comparison start that admission later refuses mid-round. An
        unobserved kind projects zero, matching the first-observation rule in
        ``can_start``.
        """
        with self._lock:
            return self._projection_locked(count, kind)

    def _projection_locked(self, count: int, kind: RolloutKind) -> float:
        observations = self._kind_observations[kind]
        if count <= 0 or not observations:
            return 0.0
        return count * self._kind_highest[kind]

    def probe_rollouts(self, count: int, *, kind: RolloutKind = "validation") -> bool:
        """Probe, with no stop side effect, whether ``count`` rollouts fit.

        Each meter in the ancestry checks the projection from its own
        observations against its remaining capped headroom. Unlike
        ``can_start`` a refusal never sets a stop reason: a helper whose
        comparison is refused ends itself, while the pipeline meter and any
        siblings keep their headroom. Genuine exhaustion still stops a meter
        through ``check``/``record``, never through this probe.

        The probe reserves nothing. Work running concurrently under the same
        capped ancestor (two helpers sharing one supplied meter) can pass it
        for the same dollars; the cap still holds, and a comparison that is
        then interrupted is discarded whole.
        """
        with self._lock:
            if self.stop_reason is not None:
                return False
            cap = self._local_cap()
            if (
                cap is not None
                and self._total() + self._projection_locked(count, kind) > cap
            ):
                return False
        return self.parent.probe_rollouts(count, kind=kind) if self.parent else True

    def check(self) -> None:
        """Prevent queued work or another tool round after a cost stop."""
        if self.parent is not None:
            self.parent.check()
        with self._lock:
            cap = self._local_cap()
            if cap is not None and self._total() >= cap:
                self.stop_reason = self.stop_reason or COST_STOP_REASON
            if self.stop_reason is not None:
                raise CostBudgetExceeded(self.stop_reason)

    def record(self, category: SpendCategory, response: ModelResponse) -> None:
        """Record exactly one new response, including an over-budget response."""
        price_error: Exception | None = None
        try:
            price_fn = self.price_fn
            ancestor = self.parent
            while ancestor is not None:
                if ancestor.price_fn is not None:
                    price_fn = ancestor.price_fn
                ancestor = ancestor.parent
            dollars = price_fn(response) if price_fn is not None else None
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
        self._record_priced(category, response, dollars, price_error)

    def _record_priced(
        self,
        category: SpendCategory,
        response: ModelResponse,
        dollars: float | None,
        price_error: Exception | None,
    ) -> None:
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
                cap = self._local_cap()
                if cap is not None and self._total() > cap:
                    self.stop_reason = self.stop_reason or COST_STOP_REASON
            # Forward even a response that exceeded the local cap. Pricing is
            # resolved once so the engine and pipeline ledgers agree exactly.
            if self.parent is not None:
                self.parent._record_priced(category, response, dollars, price_error)
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
        are training rollouts and project at the training mean. A projection
        refusal stops the requesting meter only; ancestor projections are
        probes so siblings may still fit. Actual exhaustion stops the ancestor.
        """
        return self._can_start(
            category,
            count,
            following_rollouts=following_rollouts,
            rollout_kind=rollout_kind,
            stop_on_projection=True,
        )

    def _can_start(
        self,
        category: SpendCategory,
        count: int,
        *,
        following_rollouts: int,
        rollout_kind: RolloutKind | None,
        stop_on_projection: bool,
    ) -> bool:
        """Check ancestors without turning a child's projection into a global stop."""
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
            cap = self._local_cap()
            if cap is not None:
                if self._total() >= cap:
                    self.stop_reason = COST_STOP_REASON
                    return False
                if self._total() + projection > cap:
                    if stop_on_projection:
                        self.stop_reason = COST_STOP_REASON
                    return False
            if self.parent is not None and not self.parent._can_start(
                category,
                count,
                following_rollouts=following_rollouts,
                rollout_kind=rollout_kind,
                stop_on_projection=False,
            ):
                if stop_on_projection:
                    self.stop_reason = self.parent.stop_reason or COST_STOP_REASON
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
            with (
                self.parent.step(category, kind=kind) if self.parent else nullcontext()
            ):
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
            if self._reflection_active:
                return None
            observations = self._kind_observations[kind]
            highest = self._kind_highest[kind]
            cap = self._local_cap()
            assert cap is not None
            remaining = cap - self._total() - self._reserved
            concurrency = max(1, self.max_concurrent)
            # In-flight slots ramp with the kind's observation count, so a
            # cheap first rollout cannot start a full batch reserved at an
            # underestimated high.
            limit = admission_limit(concurrency, observations, remaining, highest)
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
        if self.parent is not None:
            async with self.parent.admit_rollout(kind):
                async with self._admit_local_rollout(kind):
                    yield
        else:
            async with self._admit_local_rollout(kind):
                yield

    @asynccontextmanager
    async def _admit_local_rollout(self, kind: RolloutKind) -> AsyncIterator[None]:
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
            # Release the slot synchronously: a second cancellation queued on
            # the condition lock must not leak the reservation.
            with self._lock:
                self._active[kind] -= 1
                self._reserved -= reservation
            # If this notify acquire is itself cancelled, the run is already
            # being torn down; the slot above is released either way.
            async with self._condition:
                self._condition.notify_all()

    @asynccontextmanager
    async def admit_reflection(self) -> AsyncIterator[None]:
        """Keep composed reflection runs from racing reserved rollout dollars.

        Only child meters use this gate. Uncapped composition never waits.
        A nested reflection agent shares its outer agent's reservation.
        Under a pipeline cap this is a barrier: reflections run only between
        rollouts, after all in-flight rollouts finish. Waiting reflections do
        not have priority over new rollout admissions.
        """
        root: SpendMeter | None = None
        ancestor = self.parent
        while ancestor is not None:
            if ancestor.max_token_cost is not None:
                root = ancestor
            ancestor = ancestor.parent
        if root is None or root._in_admission.get():
            yield
            return
        async with root._condition:
            while root._reflection_active or any(root._active.values()):
                root.check()
                await root._condition.wait()
            root.check()
            if not self.can_start("reflection"):
                raise CostBudgetExceeded(self.stop_reason or COST_STOP_REASON)
            root._reflection_active = True
            token = root._in_admission.set(True)
        try:
            yield
        finally:
            root._in_admission.reset(token)
            root._reflection_active = False
            async with root._condition:
                root._condition.notify_all()

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
                stopped_by_cost=(
                    self.stop_reason is not None
                    or (
                        self.parent is not None and self.parent.report().stopped_by_cost
                    )
                ),
                stop_reason=self.stop_reason
                or (self.parent.report().stop_reason if self.parent else None),
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
        async with self.meter.admit_reflection():
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


# Internal harness context: no meter accessor is added to engine task views.
_pipeline_meter: ContextVar[SpendMeter | None] = ContextVar(
    "gepa_pipeline_meter", default=None
)
