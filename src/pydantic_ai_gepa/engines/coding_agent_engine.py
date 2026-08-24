"""An in-process managed reflection loop driven by a caller-supplied proposer."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from pydantic_evals import Case

from ..acceptance import compare_candidate_samples
from ..evaluation import EvaluationRecord, evaluate_candidate_dataset
from ..gepa_graph.models import CandidateMap
from .base import (
    BudgetExhausted,
    BudgetTracker,
    CandidateEvaluation,
    EngineConfig,
    EngineEvent,
    EngineResult,
    OptimizationTask,
    _aggregate_side_info,
)
from .registry import register_engine


@dataclass(frozen=True, slots=True)
class ReflectionContext:
    """Evidence supplied to a coding agent before it proposes a revision."""

    candidate: CandidateMap
    minibatch_records: list[EvaluationRecord]
    report: str
    iteration: int
    side_info: dict[str, Any]


Proposer: TypeAlias = Callable[[ReflectionContext], Awaitable[CandidateMap]]


@dataclass(frozen=True, slots=True)
class _CandidatePoolEntry:
    """One validation-scored candidate available for Pareto parent selection."""

    candidate: CandidateMap
    evaluation: CandidateEvaluation
    index: int


class CodingAgentEngine:
    """Optimize with a caller-supplied coding agent under a shared budget.

    The library owns minibatch sampling, evaluation, and monotonic candidate
    selection.  The caller's ``propose`` callback only receives failure
    evidence and returns a candidate to test on that same minibatch.
    """

    name = "coding_agent"

    def __init__(self, config: EngineConfig) -> None:
        """Read the required asynchronous proposal callback from configuration."""
        propose = config.engine_config.get("propose")
        if not callable(propose):
            raise TypeError(
                "engine_config['propose'] must be a callable async callback that "
                "accepts ReflectionContext and returns a CandidateMap."
            )
        self._propose = cast(Proposer, propose)
        self._engine_config = config.engine_config

    async def run(
        self,
        task: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        """Run managed baseline/reflection rounds and return the best candidate."""
        budget.check()

        minibatch_size = self._positive_int_option("minibatch_size", 5)
        concurrency = self._positive_int_option("concurrency", 5)
        max_proposals = self._positive_int_option("max_proposals_per_run", 10)
        failure_threshold = float(self._option("failure_threshold", 0.999))
        acceptance_repetitions = self._positive_int_option("acceptance_repetitions", 1)
        acceptance_max_repetitions = self._positive_int_option(
            "acceptance_max_repetitions", acceptance_repetitions
        )
        if acceptance_max_repetitions < acceptance_repetitions:
            raise ValueError(
                "engine_config['acceptance_max_repetitions'] must be greater "
                "than or equal to engine_config['acceptance_repetitions']."
            )
        acceptance_confidence = self._confidence_option("acceptance_confidence", 0.9)
        acceptance_min_delta = self._nonnegative_float_option(
            "acceptance_min_delta", 0.0
        )

        starting_spend = budget.spent
        engine_budget = BudgetTracker(config.max_metric_calls)
        history: list[EngineEvent] = []
        seed = _copy_candidate(await task.seed_candidate())
        seed_evaluation = await self._evaluate_validation(
            task=task,
            candidate=seed,
            budget=budget,
            engine_budget=engine_budget,
            history=history,
            stage="seed_validation",
        )
        if seed_evaluation is None:
            history.append(
                EngineEvent(
                    kind="summary",
                    data={
                        "iterations": 0,
                        "proposals": 0,
                        "stop_reason": "budget_exhausted",
                        "validation_evaluations": 0,
                    },
                )
            )
            return EngineResult(
                engine=self.name,
                best_candidate=seed,
                best_score=0.0,
                num_metric_calls=budget.spent - starting_spend,
                history=history,
            )

        pool = [_CandidatePoolEntry(seed, seed_evaluation, 0)]
        best_entry = pool[0]
        history.append(
            EngineEvent(
                kind="validation_evaluated",
                data=_validation_event_data(best_entry, stage="seed"),
            )
        )
        train_loader = await task.train_loader()
        all_ids = list(await train_loader.all_ids())
        epoch = 0
        iterations = 0
        proposals = 0
        stop_reason = "completed"

        while (
            (config.max_iterations is None or iterations < config.max_iterations)
            and not budget.exhausted
            and not engine_budget.exhausted
            and proposals < max_proposals
        ):
            if (
                config.stop_at_score is not None
                and best_entry.evaluation.score >= config.stop_at_score
            ):
                stop_reason = "stop_at_score"
                break
            parent = _select_pareto_parent(pool, seed=config.seed, epoch=epoch)
            minibatch_ids = _sample_minibatch(
                all_ids,
                size=minibatch_size,
                seed=config.seed,
                epoch=epoch,
            )
            epoch += 1
            minibatch = await train_loader.fetch(minibatch_ids)
            # A baseline is an indivisible provider-facing operation.  Do not
            # let `_affordable_repetitions` turn an unaffordable first batch
            # into one attempted evaluation: `_evaluate_minibatch` reserves
            # before it invokes the provider, so stopping here is both clean
            # and honest.
            if (
                len(minibatch) > budget.remaining
                or len(minibatch) > engine_budget.remaining
            ):
                history.append(
                    EngineEvent(
                        kind="budget_exhausted",
                        message="No complete baseline minibatch is affordable.",
                        data={
                            "stage": "baseline_minibatch",
                            "requested": len(minibatch),
                            "budget_remaining": budget.remaining,
                            "engine_budget_remaining": engine_budget.remaining,
                        },
                    )
                )
                stop_reason = "budget_exhausted"
                break
            effective_max_repetitions = _affordable_repetitions(
                requested=acceptance_max_repetitions,
                case_count=len(minibatch),
                shared_remaining=budget.remaining,
                engine_remaining=engine_budget.remaining,
            )
            baseline_batches: list[list[EvaluationRecord]] = []
            baseline_budget_exhausted = False
            for _ in range(effective_max_repetitions):
                try:
                    baseline_records = await self._evaluate_minibatch(
                        task=task,
                        candidate=parent.candidate,
                        minibatch=minibatch,
                        concurrency=concurrency,
                        budget=budget,
                        engine_budget=engine_budget,
                    )
                except BudgetExhausted:
                    baseline_budget_exhausted = True
                    stop_reason = "budget_exhausted"
                    break
                if not self._spend_or_record_overshoot(
                    budget=budget,
                    engine_budget=engine_budget,
                    history=history,
                    stage="baseline_minibatch",
                    records=baseline_records,
                ):
                    baseline_budget_exhausted = True
                    break
                baseline_batches.append(baseline_records)
            iterations += 1
            if baseline_budget_exhausted or not baseline_batches:
                stop_reason = "budget_overshoot"
                break

            baseline_records = baseline_batches[0]
            baseline_samples = [_mean_score(records) for records in baseline_batches]
            baseline_score = sum(baseline_samples) / len(baseline_samples)
            if budget.exhausted or engine_budget.exhausted:
                stop_reason = "budget_exhausted"
                break

            failures = [
                record
                for record in baseline_records
                if record.score < failure_threshold
            ]
            if not failures:
                history.append(
                    EngineEvent(
                        kind="clean_minibatch",
                        data={
                            "iteration": iterations,
                            "mean_score": baseline_score,
                            "minibatch_case_ids": [
                                record.case_id for record in baseline_records
                            ],
                        },
                    )
                )
                continue

            context = ReflectionContext(
                candidate=_copy_candidate(parent.candidate),
                minibatch_records=failures,
                report=_format_failure_report(baseline_records, failure_threshold),
                iteration=iterations,
                side_info=_aggregate_side_info(baseline_records),
            )
            proposal = await self._propose(context)
            proposals += 1

            proposal_samples: list[float] = []
            comparison_result = None
            initial_repetitions = min(acceptance_repetitions, len(baseline_samples))
            for _ in range(len(baseline_samples)):
                try:
                    proposal_records = await self._evaluate_minibatch(
                        task=task,
                        candidate=proposal,
                        minibatch=minibatch,
                        concurrency=concurrency,
                        budget=budget,
                        engine_budget=engine_budget,
                    )
                except BudgetExhausted:
                    stop_reason = "budget_exhausted"
                    break
                except Exception as exc:
                    raise RuntimeError(
                        "Coding-agent proposal evaluation failed."
                    ) from exc

                if not self._spend_or_record_overshoot(
                    budget=budget,
                    engine_budget=engine_budget,
                    history=history,
                    stage="proposal_minibatch",
                    records=proposal_records,
                ):
                    stop_reason = "budget_overshoot"
                    break
                proposal_samples.append(_mean_score(proposal_records))
                if len(proposal_samples) < initial_repetitions:
                    continue
                comparison_result = compare_candidate_samples(
                    baseline_samples[: len(proposal_samples)],
                    proposal_samples,
                    confidence=acceptance_confidence,
                    min_delta=acceptance_min_delta,
                )
                if comparison_result.verdict != "inconclusive":
                    break

            if comparison_result is None:
                if stop_reason in {"budget_overshoot", "budget_exhausted"}:
                    break
                raise RuntimeError(
                    "Coding-agent comparison did not collect enough proposal samples."
                )

            proposal_score = comparison_result.candidate_mean
            comparison = {
                "iteration": iterations,
                "parent_index": parent.index,
                "baseline_score": comparison_result.baseline_mean,
                "proposal_score": proposal_score,
                "minibatch_case_ids": [record.case_id for record in baseline_records],
                **comparison_result.to_dict(),
            }
            if comparison_result.improved:
                history.append(EngineEvent(kind="minibatch_improved", data=comparison))
                validation = await self._evaluate_validation(
                    task=task,
                    candidate=proposal,
                    budget=budget,
                    engine_budget=engine_budget,
                    history=history,
                    stage="proposal_validation",
                )
                if validation is None:
                    stop_reason = "budget_exhausted"
                    break
                candidate_entry = _CandidatePoolEntry(
                    _copy_candidate(proposal), validation, len(pool)
                )
                pool.append(candidate_entry)
                validation_data = {
                    **comparison,
                    **_validation_event_data(candidate_entry, stage="proposal"),
                }
                if validation.selectable and candidate_entry.index in _pareto_indices(
                    pool
                ):
                    history.append(EngineEvent(kind="accepted", data=validation_data))
                else:
                    history.append(
                        EngineEvent(kind="validation_rejected", data=validation_data)
                    )
                best_entry = _best_validation_entry(pool)
            else:
                history.append(
                    EngineEvent(kind=comparison_result.verdict, data=comparison)
                )

        if (budget.exhausted or engine_budget.exhausted) and stop_reason == "completed":
            stop_reason = "budget_exhausted"
        elif proposals >= max_proposals and stop_reason == "completed":
            stop_reason = "max_proposals_per_run"
        elif (
            config.max_iterations is not None
            and iterations >= config.max_iterations
            and stop_reason == "completed"
        ):
            stop_reason = "max_iterations"

        history.append(
            EngineEvent(
                kind="summary",
                data={
                    "iterations": iterations,
                    "proposals": proposals,
                    "stop_reason": stop_reason,
                    "validation_evaluations": len(pool),
                    "pareto_candidates": sorted(_pareto_indices(pool)),
                },
            )
        )
        return EngineResult(
            engine=self.name,
            best_candidate=_copy_candidate(best_entry.candidate),
            best_score=best_entry.evaluation.score,
            num_metric_calls=budget.spent - starting_spend,
            history=history,
        )

    async def _evaluate_minibatch(
        self,
        *,
        task: OptimizationTask,
        candidate: CandidateMap,
        minibatch: Sequence[Case[Any, Any, Any]],
        concurrency: int,
        budget: BudgetTracker,
        engine_budget: BudgetTracker,
    ) -> list[EvaluationRecord]:
        """Evaluate one candidate on the explicitly selected minibatch."""
        budget.preflight(len(minibatch))
        engine_budget.preflight(len(minibatch))
        # Reserve before the provider invocation. A rollout that raises after
        # contacting a provider remains accounted; only never-started calls
        # are rejected by preflight.
        budget.spend(len(minibatch))
        engine_budget.spend(len(minibatch))
        return await evaluate_candidate_dataset(
            agent=task.agent,
            metric=task.metric,
            dataset=minibatch,
            candidate=candidate,
            concurrency=concurrency,
            input_type=task.input_type,
            case_factory=task.case_factory,
            skills_fs=task.skills_fs,
            skills_capabilities=task.skills_capabilities,
            capture_traces=True,
        )

    async def _evaluate_validation(
        self,
        *,
        task: OptimizationTask,
        candidate: CandidateMap,
        budget: BudgetTracker,
        engine_budget: BudgetTracker,
        history: list[EngineEvent],
        stage: str,
    ) -> CandidateEvaluation | None:
        """Score one candidate on validation without exposing feedback to reflection.

        Both budgets are reserved before evaluator invocation. If a provider
        raises after starting a rollout, the call remains accounted; a
        preflight failure makes no evaluator call.
        """
        case_count = len(await task._validation_cases())
        if case_count > engine_budget.remaining:
            history.append(
                EngineEvent(
                    kind="budget_exhausted",
                    message="Validation evaluation exceeded the engine metric-call budget.",
                    data={
                        "stage": stage,
                        "requested": case_count,
                        "budget_remaining": budget.remaining,
                        "engine_budget_remaining": engine_budget.remaining,
                    },
                )
            )
            return None
        try:
            budget.preflight(case_count)
        except BudgetExhausted:
            history.append(
                EngineEvent(
                    kind="budget_exhausted",
                    message="Validation evaluation exceeded the shared metric-call budget.",
                    data={
                        "stage": stage,
                        "requested": case_count,
                        "budget_remaining": budget.remaining,
                        "engine_budget_remaining": engine_budget.remaining,
                    },
                )
            )
            return None
        engine_budget.spend(case_count)
        return await task.evaluate(candidate, budget=budget, capture_traces=False)

    def _spend_or_record_overshoot(
        self,
        *,
        budget: BudgetTracker,
        engine_budget: BudgetTracker,
        history: list[EngineEvent],
        stage: str,
        records: Sequence[EvaluationRecord],
    ) -> bool:
        """Charge an evaluation or record the budget overshoot that stops it."""
        # `_evaluate_minibatch` reserved both budgets before invoking the
        # evaluator. Keep this helper for the existing history call sites.
        return True

    def _option(self, name: str, default: Any) -> Any:
        """Return a coding-agent option from the engine-specific configuration."""
        return self._engine_config.get(name, default)

    def _positive_int_option(self, name: str, default: int) -> int:
        """Read an engine count option while rejecting invalid runtime settings."""
        value = self._option(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"engine_config[{name!r}] must be a positive integer.")
        return value

    def _confidence_option(self, name: str, default: float) -> float:
        """Read a confidence level strictly between zero and one."""

        value = float(self._option(name, default))
        if not 0.0 < value < 1.0:
            raise ValueError(f"engine_config[{name!r}] must be between 0 and 1.")
        return value

    def _nonnegative_float_option(self, name: str, default: float) -> float:
        """Read a non-negative numeric engine option."""

        value = float(self._option(name, default))
        if value < 0.0:
            raise ValueError(
                f"engine_config[{name!r}] must be greater than or equal to zero."
            )
        return value


def _sample_minibatch(
    all_ids: Sequence[Any],
    *,
    size: int,
    seed: int,
    epoch: int,
) -> list[Any]:
    """Return a deterministic, without-replacement minibatch for one epoch."""
    return random.Random(seed + epoch).sample(list(all_ids), k=min(size, len(all_ids)))


def _affordable_repetitions(
    *,
    requested: int,
    case_count: int,
    shared_remaining: int,
    engine_remaining: int,
) -> int:
    """Reserve equal baseline/candidate samples when repeated evals are affordable."""

    if case_count < 1:
        return 1
    available = min(shared_remaining, engine_remaining)
    paired_repetitions = available // (2 * case_count)
    return min(requested, max(1, paired_repetitions))


def _mean_score(records: Sequence[EvaluationRecord]) -> float:
    """Return the mean score for an evaluated minibatch."""
    return sum(record.score for record in records) / len(records) if records else 0.0


def _copy_candidate(candidate: CandidateMap) -> CandidateMap:
    """Isolate retained candidates from proposer-side mutation."""
    return {
        name: component.model_copy(deep=True) for name, component in candidate.items()
    }


def _pareto_indices(pool: Sequence[_CandidatePoolEntry]) -> set[int]:
    """Return candidates that are not dominated on complete validation scores."""

    selectable = [entry for entry in pool if entry.evaluation.selectable]
    front: set[int] = set()
    for candidate in selectable:
        scores = candidate.evaluation.per_case_scores
        if not scores:
            continue
        dominated = False
        for other in selectable:
            if other.index == candidate.index:
                continue
            other_scores = other.evaluation.per_case_scores
            if set(other_scores) != set(scores):
                continue
            if all(other_scores[key] >= score for key, score in scores.items()) and any(
                other_scores[key] > score for key, score in scores.items()
            ):
                dominated = True
                break
        if not dominated:
            front.add(candidate.index)
    return front


def _select_pareto_parent(
    pool: Sequence[_CandidatePoolEntry], *, seed: int, epoch: int
) -> _CandidatePoolEntry:
    """Select a validation-Pareto parent weighted by per-instance wins."""

    front_indices = _pareto_indices(pool)
    front = [entry for entry in pool if entry.index in front_indices]
    if not front:
        return _best_validation_entry(pool)
    shared_case_ids = set(front[0].evaluation.per_case_scores)
    for entry in front[1:]:
        shared_case_ids.intersection_update(entry.evaluation.per_case_scores)
    case_ids = sorted(shared_case_ids)
    weights = {entry.index: 0 for entry in front}
    for case_id in case_ids:
        best_score = max(entry.evaluation.per_case_scores[case_id] for entry in front)
        for entry in front:
            if entry.evaluation.per_case_scores[case_id] == best_score:
                weights[entry.index] += 1
    total = sum(weights.values())
    if total == 0:
        return _best_validation_entry(front)
    choice = random.Random(seed + epoch).randrange(total)
    cumulative = 0
    for entry in sorted(front, key=lambda item: item.index):
        cumulative += weights[entry.index]
        if choice < cumulative:
            return entry
    return front[-1]


def _best_validation_entry(
    pool: Sequence[_CandidatePoolEntry],
) -> _CandidatePoolEntry:
    """Return the highest aggregate selectable validation candidate."""

    selectable = [entry for entry in pool if entry.evaluation.selectable]
    candidates = selectable or list(pool)
    return max(candidates, key=lambda entry: (entry.evaluation.score, -entry.index))


def _validation_event_data(entry: _CandidatePoolEntry, *, stage: str) -> dict[str, Any]:
    """Expose selection scores without leaking validation feedback or outputs."""

    return {
        "stage": stage,
        "candidate_index": entry.index,
        "validation_score": entry.evaluation.score,
        "validation_case_scores": dict(entry.evaluation.per_case_scores),
        "selectable": entry.evaluation.selectable,
    }


def _format_failure_report(
    records: Sequence[EvaluationRecord], threshold: float
) -> str:
    """Format failed records in the managed CLI's markdown report style."""
    lines = ["# Eval report", ""]
    failures = [record for record in records if record.score < threshold]
    if not failures:
        lines.append("Every case in this minibatch passed; nothing to act on.")
        return "\n".join(lines)

    lines.append(
        f"{len(failures)} of {len(records)} case(s) underperformed "
        f"(score < {threshold}). Review per-case feedback and revise the candidate.\n"
    )
    for record in failures:
        lines.append(f"## {record.case_id} — score {record.score:.3f}")
        if record.feedback:
            lines.extend(["", record.feedback.rstrip()])
        lines.append("")
    return "\n".join(lines)


register_engine(CodingAgentEngine.name, CodingAgentEngine, replace=True)


__all__ = ["CodingAgentEngine", "Proposer", "ReflectionContext"]
