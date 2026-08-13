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

        best = _copy_candidate(await task.seed_candidate())
        train_loader = await task.train_loader()
        all_ids = list(await train_loader.all_ids())
        starting_spend = budget.spent
        engine_budget = BudgetTracker(config.max_metric_calls)
        history: list[EngineEvent] = []
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
                        candidate=best,
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
            if (
                config.stop_at_score is not None
                and baseline_score >= config.stop_at_score
            ):
                stop_reason = "stop_at_score"
                break
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
                candidate=_copy_candidate(best),
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
                "baseline_score": comparison_result.baseline_mean,
                "proposal_score": proposal_score,
                "minibatch_case_ids": [record.case_id for record in baseline_records],
                **comparison_result.to_dict(),
            }
            if comparison_result.improved:
                best = _copy_candidate(proposal)
                history.append(EngineEvent(kind="accepted", data=comparison))
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

        final_score, final_evaluation_charged = await self._score_final_candidate(
            task=task,
            candidate=best,
            budget=budget,
            engine_budget=engine_budget,
            history=history,
        )
        history.append(
            EngineEvent(
                kind="summary",
                data={
                    "iterations": iterations,
                    "proposals": proposals,
                    "stop_reason": stop_reason,
                    "final_evaluation_charged": final_evaluation_charged,
                },
            )
        )
        return EngineResult(
            engine=self.name,
            best_candidate=best,
            best_score=final_score,
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

    async def _score_final_candidate(
        self,
        *,
        task: OptimizationTask,
        candidate: CandidateMap,
        budget: BudgetTracker,
        engine_budget: BudgetTracker,
        history: list[EngineEvent],
    ) -> tuple[float, bool]:
        """Score the winner on the valset and charge that fair comparison.

        Both budgets are reserved before evaluator invocation. If a provider
        raises after starting a rollout, the call remains accounted; a
        preflight failure makes no evaluator call.
        """
        case_count = len(await task._validation_cases())
        if case_count > engine_budget.remaining:
            history.append(
                EngineEvent(
                    kind="budget_overshoot",
                    message="Final valset evaluation exceeded the engine metric-call budget.",
                    data={
                        "stage": "final_valset",
                        "requested": case_count,
                        "budget_remaining": budget.remaining,
                        "engine_budget_remaining": engine_budget.remaining,
                    },
                )
            )
            return 0.0, False
        try:
            budget.preflight(case_count)
        except BudgetExhausted:
            history.append(
                EngineEvent(
                    kind="budget_overshoot",
                    message="Final valset evaluation exceeded the shared metric-call budget.",
                    data={
                        "stage": "final_valset",
                        "requested": case_count,
                        "budget_remaining": budget.remaining,
                        "engine_budget_remaining": engine_budget.remaining,
                    },
                )
            )
            return 0.0, False
        engine_budget.spend(case_count)
        final_evaluation = await task.evaluate(candidate, budget=budget)
        return final_evaluation.score, True

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
