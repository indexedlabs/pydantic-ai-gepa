"""Evaluate step - runs full validation for the latest candidate."""

from __future__ import annotations

from typing import Any, Sequence

import logfire
from pydantic_graph.beta import StepContext

from pydantic_evals import Case
from ..deps import GepaDeps
from ..evaluation import EvaluationResults
from ..models import CandidateProgram, GepaState


async def evaluate_step(ctx: StepContext[GepaState, GepaDeps, None]) -> None:
    """Evaluate the most recent candidate on the validation set."""

    state = ctx.state
    previous_best_idx = state.best_candidate_idx
    previous_best_score = state.best_score
    candidate = _current_candidate(state)
    validation_batch = await _get_validation_batch(state)

    with logfire.span(
        "evaluate candidate",
        candidate_idx=candidate.idx,
        validation_batch_size=len(validation_batch),
    ):
        results = await ctx.deps.evaluator.evaluate_batch(
            candidate=candidate,
            batch=validation_batch,
            adapter=ctx.deps.adapter,
            max_concurrent=state.config.max_concurrent_evaluations,
        )

    # Serialize traces to disk if memory_exporter is available
    if ctx.deps.memory_exporter is not None:
        from pathlib import Path

        spans = ctx.deps.memory_exporter.get_finished_spans()
        if spans:
            traces_dir = Path(
                f".gepa_cache/runs/{state.run_id}/candidates/{candidate.idx}/traces"
            )
            traces_dir.mkdir(parents=True, exist_ok=True)
            traces_file = traces_dir / "traces.jsonl"

            with open(traces_file, "a", encoding="utf-8") as f:
                for span in spans:
                    f.write(span.to_json() + "\\n")

        ctx.deps.memory_exporter.clear()

    state.record_evaluation_errors(
        candidate_idx=candidate.idx,
        stage="validation",
        data_ids=results.data_ids,
        outputs=results.outputs,
    )
    validation_total, validation_avg = _summarize_scores(results.scores)
    logfire.debug(
        "EvaluateStep validation results",
        candidate_idx=candidate.idx,
        validation_total=validation_total,
        validation_average=validation_avg,
        evaluation_count=len(results.scores),
    )

    _apply_results(candidate, results)
    ctx.deps.pareto_manager.update_fronts(state, candidate.idx, results)
    state.recompute_best_candidate()
    new_best_idx = state.best_candidate_idx
    new_best_score = state.best_score
    if new_best_idx != previous_best_idx or new_best_score != previous_best_score:
        logfire.info(
            "EvaluateStep promoted best candidate",
            candidate_idx=new_best_idx,
            previous_best_idx=previous_best_idx,
            previous_best_score=previous_best_score,
            new_best_score=new_best_score,
        )
    else:
        logfire.debug(
            "EvaluateStep best candidate unchanged",
            candidate_idx=candidate.idx,
            best_candidate_idx=new_best_idx,
            best_score=new_best_score,
        )
    state.total_evaluations += len(results.data_ids)
    state.full_validations += 1
    _hydrate_missing_components(candidate, ctx.deps)

    return None


def _current_candidate(state: GepaState) -> CandidateProgram:
    if not state.candidates:
        raise ValueError("EvaluateStep requires at least one candidate in state.")
    return state.candidates[-1]


async def _get_validation_batch(state: GepaState) -> list[Case[Any, Any, Any]]:
    loader = state.validation_set
    if loader is None or len(loader) == 0:
        raise ValueError(
            "GepaState.validation_set must be populated before evaluation."
        )
    ids = list(await loader.all_ids())
    return await loader.fetch(ids)


def _apply_results(
    candidate: CandidateProgram, results: EvaluationResults[str]
) -> None:
    for data_id, score, output in results:
        candidate.record_validation(
            data_id=data_id,
            score=score,
            output=output,
        )


def _summarize_scores(scores: Sequence[float]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    total = float(sum(scores))
    return total, total / len(scores)


def _hydrate_missing_components(
    candidate: CandidateProgram,
    deps: GepaDeps,
) -> None:
    components = deps.adapter.get_components()

    missing = {
        key: value
        for key, value in components.items()
        if key not in candidate.components
    }
    if not missing:
        return

    for name, text in missing.items():
        candidate.components[name] = text.model_copy()

    seed = deps.seed_candidate or {}
    updated_seed = {name: component.model_copy() for name, component in seed.items()}
    for name, value in missing.items():
        updated_seed.setdefault(name, value.model_copy())
    deps.seed_candidate = updated_seed


__all__ = ["evaluate_step"]
