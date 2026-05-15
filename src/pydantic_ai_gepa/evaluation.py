"""Helpers for evaluating agents/candidates on datasets outside GEPA."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import logfire
from pydantic_ai import UsageLimits
from pydantic_ai.agent import AbstractAgent
from pydantic import BaseModel
from pydantic_evals import Case, Dataset

from .adapters.agent_adapter import CaseFactory, create_adapter
from .gepa_graph.models import CandidateMap, candidate_texts
from .input_type import InputSpec


@dataclass(slots=True)
class EvaluationRecord:
    """Structured result for a single dataset evaluation."""

    case_id: str
    score: float
    feedback: str | None
    payload: Mapping[str, Any]


async def evaluate_candidate_dataset(
    *,
    agent: AbstractAgent[Any, Any],
    metric,
    dataset: Sequence[Case[Any, Any, Any]] | Dataset[Any, Any],
    candidate: CandidateMap | None = None,
    concurrency: int = 20,
    agent_usage_limits: UsageLimits | None = None,
    capture_traces: bool = False,
    input_type: InputSpec[BaseModel] | None = None,
    case_factory: CaseFactory | None = None,
) -> list[EvaluationRecord]:
    """Evaluate an agent/candidate pair on a dataset in parallel.

    ``case_factory`` (when set) converts each ``Case`` into the agent's
    input model at rollout time. Use it when dataset rows carry deferred
    references — file paths, Mighty file ids, base64 blobs — that need
    to be materialized into ``BinaryContent`` (or similar) before the
    agent runs. Only honored for ``SignatureAgent`` agents.
    """

    semaphore = asyncio.Semaphore(max(1, concurrency))
    records: list[EvaluationRecord] = []

    cases: Sequence[Case[Any, Any, Any]]
    dataset_name: str | None = None
    if isinstance(dataset, Dataset):
        cases = dataset.cases
        dataset_name = dataset.name
    else:
        cases = dataset

    total_cases = len(cases)
    adapter = create_adapter(
        agent=agent,
        metric=metric,
        input_type=input_type,
        cache_manager=None,
        agent_usage_limits=agent_usage_limits,
        case_factory=case_factory,
    )

    candidate_map: CandidateMap = candidate.copy() if candidate is not None else {}
    candidate_text_map = candidate_texts(candidate_map)

    task_name = getattr(agent, "name", None) or agent.__class__.__name__
    extra_attributes: dict[str, Any] = {"gen_ai.operation.name": "experiment"}

    async def run_case(index: int, case: Case[Any, Any, Any]) -> None:
        async with semaphore:
            result = await adapter.process_case(
                case,
                index,
                capture_traces=capture_traces,
                candidate=candidate_map,
            )
            case_id = case.name or f"case-{index}"
            records.append(
                EvaluationRecord(
                    case_id=case_id,
                    score=float(result.get("score", 0.0)),
                    feedback=result.get("feedback"),
                    payload=result,
                )
            )

    with logfire.span(
        f"evaluate {task_name}",
        name=task_name,
        task_name=task_name,
        dataset_name=dataset_name,
        n_cases=total_cases,
        candidate_components=len(candidate_map),
        **extra_attributes,
    ) as eval_span:
        with adapter.apply_candidate(candidate_map):
            await asyncio.gather(
                *(run_case(idx, case) for idx, case in enumerate(cases))
            )

        experiment_metadata: dict[str, Any] = {"n_cases": total_cases}
        if dataset_name:
            experiment_metadata["dataset_name"] = dataset_name
        if candidate_text_map:
            experiment_metadata["candidate_keys"] = sorted(candidate_text_map)

        if records:
            average_score = sum(record.score for record in records) / len(records)
            experiment_metadata["average_score"] = average_score
            experiment_metadata["averages"] = {
                "assertions": average_score,
                "scores": {"metric_score": average_score},
                "labels": {},
                "metrics": {"metric_score": average_score},
            }
            eval_span.set_attribute("assertion_pass_rate", average_score)

        eval_span.set_attribute("logfire.experiment.metadata", experiment_metadata)

    return records


__all__ = [
    "EvaluationRecord",
    "evaluate_candidate_dataset",
]
