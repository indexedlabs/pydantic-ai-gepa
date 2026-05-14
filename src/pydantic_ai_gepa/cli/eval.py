"""`gepa eval` — evaluate a candidate JSON against the configured dataset."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from ..evaluation import evaluate_candidate_dataset
from ._io import write_content_file
from .candidates import Candidate
from .dataset import case_ids as dataset_case_ids
from .dataset import cases_by_id, load_dataset
from .layout import (
    GepaConfig,
    config_path,
    insert_repo_root_on_path,
    latest_run_id,
    new_run_id,
    repo_root,
    resolve_agent,
    resolve_metric,
)
from .metrics import default_substring_metric
from .runs import (
    MinibatchStore,
    ParetoLog,
    ParetoRow,
    current_commit_sha,
    utc_now_iso,
)


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = latest_run_id()
    if latest:
        return latest
    return new_run_id()


def eval_(
    candidate_file: Path = typer.Option(
        ..., "--candidate-file", help="Path to a candidate JSON file."
    ),
    minibatch_id: str | None = typer.Option(None, "--minibatch-id"),
    size: int = typer.Option(10, "--size"),
    seed: int = typer.Option(0, "--seed"),
    epoch: int = typer.Option(0, "--epoch"),
    run_id: str | None = typer.Option(None, "--run-id"),
    output_file: Path | None = typer.Option(
        None, "--output-file", help="Write JSONL results to a file (or - for stdout)."
    ),
    concurrency: int = typer.Option(5, "--concurrency"),
) -> None:
    """Evaluate a candidate JSON file against the configured dataset (or a minibatch)."""
    cfg = GepaConfig.load(config_path())
    insert_repo_root_on_path()

    agent = resolve_agent(cfg)
    metric = resolve_metric(cfg) or default_substring_metric

    dataset_path = repo_root() / cfg.dataset
    cases = load_dataset(dataset_path)
    if not cases:
        typer.echo(f"Dataset {dataset_path} is empty.", err=True)
        raise typer.Exit(code=1)

    candidate = Candidate.load(candidate_file)
    candidate_map = candidate.to_candidate_map()

    active_run_id = _resolve_run_id(run_id)
    minibatch_store = MinibatchStore(active_run_id)

    if minibatch_id:
        minibatch = minibatch_store.load(minibatch_id)
    else:
        minibatch = minibatch_store.sample(
            dataset_case_ids(cases), size=size, seed=seed, epoch=epoch
        )

    by_id = cases_by_id(cases)
    missing = [cid for cid in minibatch.case_ids if cid not in by_id]
    if missing:
        typer.echo(
            f"Minibatch references cases not present in dataset: {missing}",
            err=True,
        )
        raise typer.Exit(code=1)
    subset = [by_id[cid] for cid in minibatch.case_ids]

    records = asyncio.run(
        evaluate_candidate_dataset(
            agent=agent,
            metric=metric,
            dataset=subset,
            candidate=candidate_map,
            concurrency=concurrency,
        )
    )

    per_case = {record.case_id: record.score for record in records}
    mean = sum(per_case.values()) / len(per_case) if per_case else 0.0

    pareto = ParetoLog(active_run_id)
    pareto.append(
        ParetoRow(
            candidate_id=candidate.id,
            commit_sha=current_commit_sha(),
            component_overrides_id=str(candidate_file),
            minibatch_id=minibatch.id,
            per_case_scores=per_case,
            mean_score=mean,
            status="evaluated",
            summary=f"eval candidate {candidate.id} on minibatch {minibatch.id}",
            timestamp=utc_now_iso(),
        )
    )

    output_lines = []
    for record in records:
        output_lines.append(
            json.dumps(
                {
                    "case_id": record.case_id,
                    "score": record.score,
                    "feedback": record.feedback,
                }
            )
        )
    output_lines.append(
        json.dumps(
            {
                "summary": {
                    "candidate_id": candidate.id,
                    "minibatch_id": minibatch.id,
                    "run_id": active_run_id,
                    "mean_score": mean,
                    "n_cases": len(records),
                }
            }
        )
    )

    write_content_file(output_file, "\n".join(output_lines))
