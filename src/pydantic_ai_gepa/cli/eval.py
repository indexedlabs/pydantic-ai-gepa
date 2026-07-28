"""`gepa eval` — score the current baseline (default) or an explicit candidate.

External-reflection mode (per pydanticaigepa-spec-973) keeps the coding agent
as the reflector. The library's job is to:

  * sample a fresh minibatch (or reuse one),
  * evaluate either the current confirmed component baseline, an explicit
    ``--candidate-file``, or the current git tree,
  * enforce stage-and-confirm when the baseline is what's being evaluated and
    new component slots were discovered (per pydanticaigepa-dec-0ky),
  * append a ParetoRow + write a per-case failure report under
    ``.gepa/runs/<run_id>/reports/``,
  * enforce ``--max-iterations`` as a hard cap (per pydanticaigepa-dec-xd6).

The coding agent reads the report, edits component slots or source code, and
re-runs ``gepa eval`` — there is no separate ``propose`` verb because the
agent IS the reflector.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import typer

from ..evaluation import (
    EvaluationRecord,
    evaluate_callable_dataset,
    evaluate_candidate_dataset,
)
from ._io import write_content_file
from .candidates import (
    Candidate,
    GitCandidateError,
    GitCandidateState,
    candidate_id_from_components,
    git_candidate_state,
)
from .dataset import case_ids as dataset_case_ids
from .dataset import cases_by_id, load_dataset
from .layout import (
    GepaConfig,
    CandidateSource,
    config_path,
    insert_repo_root_on_path,
    latest_run_id,
    new_run_id,
    repo_root,
    resolve_agent,
    resolve_case_factory,
    resolve_evaluate,
    resolve_metric,
    resolve_skills,
    run_dir,
)
from .metrics import default_substring_metric
from .runs import (
    MinibatchStore,
    ParetoLog,
    ParetoRow,
    current_commit_sha,
    utc_now_iso,
)
from .store import ComponentStore


GEPA_TRACE_FILE_ENV = "GEPA_TRACE_FILE"


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = latest_run_id()
    if latest:
        return latest
    return new_run_id()


def _count_evals_in_run(run_id: str) -> int:
    return ParetoLog(run_id).count_rows()


DEFAULT_FAILURE_THRESHOLD = 0.999


@dataclass(frozen=True)
class EvalOutcome:
    records: list[EvaluationRecord]
    summary: dict[str, Any]
    report_path: Path
    trace_path: Path | None

    @property
    def n_failures(self) -> int:
        return int(self.summary["n_failures"])


def _format_failures(
    records,
    threshold: float = DEFAULT_FAILURE_THRESHOLD,
    *,
    candidate_source: CandidateSource = "components",
) -> str:
    lines = ["# Eval report", ""]
    failures = [r for r in records if r.score < threshold]
    if not failures:
        lines.append("Every case in this minibatch passed; nothing to act on.")
        return "\n".join(lines)
    next_step = (
        "Review per-case feedback and traces, edit the working-tree code or "
        "artifacts, then commit the candidate."
        if candidate_source == "git"
        else "Review per-case feedback and edit slots in `.gepa/components/` "
        "or change the agent's source."
    )
    lines.append(
        f"{len(failures)} of {len(records)} case(s) underperformed "
        f"(score < {threshold}). {next_step}\n"
    )
    for record in failures:
        lines.append(f"## {record.case_id} — score {record.score:.3f}")
        if record.feedback:
            lines.append("")
            lines.append(record.feedback.rstrip())
        lines.append("")
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(value)
    return str(value)


def current_trace_path() -> Path | None:
    """Return the trace JSONL path exposed for the current CLI evaluation."""

    value = os.environ.get(GEPA_TRACE_FILE_ENV)
    return Path(value) if value else None


@contextmanager
def _expose_trace_path(path: Path | None) -> Iterator[None]:
    previous = os.environ.get(GEPA_TRACE_FILE_ENV)
    if path is None:
        os.environ.pop(GEPA_TRACE_FILE_ENV, None)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.environ[GEPA_TRACE_FILE_ENV] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(GEPA_TRACE_FILE_ENV, None)
        else:
            os.environ[GEPA_TRACE_FILE_ENV] = previous


def _trace_file_path(
    *,
    run_id: str,
    iteration: int,
    candidate_id: str,
    minibatch_id: str,
) -> Path:
    return (
        run_dir(run_id)
        / "traces"
        / "minibatches"
        / minibatch_id
        / f"{iteration:04d}-{candidate_id}.jsonl"
    )


def _write_trace_file(
    *,
    path: Path,
    records: list[EvaluationRecord],
) -> Path | None:
    trace_rows: list[dict[str, Any]] = []
    for record in records:
        trajectory = record.payload.get("trajectory")
        if trajectory is None or not hasattr(trajectory, "to_reflective_record"):
            continue
        trace_record = trajectory.to_reflective_record()
        trace_record["case_id"] = record.case_id
        trace_record["score"] = record.score
        if record.feedback:
            trace_record["feedback"] = record.feedback
        trace_rows.append(trace_record)

    if trace_rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
    return path if path.exists() and path.stat().st_size > 0 else None


def run_eval_once(
    *,
    candidate_file: Path | None,
    minibatch_id: str | None,
    size: int,
    seed: int,
    epoch: int,
    run_id: str | None,
    concurrency: int,
    max_iterations: int,
    threshold: float,
    capture_traces: bool = False,
    candidate_source: CandidateSource | None = None,
) -> EvalOutcome:
    """Evaluate one baseline/candidate and append the standard run artifacts."""
    cfg = GepaConfig.load(config_path())
    insert_repo_root_on_path()

    source = candidate_source or cfg.candidate_source
    agent = resolve_agent(cfg) if cfg.agent else None
    evaluate = resolve_evaluate(cfg) if source == "git" else None
    metric = resolve_metric(cfg) or default_substring_metric
    case_factory = resolve_case_factory(cfg)
    skills_fs = resolve_skills(cfg)

    dataset_path = repo_root() / cfg.dataset
    cases = load_dataset(dataset_path)
    if not cases:
        typer.echo(f"Dataset {dataset_path} is empty.", err=True)
        raise typer.Exit(code=1)

    git_state: GitCandidateState | None = None
    candidate: Candidate | None = None
    candidate_overrides_id = ""
    status = ""
    if source == "git":
        if candidate_file is not None:
            typer.echo(
                "--candidate-file cannot be used with git candidates; "
                "the current working tree is the candidate.",
                err=True,
            )
            raise typer.Exit(code=2)
        if agent is None and evaluate is None:
            typer.echo(
                "Git candidate mode requires an 'agent' or plain 'evaluate' "
                "callable in gepa.toml.",
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        if agent is None:
            typer.echo(
                "Component candidate mode requires 'agent' in gepa.toml.", err=True
            )
            raise typer.Exit(code=1)
        store = ComponentStore()
        is_baseline_eval = candidate_file is None
        if is_baseline_eval:
            staged = store.detect_new_slots(agent, skills_fs=skills_fs)
            if staged:
                typer.echo(
                    "Found unconfirmed component slots; refusing to evaluate the baseline.",
                    err=True,
                )
                typer.echo("Staged stubs:", err=True)
                for slot in staged:
                    typer.echo(f"  {store.staged_path(slot)}", err=True)
                typer.echo("Confirm with:", err=True)
                for slot in staged:
                    typer.echo(f"  gepa components confirm {slot}", err=True)
                raise typer.Exit(code=2)

            introspected_names = set(
                store.effective_candidate(agent, skills_fs=skills_fs)
            )
            orphans = sorted(
                slot
                for slot in store.list_confirmed_slots()
                if slot not in introspected_names
            )
            if orphans:
                typer.echo("Orphan slot files (no longer on the agent):", err=True)
                for slot in orphans:
                    typer.echo(f"  {store.confirmed_path(slot)}", err=True)
                typer.echo(
                    "These files are ignored during eval. Delete them with "
                    "`rm` or re-init with --force to remove the cruft.",
                    err=True,
                )

            baseline_components = store.effective_candidate(agent, skills_fs=skills_fs)
            candidate = Candidate(
                id=candidate_id_from_components(baseline_components),
                components=baseline_components,
                metadata={"role": "baseline"},
            )
            candidate_overrides_id = "(baseline)"
            status = "baseline"
        else:
            candidate = Candidate.load(candidate_file)
            candidate_overrides_id = str(candidate_file)
            status = "evaluated"

    active_run_id = _resolve_run_id(run_id)
    prior_count = _count_evals_in_run(active_run_id)
    if prior_count >= max_iterations:
        typer.echo(
            f"Max iterations reached ({prior_count}/{max_iterations}). "
            "Start a new run (omit --run-id) or raise --max-iterations.",
            err=True,
        )
        raise typer.Exit(code=70)
    iteration = prior_count + 1

    run_dir(active_run_id).mkdir(parents=True, exist_ok=True)
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

    if source == "git":
        try:
            git_state = git_candidate_state(
                repo_root(), exclude_paths=[run_dir(active_run_id)]
            )
        except GitCandidateError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        candidate = Candidate(
            id=git_state.candidate_id,
            components={},
            metadata={
                "role": "baseline",
                "commit_sha": git_state.commit_sha,
                "dirty_hash": git_state.dirty_hash,
            },
        )
        candidate_overrides_id = "(git-tree)"
        status = "baseline"

    assert candidate is not None
    planned_trace_path = (
        _trace_file_path(
            run_id=active_run_id,
            iteration=iteration,
            candidate_id=candidate.id,
            minibatch_id=minibatch.id,
        )
        if capture_traces
        else None
    )
    with _expose_trace_path(planned_trace_path):
        if evaluate is not None:
            records = asyncio.run(
                evaluate_callable_dataset(
                    evaluate=evaluate,
                    metric=metric,
                    dataset=subset,
                    concurrency=concurrency,
                    case_factory=case_factory,
                )
            )
        else:
            assert agent is not None
            records = asyncio.run(
                evaluate_candidate_dataset(
                    agent=agent,
                    metric=metric,
                    dataset=subset,
                    candidate=({} if source == "git" else candidate.to_candidate_map()),
                    concurrency=concurrency,
                    case_factory=case_factory,
                    capture_traces=capture_traces,
                    skills_fs=skills_fs,
                )
            )

    per_case = {record.case_id: record.score for record in records}
    mean = sum(per_case.values()) / len(per_case) if per_case else 0.0

    pareto = ParetoLog(active_run_id)
    pareto.append(
        ParetoRow(
            candidate_id=candidate.id,
            commit_sha=(
                git_state.commit_sha
                if git_state is not None
                else current_commit_sha(repo_root())
            ),
            component_overrides_id=candidate_overrides_id,
            minibatch_id=minibatch.id,
            per_case_scores=per_case,
            mean_score=mean,
            status=status,
            summary=f"{status} eval of {candidate.id} on minibatch {minibatch.id} (mean={mean:.3f})",
            timestamp=utc_now_iso(),
        )
    )

    # Write the per-case report next to the pareto log.
    reports_dir = run_dir(active_run_id) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{iteration:04d}-{candidate.id}.md"
    report_path.write_text(
        _format_failures(records, threshold=threshold, candidate_source=source),
        encoding="utf-8",
    )
    trace_path = (
        _write_trace_file(
            path=planned_trace_path,
            records=records,
        )
        if planned_trace_path is not None
        else None
    )

    summary: dict[str, Any] = {
        "candidate_id": candidate.id,
        "candidate_role": status,
        "candidate_source": source,
        "commit_sha": (
            git_state.commit_sha
            if git_state is not None
            else current_commit_sha(repo_root())
        ),
        "dirty_tree": git_state.dirty if git_state is not None else None,
        "dirty_hash": git_state.dirty_hash if git_state is not None else None,
        "minibatch_id": minibatch.id,
        "run_id": active_run_id,
        "mean_score": mean,
        "n_cases": len(records),
        "n_failures": len([record for record in records if record.score < threshold]),
        "iterations": iteration,
        "max_iterations": max_iterations,
        "report_path": str(report_path),
        "trace_path": str(trace_path) if trace_path else None,
    }

    return EvalOutcome(
        records=records,
        summary=summary,
        report_path=report_path,
        trace_path=trace_path,
    )


def _format_output_lines(outcome: EvalOutcome) -> str:
    output_lines = []
    for record in outcome.records:
        output_lines.append(
            json.dumps(
                {
                    "case_id": record.case_id,
                    "score": record.score,
                    "feedback": record.feedback,
                }
            )
        )
    output_lines.append(json.dumps({"summary": outcome.summary}))
    return "\n".join(output_lines)


def eval_(
    candidate_file: Path | None = typer.Option(
        None,
        "--candidate-file",
        help="Path to a candidate JSON file. Omit to evaluate the current confirmed baseline in `.gepa/components/`.",
    ),
    minibatch_id: str | None = typer.Option(
        None,
        "--minibatch-id",
        help="Re-use an existing minibatch (e.g. from a previous eval summary) for a clean A/B against a slot edit.",
    ),
    size: int = typer.Option(
        10, "--size", help="Number of cases to sample when a new minibatch is drawn."
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Deterministic minibatch sampling seed. Same (seed, epoch) reproduces the same minibatch_id.",
    ),
    epoch: int = typer.Option(
        0,
        "--epoch",
        help="Bumps the minibatch identity without changing the seeding regime — use it to draw a fresh independent sample with the same seed.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Append to a specific run. Omit to use the latest existing run, or start a new one if none exists.",
    ),
    output_file: Path | None = typer.Option(
        None, "--output-file", help="Write JSONL results to a file (or - for stdout)."
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        help="Max parallel agent calls during evaluation.",
    ),
    max_iterations: int = typer.Option(
        100,
        "--max-iterations",
        help="Hard cap on eval rows in this run (per pydanticaigepa-dec-xd6). Exits 70 when exceeded.",
    ),
    threshold: float = typer.Option(
        DEFAULT_FAILURE_THRESHOLD,
        "--threshold",
        help="Score below which a case is listed as a failure in the per-case report. Default is 0.999 so any non-perfect case is flagged; lower it (e.g. 0.7) when partial-credit metrics expect imperfect scores.",
    ),
    capture_traces: bool = typer.Option(
        False,
        "--capture-traces",
        help="Persist reflective trace records for this minibatch under the run's traces directory.",
    ),
    candidate_source: str | None = typer.Option(
        None,
        "--candidate-source",
        help="Override gepa.toml candidate_source for this eval: components or git.",
    ),
) -> None:
    """Evaluate the current baseline (default) or an explicit candidate file."""
    if candidate_source not in {None, "components", "git"}:
        typer.echo("--candidate-source must be 'components' or 'git'.", err=True)
        raise typer.Exit(code=2)
    outcome = run_eval_once(
        candidate_file=candidate_file,
        minibatch_id=minibatch_id,
        size=size,
        seed=seed,
        epoch=epoch,
        run_id=run_id,
        concurrency=concurrency,
        max_iterations=max_iterations,
        threshold=threshold,
        capture_traces=capture_traces,
        candidate_source=cast(CandidateSource | None, candidate_source),
    )

    write_content_file(output_file, _format_output_lines(outcome))
