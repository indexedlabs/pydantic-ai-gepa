"""`gepa propose` — run one optimization step.

Workflow:
  1. Refuse to proceed if any introspected component slot lacks a confirmed file
     (per pydanticaigepa-dec-0ky: stage-and-confirm). Stage stubs for the agent
     and emit the gepa components confirm commands needed.
  2. Sample a fresh minibatch via MinibatchStore (or use --minibatch-id).
  3. Build the baseline candidate from .gepa/components/ (confirmed) + agent
     introspected seeds for any slot without a file.
  4. Evaluate the baseline; append a ParetoRow with status="baseline".
  5. Emit a proposal candidate JSON under .gepa/runs/<run_id>/proposals/ —
     starts as a copy of the baseline; the coding agent edits component slots
     and re-runs gepa eval. Also emits a reflection-report.md summarizing
     failure cases for the coding agent to read.
  6. Append a second ParetoRow (status="proposal").
  7. Enforce --max-iterations: count proposals under .gepa/runs/<run_id>/proposals/
     and refuse to write more.
"""

from __future__ import annotations

import asyncio
import json

import typer

from ..evaluation import evaluate_candidate_dataset
from .candidates import Candidate, candidate_id_from_components
from .dataset import case_ids as dataset_case_ids
from .dataset import cases_by_id, load_dataset
from .layout import (
    GepaConfig,
    config_path,
    insert_repo_root_on_path,
    latest_run_id,
    new_run_id,
    proposal_dir,
    repo_root,
    resolve_agent,
    resolve_metric,
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


class MaxIterationsExceeded(typer.Exit):
    """Raised when --max-iterations is hit; surfaces as exit code 70."""

    def __init__(self, count: int, limit: int) -> None:
        self.count = count
        self.limit = limit
        super().__init__(code=70)


def _count_existing_proposals(run_id: str) -> int:
    pdir = proposal_dir(run_id)
    if not pdir.is_dir():
        return 0
    return sum(1 for p in pdir.iterdir() if p.suffix == ".json")


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    latest = latest_run_id()
    if latest:
        return latest
    return new_run_id()


def _format_failures(records, threshold: float = 0.999) -> str:
    """Build a human-readable failure summary for the coding agent."""
    lines = ["# Reflection report", ""]
    failures = [r for r in records if r.score < threshold]
    if not failures:
        lines.append("Every case in this minibatch passed; no failures to reflect on.")
        return "\n".join(lines)
    lines.append(
        f"{len(failures)} of {len(records)} case(s) underperformed (score < {threshold}). Review and edit slots:\n"
    )
    for record in failures:
        lines.append(f"## {record.case_id} — score {record.score:.3f}")
        if record.feedback:
            lines.append("")
            lines.append(record.feedback.rstrip())
        lines.append("")
    return "\n".join(lines)


def propose(
    minibatch_size: int = typer.Option(10, "--minibatch-size"),
    seed: int = typer.Option(0, "--seed"),
    epoch: int = typer.Option(0, "--epoch"),
    reflection_model: str | None = typer.Option(
        None,
        "--reflection-model",
        help="Optional LLM ref for an in-process reflection step (e.g. 'openai:gpt-5'). "
        "When omitted the proposal starts as a copy of the baseline and the coding agent "
        "is expected to edit slots before running gepa eval.",
    ),
    max_iterations: int = typer.Option(
        100,
        "--max-iterations",
        help="Hard cap on proposals in this run (per pydanticaigepa-dec-xd6). Errors with exit 70 when exceeded.",
    ),
    max_rollouts: int | None = typer.Option(None, "--max-rollouts"),
    run_id: str | None = typer.Option(None, "--run-id"),
    concurrency: int = typer.Option(5, "--concurrency"),
) -> None:
    """Run one external-reflection step: eval baseline + emit proposal scaffold."""
    cfg = GepaConfig.load(config_path())
    insert_repo_root_on_path()
    agent = resolve_agent(cfg)
    metric = resolve_metric(cfg) or default_substring_metric

    dataset_path = repo_root() / cfg.dataset
    cases = load_dataset(dataset_path)
    if not cases:
        typer.echo(f"Dataset {dataset_path} is empty.", err=True)
        raise typer.Exit(code=1)

    # Stage-and-confirm gate (dec-0ky).
    store = ComponentStore()
    staged = store.detect_new_slots(agent)
    if staged:
        typer.echo(
            "Found unconfirmed component slots; refusing to propose.",
            err=True,
        )
        typer.echo("Staged stubs:", err=True)
        for slot in staged:
            typer.echo(f"  {store.staged_path(slot)}", err=True)
        typer.echo(
            "Confirm with:",
            err=True,
        )
        for slot in staged:
            typer.echo(f"  gepa components confirm {slot}", err=True)
        raise typer.Exit(code=2)

    active_run = _resolve_run_id(run_id)
    proposals_so_far = _count_existing_proposals(active_run)
    if proposals_so_far >= max_iterations:
        typer.echo(
            f"Max iterations reached ({proposals_so_far}/{max_iterations}). "
            f"Start a new run (omit --run-id) or raise --max-iterations.",
            err=True,
        )
        raise typer.Exit(code=70)

    run_dir(active_run).mkdir(parents=True, exist_ok=True)
    mb_store = MinibatchStore(active_run)
    minibatch = mb_store.sample(
        dataset_case_ids(cases), size=minibatch_size, seed=seed, epoch=epoch
    )

    by_id = cases_by_id(cases)
    subset = [by_id[cid] for cid in minibatch.case_ids if cid in by_id]

    # Baseline candidate from current .gepa/components/ + introspected seeds.
    baseline_components = store.effective_candidate(agent)
    baseline_id = candidate_id_from_components(baseline_components)
    baseline_candidate = Candidate(
        id=baseline_id, components=baseline_components, metadata={"role": "baseline"}
    )

    records = asyncio.run(
        evaluate_candidate_dataset(
            agent=agent,
            metric=metric,
            dataset=subset,
            candidate=baseline_candidate.to_candidate_map(),
            concurrency=concurrency,
        )
    )
    per_case = {r.case_id: r.score for r in records}
    mean = sum(per_case.values()) / len(per_case) if per_case else 0.0

    pareto = ParetoLog(active_run)
    commit = current_commit_sha()
    pareto.append(
        ParetoRow(
            candidate_id=baseline_id,
            commit_sha=commit,
            component_overrides_id="(baseline)",
            minibatch_id=minibatch.id,
            per_case_scores=per_case,
            mean_score=mean,
            status="baseline",
            summary=f"baseline eval on minibatch {minibatch.id} (mean={mean:.3f})",
            timestamp=utc_now_iso(),
        )
    )

    # Proposal scaffold: start as a copy of baseline. With --reflection-model
    # we'll run an in-process LLM reflection that proposes new texts; otherwise
    # leave the proposal identical to baseline (the coding agent edits slots
    # via `gepa components set` or by editing source).
    proposal_components = dict(baseline_components)
    if reflection_model:
        proposal_components = asyncio.run(
            _run_in_process_reflection(
                agent=agent,
                cfg=cfg,
                model=reflection_model,
                baseline=baseline_candidate,
                records=records,
                cases=subset,
            )
        )

    proposal_id = candidate_id_from_components(proposal_components)
    proposal_candidate = Candidate(
        id=proposal_id,
        components=proposal_components,
        metadata={
            "role": "proposal",
            "run_id": active_run,
            "baseline_id": baseline_id,
            "minibatch_id": minibatch.id,
            "reflection_model": reflection_model,
        },
    )
    proposal_path = proposal_dir(active_run) / f"{proposal_id}.json"
    proposal_candidate.write(proposal_path)

    report_path = proposal_path.with_suffix(".report.md")
    report_path.write_text(_format_failures(records), encoding="utf-8")

    pareto.append(
        ParetoRow(
            candidate_id=proposal_id,
            commit_sha=commit,
            component_overrides_id=str(proposal_path),
            minibatch_id=minibatch.id,
            per_case_scores={},
            mean_score=0.0,
            status="proposal",
            summary=f"proposal {proposal_id} from baseline {baseline_id} on minibatch {minibatch.id}",
            timestamp=utc_now_iso(),
        )
    )

    typer.echo(
        json.dumps(
            {
                "run_id": active_run,
                "baseline_id": baseline_id,
                "baseline_mean_score": mean,
                "proposal_id": proposal_id,
                "proposal_path": str(proposal_path),
                "report_path": str(report_path),
                "minibatch_id": minibatch.id,
                "iterations": proposals_so_far + 1,
                "max_iterations": max_iterations,
                "next": f"gepa eval --candidate-file {proposal_path}",
            },
            indent=2,
        )
    )


async def _run_in_process_reflection(
    *,
    agent,
    cfg: GepaConfig,
    model: str,
    baseline: Candidate,
    records,
    cases,
) -> dict[str, str]:
    """Optional in-process reflection using the existing InstructionProposalGenerator.

    Best-effort: on failure, fall back to the baseline components (the coding
    agent will edit them by hand).
    """
    try:
        from ..adapter import ReflectiveDataset
        from ..gepa_graph.models import CandidateProgram, ComponentValue
        from ..gepa_graph.proposal.instruction import InstructionProposalGenerator

        candidate_components = {
            name: ComponentValue(name=name, text=text)
            for name, text in baseline.components.items()
        }
        program = CandidateProgram(
            id=baseline.id,
            components=candidate_components,
        )
        # Build a minimal reflective dataset from the eval records: per-component
        # views are needed by the generator. The default substring metric
        # produces feedback per case; reuse it as the per-component context.
        reflective: ReflectiveDataset = {}
        for component_name in candidate_components:
            reflective[component_name] = [
                {
                    "case_id": r.case_id,
                    "score": r.score,
                    "feedback": r.feedback or "",
                }
                for r in records
            ]

        generator = InstructionProposalGenerator()
        result = await generator.propose_texts(
            candidate=program,
            reflective_data=reflective,
            components=list(candidate_components),
            model=model,
        )
        proposed = dict(baseline.components)
        proposed.update(result.texts)
        return proposed
    except Exception as exc:  # pragma: no cover - best effort fallback
        typer.echo(
            f"In-process reflection failed ({exc}); proposal remains a copy of baseline.",
            err=True,
        )
        return dict(baseline.components)
