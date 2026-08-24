"""`gepa init` — scaffold .gepa/ and seed components from the agent."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import cast

import typer

from .layout import (
    GepaConfigError,
    CandidateSource,
    current_gepa_dirname,
    default_dataset_path,
    ensure_layout,
    insert_repo_root_on_path,
    repo_root,
    resolve_module_attr,
    write_default_config,
)
from .store import ComponentStore, introspect_agent


SKILL_DEST_RELATIVE = Path(".agents") / "skills" / "gepa-optimize" / "SKILL.md"


def _install_packaged_skill(dest: Path, *, force: bool) -> Path | None:
    """Copy the bundled gepa-optimize SKILL.md to ``dest``.

    Returns the destination path if a copy happened. Returns ``None`` when the
    destination already exists and ``force`` is False (caller decides what to
    print). Raises if the packaged source is missing (should never happen for
    an installed wheel).
    """
    source = (
        importlib.resources.files("pydantic_ai_gepa")
        / "skills"
        / "gepa_optimize"
        / "SKILL.md"
    )
    if not source.is_file():
        raise GepaConfigError(
            "Bundled gepa-optimize SKILL.md is missing from the installed package."
        )

    if dest.exists() and not force:
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def init(
    agent: str | None = typer.Option(
        None,
        "--agent",
        help='Agent module ref, e.g. "mypkg.agents:my_agent". Required in component mode; optional in git mode when --evaluate is set.',
    ),
    evaluate: str | None = typer.Option(
        None,
        "--evaluate",
        help='Plain evaluation callable for git mode, e.g. "mypkg.eval:evaluate". It receives each Case (or case_factory result) and returns the pipeline output.',
    ),
    candidate_source: str = typer.Option(
        "components",
        "--candidate-source",
        help="Candidate source: components (default) or git.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help=(
            "Relative path to the dataset JSONL. "
            "Defaults to `<workspace>/dataset.jsonl` for the active --gepa-dir."
        ),
    ),
    validation_dataset: str | None = typer.Option(
        None,
        "--validation-dataset",
        help=(
            "Optional held-out validation JSONL used to select candidates. "
            "When omitted, managed runs retain legacy training-only selection."
        ),
    ),
    metric: str | None = typer.Option(
        None,
        "--metric",
        help='Optional metric module ref written to gepa.toml as `metric = "..."`, e.g. "mypkg.metrics:my_metric". When omitted, gepa falls back to a substring/equality scorer.',
    ),
    case_factory: str | None = typer.Option(
        None,
        "--case-factory",
        help='Optional case-factory module ref written to gepa.toml as `case_factory = "..."`, e.g. "mypkg.eval:my_case_factory". Use when dataset rows carry deferred references (file paths, Mighty file ids, base64 blobs) that need to be materialized into the agent\'s input model before each rollout.',
    ),
    install_skill: bool = typer.Option(
        False,
        "--install-skill",
        help="Also copy the bundled gepa-optimize skill into the repo so coding agents discover it automatically. Default destination is `.agents/skills/gepa-optimize/SKILL.md`; override with --skill-dest.",
    ),
    skill_dest: Path | None = typer.Option(
        None,
        "--skill-dest",
        help="Override the install path for --install-skill. Can be a directory (the SKILL.md is dropped inside) or a full file path. Defaults to `<repo>/.agents/skills/gepa-optimize/SKILL.md`.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Overwrite an existing .gepa/gepa.toml, re-seed every confirmed slot from agent introspection, "
            "delete orphan confirmed slot files (slots no longer present on the agent), and overwrite the "
            "installed skill (when used with --install-skill). Staged stubs in .gepa/staged/ are preserved so "
            "in-progress confirmations are not silently destroyed."
        ),
    ),
) -> None:
    """Bootstrap the workspace in the current repo: write gepa.toml + seed components."""
    insert_repo_root_on_path()

    if candidate_source not in {"components", "git"}:
        typer.echo("--candidate-source must be 'components' or 'git'.", err=True)
        raise typer.Exit(code=2)
    if candidate_source == "components" and agent is None:
        typer.echo("--agent is required in component candidate mode.", err=True)
        raise typer.Exit(code=2)
    if candidate_source == "git" and agent is None and evaluate is None:
        typer.echo(
            "Git candidate mode requires either --agent or --evaluate.", err=True
        )
        raise typer.Exit(code=2)
    if candidate_source == "components" and evaluate is not None:
        typer.echo("--evaluate is only supported in git candidate mode.", err=True)
        raise typer.Exit(code=2)

    # Sanity-check the agent ref (and metric / case_factory refs, when
    # provided) before persisting config.
    agent_obj = None
    try:
        if agent:
            agent_obj = resolve_module_attr(agent, kind="agent")
        if evaluate:
            resolve_module_attr(evaluate, kind="evaluate")
        if metric:
            resolve_module_attr(metric, kind="metric")
        if case_factory:
            resolve_module_attr(case_factory, kind="case_factory")
    except GepaConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    resolved_dataset = dataset if dataset is not None else default_dataset_path()
    ensure_layout()
    try:
        cfg_path = write_default_config(
            agent,
            resolved_dataset,
            validation_dataset=validation_dataset,
            evaluate=evaluate,
            candidate_source=cast(CandidateSource, candidate_source),
            metric=metric,
            case_factory=case_factory,
            force=force,
        )
    except GepaConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    # Seed components/ with the introspected slot values. With --force the
    # existing slot files are re-seeded; without it, existing files win so
    # user edits between init runs survive. Either way, we leave staged stubs
    # alone — those represent in-progress confirmations the user hasn't
    # finalized yet, and silently destroying them on --force was confusing.
    store = ComponentStore()
    seeds = (
        introspect_agent(agent_obj)
        if candidate_source == "components" and agent_obj is not None
        else {}
    )
    reseeded = 0
    for slot, text in seeds.items():
        if force or store.read(slot) is None:
            store.write(slot, text, clear_staged=False)
            reseeded += 1

    orphans_removed = 0
    if force and candidate_source == "components":
        introspected_names = set(seeds)
        for slot in store.list_confirmed_slots():
            if slot not in introspected_names:
                if store.delete(slot):
                    orphans_removed += 1

    typer.echo(f"Wrote {cfg_path}")
    if candidate_source == "components":
        typer.echo(
            f"Seeded {len(seeds)} component slot(s) under {store.components_dir}"
        )
    else:
        typer.echo("Configured git-native candidates; component slots were bypassed.")
    if force and reseeded > 0:
        typer.echo(f"  Re-seeded {reseeded} slot(s) from introspection.")
    if orphans_removed:
        typer.echo(f"  Removed {orphans_removed} orphan slot file(s).")

    skill_installed: Path | None = None
    if install_skill:
        target = _resolve_skill_dest(skill_dest)
        try:
            skill_installed = _install_packaged_skill(target, force=force)
        except GepaConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
        if skill_installed is None:
            typer.echo(
                f"Skill already installed at {target}; pass --force to overwrite."
            )
        else:
            typer.echo(f"Installed gepa-optimize skill at {skill_installed}")

    dirname = current_gepa_dirname()
    workspace_hint = "" if dirname == ".gepa" else f" --gepa-dir {dirname}"
    typer.echo(
        "Next steps:\n"
        f"  1. Write reflection-training cases as JSONL at {resolved_dataset}\n"
        + (
            f"     Write held-out validation cases at {validation_dataset}\n"
            if validation_dataset
            else ""
        )
        + f"  2. Run `gepa{workspace_hint} eval --size N` to score the baseline + write the per-case report"
    )


def _resolve_skill_dest(dest: Path | None) -> Path:
    """Resolve ``--skill-dest`` (or the default) to a concrete SKILL.md path.

    If the caller passes a directory (or a path that already exists as a
    directory), the bundled ``SKILL.md`` is dropped inside it. Otherwise the
    path is taken literally as the destination file.
    """
    if dest is None:
        return repo_root() / SKILL_DEST_RELATIVE
    if dest.is_dir() or dest.suffix == "":
        return dest / "SKILL.md"
    return dest
