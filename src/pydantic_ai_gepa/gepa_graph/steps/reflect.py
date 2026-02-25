"""Reflect step - implements the core reflective mutation workflow."""

from __future__ import annotations

import difflib
import hashlib
import inspect
import re
from typing import Any, Mapping, Sequence, cast

import logfire
from pydantic_graph.beta import StepContext
from pydantic_ai import FunctionToolset
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings

from ...adapter import (
    ComponentReflectiveDataset,
    ReflectiveDataset,
    SharedReflectiveDataset,
)
from ...evaluation_models import EvaluationBatch
from pydantic_evals import Case
from ..deps import GepaDeps
from ..evaluation import EvaluationResults
from ..models import CandidateMap, CandidateProgram, ComponentValue, GepaState
from ..proposal.instruction import ProposalResult
from ...skill_components import (
    apply_candidate_to_skills,
    materialize_skill_components_for_path,
    skill_description_key,
    skill_body_key,
)
from ...skills import OverlayFS, normalize_rel_path, parse_skill_md
from ...skills.models import (
    SkillFileResult,
    SkillLoadResult,
    SkillSearchResult,
    SkillSummary,
)
from ...skills.search import LocalSkillsSearchProvider
from .continue_step import IterationAction

_IMPROVEMENT_EPSILON = 1e-9
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_/-]{3,}")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def reflect_step(ctx: StepContext[GepaState, GepaDeps, None]) -> IterationAction:
    """Generate and evaluate reflective mutations for the current candidate."""

    state = ctx.state
    deps = ctx.deps

    parent_idx, parent = _select_parent(state, deps)
    minibatch = await _sample_minibatch(state, deps)
    with logfire.span(
        "evaluate first minibatch",
        parent_idx=parent_idx,
        component_versions=_component_versions(parent),
        minibatch_size=len(minibatch),
    ):
        parent_results = await _evaluate_minibatch(
            deps=deps,
            state=state,
            candidate=parent,
            batch=minibatch,
            capture_traces=True,
        )

    state.record_evaluation_errors(
        candidate_idx=parent_idx,
        stage="reflection_parent",
        data_ids=parent_results.data_ids,
        outputs=parent_results.outputs,
    )
    _record_minibatch(parent, parent_results)
    _increment_budget(state, parent_results)
    parent_total, parent_avg = _summarize_scores(parent_results.scores)

    logfire.debug(
        "ReflectStep parent minibatch results",
        parent_idx=parent_idx,
        minibatch_scores=list(parent_results.scores),
        minibatch_total=parent_total,
        minibatch_average=parent_avg,
    )

    if not parent_results.has_trajectories():
        logfire.info(
            "ReflectStep skipping reflection due to missing trajectories",
            parent_idx=parent_idx,
        )
        state.last_accepted = False
        return "continue"

    if _should_skip_perfect(parent_results.scores, state):
        logfire.info(
            "ReflectStep skipping reflection due to perfect minibatch",
            parent_idx=parent_idx,
            threshold=state.config.perfect_score,
            minibatch_total=parent_total,
        )
        state.last_accepted = False
        return "continue"

    reflection_model = _resolve_model(deps)
    components_to_update: Sequence[str] | None
    component_toolsets: list[FunctionToolset] | None = []

    if state.config.reflection_config and state.config.reflection_config.journal_file:
        from ..proposal.journal_tools import create_journal_toolset

        component_toolsets.append(
            create_journal_toolset(state.config.reflection_config.journal_file)
        )

    from ..proposal.trace_tools import create_trace_toolset

    component_toolsets.append(
        create_trace_toolset(state.run_id, parent_idx, reflection_model)
    )

    if state.config.component_selector == "reflection":
        components_to_update = None
        components_for_dataset = list(parent.components.keys())
        component_toolsets.append(
            _build_component_selection_toolset(
                state=state,
                deps=deps,
                parent_idx=parent_idx,
            )
        )
    else:
        selection = deps.component_selector.select(
            state,
            parent_idx,
            model=reflection_model,
            eval_results=parent_results,
        )
        if inspect.isawaitable(selection):
            selection = await selection
        components_to_update = list(selection)
        components_for_dataset = components_to_update

    logfire.debug(
        "ReflectStep selected components",
        parent_idx=parent_idx,
        selector=state.config.component_selector,
        components_to_update=components_to_update,
        candidate_component_count=len(parent.components),
    )

    reflective_dataset = _build_reflective_dataset(
        deps=deps,
        state=state,
        candidate=parent,
        eval_results=parent_results,
        components=components_for_dataset,
    )
    with logfire.span(
        "propose new texts",
        parent_idx=parent_idx,
        model=reflection_model,
        selector=state.config.component_selector,
        components_to_update=components_to_update,
    ):
        proposal_result = await _propose_new_texts(
            deps=deps,
            state=state,
            parent=parent,
            reflective_dataset=reflective_dataset,
            components=components_to_update,
            model=reflection_model,
            model_settings=deps.model_settings,
            component_toolsets=component_toolsets if component_toolsets else None,
        )

        # Capture and save the Reflector's own trace data (tool calls, reasoning)
        if deps.memory_exporter is not None:
            from pathlib import Path

            spans = deps.memory_exporter.get_finished_spans()
            if spans:
                reflector_dir = Path(
                    f".gepa_cache/runs/{state.run_id}/candidates/{parent_idx}/reflector_traces"
                )
                reflector_dir.mkdir(parents=True, exist_ok=True)
                reflector_file = reflector_dir / "traces.jsonl"
                with open(reflector_file, "a", encoding="utf-8") as f:
                    for span in spans:
                        f.write(span.to_json() + "\\n")
            deps.memory_exporter.clear()

        component_metadata = (
            proposal_result.component_metadata
            if state.config.track_component_hypotheses
            else None
        )
        reasoning = proposal_result.reasoning
        if reasoning is not None:
            logfire.info(
                "ReflectStep proposal reasoning",
                parent_idx=parent_idx,
                pattern=reasoning.pattern_discovery,
                hypothesis=reasoning.creative_hypothesis,
                approach=reasoning.experimental_approach,
                edge_insight=reasoning.edge_insight,
                success_checkpoint=reasoning.success_checkpoint,
                evolution_moves=reasoning.evolution_moves,
            )

    if not proposal_result.texts:
        state.last_accepted = False
        logfire.info(
            "ReflectStep skipping candidate evaluation due to no-op proposal",
            parent_idx=parent_idx,
            selector=state.config.component_selector,
            requested_components=(
                list(components_to_update) if components_to_update is not None else None
            ),
        )
        return "continue"

    new_candidate = _create_candidate(
        state=state,
        parent=parent,
        parent_idx=parent_idx,
        new_texts=proposal_result.texts,
        metadata=component_metadata,
    )
    logfire.debug(
        "ReflectStep proposed candidate",
        candidate_idx=new_candidate.idx,
        parent_idx=parent_idx,
        updated_components=sorted(
            name
            for name, value in new_candidate.components.items()
            if value.text != parent.components[name].text
        )
        or (list(components_to_update) if components_to_update is not None else []),
    )

    with logfire.span(
        "evaluate new candidate",
        candidate_idx=new_candidate.idx,
        parent_idx=parent_idx,
    ):
        new_results = await _evaluate_minibatch(
            deps=deps,
            state=state,
            candidate=new_candidate,
            batch=minibatch,
            capture_traces=False,
        )

    state.record_evaluation_errors(
        candidate_idx=new_candidate.idx,
        stage="reflection_candidate",
        data_ids=new_results.data_ids,
        outputs=new_results.outputs,
    )
    _record_minibatch(new_candidate, new_results)
    _increment_budget(state, new_results)
    new_total, new_avg = _summarize_scores(new_results.scores)
    logfire.debug(
        "ReflectStep candidate minibatch results",
        candidate_idx=new_candidate.idx,
        parent_idx=parent_idx,
        minibatch_scores=list(new_results.scores),
        minibatch_total=new_total,
        minibatch_average=new_avg,
    )

    improved = _is_strict_improvement(
        baseline_scores=parent_results.scores,
        new_scores=new_results.scores,
    )
    decision_payload = dict(
        parent_idx=parent_idx,
        candidate_idx=new_candidate.idx,
        baseline_total=parent_total,
        candidate_total=new_total,
        improvement=improved,
    )
    if improved:
        state.add_candidate(new_candidate)
        state.last_accepted = True
        state.schedule_merge(state.config.merges_per_accept)
        logfire.info(
            "ReflectStep accepted candidate",
            **cast(dict[str, Any], decision_payload),
        )
        return "evaluate"

    state.last_accepted = False
    logfire.info(
        "ReflectStep rejected candidate",
        failure_reason="not_strict_improvement",
        **cast(dict[str, Any], decision_payload),
    )
    return "continue"


def _select_parent(
    state: GepaState,
    deps: GepaDeps,
) -> tuple[int, CandidateProgram]:
    if not state.candidates:
        raise ValueError("ReflectStep requires at least one candidate in state.")
    selector = deps.candidate_selector
    select_fn = getattr(selector, "select", None)
    if select_fn is None:
        select_fn = getattr(selector, "select_candidate")
    idx = select_fn(state)
    if idx is None:
        raise RuntimeError("Candidate selector must return an index.")
    parent = state.candidates[idx]
    return idx, parent


async def _sample_minibatch(
    state: GepaState,
    deps: GepaDeps,
) -> list[Case[Any, Any, Any]]:
    loader = state.training_set
    batch = await deps.batch_sampler.sample(
        training_set=loader,
        state=state,
        size=state.config.minibatch_size,
    )
    if len(batch) < 1:
        raise ValueError("BatchSampler returned an empty minibatch.")
    return batch


async def _evaluate_minibatch(
    *,
    deps: GepaDeps,
    state: GepaState,
    candidate: CandidateProgram,
    batch: Sequence[Case[Any, Any, Any]],
    capture_traces: bool,
) -> EvaluationResults[str]:
    results = await deps.evaluator.evaluate_batch(
        candidate=candidate,
        batch=batch,
        adapter=deps.adapter,
        capture_traces=capture_traces,
        max_concurrent=state.config.max_concurrent_evaluations,
    )
    if capture_traces and deps.memory_exporter is not None:
        from pathlib import Path

        spans = deps.memory_exporter.get_finished_spans()
        if spans:
            traces_dir = Path(
                f".gepa_cache/runs/{state.run_id}/candidates/{candidate.idx}/traces"
            )
            traces_dir.mkdir(parents=True, exist_ok=True)
            traces_file = traces_dir / "traces.jsonl"
            with open(traces_file, "a", encoding="utf-8") as f:
                for span in spans:
                    f.write(span.to_json() + "\\n")
        deps.memory_exporter.clear()
    return results


def _record_minibatch(
    candidate: CandidateProgram,
    results: EvaluationResults[str],
) -> None:
    candidate.minibatch_scores = list(results.scores)


def _increment_budget(
    state: GepaState,
    results: EvaluationResults[str],
) -> None:
    state.total_evaluations += len(results.data_ids)


def _summarize_scores(scores: Sequence[float]) -> tuple[float, float]:
    total = float(sum(scores))
    avg = total / len(scores) if scores else 0.0
    return total, avg


def _should_skip_perfect(
    scores: Sequence[float],
    state: GepaState,
) -> bool:
    if not state.config.skip_perfect_score:
        return False
    perfect = state.config.perfect_score
    total = float(sum(scores))
    return total >= perfect * len(scores)


def _build_component_selection_toolset(
    *,
    state: GepaState,
    deps: GepaDeps,
    parent_idx: int,
) -> FunctionToolset:
    toolset: FunctionToolset[None] = FunctionToolset()

    def _candidate() -> CandidateProgram:
        return state.candidates[parent_idx]

    @toolset.tool
    def list_components(prefix: str | None = None) -> list[str]:
        """List available component names (optionally filtered by substring)."""
        names = sorted(_candidate().components.keys())
        if prefix:
            needle = prefix.casefold()
            names = [name for name in names if needle in name.casefold()]
        return names

    @toolset.tool
    def search_components(query: str, top_k: int = 12) -> list[str]:
        """Search component names by simple token matching."""
        if top_k <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scored: list[tuple[float, str]] = []
        for name in _candidate().components.keys():
            lower = name.casefold()
            score = sum(1.0 for t in tokens if t and t in lower)
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored[:top_k]]

    @toolset.tool
    def load_component(component_name: str) -> str:
        """Load the current text value of a component."""
        candidate = _candidate()
        component = candidate.components.get(component_name)
        if component is None:
            raise KeyError(f"Unknown component: {component_name}")
        return component.text

    skills_fs = getattr(deps.adapter, "skills_fs", None)
    if skills_fs is None:
        return toolset

    search_backend = getattr(deps.adapter, "skills_search_backend", None)
    backend = search_backend or LocalSkillsSearchProvider()

    def _resolve_skill_dir(view: OverlayFS, skill_path: str) -> str:
        try:
            normalized = normalize_rel_path(skill_path)
        except Exception as e:
            raise ModelRetry(
                f"Invalid skill_path={skill_path!r}: {e}. Use list_skills() to see valid skill paths."
            ) from e

        candidates = [
            normalized,
            normalized.replace("_", "-"),
            normalized.casefold(),
        ]
        for candidate_path in candidates:
            if candidate_path and view.exists(f"{candidate_path}/SKILL.md"):
                return candidate_path

        available = sorted(set(view.iter_skill_dirs()))
        close = difflib.get_close_matches(normalized, available, n=8, cutoff=0.45)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ModelRetry(
            f"Unknown skill_path={skill_path!r}.{hint} Use list_skills() to see valid skill paths."
        )

    @toolset.tool
    def list_skills() -> list[SkillSummary]:
        """List available skills with their name and description."""
        candidate = _candidate()
        with apply_candidate_to_skills(skills_fs, candidate.components) as view:
            items: list[SkillSummary] = []
            for skill_dir in view.iter_skill_dirs():
                if not skill_dir:
                    continue
                try:
                    raw = view.read_text(f"{skill_dir}/SKILL.md")
                    skill_md = parse_skill_md(raw)
                except Exception:
                    continue
                items.append(
                    SkillSummary(
                        skill_path=skill_dir,
                        name=skill_md.frontmatter.name,
                        description=skill_md.frontmatter.description,
                    )
                )
            return sorted(items, key=lambda s: s.skill_path)

    @toolset.tool
    async def search_skills(query: str, top_k: int = 8) -> list[SkillSearchResult]:
        """Search the enabled skills to find potentially relevant skills."""
        candidate = _candidate()
        with apply_candidate_to_skills(skills_fs, candidate.components) as view:
            return await backend.search(
                query=query,
                top_k=top_k,
                fs=view,
                candidate=candidate.components,
            )

    @toolset.tool
    def load_skill(skill_path: str) -> SkillLoadResult:
        """Load the full SKILL.md for a skill."""
        candidate = _candidate()
        with apply_candidate_to_skills(skills_fs, candidate.components) as view:
            normalized = _resolve_skill_dir(view, skill_path)
            path = f"{normalized}/SKILL.md"
            content = view.read_text(path)
        return SkillLoadResult(
            skill_path=normalized,
            content=content,
            content_hash=_hash_text(content),
        )

    @toolset.tool
    def load_skill_file(skill_path: str, path: str) -> SkillFileResult:
        """Load a file within a skill directory."""
        candidate = _candidate()
        with apply_candidate_to_skills(skills_fs, candidate.components) as view:
            normalized_skill = _resolve_skill_dir(view, skill_path)
            try:
                normalized_file = normalize_rel_path(path)
            except Exception as e:
                raise ModelRetry(
                    f"Invalid path={path!r}: {e}. Use load_skill(...) to find valid file paths within a skill."
                ) from e
            full_path = f"{normalized_skill}/{normalized_file}"
            if not view.exists(full_path):
                raise ModelRetry(
                    f"Unknown file path={normalized_file!r} for skill_path={normalized_skill!r}. "
                    "Use load_skill(...) to find valid file paths within a skill."
                )
            content = view.read_text(full_path)
        return SkillFileResult(
            skill_path=normalized_skill,
            file_path=normalized_file,
            content=content,
            content_hash=_hash_text(content),
        )

    @toolset.tool
    def activate_skill_components(
        skill_path: str, include_examples: bool = False
    ) -> list[str]:
        """Activate a skill's components so they can be optimized as `skill:*` entries."""
        candidate = _candidate()
        with apply_candidate_to_skills(skills_fs, candidate.components) as view:
            resolved_path = _resolve_skill_dir(view, skill_path)
            new_components = materialize_skill_components_for_path(
                view,
                skill_path=resolved_path,
                include_examples=include_examples,
            )

        state.activate_skill_path(resolved_path)

        activated: list[str] = []
        for program in state.candidates:
            for key, value in new_components.items():
                if key in program.components:
                    continue
                program.components[key] = ComponentValue(
                    name=key,
                    text=value.text,
                    version=0,
                    metadata=None,
                )
                activated.append(key)

        if deps.seed_candidate is not None:
            for key, value in new_components.items():
                deps.seed_candidate.setdefault(
                    key,
                    ComponentValue(
                        name=key,
                        text=value.text,
                        version=0,
                        metadata=None,
                    ),
                )

        return sorted(set(activated))

    @toolset.tool
    def create_skill(skill_path: str, description: str, body: str) -> list[str]:
        """Create a new skill from scratch and make its components available to edit. Use this to abstract reusable strategies."""
        try:
            resolved_path = normalize_rel_path(skill_path)
        except Exception as e:
            raise ModelRetry(f"Invalid skill_path: {e}")

        desc_key = skill_description_key(resolved_path)
        body_key_name = skill_body_key(resolved_path)

        new_components = {
            desc_key: ComponentValue(
                name=desc_key, text=description, version=0, metadata=None
            ),
            body_key_name: ComponentValue(
                name=body_key_name, text=body, version=0, metadata=None
            ),
        }

        state.activate_skill_path(resolved_path)

        activated: list[str] = []
        for program in state.candidates:
            for key, value in new_components.items():
                if key in program.components:
                    continue
                program.components[key] = value
                activated.append(key)

        if deps.seed_candidate is not None:
            for key, value in new_components.items():
                deps.seed_candidate.setdefault(key, value)

        return sorted(set(activated))

    @toolset.tool
    def activate_skill(skill_path: str, include_examples: bool = False) -> list[str]:
        """Deprecated alias for activate_skill_components."""
        logfire.warn(
            "activate_skill is deprecated; use activate_skill_components",
            skill_path=skill_path,
            include_examples=include_examples,
        )
        return activate_skill_components(skill_path, include_examples=include_examples)

    @toolset.tool
    def list_active_skills() -> list[str]:
        """List skill paths that have been activated so far."""
        return sorted(state.active_skill_paths)

    return toolset


def _build_reflective_dataset(
    *,
    deps: GepaDeps,
    state: GepaState,
    candidate: CandidateProgram,
    eval_results: EvaluationResults[str],
    components: Sequence[str],
) -> ReflectiveDataset:
    eval_batch = EvaluationBatch(
        outputs=list(eval_results.outputs),
        scores=list(eval_results.scores),
        trajectories=list(eval_results.trajectories)
        if eval_results.trajectories is not None
        else None,
    )

    reflection_config = state.config.reflection_config
    raw_dataset = deps.adapter.make_reflective_dataset(
        candidate=candidate.components,
        eval_batch=eval_batch,
        components_to_update=components,
        include_case_metadata=reflection_config.include_case_metadata
        if reflection_config
        else False,
        include_expected_output=reflection_config.include_expected_output
        if reflection_config
        else False,
    )
    # Preserve shared datasets to avoid duplicating identical traces per component
    # (especially when using the "all" component selector). When adapters return a
    # SharedReflectiveDataset, show the traces once in the prompt instead of
    # repeating them under every component section.
    if isinstance(raw_dataset, SharedReflectiveDataset):
        records = list(raw_dataset.records)
        sampler = state.config.reflection_sampler
        if sampler is not None:
            max_records = state.config.reflection_sampler_max_records
            records = sampler(records, max_records)
        return SharedReflectiveDataset(records=records)

    records_by_component = {
        component: list(raw_dataset.records_by_component.get(component, []))
        for component in components
    }

    sampler = state.config.reflection_sampler
    if sampler is not None:
        max_records = state.config.reflection_sampler_max_records
        records_by_component = {
            component: sampler(records, max_records)
            for component, records in records_by_component.items()
        }

    return ComponentReflectiveDataset(records_by_component=records_by_component)


def _resolve_model(
    deps: GepaDeps,
) -> Model | KnownModelName | str:
    model = deps.model
    if model is None:
        raise ValueError("model must be configured before running reflection.")
    return model


async def _propose_new_texts(
    *,
    deps: GepaDeps,
    state: GepaState,
    parent: CandidateProgram,
    reflective_dataset: ReflectiveDataset,
    components: Sequence[str] | None,
    model: Model | KnownModelName | str,
    model_settings: ModelSettings | None = None,
    component_toolsets: Sequence[FunctionToolset] | None = None,
) -> ProposalResult:
    proposal = deps.proposal_generator
    kwargs: dict[str, Any] = dict(
        candidate=parent,
        reflective_data=reflective_dataset,
        components=components,
        model=model,
        model_settings=model_settings,
        example_bank=parent.example_bank,
    )
    if component_toolsets is not None:
        propose_sig = inspect.signature(proposal.propose_texts)
        if "component_toolsets" in propose_sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in propose_sig.parameters.values()
        ):
            kwargs["component_toolsets"] = component_toolsets

    return await proposal.propose_texts(**kwargs)


def _create_candidate(
    *,
    state: GepaState,
    parent: CandidateProgram,
    parent_idx: int,
    new_texts: Mapping[str, str],
    metadata: Mapping[str, dict[str, Any]] | None = None,
) -> CandidateProgram:
    new_components: CandidateMap = {}
    for name, value in parent.components.items():
        new_components[name] = ComponentValue(
            name=name,
            text=value.text,
            version=value.version,
            metadata=None if value.metadata is None else dict(value.metadata),
        )
    for name, text in new_texts.items():
        existing = parent.components.get(name)
        base_version = existing.version if existing is not None else 0
        component_metadata = None
        if metadata and name in metadata:
            component_metadata = dict(metadata[name])
            if state.iteration >= 0:
                component_metadata.setdefault("iteration", state.iteration)
        new_components[name] = ComponentValue(
            name=name,
            text=text,
            version=base_version + 1,
            metadata=component_metadata,
        )
    # Copy example bank from parent (already modified by reflection via tool calls)
    example_bank = None
    if parent.example_bank is not None:
        example_bank = parent.example_bank.copy()

    return CandidateProgram(
        idx=len(state.candidates),
        components=new_components,
        creation_type="reflection",
        parent_indices=[parent_idx],
        discovered_at_iteration=state.iteration,
        discovered_at_evaluation=state.total_evaluations,
        example_bank=example_bank,
    )


def _is_strict_improvement(
    *,
    baseline_scores: Sequence[float],
    new_scores: Sequence[float],
) -> bool:
    improvement = sum(new_scores) - sum(baseline_scores)
    return improvement > _IMPROVEMENT_EPSILON


def _component_versions(candidate: CandidateProgram) -> Mapping[str, str]:
    return {name: component.text for name, component in candidate.components.items()}


__all__ = ["reflect_step"]
