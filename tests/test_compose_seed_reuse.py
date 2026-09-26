"""Composed sequential slices reuse the incumbent's helper-paid seed score.

Every engine slice used to re-score its seed on the full validation set even
though ``optimize_sequential`` and ``optimize_adaptive_sequential`` had
already paid the comparison budget for exactly that evaluation. These tests
pin the reuse: engine slices never spend metric calls on the seed, budget
reconciliation still balances, and the seeded views expose nothing beyond
what each engine is trusted with.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa import compose
from pydantic_ai_gepa.compose import (
    _engine_task_view,
    _InterruptedEvaluation,
    optimize_adaptive_sequential,
    optimize_sequential,
)
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    CandidateEvaluation,
    EngineConfig,
    EngineResult,
    GepaEngine,
    OptimizationTask,
    get_engine,
)
from pydantic_ai_gepa.engines.registry import _ENGINES
from pydantic_ai_gepa.engines.base import ValidationScore
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    ProposalResult,
)
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput


def _candidate(text: str) -> CandidateMap:
    return {"instructions": ComponentValue(name="instructions", text=text)}


def _counting_task(
    score_fn: Any,
    *,
    val_cases: int = 2,
    train_cases: int = 2,
) -> tuple[OptimizationTask, list[tuple[str, str]]]:
    """A deterministic task recording (case name, active candidate) per call."""
    agent = Agent(TestModel(custom_output_text="response"), instructions="candidate-0")
    calls: list[tuple[str, str]] = []

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        active = override.value if override is not None else agent._instructions
        # Active instructions render as a repr of sourced entries; reduce it
        # to the canonical candidate label for stable assertions.
        match = re.search(r"candidate-\d+|improved-\d+|improved", str(active))
        label = match.group(0) if match else str(active)
        calls.append((case.name or "", label))
        return MetricResult(
            score=score_fn(label),
            side_info={"scores": {"quality": 0.75}, "selectable": True},
        )

    train = [Case(name=f"train-{i}", inputs="input") for i in range(train_cases)]
    validation = [Case(name=f"val-{i}", inputs="input") for i in range(val_cases)]
    task = OptimizationTask(
        agent=agent, trainset=train, valset=validation, metric=metric
    )
    return task, calls


def _improving_score(text: str) -> float:
    match = re.search(r"candidate-(\d+)", text)
    return int(match.group(1)) / 10 if match else 0.0


@pytest.mark.asyncio
async def test_adaptive_best_of_n_slices_never_rescore_the_seed() -> None:
    proposals = 0

    async def propose(seed: CandidateMap) -> CandidateMap:
        nonlocal proposals
        proposals += 1
        return _candidate(f"candidate-{proposals}")

    task, calls = _counting_task(_improving_score, val_cases=2)
    config = EngineConfig(
        engine="best_of_n",
        max_metric_calls=4,
        engine_config={"n": 1, "propose": propose},
    )

    result = await optimize_adaptive_sequential(task, [config], max_metric_calls=12)

    # Three improving slices of one fresh engine instance each.
    assert len(result.results) == 3
    # Each slice evaluates only its proposed variant (2 val cases); the seed's
    # 2-call validation pass came from the comparison budget. Before the reuse
    # seam each slice reported 4.
    assert [item.num_metric_calls for item in result.results] == [2, 2, 2]
    assert result.total_metric_calls == 6
    # One seed comparison plus one per slice, each over 2 validation cases.
    assert result.comparison_metric_calls == 8
    assert result.accounted_metric_calls == len(calls) == 14
    assert result.best.best_candidate == _candidate("candidate-3")
    assert result.fair_scores == [0.1, 0.2, 0.3]
    # No slice re-scored its seed: the initial seed is only scored by the
    # opening comparison, and each adopted variant only by its own slice (as
    # the proposal) plus its comparison — never again as the next slice's seed.
    validation_calls = Counter(text for name, text in calls if name.startswith("val-"))
    assert validation_calls == {
        "candidate-0": 2,
        "candidate-1": 4,
        "candidate-2": 4,
        "candidate-3": 4,
    }


@pytest.mark.asyncio
async def test_adaptive_coding_agent_slices_never_rescore_the_seed() -> None:
    async def propose(context: Any) -> CandidateMap:
        return _candidate("candidate-1")

    task, calls = _counting_task(lambda text: 0.5, val_cases=2)
    configs = [
        EngineConfig(
            engine="coding_agent",
            max_metric_calls=7,
            max_iterations=1,
            engine_config={
                "propose": propose,
                "minibatch_size": 1,
                "acceptance_repetitions": 2,
                "acceptance_max_repetitions": 2,
            },
        )
        for _ in range(3)
    ]

    result = await optimize_adaptive_sequential(
        task, configs, max_metric_calls=21, patience=1
    )

    assert len(result.results) == 3
    # Each slice runs one iteration: selection minibatch (1) + two baseline
    # repetitions (2) + two proposal repetitions (2), all on training cases.
    # The seed's 2-call validation pass is reused. Before: 7 per slice.
    assert [item.num_metric_calls for item in result.results] == [5, 5, 5]
    assert result.total_metric_calls == 15
    assert result.comparison_metric_calls == 8
    assert result.accounted_metric_calls == len(calls) == 23
    # Every validation metric call belongs to the comparison budget.
    validation_calls = [name for name, _ in calls if name.startswith("val-")]
    assert len(validation_calls) == 8
    for item in result.results:
        seed_events = [
            event
            for event in item.history
            if event.kind == "validation_evaluated" and event.data["stage"] == "seed"
        ]
        assert len(seed_events) == 1
        assert seed_events[0].data["reused"] is True


@pytest.mark.asyncio
async def test_adaptive_gepa_slices_never_rescore_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposals = 0

    async def propose_texts(self: Any, **kwargs: Any) -> ProposalResult:
        nonlocal proposals
        proposals += 1
        # Distinct texts: the graph drops proposals identical to their parent.
        return ProposalResult(
            texts={"instructions": f"improved-{proposals}"},
            component_metadata={},
            reasoning=None,
        )

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose_texts)

    def score_fn(text: str) -> float:
        match = re.search(r"improved-(\d+)", text)
        return 0.4 + 0.1 * int(match.group(1)) if match else 0.4

    task, calls = _counting_task(score_fn, val_cases=3, train_cases=2)
    config = EngineConfig(
        engine="gepa",
        max_metric_calls=10,
        max_iterations=1,
        engine_config={
            "reflection_minibatch_size": 2,
            "reflection_config": ReflectionConfig(model=TestModel()),
        },
    )

    result = await optimize_adaptive_sequential(task, [config], max_metric_calls=30)

    # Three slices: every slice's proposal improves, so it is adopted and its
    # score seeds the next slice.
    assert len(result.results) == 3
    # Each slice: parent minibatch (2) + proposal minibatch (2) + the new
    # candidate's validation (3). The seed's 3-call validation pass is reused.
    # Before: 10 per slice.
    assert [item.num_metric_calls for item in result.results] == [7, 7, 7]
    assert result.total_metric_calls == 21
    assert result.comparison_metric_calls == 12
    assert result.accounted_metric_calls == len(calls) == 33
    assert result.best.best_candidate["instructions"].text == "improved-3"
    assert result.best.best_score == pytest.approx(0.7)
    assert result.fair_scores == pytest.approx([0.5, 0.6, 0.7])
    # No slice re-scored its seed: the initial seed is only scored by the
    # opening comparison, and each adopted candidate only by its own slice's
    # validation plus its comparison — never again as the next slice's seed.
    validation_calls = Counter(text for name, text in calls if name.startswith("val-"))
    assert validation_calls == {
        "candidate-0": 3,
        "improved-1": 6,
        "improved-2": 6,
        "improved-3": 6,
    }


@pytest.mark.asyncio
async def test_untrusted_seeded_view_leaks_no_per_case_seed_data() -> None:
    task, _ = _counting_task(_improving_score, val_cases=2)
    seed = await task.seed_candidate()
    evaluation = await task.evaluate(seed)
    assert evaluation.records and evaluation.per_case_scores

    async def propose(seed_candidate: CandidateMap) -> CandidateMap:
        return seed_candidate

    engine = get_engine(
        "best_of_n",
        EngineConfig(engine="best_of_n", engine_config={"propose": propose}),
    )
    view = _engine_task_view(task, engine, seed, seed_evaluation=evaluation)

    score = await view.seed_validation_score()
    assert isinstance(score, ValidationScore)
    assert asdict(score) == {
        "score": evaluation.score,
        "num_cases": evaluation.num_cases,
        "selectable": evaluation.selectable,
        "objective_scores": dict(evaluation.objective_scores),
    }
    for name in (
        "records",
        "outputs",
        "traces",
        "inputs",
        "side_info",
        "per_case_scores",
        "per_case_objective_scores",
    ):
        assert not hasattr(score, name)

    # The view stores no CandidateEvaluation at all: its seed-related private
    # state is the aggregate ValidationScore, and the only other privates are
    # the task itself and the captured validation context.
    stored = vars(view)
    assert set(stored) == {
        "_EngineTaskView__task",
        "_EngineTaskView__seed_score",
        "_EngineTaskView__validation_context",
    }
    assert not any(isinstance(value, CandidateEvaluation) for value in stored.values())
    seed_value = stored["_EngineTaskView__seed_score"]
    assert isinstance(seed_value, ValidationScore)
    assert asdict(seed_value) == asdict(score)
    public = {name for name in dir(view) if not name.startswith("_")}
    assert public == {
        "seed_candidate",
        "seed_validation_score",
        "train_loader",
        "validation_case_count",
        "score_validation",
        "evaluate",
        "concurrency",
        "test_set",
    }
    for name in (
        "seed_evaluation",
        "records",
        "outputs",
        "traces",
        "side_info",
        "per_case_scores",
        "per_case_objective_scores",
        "valset",
        "val_loader",
        "agent",
        "metric",
    ):
        with pytest.raises(AttributeError):
            getattr(view, name)

    # Without a pre-scored seed the accessor says so instead of failing.
    bare = _engine_task_view(task, engine, seed)
    assert await bare.seed_validation_score() is None


@pytest.mark.asyncio
async def test_trusted_seed_evaluation_matches_evaluate_field_for_field() -> None:
    task, _ = _counting_task(_improving_score, val_cases=2)
    seed = await task.seed_candidate()
    # The helper scores the seed exactly where the engine's own trusted-view
    # evaluation would: no validation_evaluation() context on either path.
    evaluation = await task.evaluate(seed)
    engine = GepaEngine(EngineConfig(engine="gepa", max_metric_calls=1))
    view = _engine_task_view(task, engine, seed, seed_evaluation=evaluation)

    reused = await view.seed_evaluation()
    assert reused is not None
    reference = await view.evaluate(seed)
    assert reused.score == reference.score == evaluation.score
    assert reused.num_cases == reference.num_cases
    assert reused.selectable == reference.selectable
    assert reused.objective_scores == reference.objective_scores
    assert reused.per_case_scores == reference.per_case_scores
    assert reused.per_case_objective_scores == reference.per_case_objective_scores
    # Scores only: no records, outputs, traces or feedback.
    assert reused.records == []
    assert reused.side_info == {}
    assert reused.per_case_scores is not evaluation.per_case_scores

    # The view itself stores only reduced values: the scores-only copy (never
    # the helper's full evaluation) on the trusted side, and just the
    # aggregate ValidationScore on the base.
    stored = vars(view)["_TrustedEngineTaskView__seed_evaluation"]
    assert isinstance(stored, CandidateEvaluation)
    assert stored is not evaluation
    assert stored.records == []
    assert stored.side_info == {}
    assert isinstance(vars(view)["_EngineTaskView__seed_score"], ValidationScore)
    # Each accessor call returns a fresh copy that callers cannot use to
    # tamper with the stored one.
    reused.per_case_scores["tamper"] = 9.9
    fresh = await view.seed_evaluation()
    assert fresh is not None and fresh is not reused
    assert "tamper" not in fresh.per_case_scores
    assert "tamper" not in stored.per_case_scores

    # The inherited aggregate accessor matches too, and both default to None.
    assert await view.seed_validation_score() is not None
    bare = _engine_task_view(task, engine, seed)
    assert await bare.seed_evaluation() is None


@pytest.mark.asyncio
async def test_proposer_cannot_mutate_the_incumbent_away_from_its_score() -> None:
    agent = Agent(TestModel(custom_output_text="response"), instructions="candidate-0")
    seen_seeds: list[str] = []

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        active = override.value if override is not None else agent._instructions
        text = str(active)
        score = 0.5 if "candidate-0" in text else 0.1 if "worse" in text else 0.0
        return MetricResult(score=score)

    task = OptimizationTask(
        agent=agent,
        trainset=[Case(name="train", inputs="input")],
        valset=[Case(name="val", inputs="input")],
        metric=metric,
    )

    async def propose(seed: CandidateMap) -> CandidateMap:
        seen_seeds.append(seed["instructions"].text)
        # In-place attack on the seed the engine handed to the proposer.
        seed["instructions"].text = "mutated"
        return _candidate("worse")

    config = EngineConfig(
        engine="best_of_n",
        max_metric_calls=2,
        engine_config={"n": 1, "propose": propose},
    )

    result = await optimize_adaptive_sequential(
        task, [config, config], max_metric_calls=4, patience=1
    )

    # Two slices ran; each slice's proposer received the unmutated incumbent.
    assert len(result.results) == 2
    assert seen_seeds == ["candidate-0", "candidate-0"]
    # Inside each slice the reused seed score still describes candidate zero,
    # which the proposer's in-place edit could not reach, so each engine
    # returns the unmutated seed with the score it actually has.
    for item in result.results:
        assert item.history[-1].data["candidate_scores"] == [0.5, 0.1]
        assert item.best_candidate == _candidate("candidate-0")
        assert item.best_score == 0.5
    assert result.fair_scores == [0.5, 0.5]
    assert result.best.engine == "seed"
    assert result.best.best_candidate == _candidate("candidate-0")
    assert result.best.best_score == 0.5


@pytest.mark.asyncio
async def test_engines_run_directly_still_score_their_seed() -> None:
    async def propose(seed: CandidateMap) -> CandidateMap:
        return _candidate("candidate-1")

    task, calls = _counting_task(_improving_score, val_cases=2)
    config = EngineConfig(
        engine="best_of_n",
        max_metric_calls=4,
        engine_config={"n": 1, "propose": propose},
    )

    result = await get_engine("best_of_n", config).run(task, config, BudgetTracker(4))

    # No composition view: the engine pays for the seed validation as before.
    assert result.num_metric_calls == 4
    seed_calls = [
        name
        for name, text in calls
        if name.startswith("val-") and text == "candidate-0"
    ]
    assert len(seed_calls) == 2


@pytest.mark.asyncio
async def test_sequential_best_of_n_stages_never_rescore_the_seed() -> None:
    # Stage two's proposal looks better inside its slice (0.9) than under the
    # harness comparison (0.6), so the stage is rejected and stage three must
    # still be seeded from stage one's adopted candidate and score.
    # Iterators are consumed per case: two val cases per evaluation.
    scores: dict[str, Any] = {
        "candidate-0": 0.5,
        "candidate-1": 0.7,
        "candidate-2": iter([0.9, 0.9, 0.6, 0.6]),
        "candidate-3": 0.8,
    }

    def score_fn(text: str) -> float:
        value = scores[text]
        return next(value) if isinstance(value, Iterator) else value

    def stage(proposal: str) -> EngineConfig:
        async def propose(seed: CandidateMap) -> CandidateMap:
            return _candidate(proposal)

        return EngineConfig(
            engine="best_of_n",
            max_metric_calls=4,
            engine_config={"n": 1, "propose": propose},
        )

    task, calls = _counting_task(score_fn, val_cases=2)
    result = await optimize_sequential(
        task,
        [stage("candidate-1"), stage("candidate-2"), stage("candidate-3")],
        max_metric_calls=12,
    )

    assert len(result.results) == 3
    # Each stage evaluates only its proposed variant (2 val cases); the seed's
    # 2-call validation pass came from the comparison budget. Before the reuse
    # seam each stage reported 4.
    assert [item.num_metric_calls for item in result.results] == [2, 2, 2]
    assert result.total_metric_calls == 6
    # One seed comparison plus one per stage, each over 2 validation cases.
    assert result.comparison_metric_calls == 8
    assert result.accounted_metric_calls == len(calls) == 14
    # The reused in-slice seed scores are exactly the comparison scores of the
    # adopted incumbents, including after stage two's rejection.
    assert [item.history[-1].data["candidate_scores"] for item in result.results] == [
        [0.5, 0.7],
        [0.7, 0.9],
        [0.7, 0.8],
    ]
    assert result.fair_scores == [0.7, 0.6, 0.8]
    assert [phase["adopted"] for phase in result.phases] == [True, False, True]
    assert result.best_index == 2
    assert result.best.best_candidate == _candidate("candidate-3")
    # No stage re-scored its seed: the initial seed is only scored by the
    # opening comparison, and each adopted variant only by its own stage (as
    # the proposal) plus its comparison — never again as the next stage's seed.
    validation_calls = Counter(text for name, text in calls if name.startswith("val-"))
    assert validation_calls == {
        "candidate-0": 2,
        "candidate-1": 4,
        "candidate-2": 4,
        "candidate-3": 4,
    }


@pytest.mark.asyncio
async def test_sequential_gepa_stages_never_rescore_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposals = 0

    async def propose_texts(self: Any, **kwargs: Any) -> ProposalResult:
        nonlocal proposals
        proposals += 1
        # Distinct texts: the graph drops proposals identical to their parent.
        return ProposalResult(
            texts={"instructions": f"improved-{proposals}"},
            component_metadata={},
            reasoning=None,
        )

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose_texts)

    def score_fn(text: str) -> float:
        match = re.search(r"improved-(\d+)", text)
        return 0.4 + 0.1 * int(match.group(1)) if match else 0.4

    task, calls = _counting_task(score_fn, val_cases=3, train_cases=2)
    configs = [
        EngineConfig(
            engine="gepa",
            max_metric_calls=10,
            max_iterations=1,
            engine_config={
                "reflection_minibatch_size": 2,
                "reflection_config": ReflectionConfig(model=TestModel()),
            },
        )
        for _ in range(3)
    ]

    result = await optimize_sequential(task, configs, max_metric_calls=30)

    # Three stages: every stage's proposal improves, so it is adopted and its
    # comparison evaluation seeds the next stage.
    assert len(result.results) == 3
    # Each stage: parent minibatch (2) + proposal minibatch (2) + the new
    # candidate's validation (3). The seed's 3-call validation pass is reused.
    # Before: 10 per stage.
    assert [item.num_metric_calls for item in result.results] == [7, 7, 7]
    assert result.total_metric_calls == 21
    assert result.comparison_metric_calls == 12
    assert result.accounted_metric_calls == len(calls) == 33
    assert result.best.best_candidate["instructions"].text == "improved-3"
    assert result.best.best_score == pytest.approx(0.7)
    assert result.fair_scores == pytest.approx([0.5, 0.6, 0.7])
    # No stage re-scored its seed: the initial seed is only scored by the
    # opening comparison, and each adopted candidate only by its own stage's
    # validation plus its comparison — never again as the next stage's seed.
    validation_calls = Counter(text for name, text in calls if name.startswith("val-"))
    assert validation_calls == {
        "candidate-0": 3,
        "improved-1": 6,
        "improved-2": 6,
        "improved-3": 6,
    }


@pytest.mark.asyncio
async def test_sequential_coding_agent_stages_never_rescore_the_seed() -> None:
    contexts: list[Any] = []

    async def propose(context: Any) -> CandidateMap:
        contexts.append(context)
        return _candidate("candidate-1")

    task, calls = _counting_task(lambda text: 0.5, val_cases=2)
    configs = [
        EngineConfig(
            engine="coding_agent",
            max_metric_calls=7,
            max_iterations=1,
            engine_config={
                "propose": propose,
                "minibatch_size": 1,
                "acceptance_repetitions": 2,
                "acceptance_max_repetitions": 2,
            },
        )
        for _ in range(3)
    ]

    result = await optimize_sequential(task, configs, max_metric_calls=21)

    assert len(result.results) == 3
    # Each stage runs one iteration: selection minibatch (1) + two baseline
    # repetitions (2) + two proposal repetitions (2), all on training cases.
    # The seed's 2-call validation pass is reused. Before: 7 per stage.
    assert [item.num_metric_calls for item in result.results] == [5, 5, 5]
    assert result.total_metric_calls == 15
    assert result.comparison_metric_calls == 8
    assert result.accounted_metric_calls == len(calls) == 23
    # Every validation metric call belongs to the comparison budget.
    validation_calls = [name for name, _ in calls if name.startswith("val-")]
    assert len(validation_calls) == 8
    for item in result.results:
        seed_events = [
            event
            for event in item.history
            if event.kind == "validation_evaluated" and event.data["stage"] == "seed"
        ]
        assert len(seed_events) == 1
        assert seed_events[0].data["reused"] is True
    # The reused scores reach selection only: proposal packets still carry
    # training-minibatch evidence and nothing from the validation cases.
    assert len(contexts) == 3
    assert all("val-" not in repr(context) for context in contexts)


@pytest.mark.asyncio
async def test_sequential_stage_views_match_engine_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, calls = _counting_task(_improving_score, val_cases=2)
    untrusted_views: list[Any] = []

    class Probe:
        name = "seed_reuse_probe"

        def __init__(self, config: EngineConfig) -> None:
            pass

        async def run(
            self, view: Any, config: EngineConfig, budget: BudgetTracker
        ) -> EngineResult:
            untrusted_views.append(view)
            return EngineResult(
                engine=self.name,
                best_candidate=await view.seed_candidate(),
                best_score=None,
                num_metric_calls=0,
            )

    monkeypatch.setitem(_ENGINES, Probe.name, Probe)
    trusted_views: list[Any] = []
    original_run = GepaEngine.run

    async def capturing_run(
        self: GepaEngine,
        view: OptimizationTask,
        config: EngineConfig,
        budget: BudgetTracker,
    ) -> EngineResult:
        trusted_views.append(view)
        return await original_run(self, view, config, budget)

    monkeypatch.setattr(GepaEngine, "run", capturing_run)

    result = await optimize_sequential(
        task,
        [
            EngineConfig(engine=Probe.name, max_metric_calls=2),
            EngineConfig(
                engine="gepa",
                max_metric_calls=2,
                max_iterations=1,
                engine_config={
                    "reflection_config": ReflectionConfig(model=TestModel())
                },
            ),
        ],
        max_metric_calls=4,
    )

    # The probe returned the seed unchanged (adopted on the tie), so the gepa
    # stage is seeded with that candidate's comparison evaluation. Its slice
    # cannot afford a minibatch, so every call belongs to a comparison.
    assert len(untrusted_views) == len(trusted_views) == 1
    assert result.accounted_metric_calls == len(calls) == 6

    # An untrusted stage gets only the aggregate the helper already paid for.
    untrusted = untrusted_views[0]
    aggregate = await untrusted.seed_validation_score()
    assert isinstance(aggregate, ValidationScore)
    assert asdict(aggregate) == {
        "score": 0.0,
        "num_cases": 2,
        "selectable": True,
        "objective_scores": {"quality": 0.75},
    }
    with pytest.raises(AttributeError):
        getattr(untrusted, "seed_evaluation")
    assert not any(
        isinstance(value, CandidateEvaluation) for value in vars(untrusted).values()
    )

    # The trusted stage's seed_evaluation() is a scores-only copy equal field
    # for field to what its own evaluate() returns for the same candidate.
    trusted = trusted_views[0]
    reused = await trusted.seed_evaluation()
    assert reused is not None
    reference = await trusted.evaluate(await trusted.seed_candidate())
    assert reused.score == reference.score
    assert reused.num_cases == reference.num_cases
    assert reused.selectable == reference.selectable
    assert reused.objective_scores == reference.objective_scores
    assert reused.per_case_scores == reference.per_case_scores
    assert reused.per_case_objective_scores == reference.per_case_objective_scores
    assert reused.records == []
    assert reused.side_info == {}


@pytest.mark.asyncio
async def test_sequential_interrupted_opening_evaluation_is_not_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_evaluate = compose._evaluate
    opening = True

    async def interrupted_once(
        task: OptimizationTask, candidate: CandidateMap, **kwargs: Any
    ) -> CandidateEvaluation:
        nonlocal opening
        if opening:
            opening = False
            return _InterruptedEvaluation(
                score=0.0, records=[], side_info={}, num_cases=0, selectable=False
            )
        return await real_evaluate(task, candidate, **kwargs)

    monkeypatch.setattr(compose, "_evaluate", interrupted_once)

    async def propose(seed: CandidateMap) -> CandidateMap:
        return _candidate("candidate-1")

    task, calls = _counting_task(_improving_score, val_cases=2)
    config = EngineConfig(
        engine="best_of_n",
        max_metric_calls=4,
        engine_config={"n": 1, "propose": propose},
    )

    result = await optimize_sequential(task, [config], max_metric_calls=4)

    assert len(result.results) == 1
    # No helper-paid evaluation survived the interruption, so the stage scores
    # its own seed (2 calls) plus its proposal (2), exactly as an unseeded run.
    assert result.results[0].num_metric_calls == 4
    # Only the stage's comparison actually ran.
    assert result.comparison_metric_calls == 2
    assert result.accounted_metric_calls == len(calls) == 6
    validation_calls = Counter(text for name, text in calls if name.startswith("val-"))
    assert validation_calls == {"candidate-0": 2, "candidate-1": 4}
    assert result.fair_scores == [0.1]
    assert result.best_index == 0
    assert result.best.best_candidate == _candidate("candidate-1")


@pytest.mark.asyncio
async def test_sequential_proposer_cannot_mutate_the_incumbent_away_from_its_score() -> (
    None
):
    agent = Agent(TestModel(custom_output_text="response"), instructions="candidate-0")
    seen_seeds: list[str] = []

    def metric(case: Case[str, str, Any], output: RolloutOutput[Any]) -> MetricResult:
        override = agent._override_instructions.get()
        active = override.value if override is not None else agent._instructions
        text = str(active)
        score = 0.5 if "candidate-0" in text else 0.1 if "worse" in text else 0.0
        return MetricResult(score=score)

    task = OptimizationTask(
        agent=agent,
        trainset=[Case(name="train", inputs="input")],
        valset=[Case(name="val", inputs="input")],
        metric=metric,
    )

    async def propose(seed: CandidateMap) -> CandidateMap:
        seen_seeds.append(seed["instructions"].text)
        # In-place attack on the seed the engine handed to the proposer.
        seed["instructions"].text = "mutated"
        return _candidate("worse")

    config = EngineConfig(
        engine="best_of_n",
        max_metric_calls=2,
        engine_config={"n": 1, "propose": propose},
    )

    result = await optimize_sequential(task, [config, config], max_metric_calls=4)

    # Two stages ran; each stage's proposer received the unmutated incumbent.
    assert len(result.results) == 2
    assert seen_seeds == ["candidate-0", "candidate-0"]
    # Inside each stage the reused seed score still describes candidate zero,
    # which the proposer's in-place edit could not reach, so each engine
    # returns the unmutated seed with the score it actually has.
    for item in result.results:
        assert item.history[-1].data["candidate_scores"] == [0.5, 0.1]
        assert item.best_candidate == _candidate("candidate-0")
        assert item.best_score == 0.5
    assert result.fair_scores == [0.5, 0.5]
    # Monotonic adoption accepts the tie: the last stage is the incumbent.
    assert result.best_index == 1
    assert result.best.best_candidate == _candidate("candidate-0")
    assert result.best.best_score == 0.5
