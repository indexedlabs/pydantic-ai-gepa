"""Composition gives arbitrary engines aggregate validation capabilities only."""

from dataclasses import asdict

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa._validation import validation_active
from pydantic_ai_gepa.compose import (
    OmniPlan,
    optimize_adaptive_sequential,
    optimize_best_of,
    optimize_omni,
    optimize_parallel,
    optimize_sequential,
    optimize_vote,
)
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    CodingAgentEngine,
    EngineConfig,
    EngineResult,
    GepaEngine,
    OptimizationTask,
    get_engine,
)
from pydantic_ai_gepa.engines.registry import _ENGINES
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig, RolloutOutput

HELPERS = [
    "parallel",
    "legacy_parallel",
    "best_of",
    "vote",
    "sequential",
    "adaptive",
    "omni",
]


@pytest.fixture(autouse=True)
def no_model_requests(monkeypatch):
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)


def candidate(text):
    return {"instructions": ComponentValue(name="instructions", text=text)}


def task():
    def metric(case, output):
        return MetricResult(
            score=0.5,
            feedback=f"{case.name} feedback",
            side_info={
                "detail": case.inputs,
                "scores": {"quality": 0.75},
                "selectable": True,
            },
        )

    return OptimizationTask(
        agent=Agent(TestModel(custom_output_text="answer"), instructions="seed"),
        trainset=[Case(name="training", inputs="training input")],
        valset=[
            Case(
                name="WITHHELD_CASE",
                inputs="WITHHELD_INPUT",
                expected_output="WITHHELD_GOLD",
            )
        ],
        test_set=[Case(name="REPORTING_CASE", inputs="REPORTING_INPUT")],
        metric=metric,
    )


async def compose(helper, task, configs):
    total = sum(config.max_metric_calls for config in configs)
    if helper == "omni":
        result = await optimize_omni(
            task,
            OmniPlan(
                phase_one=configs,
                phase_two=configs[-1],
                phase_one_metric_calls=total,
                phase_two_metric_calls=configs[-1].max_metric_calls,
            ),
        )
        return [item for item in result.results if item.engine != "seed"]
    if helper == "legacy_parallel":
        # One engine, with a configured slice larger than the shared budget.
        return await optimize_parallel(task, configs, max_metric_calls=total - 1)
    helper_fn = {
        "parallel": optimize_parallel,
        "best_of": optimize_best_of,
        "vote": optimize_vote,
        "sequential": optimize_sequential,
        "adaptive": optimize_adaptive_sequential,
    }[helper]
    kwargs = {"max_slices": len(configs)} if helper == "adaptive" else {}
    result = await helper_fn(task, configs, max_metric_calls=total, **kwargs)
    return result if isinstance(result, list) else result.results


def assert_aggregate(value):
    assert asdict(value) == {
        "score": 0.5,
        "num_cases": 1,
        "selectable": True,
        "objective_scores": {"quality": 0.75},
    }
    assert not any(
        hasattr(value, name)
        for name in (
            "records",
            "outputs",
            "side_info",
            "per_case_scores",
            "per_case_objective_scores",
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", HELPERS)
@pytest.mark.parametrize(
    "identity",
    [
        "custom",
        "gepa",
        "coding_agent",
        "gepa_subclass",
        "coding_subclass",
        "autoresearch",
    ],
)
async def test_engine_public_surface_withholds_validation(
    helper, identity, monkeypatch
):
    inspected = []

    async def inspect_view(view, config, budget):
        before = budget.spent
        seed = await view.seed_candidate()
        assert not hasattr(view, "val_loader")
        assert not hasattr(view, "valset")
        assert view.test_set is None
        assert await view.validation_case_count() == 1
        inspected.append(view)
        # Fail closed when a new public accessor appears without being audited.
        expected = {
            "seed_candidate",
            "train_loader",
            "validation_case_count",
            "score_validation",
            "evaluate",
            "agent",
            "metric",
            "input_type",
            "skills_fs",
            "skills_capabilities",
            "case_factory",
            "concurrency",
            "test_set",
        }
        public = {name for name in dir(view) if not name.startswith("_")}
        assert public == expected
        for name in sorted(public):
            value = getattr(view, name)
            if name in {"score_validation", "evaluate"}:
                kwargs = {"capture_traces": True} if name == "evaluate" else {}
                value = await value(seed, budget=budget, **kwargs)
                assert_aggregate(value)
            elif name in {"seed_candidate", "validation_case_count"}:
                value = await value()
            elif name == "train_loader":
                loader = await value()
                ids = await loader.all_ids()
                value = [ids, await loader.fetch(ids)]
            elif name == "metric":
                value = value(
                    Case(name="training", inputs="training"),
                    RolloutOutput.from_success("answer"),
                )
            assert "WITHHELD" not in repr(value), name
            assert "REPORTING" not in repr(value), name
        with pytest.raises(PermissionError):
            await view.evaluate(seed, dataset="test")
        return EngineResult(
            engine=config.engine,
            best_candidate=seed,
            best_score=0.5,
            num_metric_calls=budget.spent - before,
        )

    parent = {"gepa_subclass": GepaEngine, "coding_subclass": CodingAgentEngine}.get(
        identity, object
    )

    class ProbeEngine(parent):
        def __init__(self, config):
            self.name = config.engine

        run = staticmethod(inspect_view)

    name = identity
    options = {}
    if identity == "autoresearch":
        options["driver"] = inspect_view
    else:
        monkeypatch.setitem(_ENGINES, name, ProbeEngine)
    config = EngineConfig(engine=name, max_metric_calls=4, engine_config=options)
    results = await compose(helper, task(), [config])
    assert len(inspected) == (2 if helper == "omni" else 1)
    assert all(result.num_metric_calls == 2 for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", HELPERS)
async def test_score_validation_activates_context_inside_metric(helper, monkeypatch):
    observed = []
    evaluation_task = task()
    original = evaluation_task.metric

    def metric(case, output):
        observed.append(validation_active())
        return original(case, output)

    evaluation_task.metric = metric

    class Engine:
        name = "context_probe"

        def __init__(self, config):
            pass

        async def run(self, view, config, budget):
            seed = await view.seed_candidate()
            before = len(observed)
            value = await view.score_validation(seed, budget=budget)
            assert observed[before:] == [True]
            assert not validation_active()
            assert_aggregate(value)
            return EngineResult(
                engine=self.name,
                best_candidate=seed,
                best_score=value.score,
                num_metric_calls=1,
            )

    monkeypatch.setitem(_ENGINES, Engine.name, Engine)
    await compose(
        helper, evaluation_task, [EngineConfig(engine=Engine.name, max_metric_calls=2)]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", HELPERS)
async def test_every_builtin_composes_with_restricted_driver(helper):
    async def propose(value):
        return value.candidate if hasattr(value, "candidate") else value

    async def driver(view, config, budget):
        seed = await view.seed_candidate()
        value = await view.score_validation(seed, budget=budget)
        assert_aggregate(value)
        assert not hasattr(view, "val_loader")
        return EngineResult(
            engine="autoresearch",
            best_candidate=seed,
            best_score=value.score,
            num_metric_calls=1,
        )

    configs = [
        EngineConfig(
            engine="gepa",
            max_metric_calls=4,
            stop_at_score=0.5,
            engine_config={"reflection_config": ReflectionConfig(model=TestModel())},
        ),
        EngineConfig(
            engine="coding_agent",
            max_metric_calls=4,
            stop_at_score=0.5,
            engine_config={"propose": propose},
        ),
        EngineConfig(
            engine="best_of_n",
            max_metric_calls=4,
            engine_config={"propose": propose, "n": 1},
        ),
        EngineConfig(
            engine="autoresearch", max_metric_calls=4, engine_config={"driver": driver}
        ),
    ]
    results = await compose(helper, task(), configs)
    assert [result.engine for result in results] == [c.engine for c in configs] + (
        ["autoresearch"] if helper == "omni" else []
    )
    assert all(result.best_score == 0.5 for result in results)
    assert [result.num_metric_calls for result in results[:4]] == [1, 1, 2, 1]


@pytest.mark.asyncio
async def test_coding_pareto_parent_selection_survives_restricted_composition(
    monkeypatch,
):
    parents = []
    proposals = iter(["left", "right", "final"])
    evaluation_task = task()
    evaluation_task.valset = [
        Case(name="WITHHELD_LEFT", inputs="left"),
        Case(name="WITHHELD_RIGHT", inputs="right"),
    ]

    def metric(case, output):
        override = evaluation_task.agent._override_instructions.get()
        active = str(override.value) if override else "seed"
        name = next(
            (name for name in ("left", "right", "final") if name in active), "seed"
        )
        values = {
            "seed": [0.2, 0.2],
            "left": [0.9, 0.1],
            "right": [0.1, 0.8],
            "final": [0.3, 0.3],
        }
        score = (
            values[name][case.inputs == "right"]
            if case.name.startswith("WITHHELD")
            else {"seed": 0.1, "left": 0.3, "right": 0.5, "final": 0.7}[name]
        )
        return MetricResult(score=score, feedback=f"{case.name} feedback")

    evaluation_task.metric = metric

    async def propose(context):
        assert "WITHHELD" not in repr(context)
        parents.append(context.candidate["instructions"].text)
        return candidate(next(proposals))

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=30,
        max_iterations=3,
        seed=0,
        engine_config={
            "propose": propose,
            "minibatch_size": 1,
            "acceptance_repetitions": 2,
        },
    )
    direct = await get_engine("coding_agent", config).run(
        evaluation_task, config, BudgetTracker(30)
    )
    direct_parents = parents[:]
    assert direct_parents == ["seed", "seed", "left"]
    parents.clear()
    proposals = iter(["left", "right", "final"])

    # Run an arbitrary aggregate scorer alongside the trusted Pareto engine.
    class Scorer:
        name = "aggregate"

        def __init__(self, config):
            pass

        async def run(self, view, config, budget):
            seed = await view.seed_candidate()
            value = await view.score_validation(seed, budget=budget)
            assert value.score == 0.2
            assert not hasattr(value, "per_case_scores")
            return EngineResult(
                engine=self.name,
                best_candidate=seed,
                best_score=value.score,
                num_metric_calls=2,
            )

    monkeypatch.setitem(_ENGINES, Scorer.name, Scorer)
    results = await optimize_parallel(
        evaluation_task,
        [config, EngineConfig(engine=Scorer.name, max_metric_calls=2)],
        max_metric_calls=32,
    )
    assert parents == direct_parents
    assert results[0].best_candidate == direct.best_candidate == candidate("left")
    assert results[0].best_score == direct.best_score == 0.5
    assert results[0].num_metric_calls == direct.num_metric_calls
    assert [event.kind for event in results[0].history] == [
        event.kind for event in direct.history
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cache", [False, True])
async def test_aggregate_preserves_async_objectives_and_nonselectability(cache):
    from pydantic_ai_gepa.compose import _EngineTaskView

    evaluation_task = task()
    metric_result = MetricResult(
        score=0.5,
        feedback="WITHHELD feedback",
        side_info={
            "scores": {"quality": 0.75},
            "selectable": False,
            "detail": "WITHHELD detail",
        },
    )
    observed = []

    async def metric(case, output):
        observed.append(validation_active())
        return metric_result

    evaluation_task.metric = metric
    evaluation_task.evaluation_cache_identity = "fixed-test-v1"
    seed = await evaluation_task.seed_candidate()
    if cache:
        # A harness comparison may already have cached an unredacted result.
        raw = await evaluation_task.evaluate(seed, cache=True)
        assert "WITHHELD" in repr(raw)
    view = _EngineTaskView(evaluation_task)
    budget = BudgetTracker(1)
    value = await view.score_validation(seed, budget=budget, cache=cache)
    assert asdict(value) == {
        "score": 0.5,
        "num_cases": 1,
        "selectable": False,
        "objective_scores": {"quality": 0.75},
    }
    assert "WITHHELD" not in repr(value)
    assert budget.spent == (0 if cache else 1)
    assert observed == [not cache]
    assert metric_result.feedback == "WITHHELD feedback"
    value.objective_scores["quality"] = 99
    if cache:
        assert raw.objective_scores == {"quality": 0.75}
