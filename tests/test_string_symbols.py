"""Bridge between Git commits and CandidateMaps over declared string symbols."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa._validation import validation_evaluation
from pydantic_ai_gepa.compose import OmniPlan, optimize_omni
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    ReflectionContext,
    StringSymbolTask,
    get_engine,
)
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.gepa_graph.proposal import (
    InstructionProposalGenerator,
    ProposalResult,
)
from pydantic_ai_gepa.string_symbols import (
    StringSymbolError,
    candidate_from_commit,
    commit_from_candidate,
    string_symbol_component,
    verify_string_symbol_commit,
)
from pydantic_ai_gepa.types import MetricResult, ReflectionConfig

PROMPTS = '''"""Prompts for the fixture agent."""

import os

INSTRUCTIONS = "Be terse."  # keep this comment
GREETING: str = (
    "Hello, "
    "world."
)
FRAMED = """
Answer in one line.
"""


def helper() -> str:
    return INSTRUCTIONS + os.sep
'''

SYMBOLS = {"prompts.py": ["INSTRUCTIONS", "GREETING", "FRAMED"]}
INSTRUCTIONS = string_symbol_component("prompts.py", "INSTRUCTIONS")
GREETING = string_symbol_component("prompts.py", "GREETING")
FRAMED = string_symbol_component("prompts.py", "FRAMED")


def _git(repo: Path, *args: str, input: bytes | None = None) -> str:
    # A parent hook's GIT_DIR/GIT_INDEX_FILE must never redirect fixture git.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="Fixture",
        GIT_AUTHOR_EMAIL="fixture@example.com",
        GIT_COMMITTER_NAME="Fixture",
        GIT_COMMITTER_EMAIL="fixture@example.com",
    )
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input,
        env=env,
        check=True,
        capture_output=True,
    ).stdout.decode()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-gpg-sign", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Fixture")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "prompts.py").write_text(PROMPTS)
    (root / "README.md").write_text("fixture\n")
    _commit(root, "seed")
    # The library under test must not see a leaked repository location either.
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "not-a-repo"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "not-an-index"))
    return root


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def _candidate(**texts: str) -> CandidateMap:
    return {name: ComponentValue(name=name, text=text) for name, text in texts.items()}


def _texts(candidate: CandidateMap) -> dict[str, str]:
    return {name: component.text for name, component in candidate.items()}


def test_candidate_from_commit_reads_declared_string_constants(repo: Path) -> None:
    candidate = candidate_from_commit(repo, _head(repo), SYMBOLS)

    assert _texts(candidate) == {
        "py:prompts.py:INSTRUCTIONS": "Be terse.",
        "py:prompts.py:GREETING": "Hello, world.",
        "py:prompts.py:FRAMED": "Answer in one line.",
    }
    assert all(name == value.name for name, value in candidate.items())


def test_round_trip_rewrites_only_declared_strings(repo: Path) -> None:
    base = _head(repo)
    status_before = _git(repo, "status", "--porcelain")
    edited = _candidate(
        **{
            INSTRUCTIONS: 'Say "hi" \\ then\nexplain café ☕ in """two""" lines.',
            GREETING: "Hey'",
            FRAMED: "Answer in two lines.",
        }
    )

    sha = commit_from_candidate(repo, base, SYMBOLS, edited, "Try edited prompts")

    assert _texts(candidate_from_commit(repo, sha, SYMBOLS)) == _texts(edited)
    assert candidate_from_commit(repo, sha, SYMBOLS) == edited
    assert _git(repo, "rev-parse", f"{sha}^").strip() == base
    assert _git(repo, "log", "-1", "--format=%s", sha).strip() == "Try edited prompts"
    assert _git(repo, "diff", "--name-only", base, sha).split() == ["prompts.py"]
    # The user's checkout, index and branch are untouched.
    assert _head(repo) == base
    assert _git(repo, "status", "--porcelain") == status_before
    assert (repo / "prompts.py").read_text() == PROMPTS
    source = _git(repo, "show", f"{sha}:prompts.py")
    assert "# keep this comment" in source
    assert "def helper() -> str:\n    return INSTRUCTIONS + os.sep\n" in source
    # Framing whitespace around a stripped component value is preserved.
    assert 'FRAMED = """\nAnswer in two lines.\n"""' in source
    verify_string_symbol_commit(repo, base, sha, SYMBOLS)


def test_unchanged_candidate_returns_the_base_commit(repo: Path) -> None:
    base = _head(repo)
    seed = candidate_from_commit(repo, base, SYMBOLS)

    assert commit_from_candidate(repo, base, SYMBOLS, seed, "noop") == base
    partial = _candidate(**{GREETING: "Howdy."})
    sha = commit_from_candidate(repo, base, SYMBOLS, partial, "partial")
    assert _texts(candidate_from_commit(repo, sha, SYMBOLS)) == {
        INSTRUCTIONS: "Be terse.",
        GREETING: "Howdy.",
        FRAMED: "Answer in one line.",
    }


def test_a_bare_annotation_does_not_count_as_a_second_binding(repo: Path) -> None:
    (repo / "typed.py").write_text('X: str\nX = "value"\n')
    base = _commit(repo, "typed")

    candidate = candidate_from_commit(repo, base, {"typed.py": ["X"]})

    assert _texts(candidate) == {"py:typed.py:X": "value"}


def test_symbols_in_a_subdirectory_project_are_relative_to_it(repo: Path) -> None:
    project = repo / "agent"
    project.mkdir()
    (project / "prompts.py").write_text('SYSTEM = "nested"\n')
    base = _commit(repo, "nested")
    symbols = {"prompts.py": ["SYSTEM"]}
    name = string_symbol_component("prompts.py", "SYSTEM")

    sha = commit_from_candidate(
        project, base, symbols, _candidate(**{name: "moved"}), "nested edit"
    )

    assert _git(repo, "diff", "--name-only", base, sha).split() == ["agent/prompts.py"]
    assert _texts(candidate_from_commit(project, sha, symbols)) == {name: "moved"}


@pytest.mark.parametrize(
    ("source", "symbol", "message"),
    [
        ("OTHER = 'x'\n", "MISSING", "not assigned"),
        ("X = 1\n", "X", "plain string constant"),
        ("X = f'{1}'\n", "X", "plain string constant"),
        ("X = b'bytes'\n", "X", "plain string constant"),
        ("X = 'a' + 'b'\n", "X", "plain string constant"),
        ("X = Y = 'shared'\n", "X", "plain string constant"),
        ("X = 'first'\nX = 'second'\n", "X", "more than once"),
        ("X = 'a'\nX += 'b'\n", "X", "more than once"),
        ("if True:\n    X = 'nested'\n", "X", "top-level"),
        ("def X():\n    return 'doc'\n", "X", "plain string constant"),
        ("class E:\n    'doc'\n", "E.__doc__", "NAME = '...'"),
    ],
)
def test_refuses_symbols_that_are_not_plain_top_level_strings(
    repo: Path, source: str, symbol: str, message: str
) -> None:
    (repo / "bad.py").write_text(source)
    base = _commit(repo, "bad")

    with pytest.raises(StringSymbolError, match=message):
        candidate_from_commit(repo, base, {"bad.py": [symbol]})


@pytest.mark.parametrize(
    "symbols",
    [
        {"../prompts.py": ["INSTRUCTIONS"]},
        {"/abs/prompts.py": ["INSTRUCTIONS"]},
        {"README.md": ["INSTRUCTIONS"]},
        {"prompts.py": []},
        {"prompts.py": "INSTRUCTIONS"},
        {"prompts.py": ["INSTRUCTIONS", "INSTRUCTIONS"]},
        {"missing.py": ["INSTRUCTIONS"]},
    ],
)
def test_refuses_invalid_symbol_declarations(
    repo: Path, symbols: dict[str, Any]
) -> None:
    with pytest.raises(StringSymbolError):
        candidate_from_commit(repo, _head(repo), symbols)


def test_commit_refuses_undeclared_components(repo: Path) -> None:
    with pytest.raises(StringSymbolError, match="undeclared"):
        commit_from_candidate(
            repo,
            _head(repo),
            SYMBOLS,
            _candidate(**{"py:prompts.py:helper": "x", INSTRUCTIONS: "y"}),
            "bad",
        )


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("prompts.py", PROMPTS.replace("os.sep", "os.linesep")),
        ("prompts.py", PROMPTS.replace("# keep this comment", "# changed")),
        ("prompts.py", PROMPTS.replace('"Be terse."', 'f"Be {terse}."')),
        ("README.md", "changed\n"),
        ("new.py", "X = 1\n"),
    ],
)
def test_scope_verification_rejects_changes_outside_declared_strings(
    repo: Path, path: str, content: str
) -> None:
    base = _head(repo)
    (repo / "prompts.py").write_text(PROMPTS.replace("Be terse.", "Be brief."))
    (repo / path).write_text(content)
    sha = _commit(repo, "outside scope")

    with pytest.raises(StringSymbolError):
        verify_string_symbol_commit(repo, base, sha, SYMBOLS)


def test_scope_verification_accepts_a_declared_string_edit(repo: Path) -> None:
    base = _head(repo)
    (repo / "prompts.py").write_text(PROMPTS.replace("Be terse.", "Be brief."))
    sha = _commit(repo, "inside scope")

    verify_string_symbol_commit(repo, base, sha, SYMBOLS)


def test_scope_verification_requires_descent_from_base(repo: Path) -> None:
    base = _head(repo)
    _git(repo, "checkout", "-q", "--orphan", "other")
    (repo / "prompts.py").write_text(PROMPTS.replace("Be terse.", "Be brief."))
    unrelated = _commit(repo, "unrelated")

    with pytest.raises(StringSymbolError, match="descend"):
        verify_string_symbol_commit(repo, base, unrelated, SYMBOLS)


class _FakeEvaluator:
    """Score a commit by reading its declared instructions from Git."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls: list[tuple[str, list[str]]] = []

    async def __call__(
        self, commit: str, cases: Sequence[Case[Any, Any, Any]]
    ) -> list[MetricResult]:
        self.calls.append((commit, [case.name or "" for case in cases]))
        text = candidate_from_commit(self.repo, commit, SYMBOLS)[INSTRUCTIONS].text
        improved = "improved" in text
        return [
            MetricResult(
                score=1.0 if improved else 0.25,
                feedback=None if improved else f"{case.name}: say improved",
                side_info={"scores": {"quality": 1.0 if improved else 0.25}},
            )
            for case in cases
        ]


def _task(repo: Path, evaluator: _FakeEvaluator) -> StringSymbolTask:
    return StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=evaluator,
        trainset=[Case(name=f"train-{i}", inputs=f"question {i}") for i in range(2)],
        valset=[Case(name=f"val-{i}", inputs=f"held {i}") for i in range(2)],
        test_set=[Case(name="test-0", inputs="report")],
    )


@pytest.mark.asyncio
async def test_task_seeds_from_and_evaluates_the_base_commit(repo: Path) -> None:
    evaluator = _FakeEvaluator(repo)
    task = _task(repo, evaluator)
    base = _head(repo)

    seed = await task.seed_candidate()
    evaluation = await task.evaluate(seed)

    assert task.base_commit == base
    assert _texts(seed) == _texts(candidate_from_commit(repo, base, SYMBOLS))
    assert evaluator.calls == [(base, ["val-0", "val-1"])]
    assert evaluation.score == 0.25
    assert evaluation.per_case_scores == {"val-0": 0.25, "val-1": 0.25}
    assert evaluation.objective_scores == {"quality": 0.25}
    assert [record.feedback for record in evaluation.records] == [
        "val-0: say improved",
        "val-1: say improved",
    ]
    assert await task.commit_for(seed) == base

    with validation_evaluation():
        withheld = await task.evaluate(seed)
    assert [record.feedback for record in withheld.records] == [None, None]
    assert withheld.objective_scores == {"quality": 0.25}
    assert all("side_info" not in record.payload for record in withheld.records)


@pytest.mark.asyncio
async def test_task_rejects_a_misshapen_evaluator_result(repo: Path) -> None:
    async def short(commit: str, cases: Sequence[Case[Any, Any, Any]]) -> list[Any]:
        return [MetricResult(score=1.0)]

    task = StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=short,
        trainset=[Case(name="a", inputs="a"), Case(name="b", inputs="b")],
    )

    with pytest.raises(ValueError, match="one MetricResult per case"):
        await task.evaluate(await task.seed_candidate())


@pytest.mark.asyncio
async def test_gepa_engine_optimizes_a_git_candidate_through_the_task(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    reflective: list[Any] = []

    async def propose(self: Any, **kwargs: Any) -> ProposalResult:
        reflective.append(kwargs["reflective_data"])
        return ProposalResult(
            texts={INSTRUCTIONS: "Be terse. Say improved."},
            component_metadata={},
            reasoning=None,
        )

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose)
    evaluator = _FakeEvaluator(repo)
    task = _task(repo, evaluator)
    base = _head(repo)
    config = EngineConfig(
        engine="gepa",
        max_metric_calls=20,
        max_iterations=2,
        stop_at_score=1.0,
        engine_config={
            "reflection_minibatch_size": 2,
            "reflection_config": ReflectionConfig(model=TestModel()),
        },
    )

    result = await get_engine("gepa", config).run(task, config, BudgetTracker(20))

    assert result.best_score == 1.0
    assert result.best_candidate[INSTRUCTIONS].text == "Be terse. Say improved."
    records = reflective[0].records
    assert {record["feedback"] for record in records} == {
        "train-0: say improved",
        "train-1: say improved",
    }
    assert {record["commit"] for record in records} == {base}
    # One commit per distinct candidate, reused across per-case evaluations.
    commits = {commit for commit, _ in evaluator.calls}
    assert len(commits) == 2 and base in commits

    sha = await task.commit_for(result.best_candidate)
    assert sha in commits
    assert _texts(candidate_from_commit(repo, sha, SYMBOLS)) == _texts(
        {**await task.seed_candidate(), **result.best_candidate}
    )
    verify_string_symbol_commit(repo, base, sha, SYMBOLS)
    assert _head(repo) == base


@pytest.mark.asyncio
async def test_optimize_omni_runs_text_and_coding_engines_on_a_git_candidate(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)

    async def propose_texts(self: Any, **kwargs: Any) -> ProposalResult:
        return ProposalResult(
            texts={INSTRUCTIONS: "Say improved."}, component_metadata={}, reasoning=None
        )

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose_texts)
    contexts: list[ReflectionContext] = []

    async def propose(context: ReflectionContext) -> CandidateMap:
        contexts.append(context)
        return {
            **context.candidate,
            INSTRUCTIONS: ComponentValue(name=INSTRUCTIONS, text="improved by agent"),
        }

    evaluator = _FakeEvaluator(repo)
    task = _task(repo, evaluator)
    configs = [
        EngineConfig(
            engine="gepa",
            max_metric_calls=10,
            max_iterations=1,
            engine_config={
                "reflection_minibatch_size": 1,
                "reflection_config": ReflectionConfig(model=TestModel()),
            },
        ),
        EngineConfig(
            engine="coding_agent",
            max_metric_calls=10,
            max_iterations=1,
            engine_config={"propose": propose, "minibatch_size": 1},
        ),
    ]

    result = await optimize_omni(
        task,
        OmniPlan(
            phase_one=configs,
            phase_two=configs[0],
            phase_one_metric_calls=20,
            phase_two_metric_calls=10,
        ),
    )

    assert contexts and "say improved" in contexts[0].report
    assert "improved" in result.best.best_candidate[INSTRUCTIONS].text
    assert result.test_score == 1.0
    sha = await task.commit_for(result.best.best_candidate)
    verify_string_symbol_commit(repo, task.base_commit, sha, SYMBOLS)
