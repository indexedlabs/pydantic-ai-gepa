"""Bridge between Git commits and CandidateMaps over declared string symbols."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ImageUrl, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa._validation import validation_evaluation
from pydantic_ai_gepa.compose import OmniPlan, optimize_omni, optimize_sequential
from pydantic_ai_gepa.engines import (
    BudgetTracker,
    EngineConfig,
    OptimizationTask,
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
from pydantic_ai_gepa.inspection import InspectingModel, InspectionAborted
from pydantic_ai_gepa.spend import current_rollout_capability
from pydantic_ai_gepa.types import ExampleBankConfig, MetricResult, ReflectionConfig

MEDIA = (BinaryContent, ImageUrl)


class _PhotoQuestion(BaseModel):
    photo: BinaryContent
    question: str


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


def test_mutated_component_text_is_normalized_like_construction(
    repo: Path,
) -> None:
    base = _head(repo)
    candidate = candidate_from_commit(repo, base, SYMBOLS)
    # A proposer may mutate a copied ComponentValue, bypassing its stripping.
    candidate[INSTRUCTIONS].text = "\nimproved prompt\nline two\n"

    sha = commit_from_candidate(repo, base, SYMBOLS, candidate, "mutated")

    assert _texts(candidate_from_commit(repo, sha, SYMBOLS))[INSTRUCTIONS] == (
        "improved prompt\nline two"
    )


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


def test_refuses_an_executable_declared_file(repo: Path) -> None:
    (repo / "tool.py").write_text('X = "seed"\n')
    (repo / "tool.py").chmod(0o755)
    base = _commit(repo, "executable")

    with pytest.raises(StringSymbolError, match="non-executable"):
        candidate_from_commit(repo, base, {"tool.py": ["X"]})


def test_round_trip_keeps_a_u_prefix_and_handles_a_bom(repo: Path) -> None:
    (repo / "legacy.py").write_bytes(
        b'\xef\xbb\xbfX = u"seed"\nY = "\xc3\xa9t\xc3\xa9"\n'
    )
    base = _commit(repo, "legacy")
    symbols = {"legacy.py": ["X", "Y"]}
    x = string_symbol_component("legacy.py", "X")
    y = string_symbol_component("legacy.py", "Y")
    assert _texts(candidate_from_commit(repo, base, symbols)) == {x: "seed", y: "été"}

    edited = _candidate(**{x: "improved", y: "hiver"})
    sha = commit_from_candidate(repo, base, symbols, edited, "legacy edit")

    assert _texts(candidate_from_commit(repo, sha, symbols)) == _texts(edited)
    assert _git(repo, "show", f"{sha}:legacy.py").startswith("\ufeffX = u'improved'")


def test_round_trip_keeps_crlf_line_endings(repo: Path) -> None:
    (repo / "crlf.py").write_bytes(b'X = "one"\r\nY = 1\r\n')
    _git(repo, "config", "core.autocrlf", "false")
    base = _commit(repo, "crlf")
    symbols = {"crlf.py": ["X"]}
    x = string_symbol_component("crlf.py", "X")

    sha = commit_from_candidate(
        repo, base, symbols, _candidate(**{x: "line one\nline two"}), "crlf"
    )

    blob = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{sha}:crlf.py"],
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        check=True,
        capture_output=True,
    ).stdout
    assert blob == b'X = """line one\r\nline two"""\r\nY = 1\r\n'
    assert _texts(candidate_from_commit(repo, sha, symbols)) == {
        x: "line one\nline two"
    }


def test_a_function_local_name_is_not_a_module_binding(repo: Path) -> None:
    (repo / "local.py").write_text(
        'X = "module"\n\n\ndef f():\n    X = "local"\n    return X\n'
    )
    base = _commit(repo, "local")

    assert _texts(candidate_from_commit(repo, base, {"local.py": ["X"]})) == {
        "py:local.py:X": "module"
    }


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
        (
            "X = 'a'\ntry:\n    pass\nexcept ValueError as X:\n    pass\n",
            "X",
            "more than once",
        ),
        ("X = 'a'\nmatch 1:\n    case 1 as X:\n        pass\n", "X", "more than once"),
        ("X = 'a'\nmatch {}:\n    case {**X}:\n        pass\n", "X", "more than once"),
        ("X = 'a'\ndel X\n", "X", "more than once"),
        ("# -*- coding: latin-1 -*-\nX = 'plain'\n", "X", "only UTF-8"),
        ("X: str\nX = 'value'\n", "X", "more than once"),
        ("X: str\n", "X", "plain string constant"),
        ("X = 'a'\nt = {}\nt[(X := 'fixed')]: str\n", "X", "more than once"),
        ("X = 'a'\nfrom os.path import *\n", "X", "wildcard import"),
        ("X = 'a'\nY: (X := 'fixed')\n", "X", "more than once"),
        ("X: (X := 'fixed') = 'seed'\n", "X", "more than once"),
        ("X = (\n    'a'  # note\n    'b'\n)\n", "X", "comment"),
        ("X = 'a'\ndef f(arg=(X := 'fixed')):\n    pass\n", "X", "more than once"),
        ("X = 'a'\nY = [(X := i) for i in 'b']\n", "X", "more than once"),
        ("X = 'a'\ndef f():\n    global X\n    X = 'b'\n", "X", "more than once"),
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


def test_scope_verification_checks_unchanged_declared_files(repo: Path) -> None:
    (repo / "bad.py").write_text("Y = 123\n")
    base = _commit(repo, "bad declared file")
    (repo / "prompts.py").write_text(PROMPTS.replace("Be terse.", "Be brief."))
    sha = _commit(repo, "edit good file")
    symbols = {**SYMBOLS, "bad.py": ["Y"]}

    with pytest.raises(StringSymbolError, match="plain string constant"):
        verify_string_symbol_commit(repo, base, sha, symbols)
    # A no-op commit still checks the declared surface exists.
    with pytest.raises(StringSymbolError, match="not a file"):
        verify_string_symbol_commit(repo, sha, sha, {"absent.py": ["X"]})


def test_scope_verification_accepts_a_declared_string_edit(repo: Path) -> None:
    base = _head(repo)
    (repo / "prompts.py").write_text(PROMPTS.replace("Be terse.", "Be brief."))
    sha = _commit(repo, "inside scope")

    verify_string_symbol_commit(repo, base, sha, SYMBOLS)


def test_replace_refs_never_change_what_is_read_or_verified(repo: Path) -> None:
    base = _head(repo)
    (repo / "README.md").write_text("replaced\n")
    impostor = _commit(repo, "impostor")
    _git(repo, "reset", "-q", "--hard", base)
    _git(repo, "replace", base, impostor)

    sha = commit_from_candidate(
        repo, base, SYMBOLS, _candidate(**{INSTRUCTIONS: "Be brief."}), "edit"
    )

    changed = _git(repo, "--no-replace-objects", "diff", "--name-only", base, sha)
    assert changed.split() == ["prompts.py"]

    _git(repo, "replace", "-d", base)
    (repo / "README.md").write_text("outside scope\n")
    outside = _commit(repo, "outside")
    _git(repo, "replace", outside, sha)
    with pytest.raises(StringSymbolError, match="outside the declared"):
        verify_string_symbol_commit(repo, base, outside, SYMBOLS)


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
        self.calls: list[tuple[str, str]] = []
        self.in_flight: dict[str, int] = {}
        self.max_overlap = 0
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}

    async def __call__(self, commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        self.calls.append((commit, case.name or ""))
        split = (case.name or "").split("-")[0]
        self.in_flight[commit] = self.in_flight.get(commit, 0) + 1
        self.max_overlap = max(self.max_overlap, self.in_flight[commit])
        self.active[split] = self.active.get(split, 0) + 1
        self.max_active[split] = max(self.max_active.get(split, 0), self.active[split])
        await asyncio.sleep(0.01)
        self.in_flight[commit] -= 1
        self.active[split] -= 1
        text = candidate_from_commit(self.repo, commit, SYMBOLS)[INSTRUCTIONS].text
        improved = "improved" in text
        return MetricResult(
            score=1.0 if improved else 0.25,
            feedback=None if improved else f"{case.name}: say improved",
            side_info={"scores": {"quality": 1.0 if improved else 0.25}},
        )


def _task(repo: Path, evaluator: Any, *, concurrency: int = 20) -> StringSymbolTask:
    return StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=evaluator,
        trainset=[Case(name=f"train-{i}", inputs=f"question {i}") for i in range(2)],
        valset=[Case(name=f"val-{i}", inputs=f"held {i}") for i in range(2)],
        test_set=[Case(name="test-0", inputs="report")],
        concurrency=concurrency,
    )


def _improving_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def propose(self: Any, **kwargs: Any) -> ProposalResult:
        return ProposalResult(
            texts={INSTRUCTIONS: "Be terse. Say improved."},
            component_metadata={},
            reasoning=None,
        )

    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose)


def _gepa_config(budget: int = 20) -> EngineConfig:
    return EngineConfig(
        engine="gepa",
        max_metric_calls=budget,
        max_iterations=1,
        engine_config={
            "reflection_minibatch_size": 2,
            "reflection_config": ReflectionConfig(model=TestModel()),
        },
    )


@pytest.mark.asyncio
async def test_gepa_engine_never_promotes_a_candidate_whose_validation_failed(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _improving_proposal(monkeypatch)
    scores = _FakeEvaluator(repo)

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        result = await scores(commit, case)
        if result.score == 1.0 and case.name == "val-1":
            raise RuntimeError("judge unavailable")
        return result

    task = _task(repo, evaluator)
    config = _gepa_config()

    result = await get_engine("gepa", config).run(task, config, BudgetTracker(20))

    # The proposal scored 1.0 on val-0 (0.5 average with the failure) but its
    # validation is infrastructure-invalid, so the 0.25 seed stays best.
    assert any(name == "val-1" for _, name in scores.calls[4:])
    assert result.best_score == 0.25
    assert _texts(result.best_candidate) == _texts(await task.seed_candidate())


async def _val_1_always_fails(
    scores: _FakeEvaluator, commit: str, case: Case[Any, Any, Any]
) -> MetricResult:
    if case.name == "val-1":
        raise RuntimeError("judge unavailable")
    return await scores(commit, case)


@pytest.mark.asyncio
async def test_coding_engine_keeps_the_seed_when_no_validation_is_eligible(
    repo: Path,
) -> None:
    scores = _FakeEvaluator(repo)

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        return await _val_1_always_fails(scores, commit, case)

    async def propose(context: ReflectionContext) -> CandidateMap:
        return {
            **context.candidate,
            INSTRUCTIONS: ComponentValue(name=INSTRUCTIONS, text="improved"),
        }

    task = _task(repo, evaluator)
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=40,
        max_iterations=1,
        stop_at_score=0.1,
        engine_config={"propose": propose, "minibatch_size": 2},
    )

    result = await get_engine("coding_agent", config).run(
        task, config, BudgetTracker(40)
    )

    # Neither validation is eligible: no stop on the seed's invalid 0.125, and
    # no promotion of the proposal's invalid 0.5.
    assert any(event.kind == "validation_rejected" for event in result.history)
    assert result.best_score is None
    assert _texts(result.best_candidate) == _texts(await task.seed_candidate())


@pytest.mark.asyncio
async def test_composed_gepa_rescores_an_ineligible_seed(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _improving_proposal(monkeypatch)
    scores = _FakeEvaluator(repo)

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        return await _val_1_always_fails(scores, commit, case)

    task = _task(repo, evaluator)
    config = _gepa_config().model_copy(update={"stop_at_score": 0.1})

    result = await optimize_sequential(task, [config], max_metric_calls=20)

    # The helper's non-selectable seed evaluation is not reused as a score, so
    # stop_at_score cannot end the stage before it tries a proposal.
    gepa = result.results[-1]
    assert gepa.engine == "gepa"
    assert gepa.num_metric_calls > 0
    assert gepa.best_score is None


@pytest.mark.asyncio
async def test_evaluator_concurrency_is_bounded_across_engine_calls(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _improving_proposal(monkeypatch)
    evaluator = _FakeEvaluator(repo)
    task = _task(repo, evaluator, concurrency=1)
    config = _gepa_config()

    await get_engine("gepa", config).run(task, config, BudgetTracker(20))

    assert evaluator.max_active == {"train": 1, "val": 1}

    async def propose(context: ReflectionContext) -> CandidateMap:
        return {
            **context.candidate,
            INSTRUCTIONS: ComponentValue(name=INSTRUCTIONS, text="improved"),
        }

    coding = _FakeEvaluator(repo)
    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=12,
        max_iterations=1,
        engine_config={"propose": propose, "minibatch_size": 2, "concurrency": 1},
    )
    await get_engine("coding_agent", config).run(
        _task(repo, coding), config, BudgetTracker(12)
    )

    # The engine's own limit governs its minibatches; validation uses the task's.
    assert coding.max_active["train"] == 1
    assert coding.max_active["val"] == 2


@pytest.mark.asyncio
async def test_task_seeds_from_and_evaluates_the_base_commit(repo: Path) -> None:
    evaluator = _FakeEvaluator(repo)
    task = _task(repo, evaluator)
    base = _head(repo)

    seed = await task.seed_candidate()
    evaluation = await task.evaluate(seed)

    assert task.base_commit == base
    assert _texts(seed) == _texts(candidate_from_commit(repo, base, SYMBOLS))
    assert sorted(evaluator.calls) == [(base, "val-0"), (base, "val-1")]
    # Calls overlap on one SHA; the evaluator contract requires reentrancy.
    assert evaluator.max_overlap == 2
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
async def test_task_keeps_its_repository_after_a_working_directory_change(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(repo)
    task = StringSymbolTask(
        repo=".",
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=_FakeEvaluator(repo),
        trainset=[Case(name="a", inputs="a")],
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    sha = await task.commit_for(_candidate(**{INSTRUCTIONS: "Say improved."}))

    assert _texts(candidate_from_commit(repo, sha, SYMBOLS))[INSTRUCTIONS] == (
        "Say improved."
    )


@pytest.mark.asyncio
async def test_task_rejects_a_misshapen_evaluator_result(repo: Path) -> None:
    async def wrong(commit: str, case: Case[Any, Any, Any]) -> Any:
        return 1.0

    task = StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=wrong,
        trainset=[Case(name="a", inputs="a"), Case(name="b", inputs="b")],
    )

    with pytest.raises(ValueError, match="must return a MetricResult"):
        await task.evaluate(await task.seed_candidate())


@pytest.mark.asyncio
async def test_task_scores_an_evaluator_failure_as_a_failed_case(repo: Path) -> None:
    async def flaky(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        if case.name == "b":
            raise RuntimeError("checkout broke")
        return MetricResult(score=1.0, side_info={"scores": {"quality": 1.0}})

    task = StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=flaky,
        trainset=[Case(name="a", inputs="a"), Case(name="b", inputs="b")],
    )

    evaluation = await task.evaluate(await task.seed_candidate())

    assert evaluation.per_case_scores == {"a": 1.0, "b": 0.0}
    assert evaluation.records[1].feedback == "Commit evaluator failed: checkout broke"
    assert not evaluation.records[1].payload["output"].success
    # An infrastructure failure disqualifies the candidate instead of
    # tripping the identical-objective-keys check or counting as a real 0.0.
    assert evaluation.selectable is False
    assert evaluation.objective_scores == {"quality": 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine_config",
    [
        {"optimize_tools": True},
        {"reflection_config": ReflectionConfig(example_bank=ExampleBankConfig())},
    ],
)
async def test_gepa_engine_rejects_agent_only_options_for_the_task(
    repo: Path, engine_config: dict[str, Any]
) -> None:
    task = _task(repo, _FakeEvaluator(repo))
    config = EngineConfig(engine="gepa", engine_config=engine_config)

    with pytest.raises(ValueError, match="agent tasks only"):
        await get_engine("gepa", config).run(task, config, BudgetTracker(10))


@pytest.mark.asyncio
async def test_inspection_interception_reaches_the_caller(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals: list[Any] = []

    async def propose(self: Any, **kwargs: Any) -> ProposalResult:
        proposals.append(kwargs)
        return ProposalResult(texts={}, component_metadata={}, reasoning=None)

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose)

    async def map_messages(*args: Any) -> list[Any]:
        return [{"role": "user", "content": "captured"}]

    student = Agent(InspectingModel(TestModel(), map_messages=map_messages))

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        await student.run(str(case.inputs))
        return MetricResult(score=1.0)

    task = _task(repo, evaluator)
    with pytest.raises(InspectionAborted) as caught:
        await task.evaluate(await task.seed_candidate())
    assert caught.value.snapshot is not None

    config = _gepa_config()
    with pytest.raises(InspectionAborted):
        await get_engine("gepa", config).run(task, config, BudgetTracker(20))
    assert proposals == []


@pytest.mark.asyncio
async def test_gepa_engine_meters_model_calls_inside_the_evaluator(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    student = Agent(TestModel(custom_output_text="ok"))
    calls: list[str] = []

    async def evaluate(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        capability = current_rollout_capability()
        assert capability is not None
        await student.run(str(case.inputs), capabilities=[capability])
        calls.append(case.name or "")
        return MetricResult(score=0.5)

    task = StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=evaluate,
        trainset=[Case(name=f"val-{i}", inputs=f"held {i}") for i in range(2)],
        concurrency=1,
    )
    config = EngineConfig(
        engine="gepa",
        max_metric_calls=10,
        max_token_cost=0.10,
        engine_config={
            "price_fn": lambda response: 0.25,
            "reflection_config": ReflectionConfig(model=TestModel()),
        },
    )

    result = await get_engine("gepa", config).run(task, config, BudgetTracker(10))

    spend = next(e.data for e in result.history if e.kind == "summary")["spend_report"]
    assert spend["total_dollars"] >= 0.25
    assert spend["stopped_by_cost"] is True
    # The priced response stops the run inside the evaluator's agent call.
    assert calls == []


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
    assert evaluator.max_overlap > 1
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
    assert contexts[0].side_info["cases"][0]["side_info"] == {
        "scores": {"quality": 0.25}
    }
    assert "improved" in result.best.best_candidate[INSTRUCTIONS].text
    assert result.test_score == 1.0
    sha = await task.commit_for(result.best.best_candidate)
    verify_string_symbol_commit(repo, task.base_commit, sha, SYMBOLS)


@pytest.mark.asyncio
async def test_real_reflector_sees_training_evidence_but_not_held_out_cases(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    prompts: list[str] = []

    async def reflector(messages: Any, info: Any) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    prompts.append(content)
                elif isinstance(content, list):
                    prompts.extend(item for item in content if isinstance(item, str))
        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps(
                        {
                            "reasoning": {
                                "pattern_discovery": "Cases ask for improved.",
                                "creative_hypothesis": "Say improved.",
                                "experimental_approach": "Add the word.",
                            },
                            "updated_components": [
                                {
                                    "component_name": INSTRUCTIONS,
                                    "optimized_value": "Say improved.",
                                }
                            ],
                        }
                    )
                )
            ]
        )

    task = _task(repo, _FakeEvaluator(repo))
    config = EngineConfig(
        engine="gepa",
        max_metric_calls=20,
        max_iterations=1,
        engine_config={
            "reflection_minibatch_size": 2,
            "reflection_config": ReflectionConfig(model=FunctionModel(reflector)),
        },
    )

    result = await get_engine("gepa", config).run(task, config, BudgetTracker(20))

    prompt = "\n".join(prompts)
    assert result.best_score == 1.0
    assert "question 0" in prompt and "question 1" in prompt
    assert "train-0: say improved" in prompt
    assert "quality" in prompt
    assert "held 0" not in prompt and "val-0" not in prompt
    assert "No execution traces exist for this run" in prompt
    assert "run_python_repl" not in prompt


@pytest.mark.asyncio
async def test_agent_task_reflection_without_traces_renders_its_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # GepaEngine writes no span files, so agent tasks take the trace-less
    # prompt too: the minibatch records are rendered, compactly.
    monkeypatch.chdir(tmp_path)
    prompts: list[str] = []

    async def reflector(messages: Any, info: Any) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    prompts.append(content)
                elif isinstance(content, list):
                    prompts.extend(item for item in content if isinstance(item, str))
        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps(
                        {
                            "reasoning": {
                                "pattern_discovery": "p",
                                "creative_hypothesis": "h",
                                "experimental_approach": "e",
                            },
                            "updated_components": [
                                {
                                    "component_name": "instructions",
                                    "optimized_value": "Reply with ok, always.",
                                }
                            ],
                        }
                    )
                )
            ]
        )

    agent = Agent(TestModel(custom_output_text="nope"), instructions="Reply.")
    task = OptimizationTask(
        agent=agent,
        trainset=[Case(name=f"t-{i}", inputs=f"agent question {i}") for i in range(2)],
        metric=lambda case, output: MetricResult(
            score=0.0, feedback=f"{case.name}: wrong answer"
        ),
    )
    config = EngineConfig(
        engine="gepa",
        max_metric_calls=10,
        max_iterations=1,
        engine_config={
            "reflection_minibatch_size": 2,
            "reflection_config": ReflectionConfig(model=FunctionModel(reflector)),
        },
    )

    await get_engine("gepa", config).run(task, config, BudgetTracker(10))

    prompt = "\n".join(prompts)
    assert "No execution traces exist for this run" in prompt
    assert "t-0: wrong answer" in prompt and "t-1: wrong answer" in prompt
    assert "agent question 0" in prompt
    assert len(prompt) < 20_000


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["gepa", "coding_agent"])
@pytest.mark.parametrize(
    ("seed_fails", "proposal_fails", "proposal_score", "validated"),
    [
        # The seed's failed case must not make an inferior proposal look better.
        ("train-1", None, 0.5, False),
        # A failed case must not hide a genuine improvement.
        (None, "train-1", 1.0, True),
        # No case scored on both sides: no evidence either way.
        ("train-0", "train-1", 1.0, False),
    ],
)
async def test_failed_training_cases_are_not_acceptance_evidence(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    seed_fails: str | None,
    proposal_fails: str | None,
    proposal_score: float,
    validated: bool,
) -> None:
    _improving_proposal(monkeypatch)
    validation_calls: list[tuple[bool, str]] = []

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        text = candidate_from_commit(repo, commit, SYMBOLS)[INSTRUCTIONS].text
        improved = "improved" in text
        name = case.name or ""
        if name.startswith("val-"):
            validation_calls.append((improved, name))
        if name == (proposal_fails if improved else seed_fails):
            raise RuntimeError("judge unavailable")
        return MetricResult(
            score=proposal_score if improved else 0.75, feedback=f"{name}: improve"
        )

    async def propose(context: ReflectionContext) -> CandidateMap:
        return {
            **context.candidate,
            INSTRUCTIONS: ComponentValue(name=INSTRUCTIONS, text="Say improved."),
        }

    config = (
        _gepa_config()
        if engine == "gepa"
        else EngineConfig(
            engine="coding_agent",
            max_metric_calls=40,
            max_iterations=1,
            engine_config={"propose": propose, "minibatch_size": 2},
        )
    )

    result = await get_engine(engine, config).run(
        _task(repo, evaluator), config, BudgetTracker(config.max_metric_calls)
    )

    assert any(improved for improved, _ in validation_calls) is validated
    if validated:
        assert result.best_score == 1.0
    if engine == "coding_agent" and seed_fails and proposal_fails:
        assert any(event.kind == "infrastructure_invalid" for event in result.history)


def test_task_can_serve_consecutive_event_loops(repo: Path) -> None:
    task = _task(repo, _FakeEvaluator(repo), concurrency=1)

    async def evaluate() -> float:
        # Two cases with one slot contend the evaluator semaphore.
        return (await task.evaluate(await task.seed_candidate())).score

    assert asyncio.run(evaluate()) == 0.25
    assert asyncio.run(evaluate()) == 0.25


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["gepa", "coding_agent"])
async def test_an_evaluator_crash_alone_does_not_trigger_reflection(
    repo: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    proposals: list[Any] = []

    async def propose_texts(self: Any, **kwargs: Any) -> ProposalResult:
        proposals.append(kwargs)
        return ProposalResult(texts={}, component_metadata={}, reasoning=None)

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose_texts)

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        if case.name == "train-1":
            raise RuntimeError("judge unavailable")
        return MetricResult(score=1.0)

    async def propose(context: ReflectionContext) -> CandidateMap:
        proposals.append(context)
        return context.candidate

    config = (
        _gepa_config()
        if engine == "gepa"
        else EngineConfig(
            engine="coding_agent",
            max_metric_calls=20,
            max_iterations=1,
            engine_config={"propose": propose, "minibatch_size": 2},
        )
    )

    result = await get_engine(engine, config).run(
        _task(repo, evaluator), config, BudgetTracker(config.max_metric_calls)
    )

    assert proposals == []
    if engine == "coding_agent":
        assert [
            event.data["infrastructure_invalid_case_ids"]
            for event in result.history
            if event.kind == "clean_minibatch"
        ] == [["train-1"]]


@pytest.mark.asyncio
async def test_gepa_skips_reflection_without_selectable_rollouts(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposals: list[Any] = []

    async def propose_texts(self: Any, **kwargs: Any) -> ProposalResult:
        proposals.append(kwargs)
        return ProposalResult(texts={}, component_metadata={}, reasoning=None)

    monkeypatch.setattr(InstructionProposalGenerator, "propose_texts", propose_texts)

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        if (case.name or "").startswith("train-"):
            raise RuntimeError("judge unavailable")
        return MetricResult(score=0.5)

    base = _gepa_config()
    config = base.model_copy(
        update={"engine_config": {**base.engine_config, "skip_perfect_score": False}}
    )

    await get_engine("gepa", config).run(
        _task(repo, evaluator), config, BudgetTracker(20)
    )

    assert proposals == []


@pytest.mark.asyncio
async def test_coding_proposer_sees_no_crash_in_a_mixed_minibatch(repo: Path) -> None:
    contexts: list[ReflectionContext] = []

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        if case.name == "train-1":
            raise RuntimeError("judge unavailable")
        return MetricResult(
            score=0.25, feedback=f"{case.name}: real failure", side_info={"k": 1}
        )

    async def propose(context: ReflectionContext) -> CandidateMap:
        contexts.append(context)
        return context.candidate

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=20,
        max_iterations=1,
        engine_config={"propose": propose, "minibatch_size": 2},
    )

    await get_engine("coding_agent", config).run(
        _task(repo, evaluator), config, BudgetTracker(20)
    )

    (context,) = contexts
    assert [record.case_id for record in context.minibatch_records] == ["train-0"]
    assert "train-0: real failure" in context.report
    assert "1 of 1 case(s) underperformed" in context.report
    assert "judge unavailable" not in context.report
    assert context.side_info["feedback"] == ["train-0: real failure"]


@pytest.mark.asyncio
async def test_omni_fair_vote_tolerates_a_comparison_time_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _improving_proposal(monkeypatch)
    improved_validation_calls = 0

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        nonlocal improved_validation_calls
        text = candidate_from_commit(repo, commit, SYMBOLS)[INSTRUCTIONS].text
        improved = "improved" in text
        if improved and case.name == "val-1":
            improved_validation_calls += 1
            # The engine's own validation passes; the fair vote then fails.
            if improved_validation_calls > 1:
                raise RuntimeError("judge unavailable")
        score = 1.0 if improved else 0.25
        return MetricResult(score=score, side_info={"scores": {"quality": score}})

    task = _task(repo, evaluator)
    config = _gepa_config()

    result = await optimize_omni(
        task,
        OmniPlan(
            phase_one=[config],
            phase_two=config,
            phase_one_metric_calls=config.max_metric_calls,
            phase_two_metric_calls=config.max_metric_calls,
        ),
    )

    # The proposal's failed comparison sample disqualifies it; the seed wins.
    assert improved_validation_calls > 1
    assert not result.fair_votes[1].selectable
    assert _texts(result.best.best_candidate) == _texts(await task.seed_candidate())


@pytest.mark.parametrize("cookie", ["utf8", "UTF8", "utf-8"])
def test_utf8_coding_cookie_aliases_are_accepted(repo: Path, cookie: str) -> None:
    (repo / "cookie.py").write_text(f"# coding: {cookie}\nX = 'café'\n")
    base = _commit(repo, "cookie")
    symbols = {"cookie.py": ["X"]}
    x = string_symbol_component("cookie.py", "X")

    sha = commit_from_candidate(repo, base, symbols, _candidate(**{x: "été"}), "c")

    assert _texts(candidate_from_commit(repo, sha, symbols)) == {x: "été"}


@pytest.mark.asyncio
async def test_declared_failure_with_empty_scores_is_non_selectable(
    repo: Path,
) -> None:
    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        if case.name == "b":
            return MetricResult(
                score=0.0,
                side_info={
                    "selectable": False,
                    "infrastructure_valid": False,
                    "scores": {},
                },
            )
        return MetricResult(score=1.0, side_info={"scores": {"quality": 1.0}})

    task = StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=evaluator,
        trainset=[Case(name="a", inputs="a"), Case(name="b", inputs="b")],
    )

    with validation_evaluation():
        evaluation = await task.evaluate(await task.seed_candidate())

    assert evaluation.selectable is False
    assert evaluation.objective_scores == {"quality": 1.0}


@pytest.mark.asyncio
async def test_coding_agent_proposal_inspection_reaches_the_caller(
    repo: Path,
) -> None:
    async def map_messages(*args: Any) -> list[Any]:
        return [{"role": "user", "content": "captured"}]

    inspected = Agent(InspectingModel(TestModel(), map_messages=map_messages))

    async def evaluator(commit: str, case: Case[Any, Any, Any]) -> MetricResult:
        text = candidate_from_commit(repo, commit, SYMBOLS)[INSTRUCTIONS].text
        if "improved" in text:
            await inspected.run(str(case.inputs))
        return MetricResult(score=0.25, feedback="improve")

    async def propose(context: ReflectionContext) -> CandidateMap:
        return {
            **context.candidate,
            INSTRUCTIONS: ComponentValue(name=INSTRUCTIONS, text="Say improved."),
        }

    config = EngineConfig(
        engine="coding_agent",
        max_metric_calls=40,
        max_iterations=1,
        engine_config={"propose": propose, "minibatch_size": 2},
    )

    with pytest.raises(InspectionAborted) as caught:
        await get_engine("coding_agent", config).run(
            _task(repo, evaluator), config, BudgetTracker(40)
        )
    assert caught.value.snapshot is not None


@pytest.mark.asyncio
async def test_trace_free_reflection_attaches_media_inputs(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    attachments: list[Any] = []

    async def reflector(messages: Any, info: Any) -> ModelResponse:
        for message in messages:
            for part in message.parts:
                content = getattr(part, "content", None)
                if isinstance(content, list):
                    attachments.extend(
                        item for item in content if isinstance(item, MEDIA)
                    )
        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps(
                        {
                            "reasoning": {
                                "pattern_discovery": "p",
                                "creative_hypothesis": "h",
                                "experimental_approach": "e",
                            },
                            "updated_components": [
                                {
                                    "component_name": INSTRUCTIONS,
                                    "optimized_value": "Say improved.",
                                }
                            ],
                        }
                    )
                )
            ]
        )

    image = BinaryContent(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    modelled = BinaryContent(data=b"\x89PNG\r\n\x1a\nmodel", media_type="image/png")
    url = ImageUrl(url="https://example.com/photo.png")
    task = StringSymbolTask(
        repo=repo,
        base_commit="HEAD",
        symbols=SYMBOLS,
        evaluator=_FakeEvaluator(repo),
        trainset=[
            Case(name="train-0", inputs=image),
            Case(name="train-1", inputs={"photos": [url], "caption": "a cat"}),
            Case(name="train-2", inputs=_PhotoQuestion(photo=modelled, question="?")),
        ],
        valset=[Case(name="val-0", inputs="held")],
    )
    config = EngineConfig(
        engine="gepa",
        max_metric_calls=20,
        max_iterations=1,
        engine_config={
            "reflection_minibatch_size": 3,
            "reflection_config": ReflectionConfig(model=FunctionModel(reflector)),
        },
    )

    await get_engine("gepa", config).run(task, config, BudgetTracker(20))

    assert image in attachments and url in attachments
    # A Pydantic input model keeps its media attachable, not base64 text.
    assert modelled in attachments


@dataclass
class _Captioned:
    caption: str
    seed: InitVar[int]
    photo: BinaryContent | None = None


@pytest.mark.parametrize(
    "photo", [None, BinaryContent(data=b"x", media_type="image/png")]
)
def test_reflection_serializes_dataclasses_with_init_vars(
    photo: BinaryContent | None,
) -> None:
    from pydantic_ai_gepa.adapters.agent_adapter import _serialize_for_reflection

    value = _Captioned(caption="a cat", seed=1, photo=photo)

    serialized = _serialize_for_reflection(value)

    assert serialized == {"caption": "a cat", "photo": photo}
    assert "seed" not in serialized


def test_materialization_runs_no_git_hooks(repo: Path) -> None:
    base = _head(repo)
    status = _git(repo, "status", "--porcelain")
    hooks = repo / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    for name in ("post-index-change", "post-commit", "pre-commit"):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\necho {name} >> README.md\n")
        hook.chmod(0o755)

    sha = commit_from_candidate(
        repo, base, SYMBOLS, _candidate(**{INSTRUCTIONS: "Be brief."}), "hooks"
    )

    verify_string_symbol_commit(repo, base, sha, SYMBOLS)
    assert (repo / "README.md").read_text() == "fixture\n"
    # The fixture's own status must not fire the hook either.
    no_hooks = f"core.hooksPath={os.devnull}"
    assert _git(repo, "-c", no_hooks, "status", "--porcelain") == status
    assert _head(repo) == base


def test_reflection_never_sees_excluded_model_fields() -> None:
    from pydantic import Field, SecretStr

    from pydantic_ai_gepa.adapters.agent_adapter import _serialize_for_reflection

    class Inputs(BaseModel):
        photo: ImageUrl
        token: str = Field(exclude=True)
        password: SecretStr

    photo = ImageUrl(url="https://example.com/a.png")
    serialized = _serialize_for_reflection(
        Inputs(photo=photo, token="tok-123", password=SecretStr("hunter2"))
    )

    assert serialized["photo"] == photo
    assert "token" not in serialized
    assert "hunter2" not in repr(serialized)
