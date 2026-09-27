"""Child errors are bounded and held-out messages stay harness-private."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
from pydantic_evals import Case

from pydantic_ai_gepa.cli import harness_record, scoring_diagnostics as diagnostics
from pydantic_ai_gepa.cli import scoring_sandbox as sandbox
from pydantic_ai_gepa.cli.layout import run_dir
from tests.cli.test_scoring_sandbox import (
    clean_environment,
    commit_evaluator,
    git_repo,
    private,
    private_run,
    protocol_backend,
    score,
)
from tests.cli.test_scoring_material import raw_result

__all__ = [
    "clean_environment",
    "git_repo",
    "private",
    "private_run",
    "protocol_backend",
]


@pytest.mark.parametrize("validation", [True, False])
def test_scoring_error_is_private(
    git_repo, private, private_run, protocol_backend, validation
):
    sha = commit_evaluator(
        git_repo,
        """
def evaluate(case):
    raise ConnectionError(f"TLS failed: {case.name} {case.inputs['text']} {case.expected_output} {case.metadata['nested'][0]}")
""",
    )
    case = Case(
        name="CASE_SECRET",
        inputs={"text": "INPUT_SECRET"},
        expected_output="EXPECTED_SECRET",
        metadata={"nested": ["METADATA_SECRET"]},
    )
    records = score(git_repo, sha, validation=validation, cases=[case])
    record = harness_record.for_run("sandbox-test", git_repo)
    assert record is not None
    content = record.read("@scoring-diagnostics")
    assert content
    entry = json.loads(content)
    assert entry["class"] == "ConnectionError"
    assert entry["phase"] == ("validation" if validation else "training")
    assert entry["time"].endswith("+00:00")
    if validation:
        assert "case_id" not in entry
        assert (
            entry["message"]
            == "TLS failed: [redacted] [redacted] [redacted] [redacted]"
        )
        for secret in (
            "CASE_SECRET",
            "INPUT_SECRET",
            "EXPECTED_SECRET",
            "METADATA_SECRET",
        ):
            assert secret not in content
        assert records[0].feedback is None
    else:
        assert entry["case_id"] == "CASE_SECRET"
        assert "INPUT_SECRET" in entry["message"]
    assert records[0].payload["output"].error_message == "Sandboxed evaluation failed"
    assert "diagnostic" not in records[0].payload
    assert "TLS failed" not in json.dumps(asdict(records[0].payload["output"]))
    public = run_dir("sandbox-test", git_repo)
    before = {
        p.relative_to(public): p.read_bytes() for p in public.rglob("*") if p.is_file()
    }
    record.restore_views()
    after = {
        p.relative_to(public): p.read_bytes() for p in public.rglob("*") if p.is_file()
    }
    assert after == before
    assert not any("diagnostic" in str(path) for path in after)
    assert content.encode() not in b"\n".join(after.values())
    if validation:
        for value in after.values():
            assert b"TLS failed" not in value
            assert b"INPUT_SECRET" not in value
            assert b"METADATA_SECRET" not in value
    print("private sample:", content.strip())


@pytest.mark.parametrize(
    "value",
    [
        [],
        "text",
        {"class": "Bad.Name", "message": "text"},
        {"class": "A" * 129, "message": "text"},
        {"class": "Error", "message": []},
        {"class": "Error", "message": "text", "extra": 1},
    ],
)
def test_child_diagnostic_shape_is_untrusted(value):
    raw = raw_result()
    raw.update(failed=True, diagnostic=value)
    with pytest.raises(sandbox.ScoringSandboxError, match="diagnostic"):
        sandbox._result(raw, "case", True)


def test_scrub_before_bound_and_cap(git_repo, private, private_run, monkeypatch):
    case = Case(
        name="PRIVATE_CASE",
        inputs="a" * 1000,
        expected_output="SECRET",
        metadata={"token": "METADATA"},
    )
    value = {
        "class": "SSLCertVerificationError",
        "message": "a" * 1000 + " SECRET METADATA " + "tail " * 1000,
    }
    meter = SimpleNamespace(run_id="sandbox-test", root=git_repo)
    writes = []
    write = harness_record.Record.write

    def tracked_write(self, key, *args, **kwargs):
        writes.append(key)
        return write(self, key, *args, **kwargs)

    monkeypatch.setattr(harness_record.Record, "write", tracked_write)
    collector = diagnostics.FailureDiagnostics(meter)
    for _ in range(55):
        collector.record_failure(value, case, True)
    assert writes == ["@scoring-diagnostics"] * 50
    collector.flush()
    collector.flush()
    assert writes == ["@scoring-diagnostics"] * 50 + ["@scoring-diagnostics-dropped"]
    record = harness_record.for_run(meter.run_id, meter.root)
    assert record is not None
    content = record.read("@scoring-diagnostics")
    assert content
    entries = [json.loads(line) for line in content.splitlines()]
    assert len(entries) == 50
    assert record.read("@scoring-diagnostics-dropped") == "5"
    assert all(len(entry["message"]) == 512 for entry in entries)
    assert entries[0]["message"].startswith("[redacted] [redacted] [redacted] ")
    assert "PRIVATE_CASE" not in content


@pytest.mark.parametrize("missing", [True, False])
def test_diagnostics_unavailable_are_dropped(monkeypatch, missing):
    def unavailable(*args):
        if missing:
            return None
        raise OSError("private storage unavailable")

    monkeypatch.setattr(harness_record, "for_run", unavailable)
    collector = diagnostics.FailureDiagnostics(
        SimpleNamespace(run_id="missing", root=None)
    )
    collector.record_failure(
        {"class": "ConnectionError", "message": "TLS failed"},
        Case(inputs="secret"),
        True,
    )
    collector.dropped = 5
    collector.flush()


@pytest.mark.parametrize(
    "value, expression",
    [
        (12345, "repr(case.inputs)"),
        (12.345, "repr(case.inputs)"),
        ("Line one\nline two", "repr(case.inputs)"),
        ('a "quoted" order ☃', "json.dumps(case.inputs)"),
        ('a "quoted" order ☃', "json.dumps(case.inputs, ensure_ascii=False)"),
    ],
)
def test_scrub_numeric_and_escaped_values(
    git_repo, private, private_run, protocol_backend, value, expression
):
    sha = commit_evaluator(
        git_repo,
        f"""import json
def evaluate(case):
    raise ValueError("bad input " + {expression})
""",
    )
    case = Case(name="hidden-case", inputs={"order": value}, expected_output="expected")
    score(git_repo, sha, cases=[case])
    record = harness_record.for_run("sandbox-test", git_repo)
    assert record is not None
    content = record.read("@scoring-diagnostics")
    assert content is not None
    entry = json.loads(content)
    assert entry["class"] == "ValueError"
    assert "case_id" not in entry
    assert entry["message"].startswith("bad input ")
    for secret in {
        str(value),
        repr(value)[1:-1] if isinstance(value, str) else str(value),
        json.dumps(value)[1:-1] if isinstance(value, str) else str(value),
        json.dumps(value, ensure_ascii=False)[1:-1]
        if isinstance(value, str)
        else str(value),
    }:
        assert secret not in entry["message"]
    assert "order" not in entry["message"]


def test_scrub_does_not_collect_booleans():
    assert diagnostics._strings([True, False]) == set()


@pytest.mark.parametrize(
    "suffix, scrubbed_suffix",
    [("", ""), (" 1 ", " [redacted] "), (" attempt=1", " [redacted]=[redacted]")],
)
def test_short_values_preserve_tls_diagnostic(
    git_repo, private, private_run, protocol_backend, suffix, scrubbed_suffix
):
    message = (
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "unable to get local issuer certificate (_ssl.c:1006)"
    )
    sha = commit_evaluator(
        git_repo,
        f"def evaluate(case):\n    raise ConnectionError({message + suffix!r})\n",
    )
    score(git_repo, sha, cases=[Case(name="c", inputs={"attempt": 1})])
    record = harness_record.for_run("sandbox-test", git_repo)
    assert record is not None
    content = record.read("@scoring-diagnostics")
    assert content is not None
    entry = json.loads(content)
    assert entry["class"] == "ConnectionError"
    assert entry["message"] == message + scrubbed_suffix
    assert "case_id" not in entry


@pytest.mark.parametrize(
    "value, exception_class, expected",
    [
        ("c", "ConnectionError", "ConnectionError"),
        ("Con", "ConnectionError", "ConnectionError"),
        ("Conn", "ConnectionError", "EvaluationError"),
        ("ConnectionError", "ConnectionError", "EvaluationError"),
        ("connectionerror", "ConnectionError", "ConnectionError"),
        ("c", "c", "EvaluationError"),
    ],
)
def test_exception_class_redaction_requires_exact_or_long_value(
    value, exception_class, expected
):
    entry = diagnostics._entry(
        {"class": exception_class, "message": "failure"}, Case(inputs=value), True
    )
    assert entry["class"] == expected


@pytest.mark.parametrize(
    "value, message, expected",
    [
        (1, "1 1006 1.1 attempt=1", "[redacted] 1006 1.1 attempt=[redacted]"),
        ("abc", "abc abcdef _abc abc.def", "[redacted] abcdef _abc abc.def"),
        ("abcd", "abcd abcdef", "[redacted] [redacted]ef"),
        (1.1, "1.1 11.1", "[redacted] 11.1"),
        (1, "attempt=1. Retry 1!", "attempt=[redacted]. Retry [redacted]!"),
    ],
)
def test_short_message_values_require_token_boundaries(value, message, expected):
    entry = diagnostics._entry(
        {"class": "ValueError", "message": message}, Case(inputs=value), True
    )
    assert entry["message"] == expected


def test_escaping_evaluator_exception_aborts(
    git_repo, private, private_run, protocol_backend, monkeypatch
):
    from pydantic_ai_gepa.cli.spend import EvalSpendMeter

    sha = commit_evaluator(
        git_repo,
        """from pydantic_ai_gepa.exceptions import UsageBudgetExceeded
def evaluate(case):
    if case.name == "abort":
        raise UsageBudgetExceeded("escaping evaluator failure")
    return "good"
""",
    )
    started = []
    rollout = EvalSpendMeter.rollout

    def tracked_rollout(self):
        started.append(True)
        return rollout(self)

    monkeypatch.setattr(EvalSpendMeter, "rollout", tracked_rollout)
    with pytest.raises(sandbox.ScoringSandboxError, match="exited"):
        score(
            git_repo,
            sha,
            cases=[
                Case(name="abort", inputs="first"),
                Case(name="never-start", inputs="second"),
            ],
        )
    assert len(started) == 1


def test_drop_count_flushed_when_eval_aborts(
    git_repo, private, private_run, protocol_backend, monkeypatch
):
    monkeypatch.setattr(diagnostics, "MAX_ENTRIES", 1)
    sha = commit_evaluator(
        git_repo,
        """from pydantic_ai_gepa.exceptions import UsageBudgetExceeded
def evaluate(case):
    if case.name == "abort":
        raise UsageBudgetExceeded("escaping failure")
    raise ValueError("failure")
""",
    )
    with pytest.raises(sandbox.ScoringSandboxError, match="exited"):
        score(
            git_repo,
            sha,
            cases=[
                Case(name=name, inputs="secret")
                for name in ["first", "second", "abort"]
            ],
        )
    record = harness_record.for_run("sandbox-test", git_repo)
    assert record is not None
    assert record.read("@scoring-diagnostics-dropped") == "1"
