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


def test_scrub_before_bound_and_cap(git_repo, private, private_run):
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
    for _ in range(55):
        diagnostics.record_failure(value, case, True, meter)
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
    diagnostics.record_failure(
        {"class": "ConnectionError", "message": "TLS failed"},
        Case(inputs="secret"),
        True,
        SimpleNamespace(run_id="missing", root=None),
    )
