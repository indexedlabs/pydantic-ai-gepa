"""Training-only return channel: parity, bounded input and phase isolation."""

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_evals import Case

from pydantic_ai_gepa.cli import scoring_material as material
from pydantic_ai_gepa.cli import scoring_sandbox as sandbox
from pydantic_ai_gepa.cli.eval import (
    _expose_trace_path,
    _format_failures,
    _write_trace_file,
)
from pydantic_ai_gepa.cli.layout import GepaConfig
from pydantic_ai_gepa.evaluation import EvaluationRecord, evaluate_callable_dataset
from tests.cli import test_scoring_sandbox as existing

clean_environment = existing.clean_environment
git_repo = existing.git_repo
private = existing.private
private_run = existing.private_run
protocol_backend = existing.protocol_backend
real_backend = existing.real_backend


@pytest.fixture(autouse=True)
def fake_provider_environment(monkeypatch):
    # Environment-dump tests must never persist a developer's provider keys.
    for name in sandbox.PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)


# The same source is imported locally and executed in the scoring child.
PARITY_SOURCE = """import json
import os
from pathlib import Path
from pydantic_ai_gepa.types import MetricResult

async def evaluate(case):
    trace = Path(os.environ["GEPA_TRACE_FILE"])
    with trace.open("a") as stream:
        stream.write(json.dumps({"case_id": case.name, "stage": "classify"}) + "\\n")
    artifacts = trace.with_suffix(".cases") / case.name
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "scenario.txt").write_text("scenario evidence\\n")
    (trace.parent / "routing_evidence.json").write_text(json.dumps({"route": "review"}))
    return {"answer": "review"}

def metric(case, output):
    return MetricResult(score=0.5, feedback="Review routing", side_info={"route": "review"})
"""


def raw_result(**updates):
    return dict(
        type="result",
        score=0.5,
        feedback="feedback",
        failed=False,
        cached=False,
        material={
            "output": {"answer": "review"},
            "side_info": None,
            "metric_side_info": None,
        },
        **updates,
    )


@contextmanager
def opened(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield fd
    finally:
        os.close(fd)


def collect(path, budget=None, case_used=0):
    with opened(path) as fd:
        return material.collect_files(
            fd, budget or material.MaterialBudget(), case_used
        )


@pytest.fixture
def parity_sample(git_repo, private, private_run, protocol_backend, monkeypatch):
    sha = existing.commit_evaluator(git_repo, PARITY_SOURCE)
    case = Case(
        name="training-one", inputs="training input", expected_output="expected"
    )
    config = GepaConfig(
        candidate_source="git",
        evaluate="task_pkg.evaluation:evaluate",
        metric="task_pkg.evaluation:metric",
    )
    namespace = {}
    exec(PARITY_SOURCE, namespace)
    before = (
        git_repo
        / ".gepa"
        / "runs"
        / "before"
        / "traces"
        / "minibatches"
        / "mb"
        / "0000-eval-candidate.jsonl"
    )
    with monkeypatch.context() as patch:
        patch.delenv("GEPA_HELDOUT_DATASET")
        with _expose_trace_path(before):
            unsandboxed = asyncio.run(
                evaluate_callable_dataset(
                    evaluate=namespace["evaluate"],
                    metric=namespace["metric"],
                    dataset=[case],
                )
            )
        _write_trace_file(path=before, records=unsandboxed)
    records = existing.score(
        git_repo, sha, validation=False, cases=[case], config=config
    )
    after = (
        git_repo
        / ".gepa"
        / "runs"
        / "sandbox-test"
        / "traces"
        / "minibatches"
        / "mb"
        / "0000-eval-candidate.jsonl"
    )
    material.publish_files(root=git_repo, path=after, records=records)
    return before, after, unsandboxed, records


def test_before_after_same_evaluator(parity_sample):
    before, after, unsandboxed, records = parity_sample
    before_rows = [json.loads(line) for line in before.read_text().splitlines()]
    after_rows = [json.loads(line) for line in after.read_text().splitlines()]
    assert after_rows[:-1] == before_rows
    index = after_rows[-1]
    assert index["case_id"] == "training-one"
    artifacts = [after.parent / relative for relative in index["artifacts"]]
    assert (
        next(p for p in artifacts if p.name == "scenario.txt").read_text()
        == (before.with_suffix(".cases") / "training-one" / "scenario.txt").read_text()
    )
    assert json.loads(
        next(p for p in artifacts if p.name == "routing_evidence.json").read_text()
    ) == json.loads((before.parent / "routing_evidence.json").read_text())
    assert (
        records[0].payload["sandbox_material"]["output"]
        == unsandboxed[0].payload["output"]
    )
    assert (
        records[0].payload["sandbox_material"]["side_info"]
        == unsandboxed[0].payload["side_info"]
    )
    before_report = _format_failures(unsandboxed, candidate_source="git")
    after_report = _format_failures(records, candidate_source="git")
    assert before_report.rstrip() in after_report
    assert '"answer": "review"' in after_report
    assert '"side_info": {"route": "review"}' in after_report
    # Reflection material must not become authoritative record content.
    private_records = list(
        Path(os.environ["GEPA_HELDOUT_DATASET"]).parent.rglob("*.record.json")
    )
    assert private_records
    assert all(
        "scenario evidence" not in p.read_text()
        and "routing_evidence" not in p.read_text()
        for p in private_records
    )


def test_trajectory_rows_and_metric_side_info(
    git_repo, private, private_run, protocol_backend
):
    source = """from pydantic_ai_gepa import evaluation
from pydantic_ai_gepa.evaluation import EvaluationRecord
class Trajectory:
    metric_side_info = {"grade": "partial"}
    def to_reflective_record(self):
        return {"messages": [{"content": "training message"}]}
async def patched(**kwargs):
    return [EvaluationRecord(kwargs["dataset"][0].name, 0.5, "feedback", {"trajectory": Trajectory(), "output": "answer"})]
evaluation.evaluate_callable_dataset.__code__ = patched.__code__
evaluation.evaluate_callable_dataset.__globals__.update(EvaluationRecord=EvaluationRecord, Trajectory=Trajectory)
async def evaluate(case):
    return "unused"
"""
    sha = existing.commit_evaluator(git_repo, source)
    records = existing.score(
        git_repo, sha, validation=False, cases=[Case(name="train", inputs="input")]
    )
    row = json.loads(records[0].payload["sandbox_files"]["trace.jsonl"])
    trajectory = SimpleNamespace(
        to_reflective_record=lambda: {"messages": [{"content": "training message"}]},
        metric_side_info={"grade": "partial"},
    )
    expected = (
        git_repo / ".gepa" / "runs" / "sandbox-test" / "traces" / "expected.jsonl"
    )
    _write_trace_file(
        path=expected,
        records=[
            EvaluationRecord("train", 0.5, "feedback", {"trajectory": trajectory})
        ],
    )
    assert row == json.loads(expected.read_text())
    assert (
        records[0].payload["sandbox_material"]["metric_side_info"]
        == trajectory.metric_side_info
    )


@pytest.mark.parametrize("field", ["model_name", "error", "case_id", "path"])
def test_protocol_extra_fields_refused(field):
    raw = raw_result(**{field: "untrusted"})
    with pytest.raises(sandbox.ScoringSandboxError, match="result shape"):
        sandbox._result(raw, "parent-case", False)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"output": "x"},
        {
            "output": "x" * material.MAX_PAYLOAD_BYTES,
            "side_info": None,
            "metric_side_info": None,
        },
        {"output": float("nan"), "side_info": None, "metric_side_info": None},
    ],
)
def test_bad_material_does_not_fail_score(value):
    raw = raw_result()
    raw["material"] = value
    record = sandbox._result(raw, "parent-case", False)
    assert record.score == 0.5
    assert record.case_id == "parent-case"
    assert "sandbox_material" not in record.payload
    assert material.REFUSAL_NOTE in _format_failures([record])
    # Validation does not even inspect the structured value.
    assert sandbox._result(raw, "private-case", True).payload == {}


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "/absolute.txt",
        "a/../../escape.txt",
        "x/" * 8 + "deep.txt",
        "a" * 65 + ".txt",
        ".hidden.txt",
        "a b.txt",
        "a\\b.txt",
        "é.txt",
    ],
)
def test_invalid_relative_names(name):
    assert not material.valid_relative(name)


@pytest.mark.parametrize(
    "attack",
    [
        "oversize",
        "symlink-inside",
        "symlink-outside",
        "hardlink",
        "fifo",
        "type",
        "utf8",
        "jsonl-array",
        "jsonl-invalid",
        "jsonl-nan",
        "jsonl-infinity",
        "jsonl-duplicate",
        "json-invalid",
        "deep",
        "keys",
        "nodes",
        "overdeep-path",
        "bad-name",
    ],
)
def test_malicious_files_refused_without_losing_good_file(tmp_path, attack):
    output = tmp_path / "output"
    output.mkdir()
    (output / "good.txt").write_text("retained")
    bad = output / "bad.jsonl"
    if attack == "oversize":
        bad.write_bytes(b"x" * (material.MAX_FILE_BYTES + 1))
    elif attack.startswith("symlink"):
        target = (
            output / "good.txt"
            if attack.endswith("inside")
            else tmp_path / "secret.txt"
        )
        target.write_text("retained" if attack.endswith("inside") else "secret")
        bad.symlink_to(target)
    elif attack == "hardlink":
        target = tmp_path / "secret.txt"
        target.write_text("secret")
        os.link(target, bad)
    elif attack == "fifo":
        os.mkfifo(bad)
    elif attack == "type":
        (output / "bad.py").write_text("print('no')")
    elif attack == "utf8":
        (output / "bad.txt").write_bytes(b"\xff")
    elif attack == "deep":
        bad.write_text('{"x":' * 20 + "0" + "}" * 20)
    elif attack == "keys":
        bad.write_text(
            json.dumps({str(i): i for i in range(material.MAX_JSON_KEYS + 1)})
        )
    elif attack == "nodes":
        bad.write_text(json.dumps({"x": [None] * material.MAX_JSON_NODES}))
    elif attack == "overdeep-path":
        nested = output.joinpath(*(["dir"] * 8))
        nested.mkdir(parents=True)
        (nested / "bad.txt").write_text("no")
    elif attack == "bad-name":
        (output / "bad name.txt").write_text("no")
    elif attack == "json-invalid":
        (output / "bad.json").write_text("not JSON")
    else:
        bad.write_text(
            {
                "jsonl-array": "[]",
                "jsonl-invalid": "{",
                "jsonl-nan": '{"x":NaN}',
                "jsonl-infinity": '{"x":1e999}',
                "jsonl-duplicate": '{"x":1,"x":2}',
            }[attack]
        )
    files, refused = collect(output)
    assert refused
    assert files == {"good.txt": "retained"}


def test_case_eval_and_entry_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(material, "FILE_OVERHEAD_BYTES", 0)
    monkeypatch.setattr(material, "MAX_FILE_BYTES", 100)
    monkeypatch.setattr(material, "MAX_CASE_BYTES", 250)
    monkeypatch.setattr(material, "MAX_EVAL_BYTES", 350)
    for i in range(4):
        (tmp_path / f"{i}.txt").write_text("x" * 100)
    budget = material.MaterialBudget()
    first, refused = collect(tmp_path, budget)
    assert refused and len(first) == 2 and budget.used == 208
    second, refused = collect(tmp_path, budget)
    assert refused and len(second) == 1 and budget.used == 312
    third, refused = collect(tmp_path, budget)
    assert refused and not third
    monkeypatch.setattr(material, "MAX_ENTRIES", 1)
    files, refused = collect(tmp_path)
    assert refused and len(files) == 1


def test_payload_shares_file_budget(tmp_path, monkeypatch):
    (tmp_path / "trace.txt").write_text("x" * 50)
    record = sandbox._result(raw_result(), "case", False)
    size = len(material.checked_json(record.payload["sandbox_material"]).encode())
    monkeypatch.setattr(material, "MAX_CASE_BYTES", size + 49)
    budget = material.MaterialBudget()
    with opened(tmp_path) as fd:
        sandbox._training_material(record, fd, budget)
    assert record.payload["sandbox_files"] == {}
    assert record.payload["sandbox_material_refused"]
    assert record.score == 0.5
    assert budget.used == size


def test_source_directory_swap_cannot_redirect_collection(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "good.txt").write_text("good")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    with opened(output) as fd:
        output.rename(tmp_path / "original")
        output.symlink_to(outside, target_is_directory=True)
        files, refused = material.collect_files(fd, material.MaterialBudget(), 0)
    assert files == {"good.txt": "good"}
    assert not refused


@pytest.mark.parametrize("backend", ["protocol_backend", "real_backend"])
def test_validation_never_collects_and_next_training_is_fresh(
    request, backend, git_repo, private, private_run, monkeypatch
):
    request.getfixturevalue(backend)
    source = """import json, os
from pathlib import Path
from pydantic_ai_gepa.types import MetricResult
async def evaluate(case):
    trace = Path(os.environ["GEPA_TRACE_FILE"])
    scratch = Path(os.environ["TMPDIR"])
    if case.name.startswith("CASE_SENTINEL"):
        secret = json.dumps([case.name, case.inputs, case.expected_output])
        for folder in (trace.parent, scratch, Path.cwd()):
            try:
                (folder / "validation.txt").write_text(secret)
            except OSError:
                pass
        trace.write_text(json.dumps({"secret": secret}) + "\\n")
        (trace.parent / "routing_evidence.json").write_text(json.dumps({"secret": secret}))
        print(secret)
        return secret
    visible = {"env": dict(os.environ), "cwd": sorted(os.listdir('.')), "tmp": sorted(os.listdir(scratch)), "case": {"name": case.name, "inputs": case.inputs, "expected_output": case.expected_output}}
    # Enumerate every readable file in the isolated checkout and scratch.
    for root in (Path.cwd(), scratch):
        for file in root.rglob('*'):
            if file.is_file() and file != trace:
                try:
                    visible[str(file.relative_to(root))] = file.read_text()
                except (OSError, UnicodeError):
                    pass
    trace.write_text(json.dumps(visible) + "\\n")
    return "training output"
def metric(case, output):
    return MetricResult(score=0.5, feedback=output, side_info={"value": output})
"""
    # On the host, attempt the private dataset as well as both isolated trees.
    source = source.replace(
        'visible = {"env":',
        f"""visible = {{}}
    try:
        visible["private_read"] = Path({str(private)!r}).read_text()
    except OSError:
        visible["private_read"] = "denied"
    visible.update({{"env":""",
    ).replace(
        '"expected_output": case.expected_output}}',
        '"expected_output": case.expected_output}})',
    )
    sha = existing.commit_evaluator(git_repo, source)
    config = GepaConfig(
        candidate_source="git",
        evaluate="task_pkg.evaluation:evaluate",
        metric="task_pkg.evaluation:metric",
    )
    if backend == "protocol_backend":
        # This backend cannot assert filesystem denials; do not attempt gold.
        source = source.replace(f"Path({str(private)!r}).read_text()", '"denied"')
        sha = existing.commit_evaluator(git_repo, source)
    private_case = json.loads(private.read_text())
    private_case["expected_output"] = "EXPECTED_SENTINEL_4925"
    private.write_text(json.dumps(private_case) + "\n")
    original = sandbox.collect_files
    with monkeypatch.context() as patch:
        patch.setattr(
            sandbox,
            "collect_files",
            lambda *a: pytest.fail("validation directory collected"),
        )
        original_open = sandbox.os.open

        def checked_open(path, flags, *a, **kw):
            assert not str(path).startswith("output-"), "validation output opened"
            return original_open(path, flags, *a, **kw)

        patch.setattr(sandbox.os, "open", checked_open)
        records = existing.score(
            git_repo, sha, config=config, cases=[Case(**private_case)]
        )
        assert records[0].payload == {} and records[0].feedback is None
    assert sandbox.collect_files is original
    records = existing.score(
        git_repo,
        sha,
        config=config,
        validation=False,
        cases=[
            Case(name="train", inputs="TRAIN_INPUT", expected_output="TRAIN_EXPECTED")
        ],
    )
    path = (
        git_repo
        / ".gepa"
        / "runs"
        / "sandbox-test"
        / "traces"
        / "minibatches"
        / "mb"
        / "trace.jsonl"
    )
    material.publish_files(root=git_repo, path=path, records=records)
    visible = path.read_text() + _format_failures(records)
    # The received validation case and every file it wrote are absent.
    dump = json.loads(path.read_text())
    assert dump["private_read"] == "denied"
    assert "validation.txt" not in dump["cwd"] + dump["tmp"]
    assert "CASE_SENTINEL_4833" not in visible
    assert "INPUT_SENTINEL_4833" not in visible
    assert "EXPECTED_SENTINEL_4925" not in visible
    assert all(not key.endswith("validation.txt") for key in dump)
    assert not list((private.parent / ".gepa-heldout" / "work").iterdir())


def test_agent_trajectory_is_captured(git_repo, private, private_run, protocol_backend):
    sha = existing.commit_evaluator(
        git_repo,
        """from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa.types import MetricResult
agent = Agent(TestModel(custom_output_text="training answer"))
def metric(case, output):
    return MetricResult(score=0.5, feedback="feedback", side_info={"grade": "partial"})
""",
    )
    config = GepaConfig(
        candidate_source="git",
        agent="task_pkg.evaluation:agent",
        metric="task_pkg.evaluation:metric",
    )
    records = existing.score(
        git_repo,
        sha,
        validation=False,
        config=config,
        cases=[Case(name="train", inputs="training prompt")],
    )
    row = json.loads(records[0].payload["sandbox_files"]["trace.jsonl"])
    assert row["case_id"] == "train"
    assert row["assistant_response"] == "training answer"
    assert row["metric_side_info"] == {"grade": "partial"}
    assert records[0].payload["sandbox_material"]["output"] == "training answer"


def test_malicious_training_child_keeps_score_and_safe_material(
    git_repo, private, private_run, protocol_backend
):
    sha = existing.commit_evaluator(
        git_repo,
        """import os, json
from pathlib import Path
async def evaluate(case):
    trace = Path(os.environ["GEPA_TRACE_FILE"])
    directory = trace.parent
    trace.write_text('{"case_id":"forged-case","ok":true}\\n')
    (directory / "inside.txt").write_text("inside")
    (directory / "symlink.txt").symlink_to(directory / "inside.txt")
    outside = directory.parent / "outside.txt"
    outside.write_text("outside")
    (directory / "outside-link.txt").symlink_to(outside)
    os.link(outside, directory / "hard.txt")
    os.mkfifo(directory / "fifo.txt")
    (directory / "too-big.txt").write_bytes(b'x' * (1024 * 1024 + 1))
    for i in range(5):
        (directory / f"total-{i}.txt").write_bytes(b'x' * 1024 * 1024)
    (directory / "invalid.jsonl").write_text("[]")
    (directory / "program.py").write_text("untrusted")
    (directory / "bad name.txt").write_text("untrusted")
    (directory / "directory-link").symlink_to(directory.parent, target_is_directory=True)
    return "good"
""",
    )
    records = existing.score(
        git_repo,
        sha,
        validation=False,
        cases=[Case(name="parent-case", inputs="training", expected_output="good")],
    )
    record = records[0]
    assert record.score == 1
    assert record.payload["sandbox_material"]["output"] == "good"
    assert material.REFUSAL_NOTE in _format_failures(records)
    files = record.payload["sandbox_files"]
    assert json.loads(files["trace.jsonl"])["case_id"] == "parent-case"
    assert files["inside.txt"] == "inside"
    assert set(files) <= {
        "trace.jsonl",
        "inside.txt",
        *(f"total-{i}.txt" for i in range(5)),
    }
    assert (
        sum(len(content.encode()) for content in files.values())
        < material.MAX_CASE_BYTES
    )


def test_result_payload_cannot_spoof_health_or_case():
    raw = raw_result()
    raw["material"]["output"] = {"case_id": "fake", "success": False, "score": 999}
    record = sandbox._result(raw, "parent-case", False)
    assert record.case_id == "parent-case" and record.score == 0.5
    from pydantic_ai_gepa.evaluation_health import evaluation_infrastructure_failures

    assert not evaluation_infrastructure_failures([record])


def test_racing_file_replacement_with_fifo_is_refused(tmp_path, monkeypatch):
    (tmp_path / "bad.txt").write_text("before")
    original = material.os.open

    def racing_open(path, flags, *args, **kwargs):
        if path == "bad.txt":
            (tmp_path / "bad.txt").unlink()
            os.mkfifo(tmp_path / "bad.txt")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(material.os, "open", racing_open)
    assert collect(tmp_path) == ({}, True)


def test_parent_case_binding_is_size_checked(tmp_path, monkeypatch):
    (tmp_path / "trace.jsonl").write_text("{}\n" * 10)
    monkeypatch.setattr(material, "MAX_FILE_BYTES", 100)
    with opened(tmp_path) as fd:
        files, refused = material.collect_files(
            fd, material.MaterialBudget(), 0, case_id="long-case-name"
        )
    assert refused and not files


def test_public_destination_symlink_cannot_escape(git_repo, private, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = git_repo / ".gepa" / "runs" / "sandbox-test" / "traces"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    record = EvaluationRecord(
        "case", 0.5, None, {"sandbox_files": {"safe.txt": "evidence"}}
    )
    import typer

    with pytest.raises(typer.BadParameter):
        material.publish_files(
            root=git_repo, path=target / "trace.jsonl", records=[record]
        )
    assert list(outside.iterdir()) == []


def test_run_packet_names_training_material_only(git_repo, private, protocol_backend):
    existing.commit_evaluator(git_repo, PARITY_SOURCE)
    started = existing._run(
        "run",
        "start",
        "--size",
        "1",
        "--max-iterations",
        "16",
        "--acceptance-repetitions",
        "1",
        "--acceptance-paired-min-cases",
        "2",
    )
    assert started.exit_code == 0, (started.output, started.exception)
    state = existing._run_payload(started.output)
    run = git_repo / ".gepa" / "runs" / state["run_id"]
    packet = json.loads((run / "reflector_packet.json").read_text())
    assert packet["baseline"]["report_paths"]
    assert packet["baseline"]["trace_paths"]
    for report in packet["baseline"]["report_paths"]:
        assert '"answer": "review"' in Path(report).read_text()
    for trace in packet["baseline"]["trace_paths"]:
        trace = Path(trace)
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        assert rows[0] == {"case_id": "case-1", "stage": "classify"}
        assert all((trace.parent / name).is_file() for name in rows[-1]["artifacts"])
    public = started.output + "".join(
        p.read_text() for p in run.rglob("*") if p.is_file()
    )
    assert "CASE_SENTINEL_4833" not in public
    assert "INPUT_SENTINEL_4833" not in public


def test_jsonl_normalization_stops_at_byte_limit(tmp_path, monkeypatch):
    # Small input rows can expand substantially when bound to the parent's ID.
    # Refuse without serializing every remaining row into an unbounded list.
    (tmp_path / "trace.jsonl").write_text("{}\n" * 30)
    monkeypatch.setattr(material, "MAX_FILE_BYTES", 100)
    checked = material.checked_json
    visited = []

    def counting(value):
        visited.append(value)
        return checked(value)

    monkeypatch.setattr(material, "checked_json", counting)
    with opened(tmp_path) as fd:
        files, refused = material.collect_files(
            fd, material.MaterialBudget(), 0, case_id="long-parent-case-name"
        )
    assert refused and not files
    assert len(visited) < 5


def test_large_feedback_and_payload_preserve_score(
    git_repo, private, private_run, protocol_backend
):
    empty = {"output": "", "side_info": None, "metric_side_info": None}
    output_size = material.MAX_PAYLOAD_BYTES - len(material.checked_json(empty))
    sha = existing.commit_evaluator(
        git_repo,
        f"""from pydantic_ai_gepa.types import MetricResult
async def evaluate(case):
    return "x" * {output_size}
def metric(case, output):
    return MetricResult(score=0.5, feedback=chr(0x1f642) * 65536)
""",
    )
    config = GepaConfig(
        candidate_source="git",
        evaluate="task_pkg.evaluation:evaluate",
        metric="task_pkg.evaluation:metric",
    )
    records = existing.score(git_repo, sha, config=config, validation=False)
    record = records[0]
    assert record.score == 0.5
    assert len(record.feedback) == 65536
    assert "sandbox_material" not in record.payload
    assert material.REFUSAL_NOTE in _format_failures(records)


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_jsonl_preserves_unicode_string_separators(tmp_path, separator):
    row = {"message": "before" + separator + "after"}
    (tmp_path / "trace.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    files, refused = collect(tmp_path)
    assert not refused
    assert json.loads(files["trace.jsonl"]) == row
