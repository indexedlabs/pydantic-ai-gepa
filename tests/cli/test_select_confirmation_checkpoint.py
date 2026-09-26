"""Select resumes saved confirmation evidence and retries unfinished draws."""

from dataclasses import replace
import json

import pytest
import typer

from pydantic_ai_gepa.cli import harness_record, lane_repositories, run, select
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.evaluation import EvaluationRecord
from pydantic_ai_gepa.types import RolloutOutput
from tests.cli import test_harness_record_select_paths as flows

git_repo = flows.git_repo
lane_run = flows.lane_run


class Crash(RuntimeError):
    pass


def _state(record):
    return run.RunState.from_dict(
        json.loads(record.read("state.json"))
    ).restore_validation_evidence(record.root)


def _confirmation(record):
    files = record.load()["files"]
    return json.loads(
        next(
            value
            for key, value in files.items()
            if key.startswith("@select-confirmation/")
        )
    )


def _validation_rows(record):
    return [
        row.to_dict()
        for row in ParetoLog(record.run_id, record.root).iter_rows()
        if row.extra.get("dataset_role") == "validation"
    ]


def _assert_withheld(record):
    # Inspect the actual published files as well as their private authority.
    public = [
        value
        for key, value in record.load()["files"].items()
        if not key.startswith("@")
    ]
    public += [path.read_text() for path in record.directory.rglob("*.json")]
    public.append((record.root / ".gepa/journal.jsonl").read_text())
    for content in public:
        assert '"secret"' not in content
        assert '"secret-pair"' not in content
        assert '"scores"' not in content
        assert '"finalist"' not in content
        assert '"pending"' not in content


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize(
    "point",
    [
        "control",
        "winner_journal",
        "accepted_promotion",
        "promote",
        "promote_checkpoint",
        "journal",
        "refan",
        "rebaseline",
    ],
)
def test_confirmation_survives_every_promotion_crash(
    lane_run, monkeypatch, point, paired
):
    record, winner, _ = lane_run
    if paired:
        replace(_state(record), acceptance_paired_min_cases=2).save(record.root)
        monkeypatch.setattr(run, "_validation_schedule", lambda *_: (1, 1))
    calls = []
    evaluate = run._evaluate_validation_candidate

    def counted(*args, **kwargs):
        calls.append(kwargs.get("lane"))
        fresh, outcome = evaluate(*args, **kwargs)
        if paired:
            # The paired comparator requires two cases; inject another fake
            # case with the same deterministic score into this fixture.
            outcome.records.append(
                EvaluationRecord("secret-pair", outcome.records[0].score, None, {})
            )
        return fresh, outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", counted)

    def crash(*args, **kwargs):
        raise Crash(point)

    with monkeypatch.context() as patch:
        if point == "winner_journal":
            original = select._journal_lane_outcome

            def journal(root, entry):
                if entry["outcome"] == "promoted":
                    crash()
                return original(root, entry)

            patch.setattr(select, "_journal_lane_outcome", journal)
        elif point == "accepted_promotion":
            patch.setattr(select, "_record_accepted_promotion", crash)
        elif point == "promote":
            patch.setattr(lane_repositories.Repositories, "promote", crash)
        elif point == "promote_checkpoint":
            original = select._checkpoint

            def checkpoint(state, root, phase, ctx):
                if phase == "promote" and "accepted_promotion_count" in ctx:
                    crash()
                return original(state, root, phase, ctx)

            patch.setattr(select, "_checkpoint", checkpoint)
        elif point == "incumbent_evidence_write":
            from pydantic_ai_gepa.cli import validation

            original = validation.write_validation_evidence

            def write_evidence(*args, **kwargs):
                original(*args, **kwargs)
                if kwargs["scores"].get("secret") == 1.0:
                    crash()

            patch.setattr(validation, "write_validation_evidence", write_evidence)
        elif point == "refan":
            original = select._refan_lane

            def refan(*args, **kwargs):
                original(*args, **kwargs)
                crash()

            patch.setattr(select, "_refan_lane", refan)
        elif point in {"journal", "rebaseline"}:
            patch.setattr(select, f"_phase_{point}", crash)
        if point == "control":
            select.run_select(record.run_id)
        else:
            with pytest.raises(Crash):
                select.run_select(record.run_id)

    saved = _confirmation(record)
    assert saved["comparison"]["verdict"] == "accepted"
    assert saved["finalist"]["commit_sha"] == winner.candidate_sha
    expected_samples = [1.0] * (1 if paired else 3)
    assert [
        sample["summary"]["mean_score"] for sample in saved["samples"]
    ] == expected_samples
    expected_scores = {"secret": 1.0, **({"secret-pair": 1.0} if paired else {})}
    assert all(sample["scores"] == expected_scores for sample in saved["samples"])
    rows = _validation_rows(record)
    calls_before = list(calls)
    _assert_withheld(record)
    if point != "control":
        select.run_select(record.run_id)
    assert calls == calls_before
    assert _validation_rows(record) == rows
    assert _confirmation(record) == saved
    state = _state(record)
    assert state.best_commit_sha == winner.candidate_sha
    assert state.best_validation_samples == tuple(expected_samples)
    assert state.best_validation_per_case_scores == (expected_scores if paired else {})
    assert state.accepted_promotion_count == 1
    journal = [
        json.loads(line)
        for line in (record.root / ".gepa/journal.jsonl").read_text().splitlines()
    ]
    assert sum(row.get("outcome") == "promoted" for row in journal) == 1
    assert sum(row.get("kind") == "accepted_promotion" for row in journal) == 1

    # Neither the checkpoint nor its private case identity/scores is published
    # to state, packets, lanes, events, Pareto rows, or the reflector journal.
    _assert_withheld(record)


def test_crash_between_incumbent_evidence_and_state_save(lane_run, monkeypatch):
    test_confirmation_survives_every_promotion_crash(
        lane_run, monkeypatch, "incumbent_evidence_write", True
    )


@pytest.mark.parametrize(
    "point",
    ["between_samples", "during_draw", "rejected", "inconclusive", "changed_finalist"],
)
def test_interrupted_confirmation_sequence(lane_run, monkeypatch, point):
    record, winner, _ = lane_run
    evaluate = run._evaluate_validation_candidate
    calls = []

    def counted(state, **kwargs):
        lane = kwargs.get("lane", "")
        calls.append(lane)
        fresh, outcome = evaluate(state, **kwargs)
        if lane.endswith(":confirmation") and point in {"rejected", "inconclusive"}:
            # Incumbent is zero: identical evidence is an equivalent verdict.
            for item in outcome.records:
                item.score = 0.0
            outcome.summary["mean_score"] = 0.0
        if lane.endswith(":confirmation") and point == "during_draw":
            for item in outcome.records:
                item.score = 0.8
            outcome.summary["mean_score"] = 0.8
            if calls.count(lane) == 2:
                raise Crash(point)
        return fresh, outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", counted)
    if point == "inconclusive":
        monkeypatch.setattr(
            run,
            "_validation_improved",
            lambda *_, **__: run._inconclusive_comparison("test_inconclusive"),
        )
    with monkeypatch.context() as patch:
        if point == "between_samples":
            write = harness_record.Record.write

            def save(self, key, content, **kwargs):
                write(self, key, content, **kwargs)
                data = (
                    json.loads(content)
                    if key.startswith("@select-confirmation/")
                    else {}
                )
                if len(data.get("samples", [])) == 1 and not data.get("pending"):
                    raise Crash(point)

            patch.setattr(harness_record.Record, "write", save)
        elif point in {"rejected", "inconclusive"}:
            checkpoint = select._checkpoint

            def save(state, root, phase, ctx):
                if phase == "journal":
                    raise Crash(point)
                return checkpoint(state, root, phase, ctx)

            patch.setattr(select, "_checkpoint", save)
        elif point == "changed_finalist":
            patch.setattr(
                select,
                "_journal_lane_outcome",
                lambda *_: (_ for _ in ()).throw(Crash(point)),
            )
        with pytest.raises(Crash):
            select.run_select(record.run_id)
    saved = _confirmation(record)
    rows = _validation_rows(record)
    before = len(calls)
    budget_before = ParetoLog(record.run_id, record.root).count_budget_rows()
    if point == "changed_finalist":
        replace(winner, candidate_sha="0" * 40).save(record.root, record.run_id)
    if point == "changed_finalist":
        with pytest.raises(typer.BadParameter, match="refusing to redraw"):
            select.run_select(record.run_id)
        assert len(calls) == before
        assert _validation_rows(record) == rows
        return
    select.run_select(record.run_id)
    final = _confirmation(record)
    if point in {"between_samples", "during_draw"}:
        assert len(calls) == before + 2
        assert len(_validation_rows(record)) == len(rows) + 2
        assert final["samples"][:1] == saved["samples"]
        score = 0.8 if point == "during_draw" else 1.0
        assert _state(record).best_validation_samples == (score,) * 3
        assert final["comparison"]["verdict"] == "accepted"
        assert final["comparison"]["candidate_mean"] == pytest.approx(score)
        if point == "during_draw":
            assert saved["pending"] is True
            assert len(saved["samples"]) == 1
            assert _validation_rows(record)[: len(rows)] == rows
            # The orphan's 1.0 Pareto score is neither recovered as a sample
            # nor used as the winner's mean. Its paid row still consumes budget.
            assert rows[-1]["mean_score"] == 1.0
            assert final["iterations"] == budget_before + 2
            assert _state(record).validation_evaluations == len(
                _validation_rows(record)
            )
    else:
        assert len(calls) == before
        assert _validation_rows(record) == rows
        assert final == saved
        assert _state(record).iterations_since_acceptance == (
            1 if point == "rejected" else 0
        )
        assert _state(record).accepted_promotion_count == 0


@pytest.mark.parametrize("failures, legacy", [(1, False), (2, False), (1, True)])
def test_confirmation_infrastructure_failure_retries_kept_samples(
    lane_run, monkeypatch, failures, legacy
):
    record, winner, _ = lane_run
    evaluate = run._evaluate_validation_candidate
    calls = []
    failed_samples = []

    def counted(state, **kwargs):
        lane = kwargs.get("lane", "")
        calls.append(lane)
        fresh, outcome = evaluate(state, **kwargs)
        if lane.endswith(":confirmation") and 2 <= calls.count(lane) <= failures + 1:
            outcome.records[0].payload = {
                "output": RolloutOutput(
                    result=None,
                    success=False,
                    error_message="fake transport failure",
                    error_kind="transport",
                )
            }
            outcome.records[0].score = -1.0
            outcome.summary["mean_score"] = -1.0
            failed_samples.append(outcome)
        return fresh, outcome

    monkeypatch.setattr(run, "_evaluate_validation_candidate", counted)
    before = _state(record)
    for attempt in range(failures):
        with pytest.raises(typer.Exit) as error:
            select.run_select(record.run_id)
        assert error.value.exit_code == 1
        saved = _confirmation(record)
        paused = _state(record)
        assert paused.status == "paused_after_infrastructure_error"
        assert paused.best_candidate_id == before.best_candidate_id
        assert paused.iterations_since_acceptance == before.iterations_since_acceptance
        assert (
            paused.validation_evaluations == before.validation_evaluations + 4 + attempt
        )
        assert saved["comparison"] is None
        assert saved["pending"] is False
        assert len(saved["samples"]) == 1
        assert saved["samples"][0]["summary"]["mean_score"] == 1.0
        assert (
            saved["iterations"]
            == ParetoLog(record.run_id, record.root).count_budget_rows()
        )
        _assert_withheld(record)
    if legacy:
        # Records from the first commit stored the failed draw as a terminal
        # comparison. They must recover too, without using that failed score.
        key = paused.select_context["confirmation_checkpoint"]
        legacy_checkpoint = {
            **saved,
            "comparison": paused.last_comparison,
            "samples": saved["samples"]
            + [
                {
                    "summary": failed_samples[-1].summary,
                    "scores": {"secret": -1.0},
                }
            ],
        }
        record.write(key, json.dumps(legacy_checkpoint))
    rows = _validation_rows(record)
    count_before_retry = len(calls)
    select.run_select(record.run_id)
    final = _confirmation(record)
    state = _state(record)
    assert len(calls) == count_before_retry + 2
    assert _validation_rows(record)[: len(rows)] == rows
    assert len(_validation_rows(record)) == len(rows) + 2
    assert final["samples"][:1] == saved["samples"]
    assert len(final["samples"]) == 3
    assert final["comparison"]["verdict"] == "accepted"
    assert final["iterations"] == saved["iterations"] + 2
    assert state.validation_evaluations == paused.validation_evaluations + 2
    assert state.best_validation_samples == (1.0, 1.0, 1.0)
    assert state.best_commit_sha == winner.candidate_sha
    assert state.accepted_promotion_count == 1
    assert state.iterations_since_acceptance == 0
    assert state.status == "running"
    _assert_withheld(record)
