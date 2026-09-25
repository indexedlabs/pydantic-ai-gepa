"""Offline coverage for the managed CLI's durable dollar budget."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from pathlib import Path

import pytest
import typer
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case

from pydantic_ai_gepa.cli.eval import run_eval_once
from pydantic_ai_gepa.cli.layout import config_path, run_state_path
from pydantic_ai_gepa.cli.run import RunState, _evaluate_validation_candidate
from pydantic_ai_gepa.cli.runs import ParetoLog
from pydantic_ai_gepa.cli.spend import evaluation_spend, spend_report
from pydantic_ai_gepa.evaluation import evaluate_callable_dataset
from pydantic_ai_gepa.spend import current_rollout_capability
from tests.cli import test_run_cli, test_git_candidate_cli, test_select_cli
from tests.cli.test_run_cli import _run, _run_payload

repo = test_run_cli.repo
git_repo = test_git_candidate_cli.git_repo
lane_repo = test_select_cli.git_repo


def _price(repo: Path, dollars: float = 0.01) -> None:
    module = repo / "agent_pkg" / "pricing.py"
    module.write_text(f"def price(response):\n    return {dollars}\n")
    path = config_path(repo)
    path.write_text('price_fn = "agent_pkg.pricing:price"\n' + path.read_text())


def _start(cap: str, *args: str):
    return _run(
        "run",
        "start",
        "--max-token-cost",
        cap,
        "--size",
        "2",
        "--acceptance-repetitions",
        "1",
        "--max-iterations",
        "20",
        *args,
    )


def _eval(repo: Path, run_id: str, **kwargs):
    options = dict(
        candidate_file=None,
        minibatch_id=None,
        size=2,
        seed=0,
        epoch=0,
        run_id=run_id,
        concurrency=4,
        max_iterations=100,
        threshold=0.999,
        workspace_root=repo,
    )
    options.update(kwargs)
    return run_eval_once(**options)


@pytest.mark.parametrize("cap", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_caps(repo: Path, cap: str):
    assert _start(cap).exit_code == 2
    assert _run("eval", "--max-token-cost", cap).exit_code == 2


def test_cap_persists_old_states_load_and_resume_keeps_spend(repo: Path):
    _price(repo)
    started = _start("0.155")
    assert started.exit_code == 0, started.output
    payload = _run_payload(started.output)
    run_id = payload["run_id"]
    assert payload["max_token_cost"] == 0.155
    assert payload["spend"]["total_dollars"] == pytest.approx(0.08)
    old = json.loads(run_state_path(run_id).read_text())
    old.pop("max_token_cost")
    assert RunState.from_dict(old).max_token_cost is None
    resumed = _run("run", "resume", "--run-id", run_id, "--reflector", "replacement")
    assert resumed.exit_code == 0, resumed.output
    assert spend_report(run_id)["total_dollars"] == pytest.approx(0.08)
    continued = _run("run", "continue", "--run-id", run_id)
    assert continued.exit_code == 0, continued.output
    assert _run_payload(continued.output)["spend"]["total_dollars"] == pytest.approx(
        0.14
    )
    stopped = _run("eval", "--run-id", run_id, "--size", "2")
    assert stopped.exit_code == 70, stopped.output
    final = _run_payload(stopped.output)
    assert final["status"] == "done"
    assert final["best_candidate_id"] == payload["best_candidate_id"]
    assert final["last_comparison"]["reason_code"] == "cost_budget_exhausted"
    assert final["spend"]["total_dollars"] == pytest.approx(0.14)
    assert final["spend"]["stopped_by_cost"]
    status = _run_payload(_run("run", "status", "--run-id", run_id).output)
    assert status["spend"] == final["spend"]
    report = Path(final["final_report_path"]).read_text()
    assert '"total_dollars":' in report
    from pydantic_ai_gepa.cli.events import list_events

    assert any(event.type == "run_done" for event in list_events(run_id))


def test_start_projection_stop_keeps_best(repo: Path):
    _price(repo)
    dataset = repo / ".gepa" / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"name": "one", "inputs": "?", "expected_output": "Paris"}) + "\n"
    )
    result = _start("0.025", "--size", "1")
    assert result.exit_code == 70, result.output
    payload = _run_payload(result.output)
    assert payload["best_candidate_id"]
    assert payload["spend"]["total_dollars"] == pytest.approx(0.02)
    assert payload["spend"]["max_token_cost"] == 0.025


def test_backstop_records_overshoot_and_never_publishes_partial_eval(repo: Path):
    _price(repo, 0.1)
    with pytest.raises(typer.Exit) as stopped:
        _eval(repo, "backstop", max_token_cost=0.05)
    assert stopped.value.exit_code == 70
    report = spend_report("backstop", repo, 0.05)
    assert report["total_dollars"] == pytest.approx(0.1)
    assert report["by_model"]["test"]["requests"] == 1
    assert not ParetoLog("backstop", repo).iter_rows()
    rows = [
        json.loads(line)
        for line in (repo / ".gepa/runs/backstop/spend.jsonl").read_text().splitlines()
    ]
    assert sum(row["rollouts_started"] for row in rows) == 1
    assert sum(row["rollouts_completed"] for row in rows) == 0


def test_unpriced_model_fails_closed_but_uncapped_usage_is_reported(repo: Path):
    result = _start("1")
    assert result.exit_code == 70, result.output
    report = _run_payload(result.output)["spend"]
    assert "test" in report["stop_reason"]
    assert report["unpriced_usage"]["test"]["requests"] == 1
    outcome = _eval(repo, "uncapped")
    assert outcome.summary["spend"]["unpriced_usage"]["test"]["requests"] == 2


async def _metered_agent(name="student"):
    capability = current_rollout_capability()
    assert capability is not None
    return await Agent(TestModel(custom_output_text="ok", model_name=name)).run(
        "?", capabilities=[capability]
    )


def test_callable_and_judge_are_metered_and_context_does_not_leak(tmp_path: Path):
    async def evaluate(case):
        return (await _metered_agent()).output

    async def metric(case, output):
        await _metered_agent("judge")
        return 1.0

    with evaluation_spend(
        run_id="callable",
        root=tmp_path,
        eval_id="eval-1",
        kind="training",
        count=2,
        cap=0.2,
        price_fn=lambda response: 0.02,
    ):
        records = asyncio.run(
            evaluate_callable_dataset(
                evaluate=evaluate,
                metric=metric,
                dataset=[Case(inputs="a"), Case(inputs="b")],
                concurrency=1,
            )
        )
    assert len(records) == 2
    report = spend_report("callable", tmp_path)
    assert report["total_dollars"] == pytest.approx(0.08)
    assert report["by_model"]["student"]["requests"] == 2
    assert report["by_model"]["judge"]["requests"] == 2
    assert current_rollout_capability() is None


def test_callable_without_hook_fails_closed_only_with_cap(git_repo: Path):
    result = _run("eval", "--run-id", "no-cap", "--capture-traces")
    assert result.exit_code == 0, result.output
    assert spend_report("no-cap")["total_dollars"] == 0
    result = _run(
        "eval", "--run-id", "cap", "--capture-traces", "--max-token-cost", "1"
    )
    assert result.exit_code == 70, result.output
    assert "Evaluate callable reported no spend" in result.output
    assert not ParetoLog("cap").iter_rows()


def test_expensive_validation_uses_own_projection_and_stays_within_one_rollout(
    tmp_path: Path,
):
    async def evaluate(case):
        return (await _metered_agent()).output

    def batch(kind, count, price):
        with evaluation_spend(
            run_id="probe",
            root=tmp_path,
            eval_id=kind,
            kind=kind,
            count=count,
            cap=0.60,
            price_fn=lambda response: price,
        ):
            return asyncio.run(
                evaluate_callable_dataset(
                    evaluate=evaluate,
                    metric=lambda c, o: 1.0,
                    dataset=[Case(inputs="?") for _ in range(count)],
                    concurrency=1,
                )
            )

    batch("training", 6, 0.005)
    with pytest.raises(typer.Exit):
        batch("validation", 4, 0.50)
    report = spend_report("probe", tmp_path)
    assert report["total_dollars"] == pytest.approx(0.53)
    assert report["validation_dollars"] == pytest.approx(0.50)
    assert report["total_dollars"] <= 0.60 + 0.50
    assert report["by_model"]["student"]["requests"] == 7


def test_select_validation_refuses_projected_batch_and_preserves_incumbent(repo: Path):
    _price(repo)
    cfg = config_path(repo)
    cfg.write_text(
        f'validation_dataset = "{repo.parent / "validation.jsonl"}"\n' + cfg.read_text()
    )
    (repo.parent / "validation.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"name": f"secret-{i}", "inputs": "?", "expected_output": "Paris"}
            )
            for i in range(2)
        )
        + "\n"
    )
    started = _start("0.175")
    assert started.exit_code == 0, started.output
    run_id = _run_payload(started.output)["run_id"]
    state = RunState.from_dict(json.loads(run_state_path(run_id).read_text()))
    state, outcome = _evaluate_validation_candidate(state, workspace_root=repo)
    assert outcome.summary["spend"]["validation_dollars"] == pytest.approx(0.08)
    with pytest.raises(typer.Exit):
        _evaluate_validation_candidate(state, workspace_root=repo)
    report = spend_report(run_id, repo)
    assert report["total_dollars"] == pytest.approx(0.16)
    assert report["validation_dollars"] == pytest.approx(0.08)
    assert report["stopped_by_cost"]
    ledger = (repo / ".gepa/runs" / run_id / "spend.jsonl").read_text()
    assert "secret-" not in ledger
    assert "case_id" not in ledger and "score" not in ledger
    status = _run_payload(_run("run", "status", "--run-id", run_id).output)
    assert "secret-" not in json.dumps(status)
    assert status["spend"] == dict(report, max_token_cost=0.175)


def _lane_eval(root, name):
    try:
        _eval(Path(root), "parallel", lane=name, max_token_cost=0.2)
    except typer.Exit as exc:
        assert exc.exit_code == 70


def test_concurrent_lane_processes_share_ledger_without_lost_spend(repo: Path):
    _price(repo, 0.02)
    ctx = multiprocessing.get_context("fork")
    processes = [
        ctx.Process(target=_lane_eval, args=(str(repo), str(i))) for i in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    report = spend_report("parallel", repo)
    assert report["total_dollars"] == pytest.approx(0.12)
    assert report["by_model"]["test"]["requests"] == 6
    assert len(ParetoLog("parallel", repo).iter_rows()) == 3


def _killed_eval(root):
    with evaluation_spend(
        run_id="killed",
        root=Path(root),
        eval_id="eval",
        kind="training",
        count=1,
        cap=1,
        price_fn=lambda r: 0.01,
    ):
        asyncio.run(_metered_agent())
        os._exit(0)  # Simulate termination before evaluation's finally block.


def test_response_checkpoint_survives_process_exit(tmp_path: Path):
    process = multiprocessing.get_context("fork").Process(
        target=_killed_eval, args=(str(tmp_path),)
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 0
    assert spend_report("killed", tmp_path)["total_dollars"] == pytest.approx(0.01)


def test_cli_callable_prices_student_and_judge(git_repo: Path):
    (git_repo / "task_pkg/evaluation.py").write_text("""
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa.spend import current_rollout_capability
async def call(name):
    return await Agent(TestModel(custom_output_text="ok", model_name=name)).run(
        "?", capabilities=[current_rollout_capability()])
async def evaluate(case):
    return (await call("student")).output
async def metric(case, output):
    await call("judge")
    return 1.0
""")
    (git_repo / "task_pkg/pricing.py").write_text("def price(response): return 0.02\n")
    cfg = config_path(git_repo)
    cfg.write_text(
        'price_fn = "task_pkg.pricing:price"\nmetric = "task_pkg.evaluation:metric"\n'
        + cfg.read_text()
    )
    result = _run("eval", "--run-id", "metered", "--max-token-cost", "0.1")
    assert result.exit_code == 0, result.output
    report = spend_report("metered")
    assert report["total_dollars"] == pytest.approx(0.04)
    assert report["by_model"]["student"]["requests"] == 1
    assert report["by_model"]["judge"]["requests"] == 1


def test_reflector_replay_does_not_recharge_paid_eval(repo: Path, monkeypatch):
    from pydantic_ai_gepa.cli import run as run_module

    _price(repo)
    started = _start("0.5")
    assert started.exit_code == 0, started.output
    run_id = _run_payload(started.output)["run_id"]
    original = run_module.run_eval_once

    def interrupted(**kwargs):
        original(**kwargs)
        raise RuntimeError("process died after recording paid eval")

    monkeypatch.setattr(run_module, "run_eval_once", interrupted)
    result = _run("run", "continue", "--run-id", run_id)
    assert result.exit_code == 1
    assert spend_report(run_id)["total_dollars"] == pytest.approx(0.10)
    monkeypatch.setattr(run_module, "run_eval_once", original)
    result = _run("run", "resume", "--run-id", run_id, "--reflector", "new-worker")
    assert result.exit_code == 0, result.output
    assert spend_report(run_id)["total_dollars"] == pytest.approx(0.10)
    result = _run("run", "continue", "--run-id", run_id)
    assert result.exit_code == 0, result.output
    assert spend_report(run_id)["total_dollars"] == pytest.approx(0.14)


def test_select_command_refuses_validation_over_cap(lane_repo: Path):
    module = lane_repo / "task_pkg/evaluation.py"
    module.write_text(
        module.read_text()
        + """
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_gepa.spend import current_rollout_capability
_original = evaluate
async def evaluate(case):
    name = "validation" if case.name.startswith("secret-") else "training"
    await Agent(TestModel(custom_output_text="ok", model_name=name)).run(
        "?", capabilities=[current_rollout_capability()])
    return await _original(case)
"""
    )
    (lane_repo / "task_pkg/pricing.py").write_text(
        'def price(response): return 0.10 if response.model_name == "validation" else 0.005\n'
    )
    validation = lane_repo.parent / "validation.jsonl"
    validation.write_text(
        json.dumps({"name": "secret-holdout", "inputs": "x", "expected_output": "v"})
        + "\n"
    )
    config = config_path(lane_repo)
    config.write_text(
        f'price_fn = "task_pkg.pricing:price"\nvalidation_dataset = "{validation}"\n'
        + config.read_text()
    )
    test_select_cli._git(lane_repo, "add", ".")
    test_select_cli._git(lane_repo, "commit", "-m", "Meter fake student")
    payload = test_select_cli._start_lane_run(lane_repo, 1, "--max-token-cost", "0.45")
    run_id = payload["run_id"]
    incumbent = payload["best_candidate_id"]
    test_select_cli._drive_lane(
        lane_repo,
        run_id,
        "lane-1",
        {
            "out_case-2.txt": "b\n",
            "out_secret-holdout.txt": "v\n",
        },
    )
    before = spend_report(run_id, lane_repo)
    assert before["total_dollars"] == pytest.approx(0.405)
    selected = test_select_cli._select(lane_repo, run_id)
    assert selected.exit_code == 70, selected.output
    state = test_select_cli._state(lane_repo, run_id)
    assert state.status == "done"
    assert state.best_candidate_id == incumbent
    assert state.last_comparison["reason_code"] == "cost_budget_exhausted"
    report = spend_report(run_id, lane_repo)
    assert report["total_dollars"] == before["total_dollars"]
    assert report["validation_dollars"] == pytest.approx(0.30)
    assert "secret-holdout" not in selected.output


def test_backstop_after_an_observed_price_jump(repo: Path):
    _price(repo)
    (repo / "agent_pkg/pricing.py").write_text("""
n = 0
def price(response):
    global n
    n += 1
    return 0.005 if n < 3 else 0.1
""")
    _eval(repo, "jump", size=1, max_token_cost=0.06)
    with pytest.raises(typer.Exit):
        _eval(repo, "jump", max_token_cost=0.06)
    report = spend_report("jump", repo)
    assert report["total_dollars"] == pytest.approx(0.11)
    assert report["by_model"]["test"]["requests"] == 3
    assert len(ParetoLog("jump", repo).iter_rows()) == 1


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_or_interrupted_evaluation_keeps_spend(tmp_path: Path, error):
    with pytest.raises(error):
        with evaluation_spend(
            run_id="failed",
            root=tmp_path,
            eval_id="eval",
            kind="gate",
            count=1,
            cap=1,
            price_fn=lambda r: 0.01,
        ):
            asyncio.run(_metered_agent())
            raise error("interrupted after paid response")
    assert spend_report("failed", tmp_path)["total_dollars"] == pytest.approx(0.01)


def test_gate_complement_and_probe_have_separate_spend_kinds(repo: Path):
    _price(repo)
    gate = _eval(
        repo,
        "kinds",
        selected_case_ids=("case-paris",),
        write_pareto=False,
        max_token_cost=1,
    )
    _eval(
        repo,
        "kinds",
        selected_case_ids=("case-berlin",),
        supplemental_records=gate.records,
        max_token_cost=1,
    )
    _eval(
        repo,
        "kinds",
        case_id="case-paris",
        row_scope="probe",
        write_pareto=False,
        max_token_cost=1,
    )
    rows = [
        json.loads(line)
        for line in (repo / ".gepa/runs/kinds/spend.jsonl").read_text().splitlines()
    ]
    assert {row["kind"] for row in rows} == {"gate", "training", "probe"}
    for kind in ("gate", "training", "probe"):
        assert sum(
            row["total_dollars"] for row in rows if row["kind"] == kind
        ) == pytest.approx(0.01)
    assert spend_report("kinds", repo)["total_dollars"] == pytest.approx(0.03)


@pytest.mark.parametrize("cap,warmup", [(10, False), (None, False), (0.35, True)])
def test_adaptive_concurrency_uses_highest_cost_and_preserves_parallelism(
    tmp_path: Path, cap: float | None, warmup: bool
):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    active = peak = 0

    async def model(messages, info):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return ModelResponse(parts=[TextPart("ok")])
        finally:
            active -= 1

    agent = Agent(FunctionModel(model, model_name="fake"))

    async def evaluate(case):
        return (
            await agent.run("?", capabilities=[current_rollout_capability()])
        ).output

    def batch(eval_id, count, cap, price, concurrency):
        with evaluation_spend(
            run_id="adaptive",
            root=tmp_path,
            eval_id=eval_id,
            kind="training",
            count=count,
            cap=cap,
            price_fn=price,
            concurrency=concurrency,
        ):
            return asyncio.run(
                evaluate_callable_dataset(
                    evaluate=evaluate,
                    metric=lambda c, o: 1.0,
                    dataset=[Case(inputs="?") for _ in range(count)],
                    concurrency=concurrency,
                )
            )

    if warmup:
        costs = iter([0.1, 0.0])
        batch("seed", 2, 1, lambda r: next(costs), 1)
        peak = 0
        # Mean=.05, highest=.10. Remaining=.25 fits the complete batch's
        # mean (.20), but is below concurrency * highest (.40): serial starts.
        records = batch("near", 4, 0.35, lambda r: 0.05, 4)
        assert len(records) == 4
        assert peak == 1
    else:
        # First observation is serial; subsequent cases can use all four slots.
        records = batch("room", 8, cap, lambda r: 0.02, 4)
        assert len(records) == 8
        assert peak == 4


def _reserved_worker(root, eval_id, ready, release):
    async def evaluate(case):
        return (await _metered_agent()).output

    with evaluation_spend(
        run_id="overlap",
        root=Path(root),
        eval_id=eval_id,
        kind="training",
        count=2,
        cap=0.2,
        price_fn=lambda r: 0.02,
        concurrency=2,
    ):
        ready.put(eval_id)
        assert release.wait(10)
        asyncio.run(
            evaluate_callable_dataset(
                evaluate=evaluate,
                metric=lambda c, o: 1.0,
                dataset=[Case(inputs="a"), Case(inputs="b")],
                concurrency=2,
            )
        )


def test_reservations_allow_two_eval_processes_in_flight(repo: Path):
    _price(repo, 0.02)
    _eval(repo, "overlap", size=1, max_token_cost=0.2)
    ctx = multiprocessing.get_context("fork")
    ready, release = ctx.Queue(), ctx.Event()
    processes = [
        ctx.Process(target=_reserved_worker, args=(str(repo), str(i), ready, release))
        for i in range(2)
    ]
    for process in processes:
        process.start()
    try:
        # Both admissions finish before either eval is allowed to settle.
        assert {ready.get(timeout=10), ready.get(timeout=10)} == {"0", "1"}
    finally:
        release.set()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    assert spend_report("overlap", repo)["total_dollars"] == pytest.approx(0.10)


def _reserve_then_exit(root):
    with evaluation_spend(
        run_id="stale",
        root=Path(root),
        eval_id="dead",
        kind="training",
        count=3,
        cap=0.09,
        price_fn=lambda r: 0.02,
    ):
        os._exit(0)


def test_dead_process_reservation_is_reclaimed(repo: Path):
    _price(repo, 0.02)
    _eval(repo, "stale", size=1, max_token_cost=0.09)
    process = multiprocessing.get_context("fork").Process(
        target=_reserve_then_exit, args=(str(repo),)
    )
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 0
    reservations = repo / ".gepa/runs/stale/spend-reservations.json"
    assert json.loads(reservations.read_text())["dead"]["dollars"] == pytest.approx(
        0.06
    )
    _eval(repo, "stale", max_token_cost=0.09)
    assert json.loads(reservations.read_text()) == {}
    assert spend_report("stale", repo)["total_dollars"] == pytest.approx(0.06)


def _refused_eval(root):
    with pytest.raises(typer.Exit) as exc:
        _eval(Path(root), "reserved", max_token_cost=0.1)
    assert exc.value.exit_code == 70


def test_admission_counts_another_process_reservation(repo: Path):
    _price(repo, 0.02)
    _eval(repo, "reserved", size=1, max_token_cost=0.1)
    with evaluation_spend(
        run_id="reserved",
        root=repo,
        eval_id="in-flight",
        kind="training",
        count=3,
        cap=0.1,
        price_fn=lambda r: 0.02,
    ):
        process = multiprocessing.get_context("fork").Process(
            target=_refused_eval, args=(str(repo),)
        )
        process.start()
        process.join(timeout=20)
        assert process.exitcode == 0
        assert spend_report("reserved", repo)["total_dollars"] == pytest.approx(0.02)


def test_price_jump_drains_all_inflight_paid_responses(tmp_path: Path):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    async def evaluate(case):
        return (await _metered_agent()).output

    with evaluation_spend(
        run_id="burst",
        root=tmp_path,
        eval_id="seed",
        kind="training",
        count=1,
        cap=0.1,
        price_fn=lambda r: 0.005,
    ):
        asyncio.run(
            evaluate_callable_dataset(
                evaluate=evaluate,
                metric=lambda c, o: 1.0,
                dataset=[Case(inputs="?")],
                concurrency=1,
            )
        )

    async def expensive_batch():
        started = 0
        ready = asyncio.Event()

        async def model(messages, info):
            nonlocal started
            started += 1
            if started == 4:
                ready.set()
            await asyncio.wait_for(ready.wait(), timeout=2)
            return ModelResponse(parts=[TextPart("ok")])

        agent = Agent(FunctionModel(model, model_name="expensive"))

        async def expensive(case):
            return (
                await agent.run("?", capabilities=[current_rollout_capability()])
            ).output

        try:
            await evaluate_callable_dataset(
                evaluate=expensive,
                metric=lambda c, o: 1.0,
                dataset=[Case(inputs="?") for _ in range(8)],
                concurrency=4,
            )
        finally:
            assert started == 4

    with pytest.raises(typer.Exit):
        with evaluation_spend(
            run_id="burst",
            root=tmp_path,
            eval_id="expensive",
            kind="training",
            count=8,
            cap=0.1,
            price_fn=lambda r: 0.5,
            concurrency=4,
        ):
            asyncio.run(expensive_batch())
    report = spend_report("burst", tmp_path)
    assert report["total_dollars"] == pytest.approx(2.005)
    assert report["by_model"]["expensive"]["requests"] == 4
    assert report["stopped_by_cost"]
