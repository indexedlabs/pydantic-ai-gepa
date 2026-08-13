"""Contract tests for the outer durable Omni controller."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pydantic_ai_gepa.cli import app as gepa_app
from pydantic_ai_gepa.cli.omni import _receipt, _receipt_data


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str | dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        value if isinstance(value, str) else json.dumps(value), encoding="utf-8"
    )
    return path


@pytest.fixture
def outer_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.chdir(tmp_path)
    seed = _write(tmp_path / "seed.txt", "seed")
    minibatch = _write(tmp_path / "minibatch.json", {"case_ids": ["a", "b"]})
    candidate_a = _write(tmp_path / "candidate-a.txt", "candidate-a")
    candidate_b = _write(tmp_path / "candidate-b.txt", "candidate-b")
    continuation = _write(tmp_path / "continuation.txt", "continuation")
    manifest = _write(tmp_path / "driver-manifest.json", {"adapter": "fake-v1"})
    test_set = _write(tmp_path / "test.json", {"case_ids": ["test"]})
    workspaces = [tmp_path / "work-a", tmp_path / "work-b", tmp_path / "work-two"]
    for workspace in workspaces:
        workspace.mkdir()
    plan = {
        "seed": {"artifact_path": str(seed), "sha256": _sha(seed)},
        "evaluator_identity": "fake-evaluator-v1",
        "evaluator_sha256": "a" * 64,
        "minibatch": {
            "artifact_path": str(minibatch),
            "sha256": _sha(minibatch),
            "case_count": 2,
            "case_ids": ["a", "b"],
        },
        "phase_one": [
            {
                "child_id": "alpha",
                "engine": "gepa",
                "metric_calls": 3,
                "workspace": str(workspaces[0]),
                "driver_manifest": {
                    "artifact_path": str(manifest),
                    "sha256": _sha(manifest),
                },
            },
            {
                "child_id": "beta",
                "engine": "coding_agent",
                "metric_calls": 3,
                "workspace": str(workspaces[1]),
                "driver_manifest": {
                    "artifact_path": str(manifest),
                    "sha256": _sha(manifest),
                },
            },
        ],
        "comparison": {
            "repetitions": 2,
            "metric_calls": 12,
            "phase_two_metric_calls": 8,
            "mode": "hybrid",
        },
        "phase_two": {
            "child_id": "continue",
            "engine": "autoresearch",
            "metric_calls": 4,
            "workspace": str(workspaces[2]),
            "driver_manifest": {
                "artifact_path": str(manifest),
                "sha256": _sha(manifest),
            },
        },
        "reporting": {
            "test_set": {
                "artifact_path": str(test_set),
                "sha256": _sha(test_set),
                "case_count": 1,
                "case_ids": ["test"],
            },
            "metric_calls": 1,
        },
    }
    plan_path = _write(tmp_path / "plan.json", plan)
    return {
        "root": tmp_path,
        "plan": plan,
        "plan_path": plan_path,
        "seed": seed,
        "a": candidate_a,
        "b": candidate_b,
        "continuation": continuation,
        "manifest": manifest,
    }


def _invoke(*arguments: str):
    result = CliRunner().invoke(gepa_app, ["--no-dotenv", *arguments])
    assert result.exit_code == 0, result.output
    return result


def _start(repo: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    started = _invoke("omni", "start", "--plan", str(repo["plan_path"]))
    payload = json.loads(started.output)
    state_path = Path(payload["state_path"])
    return payload["omni_id"], state_path.parent, json.loads(state_path.read_text())


def _child_receipt(
    repo: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
    child_id: str,
    candidate: Path,
    phase: str = "phase_one",
) -> dict[str, Any]:
    plan = repo["plan"]
    children = plan["phase_one"] if phase == "phase_one" else [plan["phase_two"]]
    child = next(item for item in children if item["child_id"] == child_id)
    return {
        "phase": phase,
        "child_id": child_id,
        "engine": child["engine"],
        "plan_sha256": state["plan_sha256"],
        "seed_sha256": plan["seed"]["sha256"]
        if phase == "phase_one"
        else state["winner"]["artifact_sha256"],
        "packet_sha256": _packet_digest(run_dir, phase, child_id),
        "candidate_artifact_path": str(candidate),
        "candidate_artifact_sha256": _sha(candidate),
        "metric_calls": 2,
    }


def _packet_digest(run_dir: Path, phase: str, child_id: str) -> str:
    return _sha(run_dir / "packets" / f"{phase}-{child_id}.json")


def _dispatch(
    omni_id: str, run_dir: Path, child_id: str, phase: str = "phase_one"
) -> None:
    receipt = _write(
        run_dir / f"dispatch-{phase}-{child_id}.json",
        {
            "phase": phase,
            "child_id": child_id,
            "packet_sha256": _packet_digest(run_dir, phase, child_id),
            "pid": 123,
        },
    )
    _invoke("omni", "child-dispatched", omni_id, "--receipt", str(receipt))


def _comparison(
    repo: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
    phase: str,
    candidates: list[tuple[str, Path, float]],
) -> dict[str, Any]:
    def samples(score: float) -> list[dict[str, Any]]:
        return [
            {
                "score": score,
                "case_scores": {"a": score, "b": score},
                "objective_scores": {"quality": score},
                "per_case_objective_scores": {
                    "a": {"quality": score},
                    "b": {"quality": score},
                },
            },
            {
                "score": score,
                "case_scores": {"a": score, "b": score},
                "objective_scores": {"quality": score},
                "per_case_objective_scores": {
                    "a": {"quality": score},
                    "b": {"quality": score},
                },
            },
        ]

    return {
        "phase": phase,
        "plan_sha256": state["plan_sha256"],
        "evaluator_identity": repo["plan"]["evaluator_identity"],
        "evaluator_sha256": repo["plan"]["evaluator_sha256"],
        "minibatch_sha256": repo["plan"]["minibatch"]["sha256"],
        "packet_sha256": _comparison_packet_digest(run_dir, phase),
        "metric_calls": 12 if phase == "phase_one" else 8,
        "candidates": [
            {
                "candidate_id": name,
                "artifact_path": str(path),
                "artifact_sha256": _sha(path),
                "samples": samples(score),
            }
            for name, path, score in candidates
        ],
    }


def _comparison_packet_digest(run_dir: Path, phase: str) -> str:
    return _sha(run_dir / "packets" / f"compare-{phase}.json")


def _set_comparison_samples(
    receipt: dict[str, Any], samples_by_candidate: dict[str, list[float]]
) -> None:
    for candidate in receipt["candidates"]:
        scores = samples_by_candidate[candidate["candidate_id"]]
        candidate["samples"] = [
            {
                "score": score,
                "case_scores": {"a": score, "b": score},
                "objective_scores": {"quality": score},
                "per_case_objective_scores": {
                    "a": {"quality": score},
                    "b": {"quality": score},
                },
            }
            for score in scores
        ]


def _reporting_packet_digest(run_dir: Path) -> str:
    return _sha(run_dir / "packets" / "reporting.json")


def _mutate_packet(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tampered"] = True
    _write(path, payload)


def _advance_to_reporting(repo: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    """Drive the ordinary lifecycle to the reporting boundary for tamper tests."""
    omni_id, run_dir, state = _start(repo)
    for child_id, candidate in (("alpha", repo["a"]), ("beta", repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        submitted = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(submitted))
    state = json.loads((run_dir / "state.json").read_text())
    phase_one = _write(
        run_dir / "phase-one.json",
        _comparison(
            repo,
            run_dir,
            state,
            "phase_one",
            [
                ("seed", repo["seed"], 0.2),
                ("alpha", repo["a"], 0.8),
                ("beta", repo["b"], 0.5),
            ],
        ),
    )
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_one))
    state = json.loads((run_dir / "state.json").read_text())
    _dispatch(omni_id, run_dir, "continue", "phase_two")
    continuation = _write(
        run_dir / "continue.json",
        _child_receipt(
            repo, run_dir, state, "continue", repo["continuation"], "phase_two"
        ),
    )
    _invoke("omni", "child-submit", omni_id, "--receipt", str(continuation))
    state = json.loads((run_dir / "state.json").read_text())
    phase_two = _write(
        run_dir / "phase-two.json",
        _comparison(
            repo,
            run_dir,
            state,
            "phase_two",
            [
                ("incumbent", repo["a"], 0.8),
                ("continuation", repo["continuation"], 0.7),
            ],
        ),
    )
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_two))
    return omni_id, run_dir, json.loads((run_dir / "state.json").read_text())


def test_outer_omni_is_idempotent_and_reconciles_all_phases(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    assert len(state["outbox"]) == len(outer_repo["plan"]["phase_one"])
    assert len(list((run_dir / "packets").glob("phase_one-*.json"))) == len(
        outer_repo["plan"]["phase_one"]
    )
    for event in state["outbox"]:
        packet_path = event["payload"]["packet_path"]
        assert event["payload"]["packet_sha256"] == _sha(Path(packet_path))
    initial = json.loads(_invoke("omni", "next", omni_id, "--json").output)
    assert initial["type"] == "child_ready"
    _invoke("omni", "ack", omni_id, initial["id"])
    _invoke("omni", "ack", omni_id, initial["id"])

    alpha = _write(
        run_dir / "alpha.json",
        _child_receipt(outer_repo, run_dir, state, "alpha", outer_repo["a"]),
    )
    beta = _write(
        run_dir / "beta.json",
        _child_receipt(outer_repo, run_dir, state, "beta", outer_repo["b"]),
    )
    _dispatch(omni_id, run_dir, "alpha")
    _dispatch(omni_id, run_dir, "beta")
    _invoke("omni", "child-submit", omni_id, "--receipt", str(alpha))
    _invoke("omni", "child-submit", omni_id, "--receipt", str(alpha))
    _invoke("omni", "child-submit", omni_id, "--receipt", str(beta))
    state = json.loads((run_dir / "state.json").read_text())
    phase_one = _write(
        run_dir / "phase-one-compare.json",
        _comparison(
            outer_repo,
            run_dir,
            state,
            "phase_one",
            [
                ("seed", outer_repo["seed"], 0.2),
                ("alpha", outer_repo["a"], 0.8),
                ("beta", outer_repo["b"], 0.5),
            ],
        ),
    )
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_one))
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_one))
    state = json.loads((run_dir / "state.json").read_text())
    assert state["phase"] == "phase_two"
    assert state["winner"]["candidate"] == "alpha"

    second = _write(
        run_dir / "second.json",
        _child_receipt(
            outer_repo,
            run_dir,
            state,
            "continue",
            outer_repo["continuation"],
            "phase_two",
        ),
    )
    _dispatch(omni_id, run_dir, "continue", "phase_two")
    _invoke("omni", "child-submit", omni_id, "--receipt", str(second))
    state = json.loads((run_dir / "state.json").read_text())
    phase_two = _write(
        run_dir / "phase-two-compare.json",
        _comparison(
            outer_repo,
            run_dir,
            state,
            "phase_two",
            [
                ("incumbent", outer_repo["a"], 0.8),
                ("continuation", outer_repo["continuation"], 0.7),
            ],
        ),
    )
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_two))
    state = json.loads((run_dir / "state.json").read_text())
    assert state["phase"] == "reporting"
    reporting = _write(
        run_dir / "report.json",
        {
            "plan_sha256": state["plan_sha256"],
            "evaluator_identity": outer_repo["plan"]["evaluator_identity"],
            "evaluator_sha256": outer_repo["plan"]["evaluator_sha256"],
            "test_set_sha256": outer_repo["plan"]["reporting"]["test_set"]["sha256"],
            "packet_sha256": _reporting_packet_digest(run_dir),
            "candidate_artifact_path": str(outer_repo["a"]),
            "candidate_artifact_sha256": _sha(outer_repo["a"]),
            "metric_calls": 1,
            "score": 0.9,
        },
    )
    _invoke("omni", "reporting-submit", omni_id, "--receipt", str(reporting))
    final = json.loads((run_dir / "state.json").read_text())
    assert final["phase"] == "done"
    assert final["final"]["artifact_sha256"] == _sha(outer_repo["a"])
    assert final["usage"] == {
        "optimization_metric_calls": 6,
        "comparison_metric_calls": 20,
        "reporting_metric_calls": 1,
        "accounted_metric_calls": 27,
    }


def test_outer_omni_serializes_concurrent_child_submissions(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    alpha = _write(
        run_dir / "alpha.json",
        _child_receipt(outer_repo, run_dir, state, "alpha", outer_repo["a"]),
    )
    beta = _write(
        run_dir / "beta.json",
        _child_receipt(outer_repo, run_dir, state, "beta", outer_repo["b"]),
    )

    _dispatch(omni_id, run_dir, "alpha")
    _dispatch(omni_id, run_dir, "beta")

    def submit(receipt: Path) -> int:
        return (
            CliRunner()
            .invoke(
                gepa_app,
                [
                    "--no-dotenv",
                    "omni",
                    "child-submit",
                    omni_id,
                    "--receipt",
                    str(receipt),
                ],
            )
            .exit_code
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, [alpha, beta])) == [0, 0]
    final = json.loads((run_dir / "state.json").read_text())
    assert set(final["children"]) == {"alpha", "beta"}
    assert (
        len(
            [
                event
                for event in final["outbox"]
                if event["type"] == "fair_compare_ready"
            ]
        )
        == 1
    )


def test_outer_omni_rejects_partial_comparison_and_unbound_error(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    alpha = _write(
        run_dir / "alpha.json",
        _child_receipt(outer_repo, run_dir, state, "alpha", outer_repo["a"]),
    )
    beta = _write(
        run_dir / "beta.json",
        _child_receipt(outer_repo, run_dir, state, "beta", outer_repo["b"]),
    )
    _dispatch(omni_id, run_dir, "alpha")
    _dispatch(omni_id, run_dir, "beta")
    _invoke("omni", "child-submit", omni_id, "--receipt", str(alpha))
    _invoke("omni", "child-submit", omni_id, "--receipt", str(beta))
    state = json.loads((run_dir / "state.json").read_text())
    partial = _write(
        run_dir / "partial.json",
        _comparison(
            outer_repo, run_dir, state, "phase_one", [("seed", outer_repo["seed"], 1.0)]
        ),
    )
    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "compare-submit", omni_id, "--receipt", str(partial)],
    )
    assert result.exit_code != 0
    evidence = _write(outer_repo["root"] / "evidence.txt", "proof")
    claim = _write(
        run_dir / "claim.json",
        {
            "kind": "unattainable_evaluator_target",
            "source_receipt_sha256": "0" * 64,
            "claim": "frozen evaluator target is inconsistent",
            "requested_action": "review evaluator",
            "case_ids": ["a"],
            "evidence": [{"artifact_path": str(evidence), "sha256": _sha(evidence)}],
        },
    )
    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "error-submit", omni_id, "--claim", str(claim)],
    )
    assert result.exit_code != 0
    assert not (run_dir / "ERROR.md").exists()


def test_outer_omni_requires_confident_phase_one_and_phase_two_acceptance(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    for child_id, candidate in (("alpha", outer_repo["a"]), ("beta", outer_repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        child = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(outer_repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(child))
    state = json.loads((run_dir / "state.json").read_text())
    phase_one_data = _comparison(
        outer_repo,
        run_dir,
        state,
        "phase_one",
        [
            ("seed", outer_repo["seed"], 0.0),
            ("alpha", outer_repo["a"], 0.0),
            ("beta", outer_repo["b"], 0.0),
        ],
    )
    _set_comparison_samples(
        phase_one_data,
        {"seed": [0.4, 0.6], "alpha": [0.45, 0.65], "beta": [0.2, 0.2]},
    )
    phase_one_data["metric_calls"] = 12
    phase_one = _write(run_dir / "noisy-phase-one.json", phase_one_data)
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_one))
    state = json.loads((run_dir / "state.json").read_text())
    assert state["winner"]["candidate"] == "seed"
    assert state["winner"]["decision"]["acceptance"]["verdict"] == "inconclusive"

    _dispatch(omni_id, run_dir, "continue", "phase_two")
    continuation = _write(
        run_dir / "continue-noisy.json",
        _child_receipt(
            outer_repo,
            run_dir,
            state,
            "continue",
            outer_repo["continuation"],
            "phase_two",
        ),
    )
    _invoke("omni", "child-submit", omni_id, "--receipt", str(continuation))
    state = json.loads((run_dir / "state.json").read_text())
    phase_two_data = _comparison(
        outer_repo,
        run_dir,
        state,
        "phase_two",
        [
            ("incumbent", outer_repo["seed"], 0.0),
            ("continuation", outer_repo["continuation"], 0.0),
        ],
    )
    _set_comparison_samples(
        phase_two_data, {"incumbent": [0.4, 0.6], "continuation": [0.45, 0.65]}
    )
    phase_two_data["metric_calls"] = 8
    phase_two = _write(run_dir / "noisy-phase-two.json", phase_two_data)
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(phase_two))
    state = json.loads((run_dir / "state.json").read_text())
    assert state["final"]["candidate"] == "incumbent"
    assert state["final"]["decision"]["acceptance"]["verdict"] == "inconclusive"


def test_outer_omni_uses_actual_max_repetition_count_for_aggregates(
    outer_repo: dict[str, Any],
) -> None:
    outer_repo["plan"]["comparison"].update(
        {
            "repetitions": 3,
            "max_repetitions": 5,
            "metric_calls": 30,
            "phase_two_metric_calls": 20,
        }
    )
    _write(outer_repo["plan_path"], outer_repo["plan"])
    omni_id, run_dir, state = _start(outer_repo)
    for child_id, candidate in (("alpha", outer_repo["a"]), ("beta", outer_repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        child = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(outer_repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(child))
    state = json.loads((run_dir / "state.json").read_text())
    receipt = _comparison(
        outer_repo,
        run_dir,
        state,
        "phase_one",
        [
            ("seed", outer_repo["seed"], 0.0),
            ("alpha", outer_repo["a"], 0.0),
            ("beta", outer_repo["b"], 0.0),
        ],
    )
    _set_comparison_samples(
        receipt,
        {"seed": [0.0] * 5, "alpha": [1.0] * 5, "beta": [0.2] * 5},
    )
    receipt["metric_calls"] = 30
    compare = _write(run_dir / "five-rounds.json", receipt)
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(compare))

    winner = json.loads((run_dir / "state.json").read_text())["winner"]
    assert winner["candidate"] == "alpha"
    assert winner["mean_score"] == pytest.approx(1.0)
    assert winner["decision"]["sample_count"] == 5
    assert winner["decision"]["coordinates"]["alpha"]["case:a"] == pytest.approx(1.0)


def test_outer_omni_rechecks_selected_phase_two_seed_and_receipt_key(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    for child_id, candidate in (("alpha", outer_repo["a"]), ("beta", outer_repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        child = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(outer_repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(child))
    state = json.loads((run_dir / "state.json").read_text())
    compare = _write(
        run_dir / "phase-one-winner.json",
        _comparison(
            outer_repo,
            run_dir,
            state,
            "phase_one",
            [
                ("seed", outer_repo["seed"], 0.0),
                ("alpha", outer_repo["a"], 1.0),
                ("beta", outer_repo["b"], 0.2),
            ],
        ),
    )
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(compare))
    outer_repo["a"].write_text("winner changed after phase one", encoding="utf-8")
    phase_two_dispatch = _write(
        run_dir / "phase-two-mutated-seed.json",
        {
            "phase": "phase_two",
            "child_id": "continue",
            "packet_sha256": _packet_digest(run_dir, "phase_two", "continue"),
        },
    )
    assert (
        CliRunner()
        .invoke(
            gepa_app,
            [
                "--no-dotenv",
                "omni",
                "child-dispatched",
                omni_id,
                "--receipt",
                str(phase_two_dispatch),
            ],
        )
        .exit_code
        != 0
    )

    with pytest.raises(RuntimeError, match="receipt key"):
        _receipt_data(run_dir, "../state.json")


def test_outer_omni_deduplicates_only_bound_unattainable_claims(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    evidence = _write(outer_repo["root"] / "bound-evidence.txt", "proof")
    child = _child_receipt(outer_repo, run_dir, state, "alpha", outer_repo["a"])
    child["evidence"] = [{"artifact_path": str(evidence), "sha256": _sha(evidence)}]
    child_path = _write(run_dir / "bound-child.json", child)
    _dispatch(omni_id, run_dir, "alpha")
    _invoke("omni", "child-submit", omni_id, "--receipt", str(child_path))
    state = json.loads((run_dir / "state.json").read_text())
    claim = _write(
        run_dir / "bound-claim.json",
        {
            "kind": "unattainable_evaluator_target",
            "source_receipt_sha256": state["children"]["alpha"],
            "claim": "the frozen evaluator target is inconsistent",
            "requested_action": "review evaluator definition",
            "case_ids": ["a"],
            "evidence": [{"artifact_path": str(evidence), "sha256": _sha(evidence)}],
        },
    )
    _invoke("omni", "error-submit", omni_id, "--claim", str(claim))
    _invoke("omni", "error-submit", omni_id, "--claim", str(claim))
    index = (run_dir / "ERROR.md").read_text()
    assert index.count("errors/") == 1
    state = json.loads((run_dir / "state.json").read_text())
    assert len(state["escalations"]) == 1


def test_outer_omni_rejects_unsafe_run_ids_and_mutated_frozen_input(
    outer_repo: dict[str, Any],
) -> None:
    unsafe = CliRunner().invoke(gepa_app, ["--no-dotenv", "omni", "status", "../x"])
    assert unsafe.exit_code != 0
    Path(outer_repo["plan"]["seed"]["artifact_path"]).write_text("mutated")
    frozen = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "start", "--plan", str(outer_repo["plan_path"])],
    )
    assert frozen.exit_code != 0


def test_outer_omni_rejects_mutated_driver_manifest(outer_repo: dict[str, Any]) -> None:
    outer_repo["manifest"].write_text("mutated manifest")
    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "start", "--plan", str(outer_repo["plan_path"])],
    )
    assert result.exit_code != 0


def test_outer_omni_rechecks_driver_manifest_at_dispatch(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, _ = _start(outer_repo)
    outer_repo["manifest"].write_text("mutated after start")
    receipt = _write(
        run_dir / "manifest-dispatch.json",
        {
            "phase": "phase_one",
            "child_id": "alpha",
            "packet_sha256": _packet_digest(run_dir, "phase_one", "alpha"),
        },
    )
    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "child-dispatched", omni_id, "--receipt", str(receipt)],
    )
    assert result.exit_code != 0


def test_outer_omni_requires_dispatch_and_exact_child_packet(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    child = _child_receipt(outer_repo, run_dir, state, "alpha", outer_repo["a"])
    child_path = _write(run_dir / "undispatched.json", child)
    no_dispatch = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "child-submit", omni_id, "--receipt", str(child_path)],
    )
    assert no_dispatch.exit_code != 0
    _dispatch(omni_id, run_dir, "alpha")
    child["packet_sha256"] = "0" * 64
    _write(child_path, child)
    mismatch = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "child-submit", omni_id, "--receipt", str(child_path)],
    )
    assert mismatch.exit_code != 0


def test_outer_omni_rejects_mutated_child_packet_after_emission(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, _ = _start(outer_repo)
    original = _packet_digest(run_dir, "phase_one", "alpha")
    _mutate_packet(run_dir / "packets" / "phase_one-alpha.json")
    receipt = _write(
        run_dir / "mutated-child-dispatch.json",
        {"phase": "phase_one", "child_id": "alpha", "packet_sha256": original},
    )

    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "child-dispatched", omni_id, "--receipt", str(receipt)],
    )

    assert result.exit_code != 0
    assert json.loads((run_dir / "state.json").read_text())["dispatches"] == {}


def test_outer_omni_rejects_mutated_comparison_and_reporting_packets(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    for child_id, candidate in (("alpha", outer_repo["a"]), ("beta", outer_repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        receipt = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(outer_repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(receipt))
    state = json.loads((run_dir / "state.json").read_text())
    compare = _write(
        run_dir / "tampered-compare.json",
        _comparison(
            outer_repo,
            run_dir,
            state,
            "phase_one",
            [
                ("seed", outer_repo["seed"], 0.2),
                ("alpha", outer_repo["a"], 0.8),
                ("beta", outer_repo["b"], 0.5),
            ],
        ),
    )
    _mutate_packet(run_dir / "packets" / "compare-phase_one.json")
    rejected_compare = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "compare-submit", omni_id, "--receipt", str(compare)],
    )
    assert rejected_compare.exit_code != 0
    assert json.loads((run_dir / "state.json").read_text())["phase"] == "phase_one"

    # A distinct clean run reaches reporting; its packet has the same
    # immutable-emission binding as the child and comparison packets.
    omni_id, run_dir, state = _advance_to_reporting(outer_repo)
    report = _write(
        run_dir / "tampered-report.json",
        {
            "plan_sha256": state["plan_sha256"],
            "evaluator_identity": outer_repo["plan"]["evaluator_identity"],
            "evaluator_sha256": outer_repo["plan"]["evaluator_sha256"],
            "test_set_sha256": outer_repo["plan"]["reporting"]["test_set"]["sha256"],
            "packet_sha256": _reporting_packet_digest(run_dir),
            "candidate_artifact_path": str(outer_repo["a"]),
            "candidate_artifact_sha256": _sha(outer_repo["a"]),
            "metric_calls": 1,
            "score": 0.9,
        },
    )
    _mutate_packet(run_dir / "packets" / "reporting.json")
    rejected_reporting = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "reporting-submit", omni_id, "--receipt", str(report)],
    )
    assert rejected_reporting.exit_code != 0
    latest = json.loads((run_dir / "state.json").read_text())
    assert latest["phase"] == "reporting" and latest["reporting"] is None


def test_outer_omni_rechecks_comparison_and_reporting_datasets(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    for child_id, candidate in (("alpha", outer_repo["a"]), ("beta", outer_repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        receipt = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(outer_repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(receipt))
    state = json.loads((run_dir / "state.json").read_text())
    comparison = _write(
        run_dir / "changed-minibatch-compare.json",
        _comparison(
            outer_repo,
            run_dir,
            state,
            "phase_one",
            [
                ("seed", outer_repo["seed"], 0.2),
                ("alpha", outer_repo["a"], 0.8),
                ("beta", outer_repo["b"], 0.5),
            ],
        ),
    )
    minibatch_path = Path(outer_repo["plan"]["minibatch"]["artifact_path"])
    original_minibatch = minibatch_path.read_text(encoding="utf-8")
    minibatch_path.write_text("changed after packet emission", encoding="utf-8")
    assert (
        CliRunner()
        .invoke(
            gepa_app,
            [
                "--no-dotenv",
                "omni",
                "compare-submit",
                omni_id,
                "--receipt",
                str(comparison),
            ],
        )
        .exit_code
        != 0
    )

    # Reporting has a separate frozen dataset and must perform the same check.
    minibatch_path.write_text(original_minibatch, encoding="utf-8")
    omni_id, run_dir, state = _advance_to_reporting(outer_repo)
    report = _write(
        run_dir / "changed-test-report.json",
        {
            "plan_sha256": state["plan_sha256"],
            "evaluator_identity": outer_repo["plan"]["evaluator_identity"],
            "evaluator_sha256": outer_repo["plan"]["evaluator_sha256"],
            "test_set_sha256": outer_repo["plan"]["reporting"]["test_set"]["sha256"],
            "packet_sha256": _reporting_packet_digest(run_dir),
            "candidate_artifact_path": str(outer_repo["a"]),
            "candidate_artifact_sha256": _sha(outer_repo["a"]),
            "metric_calls": 1,
            "score": 0.9,
        },
    )
    Path(outer_repo["plan"]["reporting"]["test_set"]["artifact_path"]).write_text(
        "changed after packet emission", encoding="utf-8"
    )
    assert (
        CliRunner()
        .invoke(
            gepa_app,
            [
                "--no-dotenv",
                "omni",
                "reporting-submit",
                omni_id,
                "--receipt",
                str(report),
            ],
        )
        .exit_code
        != 0
    )


@pytest.mark.parametrize("artifact_key", ["seed", "minibatch"])
def test_outer_omni_rechecks_frozen_child_inputs_at_dispatch(
    outer_repo: dict[str, Any], artifact_key: str
) -> None:
    omni_id, run_dir, _ = _start(outer_repo)
    if artifact_key == "seed":
        path = outer_repo["seed"]
    else:
        path = Path(outer_repo["plan"]["minibatch"]["artifact_path"])
    path.write_text("changed after start", encoding="utf-8")
    receipt = _write(
        run_dir / f"changed-{artifact_key}.json",
        {
            "phase": "phase_one",
            "child_id": "alpha",
            "packet_sha256": _packet_digest(run_dir, "phase_one", "alpha"),
        },
    )
    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "child-dispatched", omni_id, "--receipt", str(receipt)],
    )
    assert result.exit_code != 0


def test_outer_omni_rejects_mutated_plan_and_receipt_bytes(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, _ = _start(outer_repo)
    plan_path = run_dir / "plan.json"
    _mutate_packet(plan_path)
    dispatch = _write(
        run_dir / "plan-dispatch.json",
        {
            "phase": "phase_one",
            "child_id": "alpha",
            "packet_sha256": _packet_digest(run_dir, "phase_one", "alpha"),
        },
    )
    assert (
        CliRunner()
        .invoke(
            gepa_app,
            [
                "--no-dotenv",
                "omni",
                "child-dispatched",
                omni_id,
                "--receipt",
                str(dispatch),
            ],
        )
        .exit_code
        != 0
    )


def test_outer_omni_duplicate_comparison_revalidates_frozen_plan(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    for child_id, candidate in (("alpha", outer_repo["a"]), ("beta", outer_repo["b"])):
        _dispatch(omni_id, run_dir, child_id)
        child = _write(
            run_dir / f"{child_id}.json",
            _child_receipt(outer_repo, run_dir, state, child_id, candidate),
        )
        _invoke("omni", "child-submit", omni_id, "--receipt", str(child))
    state = json.loads((run_dir / "state.json").read_text())
    compare = _write(
        run_dir / "duplicate-compare.json",
        _comparison(
            outer_repo,
            run_dir,
            state,
            "phase_one",
            [
                ("seed", outer_repo["seed"], 0.2),
                ("alpha", outer_repo["a"], 0.8),
                ("beta", outer_repo["b"], 0.5),
            ],
        ),
    )
    _invoke("omni", "compare-submit", omni_id, "--receipt", str(compare))
    _mutate_packet(run_dir / "plan.json")

    result = CliRunner().invoke(
        gepa_app,
        ["--no-dotenv", "omni", "compare-submit", omni_id, "--receipt", str(compare)],
    )
    assert result.exit_code != 0


def test_outer_omni_recovers_partial_event_projection_from_outbox(
    outer_repo: dict[str, Any],
) -> None:
    omni_id, run_dir, state = _start(outer_repo)
    event = state["outbox"][0]
    projection = run_dir / "events" / f"{event['sequence']:020d}.json"
    projection.write_text('{"partial":', encoding="utf-8")

    redelivered = json.loads(_invoke("omni", "next", omni_id, "--json").output)

    assert redelivered == event
    assert json.loads(projection.read_text(encoding="utf-8")) == event


def test_outer_omni_receipt_publication_recovers_partial_digest(
    outer_repo: dict[str, Any],
) -> None:
    _, run_dir, _ = _start(outer_repo)
    payload = {"kind": "receipt", "value": 1}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = run_dir / "receipts" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"partial":', encoding="utf-8")

    assert _receipt(run_dir, payload) == digest
    assert json.loads(path.read_text(encoding="utf-8")) == payload

    # A receipt is verified again before any transition reads it.
    omni_id, run_dir, state = _start(outer_repo)
    _dispatch(omni_id, run_dir, "alpha")
    dispatch_digest = json.loads((run_dir / "state.json").read_text())["dispatches"][
        "alpha"
    ]
    _mutate_packet(run_dir / "receipts" / f"{dispatch_digest}.json")
    child = _write(
        run_dir / "child-after-tampered-dispatch.json",
        _child_receipt(outer_repo, run_dir, state, "alpha", outer_repo["a"]),
    )
    assert (
        CliRunner()
        .invoke(
            gepa_app,
            ["--no-dotenv", "omni", "child-submit", omni_id, "--receipt", str(child)],
        )
        .exit_code
        != 0
    )
