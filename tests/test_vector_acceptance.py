from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from pydantic_ai_gepa.vector_acceptance import (
    VectorComparison,
    VectorComparisonRequest,
    VectorRecord,
    VectorRecordKey,
    VectorRecordStore,
    compare_vectors,
    resolve_vector_comparator,
    scorer_identity,
)
from pydantic_ai_gepa.candidate_review import (
    CandidateReviewVerdict,
    ReviewFinding,
    _verdict_from_mapping,
    resolve_candidate_reviewer,
)
from pydantic_ai_gepa.cli.runs import ParetoLog, ParetoRow, utc_now_iso


class EscalatingComparator:
    def compare(self, request: VectorComparisonRequest) -> VectorComparison:
        return VectorComparison(
            "needs_escalation" if len(request.candidate) < 3 else "accepted",
            ranking_key=(2.0, 1.0),
            display_score=0.75,
        )


def _record(
    *, candidate: str, repetition: int, inventory: str = "inventory"
) -> VectorRecord:
    return VectorRecord(
        key=VectorRecordKey(
            run_id="run",
            inventory_hash=inventory,
            scorer_identity="scorer",
            incumbent_hash="incumbent",
            candidate_hash=candidate,
            repetition=repetition,
            vector_schema_version="1",
            telemetry_schema_version="1",
        ),
        assertions={"case": {"a": {"status": "pass"}}},
        latency={"case": {"engine": 1.0}},
        display_score=0.5,
    )


def test_comparator_contract_escalates_once_then_accepts() -> None:
    incumbent = (
        _record(candidate="incumbent", repetition=1),
        _record(candidate="incumbent", repetition=2),
    )
    candidate = (
        _record(candidate="candidate", repetition=1),
        _record(candidate="candidate", repetition=2),
    )
    comparator = EscalatingComparator()
    first = compare_vectors(
        comparator, VectorComparisonRequest(incumbent, candidate, 1, 0)
    )
    assert first.verdict == "needs_escalation"
    final = compare_vectors(
        comparator,
        VectorComparisonRequest(
            incumbent, (*candidate, _record(candidate="candidate", repetition=3)), 1, 1
        ),
    )
    assert final.verdict == "accepted"
    assert final.ranking_key == (2.0, 1.0)


def test_vector_store_pools_only_compatible_scored_incumbent_reps(
    tmp_path: Path,
) -> None:
    store = VectorRecordStore(tmp_path / "vectors.jsonl")
    store.append(_record(candidate="incumbent", repetition=1))
    store.append(_record(candidate="incumbent", repetition=2))
    store.append(_record(candidate="incumbent", repetition=3, inventory="other"))
    key = _record(candidate="candidate", repetition=1).key
    assert [
        item.key.repetition for item in store.matching(key, candidate_hash="incumbent")
    ] == [1, 2]


def test_vector_comparison_refuses_mixed_inventory() -> None:
    with pytest.raises(ValueError, match="Refusing vector comparison"):
        compare_vectors(
            EscalatingComparator(),
            VectorComparisonRequest(
                (_record(candidate="incumbent", repetition=1),),
                (_record(candidate="candidate", repetition=1, inventory="other"),),
                1,
                0,
            ),
        )


def test_probe_rows_do_not_spend_or_enter_pareto_front(tmp_path: Path) -> None:
    ledger = ParetoLog("run", tmp_path)

    def row(scope: str) -> ParetoRow:
        return ParetoRow(
            candidate_id="candidate",
            commit_sha=None,
            component_overrides_id=None,
            minibatch_id="batch",
            per_case_scores={"case": 1.0},
            mean_score=1.0,
            status="evaluated",
            summary="test",
            timestamp=utc_now_iso(),
            extra={"row_scope": scope},
        )

    ledger.append(row("probe"))
    ledger.append(row("acceptance"))
    assert ledger.count_rows() == 1
    assert len(ledger.selectable_rows()) == 1


def test_scorer_identity_changes_with_resolved_source_bytes(tmp_path: Path) -> None:
    package = tmp_path / "scorers"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "metric.py"
    source.write_text("def score():\n    return 1\n", encoding="utf-8")
    first = scorer_identity(("scorers.metric:score",), root=tmp_path)
    source.write_text("def score():\n    return 2\n", encoding="utf-8")
    second = scorer_identity(("scorers.metric:score",), root=tmp_path)
    assert first != second


def test_vector_comparison_refuses_mixed_candidate_hashes() -> None:
    with pytest.raises(ValueError, match="candidate hash"):
        VectorComparisonRequest(
            (_record(candidate="incumbent", repetition=1),),
            (
                _record(candidate="candidate-a", repetition=1),
                _record(candidate="candidate-b", repetition=2),
            ),
            1,
            0,
        ).validate_compatible()


def test_vector_record_rejects_unknown_outcome() -> None:
    raw = _record(candidate="candidate", repetition=1).to_dict()
    raw["outcome"] = "timeout"
    with pytest.raises(ValueError, match="outcome"):
        VectorRecord.from_dict(raw)


def test_class_comparator_and_reviewer_are_instantiated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("test_review_providers")

    class Comparator:
        def compare(self, request: VectorComparisonRequest) -> VectorComparison:
            return VectorComparison("equivalent")

    class Reviewer:
        def review(self, request: object) -> CandidateReviewVerdict:
            return CandidateReviewVerdict("pass")

    setattr(module, "Comparator", Comparator)
    setattr(module, "Reviewer", Reviewer)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert isinstance(
        resolve_vector_comparator(f"{module.__name__}:Comparator"), Comparator
    )
    assert isinstance(
        resolve_candidate_reviewer(f"{module.__name__}:Reviewer"), Reviewer
    )


def test_pass_review_allows_non_error_advisories() -> None:
    verdict = CandidateReviewVerdict(
        "pass", (ReviewFinding(None, None, "Consider simplifying", "info"),)
    )
    assert verdict.findings[0].severity == "info"
    with pytest.raises(ValueError, match="error"):
        CandidateReviewVerdict("pass", (ReviewFinding(None, None, "Unsafe", "error"),))


def test_command_review_unknown_severity_defaults_to_error() -> None:
    verdict = _verdict_from_mapping(
        {
            "disposition": "fail",
            "findings": [{"explanation": "bad", "severity": "critical"}],
        }
    )
    assert verdict.findings[0].severity == "error"
