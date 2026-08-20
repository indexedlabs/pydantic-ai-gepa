from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai_gepa.vector_acceptance import (
    VectorComparison,
    VectorComparisonRequest,
    VectorRecord,
    VectorRecordKey,
    VectorRecordStore,
    compare_vectors,
)
from pydantic_ai_gepa.cli.runs import ParetoLog, ParetoRow, utc_now_iso


class EscalatingComparator:
    def compare(self, request: VectorComparisonRequest) -> VectorComparison:
        return VectorComparison(
            "needs_escalation" if len(request.candidate) < 3 else "accepted",
            ranking_key=(2.0, 1.0),
            display_score=0.75,
        )


def _record(*, candidate: str, repetition: int, inventory: str = "inventory") -> VectorRecord:
    return VectorRecord(
        key=VectorRecordKey(
            run_id="run", inventory_hash=inventory, scorer_identity="scorer",
            incumbent_hash="incumbent", candidate_hash=candidate,
            repetition=repetition, vector_schema_version="1", telemetry_schema_version="1",
        ),
        assertions={"case": {"a": {"status": "pass"}}},
        latency={"case": {"engine": 1.0}}, display_score=0.5,
    )


def test_comparator_contract_escalates_once_then_accepts() -> None:
    incumbent = (_record(candidate="incumbent", repetition=1), _record(candidate="incumbent", repetition=2))
    candidate = (_record(candidate="candidate", repetition=1), _record(candidate="candidate", repetition=2))
    comparator = EscalatingComparator()
    first = compare_vectors(comparator, VectorComparisonRequest(incumbent, candidate, 1, 0))
    assert first.verdict == "needs_escalation"
    final = compare_vectors(comparator, VectorComparisonRequest(incumbent, (*candidate, _record(candidate="candidate", repetition=3)), 1, 1))
    assert final.verdict == "accepted"
    assert final.ranking_key == (2.0, 1.0)


def test_vector_store_pools_only_compatible_scored_incumbent_reps(tmp_path: Path) -> None:
    store = VectorRecordStore(tmp_path / "vectors.jsonl")
    store.append(_record(candidate="incumbent", repetition=1))
    store.append(_record(candidate="incumbent", repetition=2))
    store.append(_record(candidate="incumbent", repetition=3, inventory="other"))
    key = _record(candidate="candidate", repetition=1).key
    assert [item.key.repetition for item in store.matching(key, candidate_hash="incumbent")] == [1, 2]


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
    base = dict(
        candidate_id="candidate", commit_sha=None, component_overrides_id=None,
        minibatch_id="batch", per_case_scores={"case": 1.0}, mean_score=1.0,
        status="evaluated", summary="test", timestamp=utc_now_iso(),
    )
    ledger.append(ParetoRow(**base, extra={"row_scope": "probe"}))
    ledger.append(ParetoRow(**base, extra={"row_scope": "acceptance"}))
    assert ledger.count_rows() == 1
    assert len(ledger.selectable_rows()) == 1
