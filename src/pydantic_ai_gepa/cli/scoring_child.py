"""Bounded scoring worker for Seatbelt or explicitly trusted text scoring."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from pydantic_evals import Case

from ..evaluation import (
    evaluate_callable_dataset,
    evaluate_candidate_dataset,
)
from ..evaluation_health import evaluation_infrastructure_failures
from ..spend import SpendMeter, rollout_spend
from ..types import RolloutOutput, _rollout_error_observer
from .layout import (
    GepaConfig,
    insert_repo_root_on_path,
    resolve_agent,
    resolve_case_factory,
    resolve_evaluate,
    resolve_metric,
    resolve_module_attr,
    resolve_skills,
)
from .metrics import default_substring_metric
from .scoring_material import structured_payload


def main() -> None:
    # Keep a dedicated protocol fd; candidate Python/native console output is
    # discarded at the fd level. The parent validates even this protocol fd.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    def send(value: Any) -> None:
        data = json.dumps(value, allow_nan=False) + "\n"
        # Escaped feedback and material can each fit their own limits but
        # together exceed the protocol envelope. Keep the score/feedback;
        # None asks the parent to emit the fixed material-refusal note.
        if value.get("type") == "result" and len(data.encode()) > 1024 * 1024:
            value["material"] = None
            data = json.dumps(value, allow_nan=False) + "\n"
            if len(data.encode()) > 1024 * 1024 and value.get("diagnostic"):
                value["diagnostic"]["message"] = (
                    "Exception message exceeds protocol limit"
                )
                data = json.dumps(value, allow_nan=False) + "\n"
        protocol.write(data)
        protocol.flush()

    def receive() -> Any:
        return json.loads(sys.stdin.buffer.readline(1024 * 1024 + 1))

    initialization = receive()
    config = GepaConfig.from_dict(initialization["config"])
    validation = initialization["validation"]
    if config.acceptance.trusted_scorer:
        os.environ["GEPA_CANDIDATE_COMPONENTS_JSON"] = json.dumps(
            initialization["components"], sort_keys=True
        )
    root = Path.cwd()
    guard = None
    if config.acceptance.scoring == "trusted_in_process":
        from .scoring_imports import CandidateImportGuard

        guard = CandidateImportGuard(initialization["blocked_roots"])
    insert_repo_root_on_path(root)
    if guard is not None and config.acceptance.scope_verifier:
        verifier = resolve_module_attr(
            config.acceptance.scope_verifier, expected_root=root
        )
        if verifier(components=initialization["components"]) is not True:
            raise RuntimeError("Trusted text scope verification failed.")
    evaluate = resolve_evaluate(config, expected_root=root)
    agent = resolve_agent(config, expected_root=root) if config.agent else None
    metric = resolve_metric(config, expected_root=root) or default_substring_metric
    case_factory = resolve_case_factory(config, expected_root=root)
    skills = resolve_skills(config, root=root)
    price = (
        resolve_module_attr(config.price_fn, expected_root=root)
        if config.price_fn
        else None
    )

    class Meter(SpendMeter):
        cached = False

        def declare_cached_rollout(self) -> None:
            self.cached = True

        def record(self, category: Any, response: Any) -> None:
            before = self.report().total_dollars
            super().record(category, response)
            report = self.report()
            send(
                {
                    "type": "usage",
                    "dollars": None
                    if report.unpriced_usage
                    else report.total_dollars - before,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            )
            if receive() != {"type": "ack"}:
                raise RuntimeError("Parent refused response")

    if guard is not None:
        guard.check()
    send({"type": "ready"})
    while True:
        request = receive()
        case = Case(**request["case"])
        trace_path = Path(request["output_dir"]) / "trace.jsonl"
        os.environ["GEPA_TRACE_FILE"] = str(trace_path)
        meter = Meter(price_fn=price)
        diagnostic = None

        def observe(error: Exception) -> None:
            nonlocal diagnostic
            diagnostic = {"class": type(error).__name__, "message": str(error)}

        token = _rollout_error_observer.set(observe)
        try:
            with rollout_spend(meter):
                if evaluate is not None:
                    records = asyncio.run(
                        evaluate_callable_dataset(
                            evaluate=evaluate,
                            metric=metric,
                            dataset=[case],
                            concurrency=1,
                            case_factory=case_factory,
                        )
                    )
                else:
                    if agent is None:
                        raise RuntimeError("Missing agent")
                    records = asyncio.run(
                        evaluate_candidate_dataset(
                            agent=agent,
                            metric=metric,
                            dataset=[case],
                            concurrency=1,
                            case_factory=case_factory,
                            skills_fs=skills,
                            capture_traces=not validation,
                        )
                    )
        finally:
            _rollout_error_observer.reset(token)
        record = records[0]
        failed = bool(evaluation_infrastructure_failures(records))
        if failed and diagnostic is None:
            output = record.payload.get("output")
            diagnostic = {
                "class": "EvaluationError",
                "message": getattr(output, "error_message", None)
                or "Unknown evaluation error",
            }
        material = None
        if not validation:
            # Conversion happens only in the untrusted child. The parent accepts
            # JSON, never objects, serializers or exception metadata.
            from .eval import _json_default, _write_trace_file

            try:
                _write_trace_file(path=trace_path, records=records)
                output = record.payload.get("output")
                if isinstance(output, RolloutOutput):
                    output = output.result if output.success else None
                trajectory = record.payload.get("trajectory")
                value = {
                    "output": output,
                    "side_info": record.payload.get("side_info"),
                    "metric_side_info": getattr(trajectory, "metric_side_info", None),
                }
                material = structured_payload(
                    json.loads(
                        json.dumps(value, default=_json_default, allow_nan=False)
                    )
                )
            except Exception:
                # A missing/invalid material payload is a fixed parent note,
                # never a failed rollout or child exception text.
                pass
        if guard is not None:
            guard.check()
        send(
            {
                "type": "result",
                "score": record.score,
                "feedback": None if validation else record.feedback,
                "failed": failed,
                "diagnostic": diagnostic if failed else None,
                "cached": meter.cached,
                "material": material,
            }
        )


if __name__ == "__main__":
    main()
