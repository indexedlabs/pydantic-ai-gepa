"""Untrusted scoring worker. Only launched after the OS sandbox is applied."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from pydantic_evals import Case

from ..evaluation import evaluate_callable_dataset, evaluate_candidate_dataset
from ..evaluation_health import evaluation_infrastructure_failures
from ..spend import SpendMeter, rollout_spend
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


def main() -> None:
    # Keep a dedicated protocol fd; candidate Python/native console output is
    # discarded at the fd level. The parent validates even this protocol fd.
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)

    def send(value: Any) -> None:
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()

    def receive() -> Any:
        return json.loads(sys.stdin.buffer.readline(1024 * 1024 + 1))

    initialization = receive()
    config = GepaConfig.from_dict(initialization["config"])
    validation = initialization["validation"]
    root = Path.cwd()
    insert_repo_root_on_path(root)
    # This file stays private even for training; the parent only accepts the
    # structured feedback field. Never copy arbitrary child-created artifacts.
    os.environ["GEPA_TRACE_FILE"] = str(Path(os.environ["TMPDIR"]) / "trace.jsonl")
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

    send({"type": "ready"})
    while True:
        case = Case(**receive())
        meter = Meter(price_fn=price)
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
                    )
                )
        record = records[0]
        send(
            {
                "type": "result",
                "score": record.score,
                "feedback": None if validation else record.feedback,
                "failed": bool(evaluation_infrastructure_failures(records)),
                "cached": meter.cached,
            }
        )


if __name__ == "__main__":
    main()
