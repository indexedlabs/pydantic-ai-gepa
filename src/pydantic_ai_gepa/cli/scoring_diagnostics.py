"""Bounded operator diagnostics, never published through the run's public views."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any

import typer

MAX_ENTRIES = 50
MAX_MESSAGE = 512
CLASS_PATTERN = r"[A-Za-z_][A-Za-z0-9_]{0,127}"


def valid_diagnostic(value: Any) -> bool:
    return value is None or (
        isinstance(value, dict)
        and set(value) == {"class", "message"}
        and isinstance(value["class"], str)
        and re.fullmatch(CLASS_PATTERN, value["class"]) is not None
        and isinstance(value["message"], str)
    )


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, dict):
        return set().union(*(_strings(item) for pair in value.items() for item in pair))
    if isinstance(value, (list, tuple)):
        return set().union(*(_strings(item) for item in value))
    return set()


def record_failure(value: Any, case: Any, validation: bool, meter: Any) -> None:
    from .harness_record import for_run

    if value is None or not valid_diagnostic(value):
        return
    message = value["message"]
    exception_class = value["class"]
    if validation:
        strings = _strings(
            [case.name, case.inputs, case.expected_output, case.metadata]
        )
        if strings:
            pattern = "|".join(
                re.escape(item) for item in sorted(strings, key=len, reverse=True)
            )
            message = re.sub(pattern, "[redacted]", message)
            if any(item in exception_class for item in strings):
                exception_class = "EvaluationError"
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "phase": "validation" if validation else "training",
        "class": exception_class,
        "message": message[:MAX_MESSAGE],
    }
    if not validation:
        entry["case_id"] = str(case.name or "")[:128]
    try:
        record = for_run(meter.run_id, meter.root)
        if record is None:
            return
        with record.locked():
            content = record.read("@scoring-diagnostics") or ""
            if len(content.splitlines()) >= MAX_ENTRIES:
                dropped = int(record.read("@scoring-diagnostics-dropped") or "0")
                record.write("@scoring-diagnostics-dropped", str(dropped + 1))
            else:
                record.write(
                    "@scoring-diagnostics", json.dumps(entry) + "\n", append=True
                )
    except (OSError, ValueError, typer.BadParameter):
        # Diagnostics are best effort, with no public fallback or scoring impact.
        return
