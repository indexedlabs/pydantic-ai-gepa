"""Pre-evaluation candidate review contracts and generic adapters."""

from __future__ import annotations

import json
import inspect
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, Field


ReviewDisposition = Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    component: str | None
    excerpt: str | None
    explanation: str
    severity: Literal["info", "warning", "error"] = "error"


@dataclass(frozen=True, slots=True)
class CandidateReviewRequest:
    components: Mapping[str, str]
    diff: str
    workspace_path: str
    opaque_context: Mapping[str, Any]
    attempt: int
    prior_findings: tuple[ReviewFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["components"] = dict(self.components)
        data["opaque_context"] = dict(self.opaque_context)
        return data


@dataclass(frozen=True, slots=True)
class CandidateReviewVerdict:
    disposition: ReviewDisposition
    findings: tuple[ReviewFinding, ...] = ()

    def __post_init__(self) -> None:
        if self.disposition == "pass" and any(
            finding.severity == "error" for finding in self.findings
        ):
            raise ValueError("Passing review verdicts cannot carry error findings.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "findings": [asdict(item) for item in self.findings],
        }


@runtime_checkable
class CandidateReviewer(Protocol):
    def review(self, request: CandidateReviewRequest) -> CandidateReviewVerdict: ...


def resolve_candidate_reviewer(
    ref: str, *, expected_root: Path | None = None
) -> CandidateReviewer:
    from .cli.layout import resolve_module_attr

    factory = resolve_module_attr(
        ref, kind="candidate reviewer", expected_root=expected_root
    )
    if inspect.isclass(factory):
        try:
            reviewer = factory()
        except TypeError as exc:
            raise TypeError(
                "Reviewer classes must be constructible without arguments."
            ) from exc
    else:
        reviewer = (
            factory()
            if callable(factory) and not hasattr(factory, "review")
            else factory
        )
    if inspect.isclass(reviewer):
        raise TypeError("Reviewer factory returned a class instead of an instance.")
    if not isinstance(reviewer, CandidateReviewer):
        raise TypeError("Reviewer factory must return an object with review(request).")
    return reviewer


class _AgentFinding(BaseModel):
    component: str | None = None
    excerpt: str | None = None
    explanation: str
    severity: Literal["info", "warning", "error"] = "error"


class _AgentVerdict(BaseModel):
    disposition: ReviewDisposition
    findings: list[_AgentFinding] = Field(default_factory=list)


class AgentCandidateReviewer:
    """Adapter for a Pydantic AI Agent with majority-of-N review voting."""

    def __init__(self, agent: Any, *, samples: int = 3) -> None:
        if samples < 1 or samples % 2 == 0:
            raise ValueError("Agent reviewer samples must be an odd positive number.")
        self.agent = agent
        self.samples = samples

    def review(self, request: CandidateReviewRequest) -> CandidateReviewVerdict:
        prompt = json.dumps(request.to_dict(), sort_keys=True)
        responses: list[_AgentVerdict] = []
        for _ in range(self.samples):
            result = self.agent.run_sync(prompt, output_type=_AgentVerdict)
            value = getattr(result, "output", result)
            responses.append(_AgentVerdict.model_validate(value))
        failing = [item for item in responses if item.disposition == "fail"]
        if len(failing) * 2 <= self.samples:
            advisories = tuple(
                ReviewFinding(
                    component=item.component,
                    excerpt=item.excerpt,
                    explanation=item.explanation,
                    severity=item.severity,
                )
                for response in responses
                for item in response.findings
                if item.severity != "error"
            )
            return CandidateReviewVerdict("pass", advisories)
        findings = tuple(
            ReviewFinding(
                component=item.component,
                excerpt=item.excerpt,
                explanation=item.explanation,
                severity=item.severity,
            )
            for response in failing
            for item in response.findings
        )
        return CandidateReviewVerdict("fail", findings)


class CommandCandidateReviewer:
    """Run a coding-agent command in a scratch directory and parse its verdict."""

    def __init__(
        self,
        command_template: str,
        *,
        output_path: str = "verdict.json",
        timeout_secs: float = 120.0,
        retries: int = 1,
    ) -> None:
        self.command_template = command_template
        self.output_path = output_path
        self.timeout_secs = timeout_secs
        self.retries = retries

    def review(self, request: CandidateReviewRequest) -> CandidateReviewVerdict:
        last_error: Exception | None = None
        for _ in range(self.retries + 1):
            with tempfile.TemporaryDirectory(prefix="gepa-review-") as temporary:
                root = Path(temporary)
                request_path = root / "request.json"
                request_path.write_text(
                    json.dumps(request.to_dict(), indent=2), encoding="utf-8"
                )
                output = root / self.output_path
                argv = shlex.split(
                    self.command_template.format(
                        request=shlex.quote(str(request_path)),
                        output=shlex.quote(str(output)),
                        workspace=shlex.quote(request.workspace_path),
                    )
                )
                try:
                    subprocess.run(
                        argv, cwd=root, check=True, timeout=self.timeout_secs
                    )
                    raw = json.loads(output.read_text(encoding="utf-8"))
                    return _verdict_from_mapping(raw)
                except (
                    OSError,
                    subprocess.SubprocessError,
                    TimeoutError,
                    json.JSONDecodeError,
                    FileNotFoundError,
                    ValueError,
                ) as exc:
                    last_error = exc
        return CandidateReviewVerdict(
            "fail",
            (
                ReviewFinding(
                    None, None, f"Candidate reviewer command failed: {last_error}"
                ),
            ),
        )


def _verdict_from_mapping(raw: Mapping[str, Any]) -> CandidateReviewVerdict:
    disposition = raw.get("disposition", raw.get("verdict"))
    if disposition not in {"pass", "fail"}:
        raise ValueError("Reviewer output needs disposition 'pass' or 'fail'.")
    findings_raw = raw.get("findings", [])
    if not isinstance(findings_raw, list):
        raise ValueError("Reviewer findings must be a list.")
    findings_list: list[ReviewFinding] = []
    for item in findings_raw:
        if not isinstance(item, Mapping):
            continue
        raw_severity = item.get("severity", "error")
        severity: Literal["info", "warning", "error"] = (
            raw_severity if raw_severity in {"info", "warning", "error"} else "error"
        )
        findings_list.append(
            ReviewFinding(
                component=item.get("component"),
                excerpt=item.get("excerpt"),
                explanation=str(item["explanation"]),
                severity=severity,
            )
        )
    findings = tuple(findings_list)
    return CandidateReviewVerdict(disposition, findings)
