from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_DISCOVERY_ATTR_TRUNCATION_CHARS = 4096
_SURGICAL_ATTR_TRUNCATION_CHARS = 16384
_VIEW_RESPONSE_BYTES_BUDGET = 150_000
_OVERVIEW_SAMPLE_TRACE_IDS = 20
_QUERY_LIMIT_CAP = 500
_VIEW_SPANS_LIMIT = 200
_SEARCH_MATCH_LIMIT_CAP = 200
_NOISY_FLAT_PROJECTION_RE = re.compile(
    r"^(?:llm\.(?:input|output)_messages|mcp\.tools)\.\d+\."
)


def span_to_jsonl_line(span: Any) -> str:
    """Serialize an OpenTelemetry span as one compact JSONL record."""
    data = json.loads(span.to_json())
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


@dataclass(frozen=True, slots=True)
class SpanRecord:
    data: dict[str, Any]
    raw_json: str
    raw_json_bytes: int
    ordinal: int

    @property
    def trace_id(self) -> str:
        context = _mapping(self.data.get("context"))
        return str(
            context.get("trace_id")
            or self.data.get("trace_id")
            or f"record-{self.ordinal}"
        )

    @property
    def span_id(self) -> str:
        context = _mapping(self.data.get("context"))
        return str(
            context.get("span_id") or self.data.get("span_id") or f"span-{self.ordinal}"
        )

    @property
    def parent_id(self) -> str:
        return str(self.data.get("parent_id") or self.data.get("parent_span_id") or "")

    @property
    def name(self) -> str:
        attributes = self.attributes
        return str(
            self.data.get("name")
            or attributes.get("logfire.msg")
            or attributes.get("gen_ai.operation.name")
            or f"span-{self.ordinal}"
        )

    @property
    def kind(self) -> str:
        return str(self.data.get("kind") or "")

    @property
    def status_code(self) -> str:
        status = _mapping(self.data.get("status"))
        return str(status.get("status_code") or status.get("code") or "")

    @property
    def start_time(self) -> str:
        return str(self.data.get("start_time") or "")

    @property
    def end_time(self) -> str:
        return str(self.data.get("end_time") or "")

    @property
    def attributes(self) -> dict[str, Any]:
        return dict(_mapping(self.data.get("attributes")))

    @property
    def resource_attributes(self) -> dict[str, Any]:
        resource = _mapping(self.data.get("resource"))
        return dict(_mapping(resource.get("attributes")))


class StructuredTraceStore:
    """Host-side structured query API over captured OTel/Logfire span JSON."""

    def __init__(self, spans: list[SpanRecord]) -> None:
        self._spans = spans
        traces: dict[str, list[SpanRecord]] = {}
        for span in spans:
            traces.setdefault(span.trace_id, []).append(span)
        self._traces = traces

    @classmethod
    def load(cls, path: Path) -> "StructuredTraceStore":
        return cls(list(_load_span_records(path)))

    @property
    def trace_count(self) -> int:
        return len(self._traces)

    def overview(self, filters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        traces = self._filtered_traces(filters)
        summaries = [self._trace_summary(trace_id, spans) for trace_id, spans in traces]
        status_counts: Counter[str] = Counter()
        span_name_counts: Counter[str] = Counter()
        service_names: set[str] = set()
        model_names: set[str] = set()
        agent_names: set[str] = set()
        total_input_tokens = 0
        total_output_tokens = 0
        total_raw_bytes = 0
        error_trace_count = 0

        for summary in summaries:
            status_counts.update(summary["status_counts"])
            span_name_counts.update(summary["span_name_counts"])
            service_names.update(summary["service_names"])
            model_names.update(summary["model_names"])
            agent_names.update(summary["agent_names"])
            total_input_tokens += int(summary["total_input_tokens"])
            total_output_tokens += int(summary["total_output_tokens"])
            total_raw_bytes += int(summary["raw_json_bytes"])
            if summary["has_errors"]:
                error_trace_count += 1

        return {
            "total_traces": len(summaries),
            "total_spans": sum(int(s["span_count"]) for s in summaries),
            "error_trace_count": error_trace_count,
            "service_names": sorted(service_names),
            "model_names": sorted(model_names),
            "agent_names": sorted(agent_names),
            "status_counts": dict(sorted(status_counts.items())),
            "top_span_names": span_name_counts.most_common(20),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "raw_json_bytes": total_raw_bytes,
            "sample_trace_ids": [
                str(summary["trace_id"])
                for summary in summaries[:_OVERVIEW_SAMPLE_TRACE_IDS]
            ],
        }

    def query_traces(
        self,
        filters: Mapping[str, Any] | None = None,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit = _coerce_int(limit, default=50, minimum=1, maximum=_QUERY_LIMIT_CAP)
        offset = _coerce_int(offset, default=0, minimum=0)
        traces = self._filtered_traces(filters)
        summaries = [self._trace_summary(trace_id, spans) for trace_id, spans in traces]
        return {
            "traces": summaries[offset : offset + limit],
            "total": len(summaries),
            "limit": limit,
            "offset": offset,
        }

    def count_traces(self, filters: Mapping[str, Any] | None = None) -> dict[str, int]:
        return {"total": len(self._filtered_traces(filters))}

    def view_trace(self, trace_id: str) -> dict[str, Any]:
        spans = self._spans_for_trace(trace_id)
        return self._view_response(trace_id, spans, _DISCOVERY_ATTR_TRUNCATION_CHARS)

    def view_spans(self, trace_id: str, span_ids: list[str]) -> dict[str, Any]:
        wanted = {str(span_id) for span_id in span_ids[:_VIEW_SPANS_LIMIT]}
        spans = [
            span for span in self._spans_for_trace(trace_id) if span.span_id in wanted
        ]
        return self._view_response(trace_id, spans, _SURGICAL_ATTR_TRUNCATION_CHARS)

    def search_trace(
        self,
        trace_id: str,
        regex_pattern: str,
        *,
        context_chars: int = 100,
        max_matches: int = 50,
    ) -> dict[str, Any]:
        spans = self._spans_for_trace(trace_id)
        return self._search_spans(
            trace_id,
            spans,
            regex_pattern,
            context_chars=context_chars,
            max_matches=max_matches,
        )

    def search_span(
        self,
        trace_id: str,
        span_id: str,
        regex_pattern: str,
        *,
        context_chars: int = 100,
        max_matches: int = 50,
    ) -> dict[str, Any]:
        spans = [
            span for span in self._spans_for_trace(trace_id) if span.span_id == span_id
        ]
        if not spans:
            raise KeyError(f"span_id={span_id!r} not found in trace_id={trace_id!r}")
        result = self._search_spans(
            trace_id,
            spans,
            regex_pattern,
            context_chars=context_chars,
            max_matches=max_matches,
        )
        result["span_id"] = span_id
        return result

    def _filtered_traces(
        self, filters: Mapping[str, Any] | None
    ) -> list[tuple[str, list[SpanRecord]]]:
        filters = _mapping(filters)
        rows = list(self._traces.items())
        if not filters:
            return rows
        return [
            (trace_id, spans)
            for trace_id, spans in rows
            if self._trace_matches(trace_id, spans, filters)
        ]

    def _trace_matches(
        self,
        trace_id: str,
        spans: list[SpanRecord],
        filters: Mapping[str, Any],
    ) -> bool:
        summary = self._trace_summary(trace_id, spans)

        trace_ids = _string_set(filters.get("trace_ids") or filters.get("trace_id"))
        if trace_ids and trace_id not in trace_ids:
            return False

        if "has_errors" in filters and bool(summary["has_errors"]) is not bool(
            filters["has_errors"]
        ):
            return False

        status_codes = _string_set(
            filters.get("status_codes") or filters.get("status_code")
        )
        if status_codes and not (status_codes & set(summary["status_counts"])):
            return False

        span_names = _string_set(filters.get("span_names") or filters.get("span_name"))
        if span_names and not any(span.name in span_names for span in spans):
            return False

        name_contains = filters.get("span_name_contains") or filters.get(
            "name_contains"
        )
        if name_contains:
            needle = str(name_contains).casefold()
            if not any(needle in span.name.casefold() for span in spans):
                return False

        for key in ("service_names", "model_names", "agent_names"):
            wanted = _string_set(filters.get(key) or filters.get(key[:-1]))
            if wanted and not (wanted & set(summary[key])):
                return False

        start_time_gte = filters.get("start_time_gte")
        if (
            start_time_gte
            and summary["start_time"]
            and summary["start_time"] < str(start_time_gte)
        ):
            return False

        end_time_lte = filters.get("end_time_lte")
        if (
            end_time_lte
            and summary["end_time"]
            and summary["end_time"] > str(end_time_lte)
        ):
            return False

        regex_pattern = filters.get("regex_pattern")
        if regex_pattern:
            pattern = re.compile(str(regex_pattern))
            if not any(pattern.search(span.raw_json) for span in spans):
                return False

        return True

    def _trace_summary(self, trace_id: str, spans: list[SpanRecord]) -> dict[str, Any]:
        status_counts = Counter(span.status_code for span in spans if span.status_code)
        span_name_counts = Counter(span.name for span in spans)
        service_names = {
            str(span.resource_attributes.get("service.name"))
            for span in spans
            if span.resource_attributes.get("service.name")
        }
        model_names: set[str] = set()
        agent_names: set[str] = set()
        total_input_tokens = 0
        total_output_tokens = 0

        for span in spans:
            attrs = span.attributes
            model_names.update(_model_names(attrs))
            agent_name = attrs.get("gen_ai.agent.name") or attrs.get(
                "inference.agent_name"
            )
            if agent_name:
                agent_names.add(str(agent_name))
            total_input_tokens += _int_attr(
                attrs,
                "gen_ai.usage.input_tokens",
                "inference.llm.input_tokens",
                "llm.input_tokens",
            )
            total_output_tokens += _int_attr(
                attrs,
                "gen_ai.usage.output_tokens",
                "inference.llm.output_tokens",
                "llm.output_tokens",
            )

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "start_time": min(
                (span.start_time for span in spans if span.start_time), default=""
            ),
            "end_time": max(
                (span.end_time for span in spans if span.end_time), default=""
            ),
            "has_errors": _has_error_status(status_counts),
            "status_counts": dict(sorted(status_counts.items())),
            "span_name_counts": dict(span_name_counts.most_common(20)),
            "service_names": sorted(service_names),
            "model_names": sorted(model_names),
            "agent_names": sorted(agent_names),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "raw_json_bytes": sum(span.raw_json_bytes for span in spans),
            "sample_span_ids": [span.span_id for span in spans[:20]],
        }

    def _spans_for_trace(self, trace_id: str) -> list[SpanRecord]:
        try:
            return self._traces[str(trace_id)]
        except KeyError as e:
            raise KeyError(f"trace_id={trace_id!r} not found") from e

    def _view_response(
        self,
        trace_id: str,
        spans: list[SpanRecord],
        attr_cap_chars: int,
    ) -> dict[str, Any]:
        rendered = [_render_span(span, attr_cap_chars) for span in spans]
        response = {"trace_id": trace_id, "spans": rendered, "oversized": None}
        if _json_size(response) <= _VIEW_RESPONSE_BYTES_BUDGET:
            return response

        span_sizes = [_json_size(span) for span in rendered]
        sorted_sizes = sorted(span_sizes)
        name_counts = Counter(span.name for span in spans)
        return {
            "trace_id": trace_id,
            "spans": [],
            "oversized": {
                "span_count": len(spans),
                "truncated_response_bytes": _json_size(response),
                "response_bytes_budget": _VIEW_RESPONSE_BYTES_BUDGET,
                "span_response_bytes_min": sorted_sizes[0] if sorted_sizes else 0,
                "span_response_bytes_median": sorted_sizes[len(sorted_sizes) // 2]
                if sorted_sizes
                else 0,
                "span_response_bytes_max": sorted_sizes[-1] if sorted_sizes else 0,
                "top_span_names": name_counts.most_common(10),
                "recommendation": (
                    "Use search_trace/search_span to narrow the evidence, then "
                    "view_spans with a smaller span_id set."
                ),
            },
        }

    def _search_spans(
        self,
        trace_id: str,
        spans: list[SpanRecord],
        regex_pattern: str,
        *,
        context_chars: int,
        max_matches: int,
    ) -> dict[str, Any]:
        pattern = re.compile(str(regex_pattern))
        context_chars = _coerce_int(context_chars, default=100, minimum=0, maximum=4000)
        max_matches = _coerce_int(
            max_matches,
            default=50,
            minimum=1,
            maximum=_SEARCH_MATCH_LIMIT_CAP,
        )
        matches: list[dict[str, Any]] = []
        match_count = 0
        for span_index, span in enumerate(spans):
            for match in pattern.finditer(span.raw_json):
                match_count += 1
                if len(matches) >= max_matches:
                    continue
                start = max(0, match.start() - context_chars)
                end = min(len(span.raw_json), match.end() + context_chars)
                matches.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span.span_id,
                        "span_index": span_index,
                        "span_name": span.name,
                        "kind": span.kind,
                        "status_code": span.status_code,
                        "parent_id": span.parent_id,
                        "raw_json_bytes": span.raw_json_bytes,
                        "match_text": match.group(0),
                        "matched_context": span.raw_json[start:end],
                        "match_start_char": match.start(),
                        "match_end_char": match.end(),
                    }
                )
        return {
            "trace_id": trace_id,
            "match_count": match_count,
            "returned_match_count": len(matches),
            "has_more": match_count > len(matches),
            "matches": matches,
        }


def _load_span_records(path: Path) -> list[SpanRecord]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    spans: list[SpanRecord] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        # Earlier captures wrote pretty JSON objects separated by the literal
        # characters "\\n". Accept that format so old cache entries remain usable.
        if text.startswith("\\n", pos):
            pos += 2
            continue
        if pos >= len(text):
            break

        start = pos
        obj, pos = decoder.raw_decode(text, pos)
        if not isinstance(obj, dict):
            continue
        raw = text[start:pos]
        spans.append(
            SpanRecord(
                data=obj,
                raw_json=raw,
                raw_json_bytes=len(raw.encode("utf-8")),
                ordinal=len(spans) + 1,
            )
        )
    return spans


def _render_span(span: SpanRecord, attr_cap_chars: int) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    dropped = 0
    for key, value in span.attributes.items():
        if _NOISY_FLAT_PROJECTION_RE.match(key):
            dropped += 1
            continue
        attrs[key] = _truncate_value(value, attr_cap_chars)
    if dropped:
        attrs["__dropped_flat_projections"] = (
            f"{dropped} noisy flat projection attributes were dropped; use "
            "search_trace/search_span for targeted raw inspection."
        )

    return {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "parent_id": span.parent_id,
        "name": span.name,
        "kind": span.kind,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "status_code": span.status_code,
        "attributes": attrs,
        "resource_attributes": _truncate_value(
            span.resource_attributes, attr_cap_chars
        ),
        "raw_json_bytes": span.raw_json_bytes,
    }


def _truncate_value(value: Any, cap_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= cap_chars:
            return value
        return f"{value[:cap_chars]}... [truncated {len(value) - cap_chars} chars]"
    try:
        serialized = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    if len(serialized) <= cap_chars:
        return value
    return (
        f"{serialized[:cap_chars]}"
        f"... [truncated {len(serialized) - cap_chars} serialized chars]"
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return {str(value)}


def _model_names(attrs: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in (
        "gen_ai.request.model",
        "gen_ai.response.model",
        "inference.llm.model_name",
        "llm.model_name",
    ):
        value = attrs.get(key)
        if value:
            names.add(str(value))
    return names


def _int_attr(attrs: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _has_error_status(status_counts: Mapping[str, int]) -> bool:
    return any("ERROR" in status.upper() for status in status_counts)


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        coerced = default
    coerced = max(minimum, coerced)
    if maximum is not None:
        coerced = min(maximum, coerced)
    return coerced


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
