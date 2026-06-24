from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

import pydantic_ai_gepa.gepa_graph.proposal.trace_tools as trace_tools_module
from pydantic_ai_gepa.gepa_graph.proposal.trace_store import (
    StructuredTraceStore,
    span_to_jsonl_line,
)
from pydantic_ai_gepa.gepa_graph.proposal.trace_tools import (
    ClearMessageHistoryException,
    create_trace_toolset,
)


def _write_trace_context(tmp_path):
    base_dir = tmp_path / ".gepa_cache" / "runs" / "run-1" / "candidates" / "0"
    traces_dir = base_dir / "traces"
    traces_dir.mkdir(parents=True)
    (traces_dir / "traces.jsonl").write_text(
        '{"success": true, "score": 1}\n'
        '{"success": false, "score": 0, "feedback": "missed edge"}\n',
        encoding="utf-8",
    )
    (base_dir / "components.json").write_text(
        '{"instructions": "seed instructions"}',
        encoding="utf-8",
    )
    (base_dir / "large.log").write_text(
        "".join(f"line-{idx}\n" for idx in range(1105)),
        encoding="utf-8",
    )
    (base_dir / "long.jsonl").write_text(
        '{"payload": "' + ("x" * 21050) + '"}\n',
        encoding="utf-8",
    )
    return base_dir


def _span(
    *,
    trace_id: str,
    span_id: str,
    parent_id: str = "",
    name: str = "chat test",
    status_code: str = "UNSET",
    attributes: dict | None = None,
):
    return {
        "name": name,
        "context": {
            "trace_id": trace_id,
            "span_id": span_id,
            "trace_state": "[]",
        },
        "kind": "SpanKind.CLIENT",
        "parent_id": parent_id,
        "start_time": "2026-05-03T18:03:48.000000Z",
        "end_time": "2026-05-03T18:03:49.000000Z",
        "status": {"status_code": status_code},
        "attributes": attributes or {},
        "events": [],
        "links": [],
        "resource": {
            "attributes": {
                "service.name": "gepa-test",
            },
            "schema_url": "",
        },
    }


def _write_otel_trace_context(tmp_path):
    base_dir = tmp_path / ".gepa_cache" / "runs" / "run-1" / "candidates" / "0"
    traces_dir = base_dir / "traces"
    traces_dir.mkdir(parents=True)
    spans = [
        _span(
            trace_id="trace-a",
            span_id="span-a1",
            attributes={
                "gen_ai.agent.name": "student",
                "gen_ai.request.model": "test-model",
                "gen_ai.response.model": "test-model",
                "gen_ai.usage.input_tokens": 12,
                "gen_ai.usage.output_tokens": 3,
                "gen_ai.output.messages": '[{"role":"assistant","content":"ok"}]',
            },
        ),
        _span(
            trace_id="trace-a",
            span_id="span-a2",
            parent_id="span-a1",
            name="lookup tool",
            attributes={"tool.name": "lookup"},
        ),
        _span(
            trace_id="trace-b",
            span_id="span-b1",
            name="chat test",
            status_code="ERROR",
            attributes={
                "gen_ai.agent.name": "student",
                "gen_ai.request.model": "test-model",
                "exception.message": "ValueError: bad input",
            },
        ),
    ]
    (traces_dir / "traces.jsonl").write_text(
        "".join(json.dumps(span, separators=(",", ":")) + "\n" for span in spans),
        encoding="utf-8",
    )
    (base_dir / "components.json").write_text(
        '{"instructions": "seed instructions"}',
        encoding="utf-8",
    )
    return base_dir


def test_span_to_jsonl_line_compacts_pretty_span_json() -> None:
    class FakeSpan:
        def to_json(self):
            return json.dumps(
                _span(trace_id="trace-a", span_id="span-a1"),
                indent=2,
            )

    line = span_to_jsonl_line(FakeSpan())

    assert line.endswith("\n")
    assert "\\n" not in line.rstrip("\n")
    parsed = json.loads(line)
    assert parsed["context"]["trace_id"] == "trace-a"
    assert parsed["context"]["span_id"] == "span-a1"


def test_structured_trace_store_reads_legacy_literal_newline_separator(
    tmp_path,
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        json.dumps(_span(trace_id="trace-a", span_id="span-a1"), indent=2)
        + "\\n"
        + json.dumps(_span(trace_id="trace-b", span_id="span-b1"), indent=2),
        encoding="utf-8",
    )

    store = StructuredTraceStore.load(path)

    assert store.overview()["total_traces"] == 2
    assert store.query_traces(limit=10)["traces"][1]["trace_id"] == "trace-b"


def test_structured_trace_store_filters_span_name_across_all_spans(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    spans = [
        _span(trace_id="trace-a", span_id=f"span-a{idx}", name=f"common-{idx}")
        for idx in range(25)
    ]
    spans.append(_span(trace_id="trace-a", span_id="span-rare", name="rare target"))
    spans.append(_span(trace_id="trace-b", span_id="span-b1", name="other trace"))
    path.write_text(
        "".join(json.dumps(span, separators=(",", ":")) + "\n" for span in spans),
        encoding="utf-8",
    )

    store = StructuredTraceStore.load(path)

    assert store.count_traces({"span_name": "rare target"}) == {"total": 1}
    assert (
        store.query_traces({"span_name": "rare target"})["traces"][0]["trace_id"]
        == "trace-a"
    )


@pytest.mark.asyncio
async def test_monty_repl_reads_trace_context_and_persists_state(
    monkeypatch, tmp_path
) -> None:
    _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    toolset = create_trace_toolset("run-1", 0)

    assert {
        "read_file",
        "list_dir",
        "run_python_repl",
        "clear_message_history",
        "spawn_agent",
    }.issubset(toolset.tools)

    run_python_repl = toolset.tools["run_python_repl"].function

    count_result = await run_python_repl(
        "rows = [json_loads(line) for line in read_file('traces/traces.jsonl').splitlines()]\n"
        "len(rows)"
    )
    assignment_result = await run_python_repl("answer = 42")
    variable_result = await run_python_repl("answer")
    failed_result = await run_python_repl(
        "sum(1 for row in rows if not row['success'])"
    )
    listed_result = await run_python_repl("list_dir('traces')")

    assert count_result == "2"
    assert assignment_result == "None"
    assert variable_result == "42"
    assert failed_result == "1"
    assert listed_result == "['traces/traces.jsonl']"


@pytest.mark.asyncio
async def test_structured_trace_helpers_query_view_and_search_spans(
    monkeypatch, tmp_path
) -> None:
    _write_otel_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    toolset = create_trace_toolset("run-1", 0)
    run_python_repl = toolset.tools["run_python_repl"].function

    overview_result = await run_python_repl(
        "overview = trace_overview()\n"
        "(overview['total_traces'], overview['total_spans'], "
        "overview['error_trace_count'], overview['model_names'])"
    )
    query_result = await run_python_repl(
        "errors = query_traces({'has_errors': True})\n"
        "(errors['total'], errors['traces'][0]['trace_id'], "
        "errors['traces'][0]['status_counts'])"
    )
    count_result = await run_python_repl(
        "count_traces({'model_names': ['test-model']})['total']"
    )
    view_result = await run_python_repl(
        "view = view_trace('trace-a')\n"
        "(len(view['spans']), view['spans'][0]['attributes']['gen_ai.request.model'])"
    )
    span_result = await run_python_repl(
        "selected = view_spans('trace-a', ['span-a2'])\n"
        "(len(selected['spans']), selected['spans'][0]['attributes']['tool.name'])"
    )
    search_result = await run_python_repl(
        "matches = search_trace('trace-b', 'ValueError')\n"
        "(matches['match_count'], matches['matches'][0]['span_id'], "
        "'ValueError' in matches['matches'][0]['matched_context'])"
    )

    assert overview_result == "(2, 3, 1, ['test-model'])"
    assert query_result == "(1, 'trace-b', {'ERROR': 1})"
    assert count_result == "2"
    assert view_result == "(2, 'test-model')"
    assert span_result == "(1, 'lookup')"
    assert search_result == "(1, 'span-b1', True)"


@pytest.mark.asyncio
async def test_monty_repl_timeout_does_not_poison_future_calls(
    monkeypatch, tmp_path
) -> None:
    _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        trace_tools_module,
        "_MONTY_LIMITS",
        {
            "max_duration_secs": 0.01,
            "max_memory": 128 * 1024 * 1024,
            "max_recursion_depth": 1000,
        },
    )

    toolset = create_trace_toolset("run-1", 0)
    run_python_repl = toolset.tools["run_python_repl"].function

    await run_python_repl("marker = 'still here'")
    timeout_result = await run_python_repl("while True:\n    pass")
    recovered_result = await run_python_repl("(marker, 1 + 1)")

    assert "TimeoutError: time limit exceeded" in timeout_result
    assert recovered_result == "('still here', 2)"


@pytest.mark.asyncio
async def test_host_line_helpers_inspect_large_files_without_full_read(
    monkeypatch, tmp_path
) -> None:
    base_dir = _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    toolset = create_trace_toolset("run-1", 0)
    run_python_repl = toolset.tools["run_python_repl"].function

    size_result = await run_python_repl("file_size('large.log')")
    line_count_result = await run_python_repl("line_count('large.log')")
    file_info_result = await run_python_repl("file_info('large.log')['line_count']")
    slice_result = await run_python_repl("read_lines('large.log', start=10, limit=3)")
    batch_result = await run_python_repl(
        "batch_1 = read_line_batch('large.log', limit=3)\n"
        "batch_2 = read_line_batch('large.log', offset=batch_1['next_offset'], limit=2)\n"
        "(batch_1['lines'], batch_2['lines'], batch_1['eof'], batch_2['offset'] == batch_1['next_offset'])"
    )
    scan_result = await run_python_repl(
        "offset = 0\n"
        "count = 0\n"
        "last = ''\n"
        "while True:\n"
        "    batch = read_line_batch('large.log', offset=offset, limit=500)\n"
        "    count = count + len(batch['lines'])\n"
        "    if batch['lines']:\n"
        "        last = batch['lines'][-1]\n"
        "    if batch['eof']:\n"
        "        break\n"
        "    offset = batch['next_offset']\n"
        "(count, last, batch['eof'])"
    )
    json_scan_result = await run_python_repl(
        "offset = 0\n"
        "failed = 0\n"
        "while True:\n"
        "    batch = read_line_batch('traces/traces.jsonl', offset=offset)\n"
        "    for line in batch['lines']:\n"
        "        row = json_loads(line)\n"
        "        if not row['success']:\n"
        "            failed = failed + 1\n"
        "    if batch['eof']:\n"
        "        break\n"
        "    offset = batch['next_offset']\n"
        "failed"
    )
    long_json_result = await run_python_repl(
        "line = read_line_batch('long.jsonl', limit=1)['lines'][0]\n"
        "len(json_loads(line)['payload'])"
    )
    past_eof_result = await run_python_repl(
        "batch = read_line_batch('large.log', offset=file_size('large.log'))\n"
        "(batch['lines'], batch['eof'], batch['bytes_read'])"
    )
    tail_result = await run_python_repl("tail_lines('large.log', limit=2)")
    find_result = await run_python_repl("find_lines('large.log', 'line-110', limit=3)")
    capped_result = await run_python_repl("len(read_lines('large.log', limit=5000))")
    batch_capped_result = await run_python_repl(
        "len(read_line_batch('large.log', limit=5000)['lines'])"
    )
    byte_capped_result = await run_python_repl(
        "batch = read_line_batch('large.log', limit=1000, max_bytes=20)\n"
        "(len(batch['lines']), batch['eof'])"
    )

    assert size_result == str((base_dir / "large.log").stat().st_size)
    assert line_count_result == "1105"
    assert file_info_result == "1105"
    assert slice_result == "['line-10', 'line-11', 'line-12']"
    assert batch_result == (
        "(['line-0', 'line-1', 'line-2'], ['line-3', 'line-4'], False, True)"
    )
    assert scan_result == "(1105, 'line-1104', True)"
    assert json_scan_result == "1"
    assert long_json_result == "21050"
    assert past_eof_result == "([], True, 0)"
    assert tail_result == "['line-1103', 'line-1104']"
    assert find_result == (
        "[{'line_number': 111, 'text': 'line-110'}, "
        "{'line_number': 1101, 'text': 'line-1100'}, "
        "{'line_number': 1102, 'text': 'line-1101'}]"
    )
    assert capped_result == "1000"
    assert batch_capped_result == "1000"
    assert byte_capped_result == "(2, False)"


@pytest.mark.asyncio
async def test_monty_repl_mount_is_read_only_and_context_bound(
    monkeypatch, tmp_path
) -> None:
    _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    toolset = create_trace_toolset("run-1", 0)
    run_python_repl = toolset.tools["run_python_repl"].function

    outside_result = await run_python_repl("read_file('/tmp/not-allowed')")
    outside_host_result = await run_python_repl("line_count('/tmp/not-allowed')")
    outside_batch_result = await run_python_repl("read_line_batch('/tmp/not-allowed')")
    write_result = await run_python_repl(
        "from pathlib import Path\nPath('/ctx/traces/new.txt').write_text('x')"
    )

    assert "Error executing REPL code: PermissionError" in outside_result
    assert "Error executing REPL code: PermissionError" in outside_host_result
    assert "Error executing REPL code: PermissionError" in outside_batch_result
    assert "Read-only file system" in write_result
    assert not (
        tmp_path
        / ".gepa_cache"
        / "runs"
        / "run-1"
        / "candidates"
        / "0"
        / "traces"
        / "new.txt"
    ).exists()


def test_clear_message_history_tool_raises_control_exception(tmp_path, monkeypatch):
    _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    toolset = create_trace_toolset("run-1", 0)
    clear_message_history = toolset.tools["clear_message_history"].function

    with pytest.raises(ClearMessageHistoryException) as exc_info:
        clear_message_history("continue with cached rows")

    assert exc_info.value.next_context == "continue with cached rows"


@pytest.mark.asyncio
async def test_spawn_agent_uses_fresh_monty_repl_with_trace_context(
    monkeypatch, tmp_path
) -> None:
    _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    class FakeAgent:
        calls: list[str] = []

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def run(self, current_prompt, *, toolsets):
            self.calls.append(str(current_prompt))
            run_python_repl = toolsets[0].tools["run_python_repl"].function
            output = await run_python_repl(
                "try:\n"
                "    inherited = marker\n"
                "except NameError:\n"
                "    inherited = 'missing'\n"
                "marker = 'child-local'\n"
                "count = line_count('traces/traces.jsonl')\n"
                "preview = read_line_batch('traces/traces.jsonl', limit=1)\n"
                "(inherited, count, len(preview['lines']))"
            )
            return SimpleNamespace(output=output)

    monkeypatch.setattr(
        "pydantic_ai_gepa.gepa_graph.proposal.trace_tools.Agent",
        FakeAgent,
    )

    toolset = create_trace_toolset("run-1", 0)
    parent_run_python_repl = toolset.tools["run_python_repl"].function
    spawn_agent = toolset.tools["spawn_agent"].function

    await parent_run_python_repl("marker = 'parent-local'")
    first_result = await spawn_agent("inspect child one")
    second_result = await spawn_agent("inspect child two")

    assert first_result == "('missing', 2, 1)"
    assert second_result == "('missing', 2, 1)"
    assert FakeAgent.calls == ["inspect child one", "inspect child two"]


@pytest.mark.asyncio
async def test_spawn_agent_enforces_shared_recursive_limit(
    monkeypatch, tmp_path
) -> None:
    _write_trace_context(tmp_path)
    monkeypatch.chdir(tmp_path)

    class FakeAgent:
        calls: list[str] = []

        def __init__(self, *args, **kwargs) -> None:
            pass

        async def run(self, current_prompt, *, toolsets):
            self.calls.append(str(current_prompt))
            spawn_agent = toolsets[0].tools["spawn_agent"].function
            output = await spawn_agent("nested child")
            return SimpleNamespace(output=output)

    monkeypatch.setattr(
        "pydantic_ai_gepa.gepa_graph.proposal.trace_tools.Agent",
        FakeAgent,
    )

    toolset = create_trace_toolset("run-1", 0, max_spawned_agents=1)
    spawn_agent = toolset.tools["spawn_agent"].function

    result = await spawn_agent("top child")

    assert (
        result == "Error: spawn_agent limit exceeded (1 sub-agents per proposal step)."
    )
    assert FakeAgent.calls == ["top child"]
