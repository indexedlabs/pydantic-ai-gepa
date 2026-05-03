from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    failed_result = await run_python_repl(
        "sum(1 for row in rows if not row['success'])"
    )
    listed_result = await run_python_repl("list_dir('traces')")

    assert count_result == "2"
    assert failed_result == "1"
    assert listed_result == "['traces/traces.jsonl']"


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
