Summary: Implement the TraceNavigator toolset for `analyze_trace_with_llm` and enforce Selection Mode for the Reflector to upgrade them to an RLM architecture.

## Acceptance Criteria
- `analyze_trace_with_llm` no longer injects full trace JSON into the subagent prompt.
- `TraceNavigator` toolset is implemented and provided to the subagent with tools: `get_trace_summary`, `read_messages`, `read_tool_output`, `search_trace`.
- Reflector instructions are updated to explicitly guide the Parent RLM to orchestrate trace analysis via child agents.
