Summary: Upgrade the GEPA Reflector and its Subagents to operate as Recursive Language Models (RLM) to handle massive execution traces efficiently without context bloat.

## Guarantees
- The Subagent (`analyze_trace_with_llm`) uses a `TraceNavigator` toolset to programmatically query traces instead of receiving the full trace JSON in its prompt.
- The Subagent can retrieve trace summaries, specific message slices, and full tool outputs on demand.
- The Reflector acts as a Parent RLM (Orchestrator) by offloading deep trace inspection to multiple parallel subagents.
- The Reflector starts in "Selection Mode" with a stripped initial prompt, using existing tools to fetch component snippets on demand.

## Constraints
- The full trace JSON is never injected directly into the subagent's prompt.
- Massive tool outputs are truncated or replaced with placeholders in the conversation history until explicitly requested via a tool call.

## Rationale
- Based on the "Recursive Language Models" paper (Zhang et al., 2025), treating long inputs as an external environment accessed via tools prevents KV-cache memory explosions and degraded attention.
- Currently, massive tool outputs in execution traces cause the subagent's context window to overflow. This active, tool-driven context management resolves the scaling bottleneck.

## Non-goals
- Full replacement of the Ouros Sandbox trace analysis for aggregate metrics (the Python map-reduce workflow remains valuable for global pattern detection).
