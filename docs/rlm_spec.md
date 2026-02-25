# Recursive Language Models (RLM) Specification for GEPA Reflector

Based on the paradigm introduced in [Recursive Language Models (Zhang et al., 2025)](https://arxiv.org/abs/2512.24601), this specification outlines how to upgrade the GEPA Reflector and its Subagents to operate as an RLM.

## 1. Background: The RLM Paradigm
The RLM approach fundamentally shifts how Large Language Models handle long context. Instead of passively receiving massive context dumps (which degrades attention and explodes KV-cache memory), an RLM treats long inputs as an "external environment."
- **Programmatic Interface:** The model uses tools to pull in snippets of data on-demand.
- **Recursive Decomposition:** For complex tasks, the model spawns isolated "sub-calls" (child agents) with fresh context windows to process specific segments of the data. 
- **Encapsulated State:** Parent calls only see the high-level output of child calls, keeping the parent's context window pristine and highly focused on reasoning.

## 2. Current Architecture & Limitations
Currently, GEPA's reflection system has two main scaling bottlenecks:
1. **Reflector Context Bloat:** The parent Reflector is injected with a massive prompt containing all candidate components, tool definitions, and system configurations.
2. **Subagent Context Bloat (`analyze_trace_with_llm`):** When the Reflector spawns the subagent, it injects the *entire* trace JSON into the subagent's prompt. If a trace contains huge tool outputs or long multi-turn conversations, the subagent's context window overflows, leading to degraded reasoning or token limit errors.

## 3. Proposed RLM Architecture

We will restructure both the Reflector and its Subagent to utilize an active, tool-driven context management strategy.

### Phase 1: The Subagent as an RLM (Trace Navigator)
Instead of dumping `json.dumps(target_trace)`, the `analyze_trace_with_llm` subagent will start with a minimal prompt and use tools to navigate the trace programmatically.

**New Subagent Tools:**
- `get_trace_summary()`: Returns high-level metrics (score, success, error string, total message count).
- `read_messages(start_idx: int, end_idx: int)`: Returns a slice of the conversation history, omitting massive tool outputs (replaced with placeholders).
- `read_tool_output(message_idx: int, tool_call_id: str)`: Fetches the specific, full output of a tool call only when the agent explicitly requests it.
- `search_trace(query: str)`: Greps through the trace for specific keywords or errors.

**Workflow:** 
The subagent reads the summary, identifies where the error occurred (e.g., message 15), reads messages 10-15, inspects a specific tool output if necessary, and then formulates its conclusion.

### Phase 2: The Reflector as a Parent RLM (Orchestrator)
The main Reflector will offload deep trace inspection entirely, relying on RLM sub-calls to map-reduce over the execution traces.

**New Reflector Capabilities:**
- **`delegate_trace_analysis(trace_ids: list[str], prompt: str) -> dict`**: Spawns *multiple* RLM subagents in parallel for a batch of traces. The Reflector stays focused on the meta-patterns, while the subagents handle the grueling trace navigation. 
- **Enforced Selection Mode:** The Reflector's initial system prompt will be stripped of the full component bodies. It will rely entirely on the existing `list_components()` and `load_component()` tools to fetch code snippets on-demand, matching the RLM "read_snippet" philosophy.

## 4. Implementation Details

**Target File:** `src/pydantic_ai_gepa/gepa_graph/proposal/trace_tools.py`

### Updating `analyze_trace_with_llm`:
1. Define an internal `TraceNavigator` toolset.
2. Store the `target_trace` in the toolset's state.
3. Update the agent initialization:
   ```python
   navigator_tools = create_trace_navigator_toolset(target_trace)
   agent = Agent(
       reflection_model,
       system_prompt="You are a senior debugging engineer. You must use your tools to navigate the execution trace to answer the user's question. Start by getting the summary, then read the relevant messages.",
       tools=navigator_tools
   )
   # Run the agent with just the user's question, NOT the trace JSON.
   result = await agent.run(prompt)
   ```

### Updating Reflector Instructions:
**Target File:** `src/pydantic_ai_gepa/gepa_graph/proposal/instruction.py`
- Modify the `DEFAULT_AGENT_INSTRUCTIONS` to explicitly guide the Reflector to act as a Parent RLM:
  > "You are the parent node in a Recursive Language Model. Do not attempt to guess what happened in the traces. Use `analyze_trace_with_llm` to spawn child agents that will programmatically navigate the traces and return semantic summaries to you."

## 5. Expected Benefits
- **Infinite Trace Scaling:** Traces of any length (even millions of tokens) can be analyzed without blowing up the context window.
- **Cost Efficiency:** The subagent only reads the tokens it needs (e.g., skipping the 50k token output of a successful tool call).
- **Better Reasoning:** The Reflector's context window remains uncluttered, allowing it to focus strictly on generating high-quality hypothesis and instructional updates.
