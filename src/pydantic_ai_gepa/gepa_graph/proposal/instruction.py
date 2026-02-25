"""LLM-based proposal generation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    UserContent,
    VideoUrl,
)

from pydantic_ai_gepa.inspection import InspectionAborted

from ...adapter import (
    ReflectiveDataset,
    SharedReflectiveDataset,
)
from ..example_bank import InMemoryExampleBank
from ..models import CandidateProgram, ComponentValue
from .example_bank_tools import create_example_bank_tools

DEFAULT_AGENT_INSTRUCTIONS = """Your mission is to discover instruction formats that measurably improve the student agent's performance.

## The Creative Challenge

You have execution traces showing how a student agent performed various tasks. Some succeeded, some failed, some revealed interesting patterns. Treat every trace as actionable evidence when inventing new instructions.

## Experimental Mindset

Approach this with curiosity and creativity:
- Which patterns keep repeating across traces?
- What hypotheses explain those patterns?
- Which alternative instruction styles could address the gaps?
- How can you test new ideas without discarding what already works?

## Rich Design Space

You can experiment with any format or approach. In particular, we see high leverage from **inline example banks** that juxtapose "do this" and "avoid this" snippets tied to the observed traces. Use these contrastive examples to distill the domain knowledge you observe (e.g., how certain phrases, cues, or tool-usage patterns translate into concrete actions) so the student internalizes the underlying rules, not just the surface wording.

When you add inline example banks, append them as a clearly labeled final section (e.g., "Example Bank" or "Few-Shot Reference") so the student sees the canonical examples right after the core instructions.

**Teaching Styles**
- Learning by example
- Learning by principle
- Learning by discovery
- Learning by analogy
- Learning by contrast
- Learning by practice

**Communication Formats**
- Stories that convey patterns
- Recipes for success
- Maps through problem space
- Guardrails and guidelines
- Contrastive example banks
- Mental models
- Thinking tools

**Creative Structures**
- Poetic constraints that guide
- Rhythmic patterns that stick
- Visual layouts in text
- Memorable frameworks
- Unexpected metaphors

## Evidence to Leverage

Use the traces as concrete backing for each idea:
- Failures highlight missing guardrails
- Successes show proven behaviors to preserve
- Cross-run patterns suggest reusable structures
- Edge cases reveal robustness requirements

## Analysis Framework

For each set of traces, discover:
- **Success patterns:** What approaches worked brilliantly?
- **Failure modes:** Where did things go wrong?
- **Hidden connections:** What patterns link different outcomes?
- **Knowledge gaps:** What understanding is missing?
- **Efficiency opportunities:** How could tasks be done better?
- **Structural insights:** What systemic issues appear?

## Scratchpad Relay Protocol

The "Pattern Discovery", "Creative Hypothesis", and "Experimental Approach" fields act as a multi-step scratchpad. Treat them like a baton pass to the next reflection:
- Start each field with labels such as `Keep:`, `Change:`, `Experiment:` so lineage is obvious.
- Cite the evidence (trace IDs, failure themes) that motivated each bullet.
- Explicitly connect your proposed changes to specific failures in the traces.
- End the Experimental Approach with a checkpoint describing how to measure whether the change worked (what behaviors or metrics should improve next time).

## Output Requirements

Your updated components should:
- Address specific patterns from the traces
- Introduce concise, testable hypotheses
- Match the complexity to the evidence
- Balance clarity with creativity
- Work together as a unified system
- Whenever feasible, include a short bank of positive vs. negative examples (or success vs. failure traces) that encode the domain knowledge extracted from the traces—spell out the interpretation rule, then show the matching and mismatching code. Place this example bank at the end of the instructions so it reads like a few-shot appendix the student can reference quickly.

## Critical: Student-Facing Language

The student agent does NOT see traces or trace IDs—those are internal to this reflection process. When writing instructions for the student:
- NEVER reference "Trace 1", "Trace 7", or any trace numbering
- NEVER say "as seen in the traces" or similar—the student has no context for this
- DO describe patterns, rules, and examples in terms the student can understand
- DO name patterns descriptively rather than by trace number

Bad: "### ✓ Simple Split - Trace 7 pattern"
Good: "### ✓ Pattern name that describes the behavior"

## Instruction Design Goal

Produce instructions that are clear, memorable, and grounded in observed behavior. Help the student see patterns, avoid pitfalls, and execute reliable solutions. Let the evidence steer you toward new hypotheses instead of repeating boilerplate. Favor ideas that raise the student's ability to reason about *any* domain, not just the current dataset.

## Hypothesis Scratchpad Discipline

- Before proposing new instructions, reread the stored hypotheses above the configuration and explicitly state how you are extending or revising them.
- Tie each hypothesis directly to the traces and components it informed—cite successes, failures, or surprises.
- Call out which parts of the hypothesis stay valid, which parts need tweaks, and which parts you are discarding.
- Keep it concise and component-aware so the next reflection can quickly inherit the right mental model.

Always connect the *latest* evidence back to its originating hypothesis before proposing new instructions, and let the scratchpad capture the causal reasoning you want to hand off."""

EXAMPLE_BANK_TOOLS_INSTRUCTIONS = """## Searchable Example Bank Tools

You have access to a **searchable example bank** - a persistent library of few-shot examples that the student can query at runtime. This is different from inline example banks (examples embedded directly in instructions).

**When to use the searchable example bank vs. inline examples:**
- **Inline examples** (in instructions): Best for a small number of critical examples (3-10) that should always be visible to the student. Start here.
- **Searchable example bank**: Use when you need tens of examples that would bloat the context if inlined. The student retrieves only relevant examples on demand.

**Available tools:**
- `add_example(title, keywords, content)` - Add an example (title + keywords + content).
- `remove_example(example_id)` - Remove an example that's no longer helpful or is causing confusion.
- `list_examples()` - See all examples currently in the bank.
- `read_example(example_id)` - Read the full content of an example.
- `test_retrieval(query)` - Preview what the student would see when searching with a given query.

**When to add examples to the bank:**
- When inline examples aren't enough and you need a larger reference library
- When you see many distinct failure patterns that each warrant their own example
- When the traces reveal domain-specific knowledge that's hard to encode in general instructions

**When to remove examples:**
- If an example is too narrow and doesn't generalize
- If the student is retrieving and misapplying an example
- If the example conflicts with updated instructions

**Important - Teaching the student to use the example bank:**
When you add examples to the bank, also update the student's instructions to tell them when and how to use the `search_examples` tool. For example, if you add examples about handling edge cases, include guidance like "When encountering an unfamiliar pattern, use `search_examples` to find relevant examples before proceeding." Without explicit instructions, the student may not know to search the example bank.

**Example structure tips:**
- Title: Brief, descriptive (e.g., "Handling null responses from API")
- Keywords: Terms the student would search for (e.g., ["null", "API", "error handling", "empty response"])
- Content: The actual example showing the pattern, ideally with both correct and incorrect approaches"""


COMPONENT_DISCOVERY_TOOLS_INSTRUCTIONS = """## Component discovery tools

You can use these tools to choose what to edit:
- `list_components(prefix)` - List available component names (optional substring filter).
- `search_components(query, top_k)` - Find likely relevant component names.
- `load_component(component_name)` - Read the current text for a component.
"""


SKILLS_DISCOVERY_TOOLS_INSTRUCTIONS = """## Skills discovery tools

When skills are enabled, you may also use:
- `list_skills()` - List available skills (name + description).
- `search_skills(query, top_k)` - Find relevant skills by keyword.
- `load_skill(skill_path)` - Read the full SKILL.md for a skill (without activating it for editing).
- `load_skill_file(skill_path, path)` - Read a file referenced by a skill (e.g. examples) without activating.
- `activate_skill_components(skill_path, include_examples)` - Make a skill's description/body (and optionally examples) available as components to edit.
- `create_skill(skill_path, description, body)` - Create a new skill from scratch, immediately making its components available to edit. Use this to abstract reusable strategies out of the main instructions.
- `list_active_skills()` - List which skills have been activated so far.
"""

JOURNAL_TOOLS_INSTRUCTIONS = """## Durable Journal Tools (Reflection Ledger)

You have access to a persistent journal via the `read_journal_entries` and `append_journal_entry` tools. This journal acts as a long-term memory across independent optimization runs.

**CRITICAL PROTOCOL:**
1. **Always read first:** If the journal tools are available, you should use `read_journal_entries` to see what past iterations have learned before you make any decisions.
2. **Incorporate insights:** Use the past strategies to inform your new proposals (e.g. avoid repeating known mistakes, apply discovered rules).
3. **Record new discoveries:** If you identify a concrete, reusable strategy or edge-case insight from analyzing the current evaluation traces, you MUST use `append_journal_entry` to log it for future iterations before finalizing your proposal.
"""


def _toolset_has_tool(toolset: object, tool_name: str) -> bool:
    tools = getattr(toolset, "tools", None)
    if isinstance(tools, dict):
        return tool_name in tools
    return False


def _skills_tools_enabled(component_toolsets: Sequence[AbstractToolset[None]]) -> bool:
    # Detect whether the reflection toolset included skills tools (which only happens
    # when `skills` is enabled in the runner).
    return any(
        _toolset_has_tool(toolset, "list_skills")
        or _toolset_has_tool(toolset, "search_skills")
        or _toolset_has_tool(toolset, "load_skill")
        or _toolset_has_tool(toolset, "load_skill_file")
        or _toolset_has_tool(toolset, "activate_skill_components")
        or _toolset_has_tool(toolset, "create_skill")
        or _toolset_has_tool(toolset, "list_active_skills")
        for toolset in component_toolsets
    )


def _journal_tools_enabled(component_toolsets: Sequence[AbstractToolset[None]]) -> bool:
    return any(
        _toolset_has_tool(toolset, "read_journal_entries")
        or _toolset_has_tool(toolset, "append_journal_entry")
        for toolset in component_toolsets
    )


class TrajectoryAnalysis(BaseModel):
    """Analysis of what happened in the traces before proposing changes."""

    pattern_discovery: str = Field(
        description="What patterns emerge from the traces? Look for connections between successes and failures.",
    )
    creative_hypothesis: str = Field(
        description="Based on the patterns, what approach could address the failures? Why might this work?",
    )
    experimental_approach: str = Field(
        description="What specific instructional changes will you make? How do they address the observed failures?",
    )
    edge_insight: str = Field(
        default="",
        description="Optional: Note any recurring error pattern that's hard to fix and why it persists.",
    )
    evolution_moves: list[str] = Field(
        default_factory=list,
        description="Optional: List the key changes being made (useful for tracking what was tried).",
    )
    success_checkpoint: str = Field(
        default="",
        description="Optional: How will we know this change worked (what should improve in the next evaluation)?",
    )


class ComponentUpdate(BaseModel):
    """Structured representation of a component update."""

    component_name: str = Field(
        description="Name of the component that was updated.",
    )
    optimized_value: str = Field(
        description="Fully rewritten text for the component.",
    )


class InstructionProposalOutput(BaseModel):
    """Agent output schema for instruction proposals."""

    reasoning: TrajectoryAnalysis = Field(
        description="Creative analysis of patterns in the traces, forming the basis for innovative instruction design.",
    )
    updated_components: list[ComponentUpdate] = Field(
        default_factory=list,
        description="Updates for each requested component, implementing the creative approach described above.",
    )


@dataclass(slots=True)
class ProposalResult:
    """Resolved results from the instruction proposal call."""

    texts: dict[str, str]
    component_metadata: dict[str, dict[str, Any]]
    reasoning: TrajectoryAnalysis | None


class InstructionProposalGenerator:
    """Generate improved component texts via a structured agent call."""

    def __init__(
        self,
        instructions: str | None = None,
        *,
        include_hypothesis_metadata: bool = False,
        additional_instructions: str | None = None,
    ) -> None:
        self._agent = Agent(
            instructions=instructions or DEFAULT_AGENT_INSTRUCTIONS,
            output_type=InstructionProposalOutput,
        )
        self._include_hypothesis_metadata = include_hypothesis_metadata
        self._additional_instructions = additional_instructions

    async def propose_texts(
        self,
        *,
        candidate: CandidateProgram,
        reflective_data: ReflectiveDataset,
        components: Sequence[str] | None,
        model: Model | KnownModelName | str,
        model_settings: ModelSettings | None = None,
        example_bank: InMemoryExampleBank | None = None,
        component_toolsets: Sequence[AbstractToolset[None]] | None = None,
    ) -> ProposalResult:
        """Propose new texts for each component via the structured agent.

        Args:
            candidate: The candidate program to optimize
            reflective_data: Training data with execution traces
            components: Optional component names to update. When None, the reflection agent
                is responsible for selecting which components to update using tools.
            model: Model to use for proposal generation
            model_settings: Optional model settings (e.g., temperature, max_tokens)
            example_bank: Optional example bank for the reflection agent to manage.
                If provided, the agent can add/remove examples via tool calls.
            component_toolsets: Optional toolsets for component discovery/activation.
        """
        if components is not None and not components:
            return ProposalResult(texts={}, component_metadata={}, reasoning=None)

        actionable: list[str] = []
        if components is not None:
            untouched: dict[str, str] = {}
            for component in components:
                if component not in candidate.components:
                    raise KeyError(f"Component '{component}' not found in candidate.")

                records = list(self._records_for_component(reflective_data, component))
                if records:
                    actionable.append(component)
                else:
                    untouched[component] = candidate.components[component].text

            if not actionable:
                return ProposalResult(texts={}, component_metadata={}, reasoning=None)
        else:
            untouched = {}

        prompt = self._build_user_prompt(
            candidate=candidate,
            reflective_data=reflective_data,
            components=actionable,
            example_bank=example_bank,
        )

        try:
            toolsets: list[AbstractToolset[None]] = []
            runtime_instructions_parts: list[str] = []
            if example_bank is not None:
                toolsets.append(create_example_bank_tools(example_bank))
                runtime_instructions_parts.append(EXAMPLE_BANK_TOOLS_INSTRUCTIONS)
            if component_toolsets:
                toolsets.extend(component_toolsets)
                runtime_instructions_parts.append(
                    COMPONENT_DISCOVERY_TOOLS_INSTRUCTIONS
                )
                if _skills_tools_enabled(component_toolsets):
                    runtime_instructions_parts.append(
                        SKILLS_DISCOVERY_TOOLS_INSTRUCTIONS
                    )
                if _journal_tools_enabled(component_toolsets):
                    runtime_instructions_parts.append(JOURNAL_TOOLS_INSTRUCTIONS)
            if components is None:
                runtime_instructions_parts.append(
                    "Select which component(s) to update based on the traces, then include only those in `updated_components`."
                )
            if self._additional_instructions:
                runtime_instructions_parts.append(self._additional_instructions)
            runtime_instructions = (
                "\n\n".join(runtime_instructions_parts)
                if runtime_instructions_parts
                else None
            )

            from .trace_tools import ClearMessageHistoryException

            current_prompt: str | Sequence[UserContent] = prompt
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count > 20:
                    raise RuntimeError("Agent exceeded maximum clear_message_history loops (20).")
                try:
                    result = await self._agent.run(
                        current_prompt,
                        model=model,
                        model_settings=model_settings,
                        toolsets=toolsets if toolsets else None,
                        instructions=runtime_instructions,
                        usage_limits=UsageLimits(request_limit=15),
                    )
                    break
                except ClearMessageHistoryException as e:
                    current_prompt = self._join_user_content([
                        f"History cleared. You previously left yourself this note to continue:\n\n{e.next_context}\n\n",
                        "Your Python REPL state is intact."
                    ])

        except InspectionAborted:
            raise
        except Exception:
            return ProposalResult(
                texts={},
                component_metadata={},
                reasoning=None,
            )

        updates = {
            update.component_name: update.optimized_value
            for update in result.output.updated_components
            if update.component_name
        }

        updated: dict[str, str] = {}
        for name, text in updates.items():
            if name not in candidate.components:
                continue
            current = candidate.components[name].text
            if str(text) != current:
                updated[name] = str(text)
        if components is not None:
            # Ensure explicitly requested components don't get dropped when the agent omits them.
            for component in actionable:
                if component in updated:
                    continue
                proposed = updates.get(component)
                if (
                    proposed is not None
                    and str(proposed) != candidate.components[component].text
                ):
                    updated[component] = str(proposed)
        metadata = self._build_component_metadata(
            reasoning=result.output.reasoning
            if self._include_hypothesis_metadata
            else None,
            components=sorted(updated.keys()),
        )

        return ProposalResult(
            texts=updated,
            component_metadata=metadata,
            reasoning=result.output.reasoning,
        )

    def _build_user_prompt(
        self,
        *,
        candidate: CandidateProgram,
        reflective_data: ReflectiveDataset,
        components: Sequence[str],
        example_bank: InMemoryExampleBank | None = None,
    ) -> Sequence[UserContent]:
        selection_mode = not components
        lines: list[Any] = [
            "# Creative Instruction Design Challenge",
            "",
            "Transform the student agent's performance through innovative instruction formats.",
            "",
        ]

        lines.extend(
            [
                "## Context",
                "- A student agent has been running with the configuration shown below",
                "- We've collected traces from real production runs",
                "- Your job is to improve specific components so the student agent performs better",
                "",
            ]
        )

        catalog_tool_defs = self._build_tool_definitions_from_candidate(candidate)

        metadata_groups: list[dict[str, Any]] = []
        metadata_components: list[list[str]] = []
        metadata_index: dict[tuple[tuple[str, str], ...], int] = {}
        target_components = {component for component in components}

        component_sections: list[str] = []
        candidate_component_names = list(candidate.components.keys())

        if selection_mode:
            max_listed_components = 200
            listed_component_names = candidate_component_names[:max_listed_components]
            remaining_components = len(candidate_component_names) - len(
                listed_component_names
            )
            prefix_counts: dict[str, int] = {}
            for name in candidate_component_names:
                if name == "instructions":
                    prefix = "core"
                elif name.startswith("signature:"):
                    prefix = "signature"
                elif name.startswith("tool:"):
                    prefix = "tool"
                elif name.startswith("skill:"):
                    prefix = "skill"
                else:
                    prefix = name.split(":", 1)[0] if ":" in name else "core"
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            prefix_summary = ", ".join(
                f"{prefix}={count}"
                for prefix, count in sorted(
                    prefix_counts.items(), key=lambda item: item[0]
                )
            )
            lines.extend(
                [
                    "## Optimizable components (current candidate)",
                    "",
                    "This run is in component-selection mode: choose which component(s) to update based on the traces.",
                    "The full set of optimizable components is listed here; load any component as needed using the discovery tools.",
                    "",
                    f"Total components: {len(candidate_component_names)}",
                    f"Component groups: {prefix_summary}",
                    "",
                    "Component names:",
                    "```",
                    *listed_component_names,
                    *(
                        [
                            f"... ({remaining_components} more; use list_components()/search_components() to narrow)"
                        ]
                        if remaining_components > 0
                        else []
                    ),
                    "```",
                    "",
                ]
            )
            skill_components_present = any(
                name.startswith("skill:") for name in candidate_component_names
            ) or any(
                name.startswith(
                    (
                        "tool:list_skills:",
                        "tool:search_skills:",
                        "tool:load_skill:",
                        "tool:load_skill_file:",
                    )
                )
                for name in candidate_component_names
            )
            if skill_components_present:
                lines.extend(
                    [
                        "Skills note:",
                        "- Skills content is lazily loaded, so `skill:*` components won't appear until you activate them.",
                        "- Use `list_skills()`, optionally `load_skill(...)` / `load_skill_file(...)` to inspect, then `activate_skill_components(...)` to make specific skills editable as `skill:*` components.",
                        "",
                    ]
                )

        # Show all components in the candidate (explicit mode), or a minimal snapshot
        # (selection mode) to avoid bloating the prompt when many components exist.
        included_component_names: list[str]
        if selection_mode:
            included_component_names = [
                name
                for name in candidate_component_names
                if name == "instructions" or name.startswith("signature:")
            ]
            if not included_component_names:
                included_component_names = [
                    name
                    for name in candidate_component_names
                    if not name.startswith("skill:")
                ][:5]
        else:
            included_component_names = candidate_component_names

        # Use clear boundary markers that won't conflict with content.
        for component_name in included_component_names:
            component_value = candidate.components[component_name]
            component_sections.append(
                f"=== start component: `{component_name}` given to student ==="
            )
            component_sections.append(component_value.text.strip())
            component_sections.append("=== end ===")
            component_sections.append("")

            if (
                self._include_hypothesis_metadata
                and component_name in target_components
            ):
                metadata_entry = self._extract_component_metadata(component_value)
                if metadata_entry:
                    signature = self._metadata_signature(metadata_entry)
                    idx = metadata_index.get(signature)
                    if idx is None:
                        idx = len(metadata_groups)
                        metadata_index[signature] = idx
                        metadata_groups.append(metadata_entry)
                        metadata_components.append([component_name])
                    else:
                        metadata_components[idx].append(component_name)

        if metadata_groups:
            lines.extend(
                [
                    "## Stored hypotheses from previous reflections",
                    "",
                ]
            )
            for metadata_entry, component_list in zip(
                metadata_groups, metadata_components
            ):
                component_names = ", ".join(f"`{name}`" for name in component_list)
                lines.append(f"- Components: {component_names}")
                iteration = metadata_entry.get("iteration")
                if iteration is not None:
                    lines.append(f"  - Iteration: {iteration}")
                if "pattern" in metadata_entry:
                    lines.append(f"  - Pattern: {metadata_entry['pattern']}")
                if "hypothesis" in metadata_entry:
                    lines.append(f"  - Hypothesis: {metadata_entry['hypothesis']}")
                if "approach" in metadata_entry:
                    lines.append(f"  - Plan: {metadata_entry['approach']}")
                moves = metadata_entry.get("moves")
                if moves:
                    joined_moves = ", ".join(str(move) for move in moves)
                    lines.append(f"  - Moves: {joined_moves}")
                elif iteration is not None:
                    lines.append("  - Moves: (not provided)")
                if "edge_insight" in metadata_entry:
                    lines.append(f"  - Edge insight: {metadata_entry['edge_insight']}")
                elif iteration is not None:
                    lines.append("  - Edge insight: (not provided)")
                if "checkpoint" in metadata_entry:
                    lines.append(f"  - Checkpoint: {metadata_entry['checkpoint']}")
                elif iteration is not None:
                    lines.append("  - Checkpoint: (not provided)")
                lines.append("")

        lines.extend(
            [
                "---",
                "",
                "## Full student agent configuration",
                "",
                "This is the complete configuration the student agent was running with:",
                "",
            ]
        )

        lines.extend(component_sections)

        # Collect and show tools if present in evidence
        tools = self._collect_tools(reflective_data)
        tool_map: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = self._extract_tool_name(tool)
            if name:
                tool_map[name] = tool

        for name, catalog_tool in catalog_tool_defs.items():
            if name in tool_map:
                self._merge_tool_definitions(tool_map[name], catalog_tool)
            else:
                tools.append(catalog_tool)
                tool_map[name] = catalog_tool

        output_tool_names: list[str] = []
        if tools:
            for tool in tools:
                if isinstance(tool, Mapping) and tool.get("kind") == "output":
                    function_block = tool.get("function")
                    if isinstance(function_block, Mapping):
                        name = function_block.get("name")
                        if isinstance(name, str) and name:
                            output_tool_names.append(name)
            lines.append("**Tools available to student (JSON Schema):**")
            lines.append("```json")
            lines.append(json.dumps(tools, indent=2))
            lines.append("```")
            lines.append("")
            if output_tool_names:
                sample_name = output_tool_names[0]
                lines.append(
                    f'_Tools with `"kind": "output"` (e.g., `{sample_name}`) end the run.'
                    " Teach the student to call the appropriate output tool when finalizing their answer._"
                )
                lines.append("")

        lines.extend(
            [
                "---",
                "",
                "## Production traces from student agent runs",
                "",
                "Each trace contains:",
                "- `messages`: Full conversation history with system prompts, user inputs, assistant responses, tool calls, and tool returns",
                "- `tools`: Tool definitions that were available (if any)",
                "- `score`: Performance score (0.0-1.0, higher is better)",
                "- `success`: Whether the run completed successfully",
                "- `feedback`: Evaluator feedback on this specific run",
                "",
            ]
        )

        # Instead of serializing full traces, we provide high-level metadata and let the agent use trace analysis tools
        total_records = 0
        success_records = 0

        if isinstance(reflective_data, SharedReflectiveDataset):
            records = reflective_data.records
            total_records = len(records)
            success_records = sum(1 for r in records if r.get("success"))
        else:
            for component_records in reflective_data.records_by_component.values():
                total_records += len(component_records)
                success_records += sum(1 for r in component_records if r.get("success"))

        failed_records = total_records - success_records

        lines.extend(
            [
                "",
                "## Execution Traces Available for Analysis",
                "",
                f"{total_records} traces available from the execution: {success_records} succeeded, {failed_records} failed.",
                "The traces are stored on disk as `traces/traces.jsonl`.",
                "You must use the `run_python_repl(python_code: str)` tool to write and execute python scripts to parse these structured files.",
                "You may also use `spawn_agent(instructions: str)` to spawn a Recursive Language Model (RLM) sub-agent to deeply inspect specific traces for semantic failures.",
                "",
            ]
        )

        lines.extend(
            [
                "",
                "### Analysis guidance",
                "- Use `run_python_repl` to aggregate errors or find common failure modes across `traces.jsonl`.",
                "- Use `spawn_agent` to understand *why* a specific trace failed if the python analysis is insufficient.",
                "- What failure patterns repeat across runs?",
                "- Are components misaligned (e.g., instructions referencing tools that don't exist)?",
                "- Which successful patterns should be preserved or extended?",
                "- What domain knowledge should be codified in the prompts?",
                "- Are there patterns of inefficient tool usage (redundant calls, speculative calls, lack of planning)?",
                "- How can prompts guide the student to gather what's needed in fewer, well-targeted tool calls?",
                "",
                "---",
                "",
                "## Components to update",
                "",
                (
                    "Select which component(s) to update based on the evidence above, then rewrite only those components."
                    if selection_mode
                    else "Rewrite these components as a coordinated update based on the evidence above:"
                ),
                *(
                    [
                        "",
                        "Selection tips:",
                        *(
                            [
                                "- If you need to edit skills content, first use `search_skills(...)`, then `activate_skill_components(...)` to materialize `skill:*` components, then update those `skill:*` components.",
                            ]
                            if any(
                                name.startswith("skill:")
                                for name in candidate.components
                            )
                            or any(
                                name.startswith(
                                    (
                                        "tool:list_skills:",
                                        "tool:search_skills:",
                                        "tool:load_skill:",
                                        "tool:load_skill_file:",
                                    )
                                )
                                for name in candidate.components
                            )
                            else []
                        ),
                        "- If tool usage is inefficient or confusing, update the relevant `tool:*` components (e.g., tool descriptions/parameter docs) so the student uses tools correctly.",
                    ]
                    if selection_mode
                    else []
                ),
                "",
            ]
        )

        # Show each component to update (use clear boundary markers)
        for component in components:
            component_value = candidate.components[component]
            lines.append(f"=== start component: `{component}` current value ===")
            lines.append(component_value.text.strip())
            lines.append("=== end ===")
            lines.append("")

        return self._join_user_content(lines)

    @staticmethod
    def _join_user_content(content: list[Any]) -> list[UserContent]:
        result: list[UserContent] = []
        current_str: list[str] = []
        for item in content:
            if isinstance(item, str):
                current_str.append(item)
            else:
                if current_str:
                    result.append("\n".join(current_str))
                    current_str = []
                result.append(item)
        if current_str:
            result.append("\n".join(current_str))
        return result

    def _build_component_metadata(
        self,
        *,
        reasoning: TrajectoryAnalysis | None,
        components: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        if not self._include_hypothesis_metadata or reasoning is None:
            return {}

        base: dict[str, Any] = {
            "pattern": reasoning.pattern_discovery.strip(),
            "hypothesis": reasoning.creative_hypothesis.strip(),
            "approach": reasoning.experimental_approach.strip(),
        }
        edge_insight = reasoning.edge_insight.strip()
        if edge_insight:
            base["edge_insight"] = edge_insight
        checkpoint = reasoning.success_checkpoint.strip()
        if checkpoint:
            base["checkpoint"] = checkpoint
        moves = [move.strip() for move in reasoning.evolution_moves if move.strip()]
        if moves:
            base["moves"] = moves
        filtered = {key: value for key, value in base.items() if value}
        if not filtered:
            return {}

        return {component: dict(filtered) for component in components}

    def _extract_component_metadata(
        self, component_value: ComponentValue
    ) -> dict[str, Any]:
        metadata = component_value.metadata or {}
        if not isinstance(metadata, dict) or not metadata:
            return {}

        hypothesis = str(metadata.get("hypothesis", "")).strip()
        pattern = str(metadata.get("pattern", "")).strip()
        approach = str(metadata.get("approach", "")).strip()
        edge_insight = str(metadata.get("edge_insight", "")).strip()
        checkpoint = str(metadata.get("checkpoint", "")).strip()
        raw_moves = metadata.get("moves")
        moves: list[str] = []
        if isinstance(raw_moves, list):
            moves = [str(move).strip() for move in raw_moves if str(move).strip()]
        iteration = metadata.get("iteration")

        if not any(
            [hypothesis, pattern, approach, edge_insight, checkpoint, moves, iteration]
        ):
            return {}

        entry: dict[str, Any] = {}
        if iteration is not None:
            entry["iteration"] = iteration
        if pattern:
            entry["pattern"] = pattern
        if hypothesis:
            entry["hypothesis"] = hypothesis
        if approach:
            entry["approach"] = approach
        if edge_insight:
            entry["edge_insight"] = edge_insight
        if checkpoint:
            entry["checkpoint"] = checkpoint
        if moves:
            entry["moves"] = moves
        return entry

    def _metadata_signature(
        self, metadata: Mapping[str, Any]
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (key, str(value))
            for key, value in sorted(metadata.items(), key=lambda item: item[0])
        )

    def _collect_tools(
        self, reflective_data: ReflectiveDataset
    ) -> list[dict[str, Any]]:
        """Collect unique tools from all records in the reflective dataset."""
        tools_map: dict[str, dict[str, Any]] = {}

        records: Sequence[Mapping[str, Any]]
        if isinstance(reflective_data, SharedReflectiveDataset):
            records = reflective_data.records
        else:
            # Flatten all component records
            all_records: list[Mapping[str, Any]] = []
            for component_records in reflective_data.records_by_component.values():
                all_records.extend(component_records)
            records = all_records

        def merge_tool_entries(
            entries: Any,
            *,
            default_kind: str | None = None,
        ) -> None:
            if not entries:
                return
            for tool in entries:
                if not isinstance(tool, Mapping):
                    continue
                function = tool.get("function")
                if not isinstance(function, Mapping):
                    continue
                tool_name = function.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue
                if tool_name in tools_map:
                    continue
                normalized_tool = dict(tool)
                if default_kind and not normalized_tool.get("kind"):
                    normalized_tool["kind"] = default_kind
                tools_map[tool_name] = normalized_tool

        for record in records:
            merge_tool_entries(record.get("tools"))
            merge_tool_entries(record.get("output_tools"), default_kind="output")

        return list(tools_map.values())

    @staticmethod
    def _extract_tool_name(tool: Mapping[str, Any]) -> str | None:
        function_block = tool.get("function")
        if isinstance(function_block, Mapping):
            name = function_block.get("name")
            if isinstance(name, str) and name.strip():
                return name
        return None

    @staticmethod
    def _merge_tool_definitions(
        base: dict[str, Any],
        supplement: dict[str, Any],
    ) -> None:
        if "kind" not in base and supplement.get("kind"):
            base["kind"] = supplement["kind"]

        base_function = base.get("function")
        supplement_function = supplement.get("function")
        if not isinstance(base_function, Mapping) or not isinstance(
            supplement_function, Mapping
        ):
            return

        if not isinstance(base_function, dict):
            base_function = dict(base_function)
            base["function"] = base_function

        if not base_function.get("description") and supplement_function.get(
            "description"
        ):
            base_function["description"] = supplement_function["description"]

        if not base_function.get("parameters") and supplement_function.get(
            "parameters"
        ):
            base_function["parameters"] = supplement_function["parameters"]

    def _build_tool_definitions_from_candidate(
        self,
        candidate: CandidateProgram,
    ) -> dict[str, dict[str, Any]]:
        tool_defs: dict[str, dict[str, Any]] = {}

        for component_name, component_value in candidate.components.items():
            if not component_name.startswith("tool:"):
                continue

            remainder = component_name[len("tool:") :]
            if ":" not in remainder:
                continue

            tool_name, _, key = remainder.partition(":")
            if not tool_name:
                continue

            entry = tool_defs.setdefault(tool_name, self._init_tool_entry(tool_name))
            function_block = entry["function"]
            text = component_value.text.strip()
            if not text:
                continue

            if key == "description":
                function_block["description"] = text
            elif key.startswith("param:"):
                path = key[len("param:") :]
                parameters = function_block.setdefault(
                    "parameters",
                    {"type": "object", "properties": {}},
                )
                self._inject_catalog_parameter(parameters, path, text)

        return {
            name: entry
            for name, entry in tool_defs.items()
            if self._tool_entry_has_content(entry)
        }

    @staticmethod
    def _tool_entry_has_content(entry: Mapping[str, Any]) -> bool:
        function_block = entry.get("function")
        if not isinstance(function_block, Mapping):
            return False
        if function_block.get("description"):
            return True
        parameters = function_block.get("parameters")
        if isinstance(parameters, Mapping):
            properties = parameters.get("properties")
            if isinstance(properties, Mapping) and properties:
                return True
        return False

    def _init_tool_entry(self, tool_name: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        kind = self._guess_tool_kind(tool_name)
        if kind:
            entry["kind"] = kind
        return entry

    @staticmethod
    def _guess_tool_kind(tool_name: str) -> str | None:
        normalized = tool_name.casefold()
        if normalized == "final_result" or normalized.startswith("final_result_"):
            return "output"
        return None

    def _inject_catalog_parameter(
        self,
        schema: dict[str, Any],
        path: str,
        description: str,
    ) -> None:
        if not path:
            return

        segments = [segment for segment in path.split(".") if segment]
        if not segments:
            return

        node = schema
        node.setdefault("type", "object")
        for idx, raw_segment in enumerate(segments):
            is_array = raw_segment.endswith("[]")
            segment = raw_segment[:-2] if is_array else raw_segment
            if not segment:
                continue

            properties = node.setdefault("properties", {})
            child = properties.setdefault(segment, {})

            if is_array:
                child.setdefault("type", "array")
                items = child.setdefault("items", {"type": "object"})
                node = items
                continue

            if idx < len(segments) - 1:
                child.setdefault("type", "object")
                node = child
            else:
                node = child

        if "type" not in node:
            node["type"] = "string"
        node["description"] = description

    @staticmethod
    def _records_for_component(
        dataset: ReflectiveDataset,
        component: str,
    ) -> Sequence[Mapping[str, Any]]:
        if isinstance(dataset, SharedReflectiveDataset):
            return dataset.records
        return dataset.records_by_component.get(component, ())

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value).strip()

    @staticmethod
    def _format_side_info(side_info: Any) -> list[Any]:
        items: list[Any] = []

        def _walk(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{path}.{k}" if path else k
                    _walk(v, p)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    p = f"{path}[{i}]"
                    _walk(v, p)
            elif isinstance(
                obj, (ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent)
            ):
                items.append(f"  - **{path}**:")
                items.append(obj)
            else:
                items.append(f"  - **{path}**: {obj}")

        _walk(side_info)
        return items

    def _format_trace_sections(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        label: str,
    ) -> list[Any]:
        """Render reflective records in a readable structure."""
        if not records:
            return ["No reflective examples were provided.", ""]

        lines: list[Any] = []
        for idx, record in enumerate(records, start=1):
            lines.append(f"=== start {label.lower()} {idx} ===")
            lines.append("")

            summary_lines: list[str] = []
            summary_fields = (
                "score",
                "success",
                "feedback",
                "assistant_response",
                "error",
            )
            for field in summary_fields:
                value = record.get(field)
                if value not in (None, ""):
                    formatted_label = field.replace("_", " ").title()
                    summary_lines.append(
                        f"- **{formatted_label}:** {self._format_scalar(value)}"
                    )

            if summary_lines:
                lines.extend(summary_lines)
                lines.append("")

            side_info = record.get("side_info")
            if side_info:
                lines.append("- **Side Info:**")
                lines.extend(self._format_side_info(side_info))
                lines.append("")

            messages = record.get("messages")
            primary_instructions = self._extract_primary_instructions(messages)
            if messages:
                sanitized_messages = self._strip_duplicate_instructions(
                    messages,
                    primary_instructions,
                )
                lines.append("- **Messages:**")
                lines.append("```json")
                lines.append(json.dumps(sanitized_messages, indent=2))
                lines.append("```")

            run_usage = record.get("run_usage")
            if run_usage:
                lines.append("- **Usage:**")
                lines.append("```json")
                lines.append(json.dumps(run_usage, indent=2))
                lines.append("```")

            lines.append("")
            lines.append(f"=== end {label.lower()} {idx} ===")
            lines.append("")

        return lines

    @staticmethod
    def _extract_primary_instructions(messages: Any) -> str | None:
        if not isinstance(messages, Sequence):
            return None
        for message in messages:
            if isinstance(message, Mapping):
                instructions = message.get("instructions")
                if isinstance(instructions, str) and instructions.strip():
                    return instructions.strip()
        return None

    @staticmethod
    def _strip_duplicate_instructions(
        messages: Any,
        canonical: str | None,
    ) -> Any:
        if not isinstance(messages, list):
            return messages

        sanitized: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                sanitized.append(message)
                continue

            message_copy = dict(message)
            instructions = message_copy.get("instructions")
            if (
                isinstance(instructions, str)
                and canonical
                and instructions.strip() == canonical
            ):
                message_copy.pop("instructions", None)

            sanitized.append(message_copy)
        return sanitized


__all__ = [
    "InstructionProposalGenerator",
    "DEFAULT_AGENT_INSTRUCTIONS",
    "JOURNAL_TOOLS_INSTRUCTIONS",
    "InstructionProposalOutput",
    "ComponentUpdate",
    "TrajectoryAnalysis",
]
