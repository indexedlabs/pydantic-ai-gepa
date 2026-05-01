"""Tools for the reflection agent to manage the example bank."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import FunctionToolset

from ..example_bank import BankedExample, InMemoryExampleBank


class ExampleInput(BaseModel):
    """Input schema for adding an example to the bank."""

    title: str = Field(
        description="Short descriptive name (e.g., 'Handle nested JSON')"
    )
    keywords: list[str] = Field(
        description="Semantic keywords for retrieval - what user queries should find this example? (e.g., ['json', 'nested', 'parsing'])"
    )
    content: str = Field(
        description="The full example content. Can be any format: input/output pairs, reasoning chains, do/don't comparisons, etc."
    )


def create_example_bank_tools(bank: InMemoryExampleBank) -> FunctionToolset:
    """Create tools for the reflection agent to manage examples.

    These tools allow the reflection agent to iteratively curate the example
    bank by adding, removing, and testing examples.
    """
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain
    def add_example(title: str, keywords: list[str], content: str) -> str:
        """Add a few-shot example to help the student agent handle similar cases.

        Use this when you identify patterns in the failures that could be addressed
        by showing the student concrete examples of correct behavior. You can add
        multiple examples over time to build a comprehensive reference library.
        """
        ex = ExampleInput(title=title, keywords=keywords, content=content)
        example = BankedExample(
            title=ex.title,
            keywords=ex.keywords,
            content=ex.content,
        )
        bank.add(example)
        return f"Added example '{ex.title}' (id: {example.id})"

    @toolset.tool_plain
    def remove_example(example_id: str) -> str:
        """Remove an example that isn't helping or is causing issues.

        Use this when an example is:
        - Redundant with another example
        - Causing confusion or incorrect behavior
        - No longer relevant after prompt changes
        """
        if bank.remove(example_id):
            return f"Removed example {example_id}"
        return f"Example {example_id} not found"

    @toolset.tool_plain
    def list_examples() -> str:
        """View all current examples in the bank (titles and keywords only).

        Use this to understand what examples already exist before adding
        new ones, or to identify examples that should be removed.
        Use read_example() to see the full content of a specific example.
        """
        if len(bank) == 0:
            return "Example bank is empty."
        lines = ["Current examples in bank:"]
        for ex in bank:
            kw_str = ", ".join(ex.keywords) if ex.keywords else "(no keywords)"
            lines.append(f"- [{ex.id}] {ex.title}")
            lines.append(f"  Keywords: {kw_str}")
        return "\n".join(lines)

    @toolset.tool_plain
    def read_example(example_id: str) -> str:
        """Read the full content of an example.

        Use this after list_examples() to see the complete content of
        examples you want to review, modify, or use as reference.
        """
        example = bank.get(example_id)
        if example is None:
            return f"[{example_id}] Not found"
        lines = [
            f"[{example.id}] {example.title}",
            f"Keywords: {', '.join(example.keywords) if example.keywords else '(none)'}",
            "",
            example.content,
        ]
        return "\n".join(lines)

    @toolset.tool_plain
    def test_retrieval(query: str) -> str:
        """Test what examples would be retrieved for a given query.

        Use this to verify that your keywords are working correctly and
        that the right examples would surface for expected user inputs.
        """
        results = bank.search(query, k=3)
        if not results:
            return f"No examples would be retrieved for query: '{query}'"
        lines = [f"Query '{query}' would retrieve:"]
        for i, ex in enumerate(results, 1):
            lines.append(f"{i}. {ex.title} (id: {ex.id})")
        return "\n".join(lines)

    return toolset


__all__ = ["create_example_bank_tools"]
