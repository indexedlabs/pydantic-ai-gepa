import json
from pathlib import Path
from pydantic_ai import FunctionToolset
import logfire


def create_journal_toolset(journal_file: str) -> FunctionToolset[None]:
    toolset = FunctionToolset()
    journal_path = Path(journal_file)

    @toolset.tool
    def read_journal_entries() -> str:
        """Read past insights and strategies recorded during previous optimization runs."""
        if not journal_path.exists():
            return "No previous journal entries found."
        try:
            return journal_path.read_text()
        except Exception as e:
            return f"Failed to read journal: {e}"

    @toolset.tool
    def append_journal_entry(insight: str, strategy: str) -> str:
        """Record a newly discovered insight or strategy to the durable journal so it can be used in future optimization runs.

        Args:
            insight: A description of why a particular approach failed or what was learned from the execution traces.
            strategy: A concrete rule or strategy to avoid this failure in the future.
        """
        try:
            entry = json.dumps({"insight": insight, "strategy": strategy}) + "\n"
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with journal_path.open("a") as f:
                f.write(entry)
            logfire.info("Appended journal entry", insight=insight, strategy=strategy)
            return "Successfully recorded journal entry."
        except Exception as e:
            return f"Failed to write journal entry: {e}"

    return toolset
