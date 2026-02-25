from pathlib import Path
from pydantic_ai_gepa.gepa_graph.proposal.journal_tools import create_journal_toolset


def test_journal_tools(tmp_path: Path):
    journal_file = tmp_path / "journal.jsonl"
    toolset = create_journal_toolset(str(journal_file))

    # Test reading empty journal
    result = toolset.tools["read_journal_entries"].function()
    assert "No previous journal" in result

    # Test appending an entry
    result = toolset.tools["append_journal_entry"].function(
        insight="The API is rate limited", strategy="Add exponential backoff"
    )
    assert "Successfully" in result

    # Test reading the journal
    result = toolset.tools["read_journal_entries"].function()
    assert "rate limited" in result
    assert "backoff" in result
