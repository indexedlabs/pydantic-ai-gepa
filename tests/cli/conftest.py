"""Legacy CLI flow fixtures use in-process fake evaluators.

These tests exercise controller state transitions, often monkeypatching Python
callables that cannot cross an exec boundary. OS isolation and the actual child
protocol are covered separately in test_scoring_sandbox.py. This injection is
Python-only; no production CLI/environment setting can select this backend.
"""

import pytest


@pytest.fixture(autouse=True)
def legacy_scoring_flow_backend(request, monkeypatch):
    if request.node.path.name not in {"test_scoring_sandbox.py", "test_safe_git.py"}:
        monkeypatch.setattr(
            "pydantic_ai_gepa.cli.scoring_sandbox.required", lambda: False
        )
        # Preserve legacy controller transition tests for the deferred lane
        # mutation design. Production has no switch for either injected backend;
        # test_safe_git exercises the real refusal with a hostile repository.
        monkeypatch.setattr(
            "pydantic_ai_gepa.cli.safe_git.refuse_heldout_git_mutations", lambda: None
        )
