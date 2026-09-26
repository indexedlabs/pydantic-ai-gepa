"""The test runner cannot write the developer's real GEPA workspace."""

import pytest

from tests.conftest import _REAL_GEPA


def test_real_home_workspace_writes_are_refused() -> None:
    with pytest.raises(AssertionError, match="real home GEPA workspace"):
        (_REAL_GEPA / "gepa.toml").write_text("must never be written")
