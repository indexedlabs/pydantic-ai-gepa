from __future__ import annotations


import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai import ToolDefinition
from pydantic_ai.models.test import TestModel
from inline_snapshot import snapshot

from pydantic_ai_gepa.components import extract_seed_candidate, get_component_names
from pydantic_ai_gepa.gepa_graph.models import ComponentValue
from pydantic_ai_gepa.tool_components import (
    get_or_create_output_tool_optimizer,
    _output_parameter_key,
    _output_description_key,
)


class OutputModel(BaseModel):
    """Test output model."""

    result: str = Field(description="original result description")


def _make_agent() -> Agent[object, OutputModel]:
    return Agent(
        TestModel(custom_output_args=OutputModel(result="ok")),
        output_type=OutputModel,
        instructions="Be helpful",
        name="output-test",
    )


def test_seed_includes_output_components_when_enabled():
    agent = _make_agent()
    # Install optimizer to hydrate seed components for output tools
    get_or_create_output_tool_optimizer(agent)

    seed = extract_seed_candidate(agent, optimize_output_type=True)

    description_key = _output_description_key("final_result")
    param_key = _output_parameter_key("final_result", ("result",))

    assert description_key in seed
    assert param_key in seed
    assert seed[param_key].text == "original result description"


def test_component_names_include_output_when_enabled():
    agent = _make_agent()
    get_or_create_output_tool_optimizer(agent)

    names = get_component_names(agent, optimize_output_type=True)
    assert names == snapshot(
        [
            "instructions",
            "output:final_result:description",
            "output:final_result:param:result",
        ]
    )


@pytest.mark.asyncio
async def test_candidate_applies_to_output_tool_definitions():
    agent = _make_agent()
    optimizer = get_or_create_output_tool_optimizer(agent)

    # Capture existing output tool definitions
    toolset = getattr(agent, "_output_toolset", None)
    tool_defs = list(getattr(toolset, "_tool_defs", []) or [])

    description_key = _output_description_key(tool_defs[0].name)
    param_key = _output_parameter_key(tool_defs[0].name, ("result",))

    candidate = {
        description_key: ComponentValue(
            name=description_key, text="better description"
        ),
        param_key: ComponentValue(name=param_key, text="improved result guidance"),
    }

    with optimizer.candidate_context(candidate):
        prepared = await optimizer._prepare_wrapper(None, tool_defs)  # type: ignore[arg-type]

    assert prepared is not None
    updated = prepared[0]
    assert updated == snapshot(
        ToolDefinition(
            name="final_result",
            parameters_json_schema={
                "properties": {
                    "result": {
                        "description": "improved result guidance",
                        "type": "string",
                    }
                },
                "required": ["result"],
                "title": "OutputModel",
                "type": "object",
            },
            description="better description",
            kind="output",
        )
    )
