from __future__ import annotations

from typing import Annotated, Literal


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

    prepared = await optimizer.prepare_output_tools(
        None,  # type: ignore[arg-type]
        tool_defs,
        candidate=candidate,
    )

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


class DecisionOutput(BaseModel):
    """Output whose yes/no field carries described options (criteria)."""

    title: (
        Annotated[Literal[True], Field(description="title follows the rules")]
        | Annotated[Literal[False], Field(description="title breaks a rule")]
    ) = Field(description="Does the title follow the rules?")


def _make_decision_agent() -> Agent[object, DecisionOutput]:
    return Agent(
        TestModel(custom_output_args=DecisionOutput(title=True)),
        output_type=DecisionOutput,
        name="decision-test",
    )


def test_described_options_are_output_components():
    agent = _make_decision_agent()
    get_or_create_output_tool_optimizer(agent)

    seed = extract_seed_candidate(agent, optimize_output_type=True)

    assert (
        seed["output:final_result:param:title"].text
        == "Does the title follow the rules?"
    )
    assert (
        seed["output:final_result:param:title=true"].text == "title follows the rules"
    )
    assert seed["output:final_result:param:title=false"].text == "title breaks a rule"


@pytest.mark.asyncio
async def test_candidate_rewrites_described_options():
    agent = _make_decision_agent()
    optimizer = get_or_create_output_tool_optimizer(agent)
    tool_defs = list(
        getattr(getattr(agent, "_output_toolset", None), "_tool_defs", []) or []
    )
    yes_key = "output:final_result:param:title=true"
    no_key = "output:final_result:param:title=false"

    prepared = await optimizer.prepare_output_tools(
        None,  # type: ignore[arg-type]
        tool_defs,
        candidate={
            yes_key: ComponentValue(name=yes_key, text="new yes criterion"),
            no_key: ComponentValue(name=no_key, text="new no criterion"),
        },
    )

    assert prepared is not None
    options = prepared[0].parameters_json_schema["properties"]["title"]["anyOf"]
    assert {o["const"]: o["description"] for o in options} == {
        True: "new yes criterion",
        False: "new no criterion",
    }


async def _prepare(agent, candidate_text: dict[str, str]):
    optimizer = get_or_create_output_tool_optimizer(agent)
    tool_defs = list(
        getattr(getattr(agent, "_output_toolset", None), "_tool_defs", []) or []
    )
    prepared = await optimizer.prepare_output_tools(
        None,  # type: ignore[arg-type]
        tool_defs,
        candidate={
            k: ComponentValue(name=k, text=v) for k, v in candidate_text.items()
        },
    )
    assert prepared is not None
    return prepared[0].parameters_json_schema


class MixedTypeOutput(BaseModel):
    """Options whose consts differ only by JSON type."""

    choice: (
        Annotated[Literal[True], Field(description="boolean yes")]
        | Annotated[Literal["true"], Field(description="string yes")]
    ) = Field(description="Pick one")


@pytest.mark.asyncio
async def test_options_keep_their_json_type():
    agent = Agent(
        TestModel(custom_output_args=MixedTypeOutput(choice=True)),
        output_type=MixedTypeOutput,
        name="mixed-test",
    )
    get_or_create_output_tool_optimizer(agent)
    seed = extract_seed_candidate(agent, optimize_output_type=True)
    assert seed["output:final_result:param:choice=true"].text == "boolean yes"
    assert seed['output:final_result:param:choice="true"'].text == "string yes"

    schema = await _prepare(
        agent,
        {
            "output:final_result:param:choice=true": "new boolean yes",
            'output:final_result:param:choice="true"': "new string yes",
        },
    )
    options = schema["properties"]["choice"]["anyOf"]
    assert {repr(o["const"]): o["description"] for o in options} == {
        "True": "new boolean yes",
        "'true'": "new string yes",
    }


class AliasedOutput(BaseModel):
    """A boolean field with criteria, and aliases that contain '='."""

    title: (
        Annotated[Literal[True], Field(description="yes criterion")]
        | Annotated[Literal[False], Field(description="no criterion")]
    ) = Field(description="Does the title follow the rules?")
    other: str = Field(alias="title=true", description="an unrelated field")
    decision: str = Field(alias="=decision", description="original")


@pytest.mark.asyncio
async def test_aliases_with_equals_are_properties_not_options():
    agent = Agent(
        TestModel(
            custom_output_args={"title": True, "title=true": "x", "=decision": "y"}
        ),
        output_type=AliasedOutput,
        name="alias-test",
    )
    get_or_create_output_tool_optimizer(agent)
    seed = extract_seed_candidate(agent, optimize_output_type=True)
    assert seed["output:final_result:param:title=true"].text == "yes criterion"
    assert seed["output:final_result:param:title\\=true"].text == "an unrelated field"
    assert seed["output:final_result:param:\\=decision"].text == "original"

    schema = await _prepare(
        agent,
        {
            "output:final_result:param:title=true": "new yes criterion",
            "output:final_result:param:\\=decision": "rewritten",
        },
    )
    props = schema["properties"]
    yes = next(o for o in props["title"]["anyOf"] if o["const"] is True)
    assert yes["description"] == "new yes criterion"
    assert props["title=true"]["description"] == "an unrelated field"
    assert props["=decision"]["description"] == "rewritten"
