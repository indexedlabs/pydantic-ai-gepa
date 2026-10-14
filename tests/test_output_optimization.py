from __future__ import annotations

import copy

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
    OutputToolComponentCatalog,
    _iter_schema_descriptions,
    _set_schema_description,
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
    assert seed["output:final_result:option:title@0"].text == "title follows the rules"
    assert seed["output:final_result:option:title@1"].text == "title breaks a rule"


@pytest.mark.asyncio
async def test_candidate_rewrites_described_options():
    agent = _make_decision_agent()
    optimizer = get_or_create_output_tool_optimizer(agent)
    tool_defs = list(
        getattr(getattr(agent, "_output_toolset", None), "_tool_defs", []) or []
    )
    yes_key = "output:final_result:option:title@0"
    no_key = "output:final_result:option:title@1"

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


class RepeatedConstOutput(BaseModel):
    """Options are identified by position: consts may repeat or differ only by type."""

    answer: (
        Annotated[Literal["yes"], Field(description="criterion A")]
        | Annotated[Literal["yes"], Field(description="criterion B")]
        | Annotated[Literal[True], Field(description="boolean yes")]
        | Annotated[Literal["N/A"], Field(description="not applicable")]
    ) = Field(description="Answer")


@pytest.mark.asyncio
async def test_options_are_addressed_by_position():
    agent = Agent(
        TestModel(custom_output_args={"answer": "yes"}),
        output_type=RepeatedConstOutput,
        name="repeated-test",
    )
    get_or_create_output_tool_optimizer(agent)
    seed = extract_seed_candidate(agent, optimize_output_type=True)
    keys = [f"output:final_result:option:answer@{i}" for i in range(4)]
    assert [seed[k].text for k in keys] == [
        "criterion A",
        "criterion B",
        "boolean yes",
        "not applicable",
    ]

    schema = await _prepare(agent, {keys[1]: "new B", keys[2]: "new boolean"})
    options = schema["properties"]["answer"]["anyOf"]
    assert [o["description"] for o in options] == [
        "criterion A",
        "new B",
        "new boolean",
        "not applicable",
    ]


def test_nested_and_array_options_are_components():
    # Inlined schemas (the walker does not follow $ref, as for any description).
    option = {
        "anyOf": [
            {"const": True, "description": "inner yes"},
            {"const": False, "description": "inner no"},
        ]
    }
    inner = {"type": "object", "properties": {"ok": option}}
    schema = {
        "type": "object",
        "properties": {"item": inner, "items": {"type": "array", "items": inner}},
    }
    paths = dict(_iter_schema_descriptions(schema))
    keys = {_output_parameter_key("final_result", p): p for p in paths}
    assert set(keys) == {
        "output:final_result:option:item.ok@0",
        "output:final_result:option:item.ok@1",
        "output:final_result:option:items[].ok@0",
        "output:final_result:option:items[].ok@1",
    }
    updated = copy.deepcopy(schema)
    assert _set_schema_description(
        updated, keys["output:final_result:option:items[].ok@1"], "array no"
    )
    array_options = updated["properties"]["items"]["items"]["properties"]["ok"]
    assert [o["description"] for o in array_options["anyOf"]] == [
        "inner yes",
        "array no",
    ]
    assert schema["properties"]["item"]["properties"]["ok"] == option


class AliasedOutput(BaseModel):
    """Aliases containing '=' or '@' keep the keys they had before options existed."""

    title: (
        Annotated[Literal[True], Field(description="yes criterion")]
        | Annotated[Literal[False], Field(description="no criterion")]
    ) = Field(description="Does the title follow the rules?")
    other: str = Field(alias="a=b", description="an unrelated field")
    decision: str = Field(alias="=decision", description="original")


@pytest.mark.asyncio
async def test_property_keys_are_unchanged_by_options():
    agent = Agent(
        TestModel(custom_output_args={"title": True, "a=b": "x", "=decision": "y"}),
        output_type=AliasedOutput,
        name="alias-test",
    )
    get_or_create_output_tool_optimizer(agent)
    seed = extract_seed_candidate(agent, optimize_output_type=True)
    assert seed["output:final_result:option:title@0"].text == "yes criterion"
    assert seed["output:final_result:param:a=b"].text == "an unrelated field"
    assert seed["output:final_result:param:=decision"].text == "original"

    schema = await _prepare(
        agent,
        {
            "output:final_result:option:title@0": "new yes criterion",
            "output:final_result:param:=decision": "rewritten",
        },
    )
    props = schema["properties"]
    yes = next(o for o in props["title"]["anyOf"] if o["const"] is True)
    assert yes["description"] == "new yes criterion"
    assert props["a=b"]["description"] == "an unrelated field"
    assert props["=decision"]["description"] == "rewritten"


@pytest.mark.parametrize("option_first", [True, False])
def test_property_named_like_an_option_never_shares_its_key(option_first):
    title = {
        "anyOf": [{"const": True, "description": "yes criterion"}],
        "description": "Title ok?",
    }
    real = {"type": "string", "description": "a real property"}
    properties = (
        {"title": title, "title@0": real}
        if option_first
        else {"title@0": real, "title": title}
    )
    tool_def = ToolDefinition(
        name="final_result",
        parameters_json_schema={"type": "object", "properties": properties},
    )
    catalog = OutputToolComponentCatalog()
    catalog.ingest([tool_def])
    seeds = catalog.seed_snapshot()
    assert seeds["output:final_result:param:title@0"] == "a real property"
    assert seeds["output:final_result:option:title@0"] == "yes criterion"

    metadata = catalog.metadata_for("final_result")
    assert metadata is not None
    schema = copy.deepcopy(tool_def.parameters_json_schema)
    for key, path in metadata.parameter_paths.items():
        _set_schema_description(schema, path, seeds[key])
    assert schema == tool_def.parameters_json_schema


def test_properties_formatting_alike_keep_the_later_path():
    # Unchanged from before options existed: no error, the later path wins.
    catalog = OutputToolComponentCatalog()
    catalog.ingest(
        [
            ToolDefinition(
                name="final_result",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "a.b": {"type": "string", "description": "flat"},
                        "a": {
                            "type": "object",
                            "properties": {
                                "b": {"type": "string", "description": "nested"}
                            },
                        },
                    },
                },
            )
        ]
    )
    metadata = catalog.metadata_for("final_result")
    assert metadata is not None
    assert metadata.parameter_paths["output:final_result:param:a.b"] == ("a", "b")


def test_options_in_any_of_and_one_of_get_distinct_positions():
    schema = {
        "type": "object",
        "properties": {
            "x": {
                "anyOf": [{"const": True, "description": "any yes"}],
                "oneOf": [{"const": True, "description": "one yes"}],
            }
        },
    }
    keys = {
        _output_parameter_key("final_result", p): (p, d)
        for p, d in _iter_schema_descriptions(schema)
    }
    assert {k: d for k, (_, d) in keys.items()} == {
        "output:final_result:option:x@0": "any yes",
        "output:final_result:option:x@1": "one yes",
    }
    updated = copy.deepcopy(schema)
    assert _set_schema_description(
        updated, keys["output:final_result:option:x@1"][0], "NEW"
    )
    assert updated["properties"]["x"]["anyOf"][0]["description"] == "any yes"
    assert updated["properties"]["x"]["oneOf"][0]["description"] == "NEW"


def test_option_slots_round_trip_through_the_store(tmp_path):
    from pydantic_ai_gepa.cli.store import ComponentStore

    store = ComponentStore(tmp_path)
    slot = "output:final_result:option:status@1"
    store.stage(slot, "not applicable")
    store.confirm_staged(slot)
    assert slot in store.list_confirmed_slots()
