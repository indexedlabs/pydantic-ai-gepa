from typing import ClassVar

import pydantic_ai_gepa.input_type as input_type_module
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_gepa import SignatureAgent
from pydantic_ai_gepa.input_type import generate_user_content


class MyInput(BaseModel):
    name: str

    base_encoder_script: ClassVar[str] = """
def encode(data):
    return f"My Markdown Input:\\n- Name: {data['name']}"

encode(data)
"""


def test_custom_encoder_via_classvar():
    agent = Agent("test", output_type=str)
    sig_agent = SignatureAgent(agent, input_type=MyInput)

    comps = sig_agent.input_spec.get_gepa_components()
    assert (
        comps["signature:MyInput:encoder"].strip()
        == MyInput.base_encoder_script.strip()
    )


def test_custom_encoder_via_init():
    custom_script = "def x(data): return 'X'\\nx(data)"
    agent = Agent("test", output_type=str)
    sig_agent = SignatureAgent(
        agent, input_type=MyInput, base_encoder_script=custom_script
    )

    comps = sig_agent.input_spec.get_gepa_components()
    assert comps["signature:MyInput:encoder"].strip() == custom_script.strip()


def test_custom_encoder_executes_in_monty_session():
    candidate = {
        "signature:MyInput:encoder": MyInput.base_encoder_script,
    }

    content = generate_user_content(MyInput(name="Ada"), candidate=candidate)

    assert content == ["My Markdown Input:\n- Name: Ada"]


def test_runaway_custom_encoder_falls_back_to_xml(monkeypatch):
    monkeypatch.setattr(
        input_type_module,
        "_MONTY_ENCODER_LIMITS",
        {
            "max_duration_secs": 0.01,
            "max_memory": 128 * 1024 * 1024,
            "max_recursion_depth": 1000,
        },
    )
    candidate = {
        "signature:MyInput:encoder": "while True:\n    pass",
    }

    content = generate_user_content(MyInput(name="Ada"), candidate=candidate)

    assert content == ["<name>Ada</name>"]
