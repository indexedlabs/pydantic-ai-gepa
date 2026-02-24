from typing import ClassVar
from pydantic import BaseModel
from pydantic_ai_gepa import SignatureAgent
from pydantic_ai import Agent

class MyInput(BaseModel):
    name: str

    base_encoder_script: ClassVar[str] = """
def encode(data):
    return f"My Markdown Input:\\n- Name: {data['name']}"

encode(data)
"""

def test_custom_encoder_via_classvar():
    agent = Agent('test', output_type=str)
    sig_agent = SignatureAgent(agent, input_type=MyInput)
    
    comps = sig_agent.input_spec.get_gepa_components()
    assert comps["signature:MyInput:encoder"].strip() == MyInput.base_encoder_script.strip()

def test_custom_encoder_via_init():
    custom_script = "def x(data): return 'X'\\nx(data)"
    agent = Agent('test', output_type=str)
    sig_agent = SignatureAgent(agent, input_type=MyInput, base_encoder_script=custom_script)
    
    comps = sig_agent.input_spec.get_gepa_components()
    assert comps["signature:MyInput:encoder"].strip() == custom_script.strip()
