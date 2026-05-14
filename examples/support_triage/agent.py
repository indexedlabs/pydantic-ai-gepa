"""Support-triage agent — routes a customer message to the right tool.

The demo lives at the intersection of two things `pydantic-ai-gepa` is good at:
the agent's instructions are vague, and each tool's description is intentionally
underspecified, so the optimizer has somewhere to go. Run `gepa propose` /
`gepa eval` against this agent to watch tool descriptions improve.

Override the model via the `GEPA_EXAMPLE_MODEL` env var. The default is
`openai:gpt-4o-mini` because it picks up reasonable tool-routing signal without
costing much. For deterministic CI runs we patch the model to a `FunctionModel`
inside the test suite.

The optimizable surface (what `gepa components list` will surface):

- `instructions`                                       — the agent's system prompt
- `tool:open_ticket:description`                       — vague: "handles tickets"
- `tool:open_ticket:param:summary`                     — short summary string
- `tool:lookup_order:description`                      — vague: "order stuff"
- `tool:lookup_order:param:order_id`                   — order id format
- `tool:escalate_to_human:description`                 — vague: "escalation"
- `tool:escalate_to_human:param:reason`
- `tool:send_reset_link:description`                   — vague: "auth"
- `tool:send_reset_link:param:email`
- `tool:check_shipment_status:description`             — vague: "shipping"
- `tool:check_shipment_status:param:tracking_number`
"""

from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model


DEFAULT_MODEL: KnownModelName = "openai:gpt-4o-mini"


def _resolve_model() -> Model | KnownModelName | str:
    return os.environ.get("GEPA_EXAMPLE_MODEL", DEFAULT_MODEL)


agent: Agent[None, str] = Agent(
    _resolve_model(),
    instructions=(
        "You handle customer service messages. Pick a tool and call it. Then "
        "return exactly what the tool returned, no extra commentary."
    ),
    name="support-triage",
)


@agent.tool_plain
def open_ticket(summary: str) -> str:
    """Handles tickets.

    Args:
        summary: A summary string.
    """
    return f"open_ticket: opened a ticket — {summary}"


@agent.tool_plain
def lookup_order(order_id: str) -> str:
    """Order stuff.

    Args:
        order_id: The order id.
    """
    return f"lookup_order: looked up order {order_id}"


@agent.tool_plain
def escalate_to_human(reason: str) -> str:
    """Escalation.

    Args:
        reason: Reason for escalation.
    """
    return f"escalate_to_human: escalated — {reason}"


@agent.tool_plain
def send_reset_link(email: str) -> str:
    """Auth.

    Args:
        email: User email.
    """
    return f"send_reset_link: sent reset link to {email}"


@agent.tool_plain
def check_shipment_status(tracking_number: str) -> str:
    """Shipping.

    Args:
        tracking_number: A tracking number.
    """
    return f"check_shipment_status: shipment {tracking_number} is in transit"
