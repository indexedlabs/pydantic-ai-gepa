from __future__ import annotations

from inline_snapshot import snapshot
from pydantic import BaseModel, Field
from pydantic_ai_gepa.gepa_graph.models import CandidateMap, ComponentValue
from pydantic_ai_gepa.input_type import (
    apply_candidate_to_input_model,
    generate_system_instructions,
    generate_user_content,
    get_gepa_components,
)


class Address(BaseModel):
    """Address information."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    zip_code: str = Field(description="ZIP or postal code")


class CustomerQuery(BaseModel):
    """Process customer inquiries with full context."""

    customer_name: str = Field(description="Full name of the customer")
    query: str = Field(description="The customer's question or issue")
    billing_address: Address = Field(description="Customer's billing address")
    shipping_address: Address | None = Field(
        default=None, description="Optional shipping address"
    )


class SimpleQuery(BaseModel):
    """Process simple queries."""

    question: str = Field(description="The question to answer")


def _candidate_map(entries: dict[str, str]) -> CandidateMap:
    return {
        name: ComponentValue(name=name, text=value) for name, value in entries.items()
    }


def test_signature_component_extraction_with_nested_models():
    """Test that nested models don't cause key collisions."""
    # Extract components from CustomerQuery
    customer_components = get_gepa_components(CustomerQuery)

    # Should have class-specific keys
    assert "signature:CustomerQuery:instructions" in customer_components
    assert "signature:CustomerQuery:customer_name:desc" in customer_components
    assert "signature:CustomerQuery:billing_address:desc" in customer_components

    # Extract components from SimpleQuery
    simple_components = get_gepa_components(SimpleQuery)

    # Should also have class-specific keys
    assert "signature:SimpleQuery:instructions" in simple_components
    assert "signature:SimpleQuery:question:desc" in simple_components

    # Verify no key collisions - each signature has unique keys
    assert len(set(customer_components.keys()) & set(simple_components.keys())) == 0


def test_apply_candidate_with_class_specific_keys():
    """Test that candidates are applied correctly with class-specific keys."""
    # Create a candidate with optimized text
    candidate = _candidate_map(
        {
            "signature:CustomerQuery:instructions": "OPTIMIZED: Handle customer issues professionally",
            "signature:CustomerQuery:customer_name:desc": "OPTIMIZED: Customer full legal name",
            "signature:SimpleQuery:instructions": "OPTIMIZED: Answer concisely",
            "signature:SimpleQuery:question:desc": "OPTIMIZED: The user question",
        }
    )

    # Apply to CustomerQuery
    original_customer_doc = CustomerQuery.__doc__
    original_customer_desc = CustomerQuery.model_fields["customer_name"].description
    original_simple_doc = SimpleQuery.__doc__
    original_simple_desc = SimpleQuery.model_fields["question"].description

    with apply_candidate_to_input_model(CustomerQuery, candidate):
        assert (
            CustomerQuery.__doc__ == "OPTIMIZED: Handle customer issues professionally"
        )
        assert (
            CustomerQuery.model_fields["customer_name"].description
            == "OPTIMIZED: Customer full legal name"
        )

    with apply_candidate_to_input_model(SimpleQuery, candidate):
        assert SimpleQuery.__doc__ == "OPTIMIZED: Answer concisely"
        assert (
            SimpleQuery.model_fields["question"].description
            == "OPTIMIZED: The user question"
        )

    assert CustomerQuery.__doc__ == original_customer_doc
    assert (
        CustomerQuery.model_fields["customer_name"].description
        == original_customer_desc
    )
    assert SimpleQuery.__doc__ == original_simple_doc
    assert SimpleQuery.model_fields["question"].description == original_simple_desc


def test_to_user_content_with_nested_models():
    """Test that nested models are formatted correctly in user content."""
    # Create an instance with nested models
    query = CustomerQuery(
        customer_name="John Doe",
        query="Where is my order?",
        billing_address=Address(
            street="123 Main St", city="Springfield", zip_code="12345"
        ),
    )

    # Convert to user content
    content = generate_user_content(query)
    assert content == snapshot(
        [
            """\
<customer_name>John Doe</customer_name>

<query>Where is my order?</query>

<billing_address>
  <street>123 Main St</street>
  <city>Springfield</city>
  <zip_code>12345</zip_code>
</billing_address>\
"""
        ]
    )

    # Test with optimized candidate
    candidate = {
        "signature:CustomerQuery:instructions": "Help the customer quickly",
        "signature:CustomerQuery:customer_name:desc": "OPTIMIZED: Customer full legal name",
    }

    system_instructions = generate_system_instructions(query, candidate=candidate)
    assert system_instructions == snapshot("""\

Help the customer quickly

Inputs:

- `<customer_name>` (str): OPTIMIZED: Customer full legal name
- `<query>` (str): The customer's question or issue
- `<billing_address>` (Address): Customer's billing address
- `<shipping_address>` (Union[Address, NoneType]): Optional shipping address

Schemas:

Address
  - `<street>` (str): Street address
  - `<city>` (str): City name
  - `<zip_code>` (str): ZIP or postal code\
""")

    content_optimized = generate_user_content(query)
    assert content_optimized == snapshot(
        [
            """\
<customer_name>John Doe</customer_name>

<query>Where is my order?</query>

<billing_address>
  <street>123 Main St</street>
  <city>Springfield</city>
  <zip_code>12345</zip_code>
</billing_address>\
"""
        ]
    )
