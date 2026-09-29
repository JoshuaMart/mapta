"""The advertised schema is derived from the signature and the docstring."""

from mapta.application.tools import ToolContext, tool
from mapta.application.tools.schema import parse_docstring


@tool
async def sample(ctx: ToolContext, path: str, timeout: int = 120, note: str | None = None) -> str:
    """Do something useful.

    Args:
        path: Where to do it.
        timeout: Seconds before giving up.
        note: An optional note.

    Returns:
        Something.
    """
    return path


def test_description_stops_at_the_first_section():
    description, args = parse_docstring(sample.func.__doc__)
    assert description == "Do something useful."
    assert args["path"] == "Where to do it."


def test_ctx_is_never_advertised():
    assert "ctx" not in sample.parameters["properties"]


def test_every_property_is_required_for_strict_mode():
    properties = sample.parameters["properties"]
    assert sample.parameters["required"] == list(properties)
    assert sample.strict is True


def test_defaulted_arguments_accept_null():
    properties = sample.parameters["properties"]
    assert properties["timeout"]["type"] == ["integer", "null"]
    assert properties["note"]["type"] == ["string", "null"]
    assert properties["path"]["type"] == "string"


def test_docstring_arg_descriptions_reach_the_schema():
    assert sample.parameters["properties"]["timeout"]["description"] == "Seconds before giving up."


def test_spec_is_provider_neutral():
    spec = sample.spec()
    assert spec.name == "sample"
    assert spec.parameters is sample.parameters
    # No wire-format nesting leaks into the application layer.
    assert set(spec.as_dict()) == {"name", "description", "parameters", "strict"}
