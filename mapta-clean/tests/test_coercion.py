"""Arguments are checked against the tool's schema before the tool runs.

The regression this pins down: a model sent `timeout` as an object, which
reached `asyncio.timeout()` and crashed the round with a TypeError.
"""

import pytest

from mapta.application.arguments import coerce_tool_arguments, describe_parameters
from mapta.domain import ToolArgumentError

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout": {"type": ["integer", "null"]},
        "ratio": {"type": "number"},
        "detach": {"type": "boolean"},
        "args": {"type": "array", "items": {}},
        "env": {"type": "object", "additionalProperties": True},
        "anything": {},
        "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    },
    "required": ["command"],
    "additionalProperties": False,
}


def coerce(**arguments):
    return coerce_tool_arguments(SCHEMA, arguments)


def test_an_object_where_an_integer_was_declared_is_refused():
    with pytest.raises(ToolArgumentError, match="`timeout` expected integer or null, got object"):
        coerce(command="ls", timeout={"type": "integer"})


def test_well_typed_arguments_pass_through():
    assert coerce(command="ls", timeout=30) == {"command": "ls", "timeout": 30}


def test_null_is_kept_when_the_schema_allows_it():
    assert coerce(command="ls", timeout=None) == {"command": "ls", "timeout": None}


def test_numeric_strings_are_coerced():
    assert coerce(command="ls", timeout="30")["timeout"] == 30


def test_integral_floats_are_coerced():
    assert coerce(command="ls", timeout=30.0)["timeout"] == 30


def test_a_fractional_float_is_not_an_integer():
    with pytest.raises(ToolArgumentError, match="`timeout`"):
        coerce(command="ls", timeout=30.5)


def test_booleans_are_not_integers():
    with pytest.raises(ToolArgumentError, match="`timeout`"):
        coerce(command="ls", timeout=True)


def test_boolean_strings_are_coerced():
    assert coerce(command="ls", detach="yes")["detach"] is True
    assert coerce(command="ls", detach="0")["detach"] is False


def test_a_structure_is_never_stringified_into_a_command():
    with pytest.raises(ToolArgumentError, match="`command` expected string, got object"):
        coerce(command={"cmd": "ls"})


def test_numbers_are_accepted_where_a_string_is_declared():
    assert coerce(command=42)["command"] == "42"


def test_arrays_and_objects():
    assert coerce(command="ls", args=["-l"], env={"A": "1"}) == {
        "command": "ls",
        "args": ["-l"],
        "env": {"A": "1"},
    }


def test_any_of_accepts_either_branch():
    assert coerce(command="ls", either="x")["either"] == "x"
    assert coerce(command="ls", either=7)["either"] == 7


def test_a_permissive_schema_accepts_anything():
    assert coerce(command="ls", anything={"deep": [1]})["anything"] == {"deep": [1]}


def test_unknown_parameters_are_reported():
    with pytest.raises(ToolArgumentError, match="`nope` is not a parameter"):
        coerce(command="ls", nope=1)


def test_every_problem_is_reported_at_once():
    with pytest.raises(ToolArgumentError) as info:
        coerce(command={"a": 1}, timeout={"b": 2}, nope=3)
    message = str(info.value)
    assert "`command`" in message and "`timeout`" in message and "`nope`" in message


def test_parameters_are_described_for_the_retry_hint():
    described = describe_parameters(SCHEMA)
    assert described["timeout"] == "integer or null"
    assert described["either"] == "string or integer"
    assert described["anything"] == "any"
