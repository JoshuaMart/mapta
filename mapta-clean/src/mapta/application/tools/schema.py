"""Derive a JSON schema for a tool from its signature and docstring.

Docstrings follow the Google style: a free-form description, then an ``Args:``
(or ``Arguments:``/``Parameters:``) block of ``name: description`` entries.
Python 3.14 evaluates annotations lazily (PEP 649), so hints are resolved
through :func:`typing.get_type_hints` rather than read raw.
"""

import inspect
import re
import types
import typing
from collections.abc import Callable
from typing import Any

__all__ = ["ToolSchema", "build_tool_schema", "parse_docstring"]

_SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Returns|Return|Raises|Yields|Examples?|Notes?)\s*:\s*$",
    re.IGNORECASE,
)
_ARGS_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_ARG_LINE_RE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")

_PRIMITIVES: dict[Any, str] = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
}


class ToolSchema(typing.NamedTuple):
    """The three things a provider needs to advertise a tool."""

    description: str
    parameters: dict[str, Any]
    strict: bool
    #: Names with a Python default, expressed as nullable types in the schema.
    optional_params: frozenset[str]
    #: Names the Python function genuinely needs, which the schema's own
    #: ``required`` list cannot express under strict mode.
    required_params: frozenset[str]


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into ``(description, {arg_name: arg_description})``."""
    if not doc:
        return "", {}

    description: list[str] = []
    arg_docs: dict[str, str] = {}
    in_args = False
    seen_section = False
    current_arg: str | None = None

    for line in inspect.cleandoc(doc).splitlines():
        if _SECTION_RE.match(line):
            in_args = bool(_ARGS_SECTION_RE.match(line))
            seen_section = True
            current_arg = None
            continue

        if not in_args:
            # Text after the first section header belongs to that section,
            # not to the tool description.
            if not seen_section:
                description.append(line)
            continue

        match = _ARG_LINE_RE.match(line)
        if match and not line.startswith((" " * 8, "\t\t")):
            current_arg = match.group(1).lstrip("*")
            arg_docs[current_arg] = match.group(3).strip()
        elif current_arg and line.strip():
            arg_docs[current_arg] = f"{arg_docs[current_arg]} {line.strip()}".strip()

    return "\n".join(description).strip(), arg_docs


def _schema_for(annotation: Any) -> tuple[dict[str, Any], bool]:
    """Return ``(schema, allows_none)`` for one annotation.

    An empty schema means "unknown type": permissive, and strict mode is
    switched off for the whole tool.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}, False
    if annotation is types.NoneType:
        return {"type": "null"}, True
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}, False

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (typing.Union, types.UnionType):
        allows_none = types.NoneType in args
        members = [a for a in args if a is not types.NoneType]
        sub_schemas: list[dict[str, Any]] = []
        for member in members:
            sub_schema, _ = _schema_for(member)
            if not sub_schema:  # unknown member type -> stay permissive
                return {}, allows_none
            sub_schemas.append(sub_schema)
        if len(sub_schemas) == 1:
            return sub_schemas[0], allows_none
        return {"anyOf": sub_schemas}, allows_none

    if origin in (list, set, tuple, frozenset):
        item_schema: dict[str, Any] = {}
        if args and args[0] is not Ellipsis:
            item_schema, _ = _schema_for(args[0])
        return {"type": "array", "items": item_schema or {}}, False

    if origin is dict or annotation is dict:
        return {"type": "object", "additionalProperties": True}, False
    if annotation in (list, set, tuple, frozenset):
        return {"type": "array", "items": {}}, False

    return {}, False


def _allow_null(schema: dict[str, Any]) -> dict[str, Any]:
    """Widen a schema so the model may pass ``null`` to skip the argument.

    Strict mode requires every property to be listed in ``required``, so an
    omitted argument is expressed as an explicit null instead.
    """
    if declared := schema.get("type"):
        types_ = declared if isinstance(declared, list) else [declared]
        if "null" not in types_:
            return schema | {"type": [*types_, "null"]}
    elif "anyOf" in schema and {"type": "null"} not in schema["anyOf"]:
        return schema | {"anyOf": [*schema["anyOf"], {"type": "null"}]}
    return schema


def build_tool_schema(
    func: Callable[..., Any],
    *,
    skip: frozenset[str] = frozenset({"self", "cls"}),
    strict_mode: bool = True,
) -> ToolSchema:
    """Build the advertised schema for ``func``."""
    description, arg_docs = parse_docstring(func.__doc__)
    signature = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # pragma: no cover - unresolvable forward refs
        hints = getattr(func, "__annotations__", {}) or {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    optional: set[str] = set()
    strict = True

    for name, param in signature.parameters.items():
        if name in skip:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # *args / **kwargs cannot be described faithfully in strict mode.
            strict = False
            continue

        schema, annotation_allows_none = _schema_for(hints.get(name, param.annotation))
        if not schema:
            strict = False

        has_default = param.default is not param.empty
        if has_default:
            optional.add(name)
        if doc := arg_docs.get(name):
            schema = schema | {"description": doc}
        if has_default or annotation_allows_none:
            schema = _allow_null(schema)

        properties[name] = schema
        required.append(name)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return ToolSchema(
        description=description,
        parameters=parameters,
        strict=bool(strict_mode and strict),
        optional_params=frozenset(optional),
        required_params=frozenset(properties) - frozenset(optional),
    )
