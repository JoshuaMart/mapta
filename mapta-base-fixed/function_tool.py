"""Minimal `@function_tool` decorator used to expose async Python functions as
OpenAI Responses API tools.

A decorated function stays directly awaitable (``await my_tool(...)``) and gains
the metadata the Responses API needs:

* ``name``                 tool name (defaults to the function name)
* ``description``          first paragraph of the docstring
* ``params_json_schema``   JSON schema derived from the signature + docstring
* ``strict_json_schema``   whether the schema is strict-mode compatible

Usage::

    @function_tool
    async def my_tool(path: str, timeout: int = 120):
        '''Do something.

        Args:
            path: Where to do it.
            timeout: Seconds before giving up.
        '''

    @function_tool(name_override="short_name")
    async def a_much_longer_function_name(...):
        ...

Docstrings follow the Google style: a free-form description, then an ``Args:``
(or ``Arguments:``/``Parameters:``) block of ``name: description`` entries.
"""

from __future__ import annotations

import functools
import inspect
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = ["function_tool", "FunctionTool"]

# Sections that terminate the free-form description / the Args block.
_SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Returns|Return|Raises|Yields|Examples?|Notes?)\s*:\s*$",
    re.IGNORECASE,
)
_ARGS_SECTION_RE = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_ARG_LINE_RE = re.compile(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$")

_PRIMITIVES: Dict[Any, str] = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
}


def _parse_docstring(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Split a docstring into (description, {arg_name: arg_description})."""
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    description: List[str] = []
    arg_docs: Dict[str, str] = {}

    in_args = False
    seen_section = False
    current_arg: Optional[str] = None

    for line in lines:
        if _SECTION_RE.match(line):
            in_args = bool(_ARGS_SECTION_RE.match(line))
            seen_section = True
            current_arg = None
            continue

        if not in_args:
            # Everything after the first section header (Returns:, Raises:, ...)
            # belongs to that section, not to the tool description.
            if not seen_section:
                description.append(line)
            continue

        match = _ARG_LINE_RE.match(line)
        if match and not line.startswith((" " * 8, "\t\t")):
            current_arg = match.group(1).lstrip("*")
            arg_docs[current_arg] = match.group(3).strip()
        elif current_arg and line.strip():
            # Continuation of the previous argument's description.
            arg_docs[current_arg] = f"{arg_docs[current_arg]} {line.strip()}".strip()

    return "\n".join(description).strip(), arg_docs


def _schema_for_annotation(annotation: Any) -> Tuple[Dict[str, Any], bool]:
    """Return (schema, is_optional) for a type annotation.

    ``is_optional`` is True when the annotation explicitly allows ``None``.
    Unknown types fall back to a permissive schema (``{}``), which also disables
    strict mode for the whole tool.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}, False

    if annotation is type(None):
        return {"type": "null"}, True

    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}, False

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is typing.Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        is_optional = type(None) in args
        members = [a for a in args if a is not type(None)]
        sub_schemas = []
        for member in members:
            sub_schema, _ = _schema_for_annotation(member)
            if not sub_schema:  # unknown member type -> stay permissive
                return {}, is_optional
            sub_schemas.append(sub_schema)
        if len(sub_schemas) == 1:
            return sub_schemas[0], is_optional
        return {"anyOf": sub_schemas}, is_optional

    if origin in (list, set, tuple, frozenset):
        item_schema: Dict[str, Any] = {}
        if args and args[0] is not Ellipsis:
            item_schema, _ = _schema_for_annotation(args[0])
        return {"type": "array", "items": item_schema or {}}, False

    if origin is dict:
        return {"type": "object", "additionalProperties": True}, False

    if annotation in (list, set, tuple):
        return {"type": "array", "items": {}}, False

    if annotation is dict:
        return {"type": "object", "additionalProperties": True}, False

    # Unsupported annotation: permissive schema, and strict mode is off.
    return {}, False


def _build_schema(
    func: Callable[..., Any], arg_docs: Dict[str, str]
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """Build the JSON schema for ``func``'s parameters.

    Returns (schema, strict, names_with_defaults).
    """
    signature = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # pragma: no cover - unresolvable forward refs
        hints = getattr(func, "__annotations__", {}) or {}

    properties: Dict[str, Any] = {}
    required: List[str] = []
    defaulted: List[str] = []
    strict = True

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # *args / **kwargs cannot be described faithfully in strict mode.
            strict = False
            continue

        annotation = hints.get(name, param.annotation)
        schema, annotation_is_optional = _schema_for_annotation(annotation)
        if not schema:
            strict = False

        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            defaulted.append(name)

        description = arg_docs.get(name)
        if description:
            schema = {**schema, "description": description}

        # Strict mode requires every property in `required`, so an argument the
        # model wants to omit is expressed as an explicit null instead.
        if has_default or annotation_is_optional:
            if schema.get("type"):
                declared = schema["type"]
                types = declared if isinstance(declared, list) else [declared]
                if "null" not in types:
                    schema = {**schema, "type": [*types, "null"]}
            elif "anyOf" in schema and {"type": "null"} not in schema["anyOf"]:
                schema = {**schema, "anyOf": [*schema["anyOf"], {"type": "null"}]}

        properties[name] = schema
        required.append(name)

    json_schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return json_schema, strict, defaulted


class FunctionTool:
    """Awaitable wrapper carrying Responses-API tool metadata."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        strict_mode: bool = True,
    ) -> None:
        description, arg_docs = _parse_docstring(func.__doc__)
        schema, schema_is_strict, defaulted = _build_schema(func, arg_docs)

        # Copy __name__/__doc__/__wrapped__ first: it also copies func.__dict__,
        # so our own attributes have to be assigned afterwards.
        functools.update_wrapper(self, func)

        self.func = func
        self.name = name_override or func.__name__
        self.description = description_override or description or f"Call {self.name}."
        self.params_json_schema = schema
        self.strict_json_schema = bool(strict_mode and schema_is_strict)
        self.is_async = inspect.iscoroutinefunction(func)
        self._defaulted = set(defaulted)

    def _clean_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop explicit nulls for arguments that have a Python default."""
        return {
            key: value
            for key, value in kwargs.items()
            if not (value is None and key in self._defaulted)
        }

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = self._clean_kwargs(kwargs)
        if self.is_async:
            return await self.func(*args, **kwargs)
        return self.func(*args, **kwargs)

    def as_tool_definition(self) -> Dict[str, Any]:
        """Return the tool definition dict for the Responses API."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.params_json_schema,
            "strict": self.strict_json_schema,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FunctionTool {self.name}>"


def function_tool(
    func: Optional[Callable[..., Any]] = None,
    *,
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
    strict_mode: bool = True,
) -> Any:
    """Expose an (async) function as a Responses API tool.

    Works bare (``@function_tool``) or with arguments
    (``@function_tool(name_override="foo")``).
    """

    def decorator(target: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            target,
            name_override=name_override,
            description_override=description_override,
            strict_mode=strict_mode,
        )

    if func is not None:
        return decorator(func)
    return decorator
