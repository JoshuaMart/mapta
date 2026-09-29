"""Decoding and validation of a tool call's ``arguments`` payload.

Models routinely emit arguments that are neither valid JSON nor the types the
tool declared. ``decode_tool_arguments`` recovers the usual malformations
(markdown fences, Python literals, single quotes, trailing commas, surrounding
prose, double encoding); ``coerce_tool_arguments`` then checks the result
against the tool's schema. Both turn a bad call into something the model can
read and retry from.

A payload cut off mid-write is repaired only by *dropping* what was half
written, never by completing it: a truncated value is a different value, and
for a shell tool a different command.
"""

import ast
import json
import logging
import re
import typing
from collections.abc import Callable, Mapping
from typing import Any

from ..domain import JSONObject, ToolArgumentError

__all__ = [
    "coerce_tool_arguments",
    "decode_tool_arguments",
    "describe_parameters",
    "looks_truncated",
]

logger = logging.getLogger(__name__)

_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```$")

_ATTEMPTS: tuple[Callable[[str], Any], ...] = (
    json.loads,
    # Allow raw newlines/tabs inside string values (very common).
    lambda text: json.loads(text, strict=False),
    # Take the first complete object, ignoring trailing garbage or a second
    # concatenated object.
    lambda text: json.JSONDecoder(strict=False).raw_decode(text)[0],
    # Python's repr of a dict: single quotes, None/True/False, trailing commas.
    # literal_eval only builds literals, so nothing here is executed.
    ast.literal_eval,
)


def decode_tool_arguments(raw: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the arguments as a dict, or raise ``ValueError``."""
    match raw:
        case None | "":
            return {}
        case Mapping():
            return dict(raw)
        case str():
            return _decode_text(raw)
        case _:
            raise ValueError(f"unsupported arguments type {type(raw).__name__}")


def _decode_text(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):  # some providers wrap the JSON in a markdown fence
        text = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", text)).strip()

    first_error: Exception | None = None
    # The second pass drops any prose the model wrote before the object.
    for candidate in _candidates(text):
        for attempt in _ATTEMPTS:
            try:
                parsed = attempt(candidate)
            except Exception as exc:
                first_error = first_error or exc
                continue
            if isinstance(parsed, dict):
                return parsed
            first_error = first_error or ValueError(
                f"arguments decoded to {type(parsed).__name__}, not an object"
            )

    # Cut off mid-write: close what is open, or drop the incomplete tail.
    for repaired in _truncation_repairs(text):
        try:
            parsed = json.loads(repaired, strict=False)
        except Exception:
            continue
        if isinstance(parsed, dict):
            logger.warning(
                "Recovered a truncated tool-call payload (%d chars); the model was cut "
                "off mid-write and any trailing argument was lost.",
                len(text),
            )
            return parsed

    # Double-encoded JSON: a string whose content is itself JSON.
    try:
        inner = json.loads(text, strict=False)
        if isinstance(inner, str):
            reparsed = json.loads(inner, strict=False)
            if isinstance(reparsed, dict):
                return reparsed
    except Exception:
        pass

    raise ValueError(str(first_error or "could not parse arguments"))


class _Unclosed(typing.NamedTuple):
    """What a payload left open, and where it could be cut back to."""

    closers: str
    in_string: bool
    member_commas: tuple[int, ...]

    @property
    def truncated(self) -> bool:
        return bool(self.closers) or self.in_string


def _scan(text: str) -> _Unclosed:
    """Find the unterminated structures in ``text``, ignoring string contents."""
    stack: list[str] = []
    commas: list[int] = []
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if escape:
            escape = False
        elif in_string and char == "\\":
            escape = True
        elif char == '"':
            in_string = not in_string
        elif in_string:
            continue
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
        elif char == "," and len(stack) == 1:
            # A comma at depth 1 separates two arguments, so it is a safe
            # place to cut an object back to its complete members.
            commas.append(index)

    return _Unclosed("".join(reversed(stack)), in_string, tuple(commas))


def looks_truncated(raw: str | Mapping[str, Any] | None) -> bool:
    """Whether a payload was cut off rather than merely malformed."""
    return isinstance(raw, str) and _scan(raw.strip()).truncated


def _truncation_repairs(text: str) -> tuple[str, ...]:
    """Candidate repairs, most faithful first."""
    scan = _scan(text)
    if not scan.truncated:
        return ()

    # Only ever drop what was half-written; never invent the missing end of a
    # value. Closing an open string would turn `"rm -rf /tmp/x/y` into the
    # command `rm -rf /tmp/x`, which is a different and more destructive one.
    repairs = [text + scan.closers]
    repairs.extend(text[:position] + scan.closers for position in reversed(scan.member_commas))
    return tuple(repairs)


def _candidates(text: str) -> tuple[str, ...]:
    """The payload as sent, then the same starting at its first ``{``."""
    start = text.find("{")
    if start <= 0:
        return (text,)
    return (text, text[start:])


# --- validation against the tool's own schema ----------------------------


class _Mismatch(Exception):
    """One argument does not fit its declared schema."""


def coerce_tool_arguments(
    parameters: JSONObject,
    arguments: Mapping[str, Any],
    *,
    required: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return the arguments coerced to their declared types.

    ``required`` is what the Python function genuinely needs, which is not the
    schema's ``required`` list: strict mode puts every property in there and
    expresses "may be omitted" as a nullable type instead.

    Raises :class:`~mapta.domain.errors.ToolArgumentError` listing every problem
    at once, so the model can fix the whole call in one retry.
    """
    properties: dict[str, Any] = parameters.get("properties", {}) or {}
    coerced: dict[str, Any] = {}
    problems: list[str] = []

    for name in sorted(required - set(arguments)):
        problems.append(f"`{name}` is required but was not provided")

    for key, value in arguments.items():
        schema = properties.get(key)
        if schema is None:
            problems.append(f"`{key}` is not a parameter of this tool")
            continue
        try:
            coerced[key] = _coerce(value, schema)
        except _Mismatch as exc:
            problems.append(f"`{key}` {exc}")

    if problems:
        raise ToolArgumentError("; ".join(problems))
    return coerced


def _allowed_types(schema: JSONObject) -> tuple[str, ...]:
    """The JSON types a schema accepts, flattening ``anyOf``."""
    if declared := schema.get("type"):
        return tuple(declared) if isinstance(declared, list) else (declared,)
    if branches := schema.get("anyOf"):
        return tuple(
            name for branch in branches for name in _allowed_types(branch)
        )
    return ()


def _coerce(value: Any, schema: JSONObject) -> Any:
    """Convert ``value`` to one of the types ``schema`` allows.

    Types the value already has win over types it could be converted to, so an
    integer offered to ``anyOf: [string, integer]`` stays an integer instead of
    being stringified by whichever branch happens to come first.
    """
    allowed = _allowed_types(schema)
    if not allowed:  # a permissive schema accepts whatever arrived
        return value

    exact = tuple(name for name in allowed if _IS_EXACTLY.get(name, _never)(value))
    for name in (*exact, *allowed):
        if (converter := _CONVERTERS.get(name)) is None:
            continue
        try:
            return converter(value)
        except (_Mismatch, TypeError, ValueError):
            continue
    raise _Mismatch(f"expected {_join(allowed)}, got {_json_type(value)}")


def _as_string(value: Any) -> str:
    # A dict or a list here means the model sent structure where prose was
    # asked for -- stringifying it would produce a nonsense command.
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float):
        return str(value)
    raise _Mismatch("is not a string")


def _as_integer(value: Any) -> int:
    if isinstance(value, bool):  # bool is an int in Python, but not in JSON
        raise _Mismatch("is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise _Mismatch("is not an integer")


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        raise _Mismatch("is not a number")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise _Mismatch("is not a number")


_TRUE = frozenset({"true", "yes", "1", "on"})
_FALSE = frozenset({"false", "no", "0", "off"})


def _as_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
    raise _Mismatch("is not a boolean")


def _as_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    raise _Mismatch("is not an array")


def _as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise _Mismatch("is not an object")


def _as_null(value: Any) -> None:
    if value is None:
        return None
    raise _Mismatch("is not null")


def _never(value: Any) -> bool:
    return False


#: Whether a value already *is* the JSON type, before any conversion.
_IS_EXACTLY: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, Mapping),
    "null": lambda v: v is None,
}

_CONVERTERS: dict[str, Callable[[Any], Any]] = {
    "string": _as_string,
    "integer": _as_integer,
    "number": _as_number,
    "boolean": _as_boolean,
    "array": _as_array,
    "object": _as_object,
    "null": _as_null,
}


def _json_type(value: Any) -> str:
    match value:
        case bool():
            return "boolean"
        case int() | float():
            return "number"
        case str():
            return "string"
        case Mapping():
            return "object"
        case list() | tuple():
            return "array"
        case None:
            return "null"
        case _:
            return type(value).__name__


def _join(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def describe_parameters(parameters: JSONObject) -> dict[str, str]:
    """``{name: "integer or null"}``, attached to an error so the model can retry."""
    properties: dict[str, Any] = parameters.get("properties", {}) or {}
    return {
        name: _join(types) if (types := _allowed_types(schema)) else "any"
        for name, schema in properties.items()
    }
