"""Tool arguments arrive malformed often enough that recovery is a feature."""

import pytest

from mapta.application import decode_tool_arguments


def test_plain_json():
    assert decode_tool_arguments('{"command": "ls"}') == {"command": "ls"}


def test_empty_payloads_mean_no_arguments():
    assert decode_tool_arguments(None) == {}
    assert decode_tool_arguments("") == {}


def test_mapping_passes_through():
    assert decode_tool_arguments({"a": 1}) == {"a": 1}


def test_markdown_fence_is_stripped():
    assert decode_tool_arguments('```json\n{"a": 1}\n```') == {"a": 1}


def test_raw_newlines_inside_strings():
    assert decode_tool_arguments('{"code": "a\nb"}') == {"code": "a\nb"}


def test_trailing_garbage_is_ignored():
    assert decode_tool_arguments('{"a": 1} {"b": 2}') == {"a": 1}


def test_double_encoded_json():
    assert decode_tool_arguments('"{\\"a\\": 1}"') == {"a": 1}


def test_unusable_payload_raises():
    with pytest.raises(ValueError):
        decode_tool_arguments("not json at all")


def test_non_object_json_raises():
    with pytest.raises(ValueError):
        decode_tool_arguments("[1, 2, 3]")


# --- recovery of the malformations seen in real runs ---------------------


def test_python_literals_are_recovered():
    """`None`/`True`/`False` are the single most common cause of a failed call."""
    assert decode_tool_arguments('{"command": "ls", "timeout": None}') == {
        "command": "ls",
        "timeout": None,
    }
    assert decode_tool_arguments('{"command": "ls", "detach": True}')["detach"] is True


def test_single_quoted_payloads_are_recovered():
    assert decode_tool_arguments("{'command': 'ls -la'}") == {"command": "ls -la"}


def test_trailing_commas_are_recovered():
    assert decode_tool_arguments('{"command": "ls",}') == {"command": "ls"}


def test_prose_before_the_object_is_skipped():
    assert decode_tool_arguments('Here you go: {"command": "ls"}') == {"command": "ls"}


def test_literal_eval_never_executes_anything():
    with pytest.raises(ValueError):
        decode_tool_arguments('{"command": __import__("os").system("touch /tmp/pwned")}')


def test_a_value_cut_in_half_is_never_completed():
    """Closing the string would hand `rm -rf /tmp/x` to the shell instead."""
    with pytest.raises(ValueError):
        decode_tool_arguments('{"command": "rm -rf /tmp/scratch/sub')


def test_a_half_written_value_is_dropped_not_guessed():
    # The complete member survives; the truncated one is discarded, so the
    # caller sees a missing argument rather than a shortened one.
    assert decode_tool_arguments('{"limit": 5, "command": "rm -rf /tmp/scratch/sub') == {
        "limit": 5
    }
