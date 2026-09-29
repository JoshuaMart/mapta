"""CLI wiring: flags, targets, exit codes."""

import json

import pytest

from mapta.config import Settings
from mapta.domain import ConfigurationError
from mapta.interface.cli import apply_overrides, build_parser, main, resolve_targets


def test_list_tools_needs_no_api_key(capsys):
    assert main(["--list-tools"]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert {"sandbox_agent", "validator_agent", "send_telegram_alert"} <= {
        t["name"] for t in tools
    }


def test_targets_come_from_repeated_flags():
    args = build_parser().parse_args(["-t", "https://a.test", "-t", "https://b.test"])
    assert [t.url for t in resolve_targets(args)] == ["https://a.test", "https://b.test"]


def test_targets_come_from_a_file(tmp_path):
    path = tmp_path / "targets.txt"
    path.write_text("# header\nhttps://a.test\n\n")
    args = build_parser().parse_args(["-f", str(path)])
    assert [t.url for t in resolve_targets(args)] == ["https://a.test"]


def test_missing_targets_file_is_an_error(tmp_path):
    args = build_parser().parse_args(["-f", str(tmp_path / "nope.txt")])
    with pytest.raises(ConfigurationError):
        resolve_targets(args)


def test_empty_targets_file_is_an_error(tmp_path):
    path = tmp_path / "targets.txt"
    path.write_text("# only comments\n")
    args = build_parser().parse_args(["-f", str(path)])
    with pytest.raises(ConfigurationError):
        resolve_targets(args)


def test_flags_override_the_environment():
    settings = Settings.from_env({"OPENROUTER_API_KEY": "sk-or", "MAX_ROUNDS": "100"})
    args = build_parser().parse_args(["-r", "5", "-o", "/tmp/out", "-p", "hit {target_url}"])
    overridden = apply_overrides(settings, args)
    assert overridden.scan.max_rounds == 5
    assert overridden.reports.output_dir == "/tmp/out"
    assert overridden.scan.user_prompt == "hit {target_url}"


def test_bad_configuration_exits_with_code_two(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("mapta.interface.cli.load_dotenv", lambda: None)
    assert main(["-t", "https://a.test", "--log-file", "/dev/null"]) == 2
    assert "Error:" in capsys.readouterr().err
