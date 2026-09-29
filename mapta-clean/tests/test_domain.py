"""Domain rules that the rest of the code relies on."""

from mapta.domain import ScanTally, Severity, Target, targets_from_lines


def test_slug_is_filesystem_safe():
    assert Target("https://example.com/a/b?c=1").slug == "example.com_a_b_c_1"


def test_host_drops_scheme_and_path():
    assert Target("https://example.com/a/b").host == "example.com"


def test_slug_never_empty():
    assert Target("https://").slug == "target"


def test_severity_parsing_is_case_insensitive():
    assert Severity.parse("critical") is Severity.CRITICAL
    assert Severity.parse("  HIGH ") is Severity.HIGH


def test_unknown_severity_falls_back_to_info():
    assert Severity.parse("catastrophic") is Severity.INFO


def test_headline_severity_is_the_worst_present():
    assert ScanTally(total=3, medium=2, low=1).headline_severity is Severity.MEDIUM
    assert ScanTally(total=0).headline_severity is None


def test_targets_file_skips_blanks_and_comments():
    lines = ["# comment", "", "  https://a.test  ", "https://b.test"]
    assert [t.url for t in targets_from_lines(lines)] == ["https://a.test", "https://b.test"]
