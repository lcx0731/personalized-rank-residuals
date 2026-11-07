import json
import subprocess
import sys

import pytest


def run(operation, path):
    return subprocess.run([sys.executable, "-m", "rank_residuals.cli", operation, str(path)],
                          capture_output=True, text=True, timeout=15)


def test_cli_roundtrip_independent_verification(tmp_path):
    request = {"nodes": 3, "edges": [[0, 1, 1], [1, 2, 1]],
               "alpha": "1/2", "personalization": [1, 0, 0]}
    path = tmp_path / "rank.json"
    raw = json.dumps(request)
    path.write_text(raw)
    ranked = run("rank", path)
    assert ranked.returncode == 0
    result = json.loads(ranked.stdout)
    assert path.read_text() == raw
    request["scores"] = result["scores"]
    check = tmp_path / "verify.json"
    check.write_text(json.dumps(request))
    checked = run("verify", check)
    assert checked.returncode == 0
    assert json.loads(checked.stdout) == result["certificate"]


def test_budget_exhaustion_has_distinct_exit_code_and_usable_report(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"nodes": 2, "edges": [[0, 1, 1]], "max_pushes": 0}))
    result = run("rank", path)
    assert result.returncode == 3
    assert json.loads(result.stdout)["certificate"]["exact_error_bound_l1"] == "1"


@pytest.mark.parametrize("raw", [
    '{}', '[]', '{"nodes":2,"edges":[],"x":1}', '{"nodes":2,"nodes":2,"edges":[]}',
    '{"nodes":2,"edges":[],"alpha":0.85}', '{"nodes":2,"edges":[],"tolerance":NaN}',
    '{broken', ' ' * (1024*1024+1),
], ids=["missing", "array", "unknown", "duplicate", "float-alpha", "nan", "invalid-json", "oversize"])
def test_bad_cli_inputs(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    result = run("rank", path)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "rank-residuals:" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_file_and_arguments(tmp_path):
    assert run("rank", tmp_path / "absent.json").returncode == 1
    assert subprocess.run([sys.executable, "-m", "rank_residuals.cli"], capture_output=True).returncode == 2
