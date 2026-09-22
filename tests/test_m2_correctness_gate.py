import json

from scripts.validate_m2_correctness import main, run_gate


def test_m2_debug_gate_passes_every_check():
    report = run_gate()
    assert report["status"] == "passed"
    assert report["failures"] == []
    assert all(report["checks"].values())
    assert report["layout"]["latent_shape"] == [2, 4, 3, 5, 7]
    assert report["layout"]["token_count"] == 36


def test_gate_cli_writes_machine_readable_report(tmp_path, monkeypatch):
    output = tmp_path / "gate.json"
    monkeypatch.setattr("sys.argv", ["validate_m2_correctness", "--output", str(output)])
    assert main() == 0
    assert json.loads(output.read_text())["status"] == "passed"

