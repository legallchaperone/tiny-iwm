from pathlib import Path

import yaml


REGISTRY = Path(__file__).parents[1] / "third_party" / "UPSTREAMS.yaml"


def test_upstream_registry_has_complete_pinned_records():
    registry = yaml.safe_load(REGISTRY.read_text())

    assert registry["schema_version"] == 1
    assert set(registry["sources"]) == {
        "research-template",
        "wan-2.1",
        "causal-forcing",
        "sana",
        "matrix-game-3.5",
        "prope",
    }

    for source in registry["sources"].values():
        assert source["repository"].startswith("https://github.com/")
        assert len(source["commit"]) == 40
        assert source["intended_local_targets"]
        assert source["adaptation_notes"].strip()
        assert source["reuse_kind"] in {"adapted", "mixed", "reference-only", "vendored"}


def test_missing_license_prevents_source_reuse():
    sources = yaml.safe_load(REGISTRY.read_text())["sources"]

    for source in sources.values():
        license_record = source["license"]
        if license_record["status"] == "verified":
            assert license_record["path"]
            assert license_record["url"].endswith(license_record["path"])
        else:
            assert source["reuse_kind"] == "reference-only"
            assert "do not copy" in source["adaptation_notes"].lower()
