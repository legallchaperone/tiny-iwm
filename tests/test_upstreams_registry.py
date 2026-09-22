from pathlib import Path

import pytest
import yaml


REGISTRY = Path(__file__).parents[1] / "third_party" / "UPSTREAMS.yaml"
REPOSITORY_ROOT = REGISTRY.parents[1]


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


def test_existing_template_scaffold_has_per_file_source_and_verification():
    source = yaml.safe_load(REGISTRY.read_text())["sources"]["research-template"]
    mappings = source["existing_local_mappings"]
    by_path = {mapping["local_path"]: mapping for mapping in mappings}

    assert len(mappings) == 73
    assert len(by_path) == len(mappings)
    assert by_path["utils/wandb_utils.py"]["upstream_path"] == "utils/wandb_utils.py"
    assert by_path["algorithms/examples/classifier/classifier.py"]["reuse_kind"] == "vendored"
    assert by_path["README.md"]["reuse_kind"] == "adapted"

    for local_path, mapping in by_path.items():
        assert (REPOSITORY_ROOT / local_path).exists()
        assert mapping["upstream_path"] == local_path
        assert mapping["reuse_kind"] in {"vendored", "adapted"}
        verification = mapping["verification"]
        assert verification["method"] == "tree-match-verified-in-CWX-5"
        assert len(verification["imported_local_commit"]) == 40
        assert verification["upstream_commit"] == source["commit"]


def test_world_model_targets_follow_the_planned_package_layout():
    sources = yaml.safe_load(REGISTRY.read_text())["sources"]

    for source_name in ("wan-2.1", "causal-forcing", "matrix-game-3.5", "prope"):
        targets = sources[source_name]["intended_local_targets"]
        for target in targets:
            if target.startswith(("models/", "training/")):
                pytest.fail(f"{source_name} uses an unqualified target path: {target}")

    assert "algorithms/world_model/models/dit.py" in sources["wan-2.1"]["intended_local_targets"]
    assert "algorithms/world_model/training_batch.py" in sources["causal-forcing"]["intended_local_targets"]
    assert sources["matrix-game-3.5"]["intended_local_targets"] == [
        "algorithms/world_model/models/prope.py"
    ]
