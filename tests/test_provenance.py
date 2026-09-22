import json
import subprocess
from pathlib import Path

import yaml
from omegaconf import OmegaConf
import pytest

from scripts.write_run_records import compose_and_write
import utils.provenance as provenance_module
from utils.provenance import write_run_records


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _make_repository(path: Path) -> None:
    _run_git(path, "init")
    _run_git(path, "config", "user.name", "Test User")
    _run_git(path, "config", "user.email", "test@example.com")
    (path / "tracked.txt").write_text("tracked\n")
    _run_git(path, "add", "tracked.txt")
    _run_git(path, "commit", "-m", "initial")


def test_write_run_records_captures_config_git_runtime_upstreams_and_seeds(tmp_path):
    project_root = tmp_path / "repository"
    project_root.mkdir()
    _make_repository(project_root)
    registry_dir = project_root / "third_party"
    registry_dir.mkdir()
    (registry_dir / "UPSTREAMS.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": {
                    "example": {
                        "repository": "https://github.com/example/upstream",
                        "commit": "a" * 40,
                    }
                }
            }
        )
    )
    _run_git(project_root, "add", "third_party/UPSTREAMS.yaml")
    _run_git(project_root, "commit", "-m", "add upstream registry")

    cfg = OmegaConf.create({"name": "test-run", "seed": 17, "rollout": {"generation_seeds": [23, 29]}})
    output_dir = tmp_path / "outputs" / "test-run"
    provenance = write_run_records(cfg, output_dir, project_root)

    assert OmegaConf.load(output_dir / "resolved_config.yaml") == cfg
    assert json.loads((output_dir / "provenance.json").read_text()) == provenance
    assert provenance["git"]["revision"]
    assert provenance["git"]["dirty"] is False
    assert provenance["upstream_revisions"] == {"example/upstream": "a" * 40}
    assert provenance["seeds"] == {"seed": 17, "rollout.generation_seeds": [23, 29]}
    assert provenance["runtime"]["python"]
    assert "packages" in provenance["runtime"]


def test_write_run_records_reports_dirty_worktree(tmp_path):
    project_root = tmp_path / "repository"
    project_root.mkdir()
    _make_repository(project_root)
    (project_root / "tracked.txt").write_text("changed\n")

    provenance = write_run_records(OmegaConf.create({"name": "dirty"}), tmp_path / "output", project_root)

    assert provenance["git"]["dirty"] is True
    assert provenance["git"]["status_porcelain"] == [" M tracked.txt"]


def test_no_training_command_composes_and_writes_without_wandb_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    provenance = compose_and_write(
        tmp_path / "run",
        ["+name=record-only", "wandb.mode=disabled"],
    )

    resolved = OmegaConf.load(tmp_path / "run" / "resolved_config.yaml")
    assert resolved.name == "record-only"
    assert resolved.wandb.mode == "disabled"
    assert (tmp_path / "run" / "provenance.json").is_file()
    assert provenance["git"]["revision"]


def test_existing_run_records_are_never_replaced(tmp_path):
    project_root = tmp_path / "repository"
    project_root.mkdir()
    _make_repository(project_root)
    output_dir = tmp_path / "run"
    write_run_records(OmegaConf.create({"name": "original"}), output_dir, project_root)
    original_config = (output_dir / "resolved_config.yaml").read_text()
    original_provenance = (output_dir / "provenance.json").read_text()

    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_run_records(OmegaConf.create({"name": "replacement"}), output_dir, project_root)

    assert (output_dir / "resolved_config.yaml").read_text() == original_config
    assert (output_dir / "provenance.json").read_text() == original_provenance


def test_remote_credentials_are_removed_from_provenance(tmp_path):
    project_root = tmp_path / "repository"
    project_root.mkdir()
    _make_repository(project_root)
    _run_git(
        project_root,
        "remote",
        "add",
        "origin",
        "https://user:top-secret@github.com:notaport/example/project.git?access_token=query-secret#fragment-secret",
    )

    provenance = write_run_records(OmegaConf.create({"name": "safe"}), tmp_path / "run", project_root)

    assert provenance["git"]["origin"] == "https://github.com:notaport/example/project.git"
    persisted = (tmp_path / "run" / "provenance.json").read_text()
    assert "top-secret" not in persisted
    assert "query-secret" not in persisted
    assert "fragment-secret" not in persisted


def test_unparsable_remote_authority_is_sanitized_as_opaque_text():
    remote = "https://user:secret@[example.com/repo.git?access_token=query-secret"

    sanitized = provenance_module._sanitize_remote_url(remote)

    assert sanitized == "https://[example.com/repo.git"
    assert "secret" not in sanitized


def test_temporary_record_is_removed_when_write_fails(tmp_path, monkeypatch):
    original_factory = provenance_module.tempfile.NamedTemporaryFile
    temporary_paths = []

    class FailingWrite:
        def __init__(self, temporary):
            self.temporary = temporary
            self.name = temporary.name

        def __enter__(self):
            self.temporary.__enter__()
            return self

        def __exit__(self, *args):
            return self.temporary.__exit__(*args)

        def write(self, _content):
            raise OSError("simulated disk failure")

    def failing_factory(*args, **kwargs):
        temporary = original_factory(*args, **kwargs)
        temporary_paths.append(Path(temporary.name))
        return FailingWrite(temporary)

    monkeypatch.setattr(provenance_module.tempfile, "NamedTemporaryFile", failing_factory)

    with pytest.raises(OSError, match="simulated disk failure"):
        provenance_module._atomic_create_files({tmp_path / "record.json": "secret"})

    assert temporary_paths
    assert all(not path.exists() for path in temporary_paths)
