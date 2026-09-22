"""Write local, reproducible run records without relying on W&B."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

import yaml
from omegaconf import DictConfig, OmegaConf


PINNED_UPSTREAM_REVISIONS = {
    "legallchaperone/research-template": "e4a3f528d4aad7872adf6ad9cd94f80c06d91e48",
    "Wan-Video/Wan2.1": "9737cba9c1c3c4d04b33fcad41c111989865d315",
    "thu-ml/Causal-Forcing": "31a43313af7c805af10e4e2bcbad1fa036e0ced9",
    "NVlabs/Sana": "f9178744c096dcf2a2ea773da183e341bcbeb044",
    "Riemann-Dynamics/Matrix-Game-3.5": "fbf7def0693ae14f745ba35bf2a26215d4ef991d",
    "liruilong940607/prope": "4c11297761225d25258e5ec21c61e9b19ab61e38",
}


def _git(project_root: Path, *args: str) -> Optional[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    # Preserve leading spaces because they are meaningful in `git status
    # --porcelain` (index status in column one, worktree status in column two).
    return result.stdout.rstrip("\n")


def _git_record(project_root: Path) -> Dict[str, Any]:
    status = _git(project_root, "status", "--porcelain", "--untracked-files=normal")
    origin = _git(project_root, "remote", "get-url", "origin")
    return {
        "revision": _git(project_root, "rev-parse", "HEAD"),
        "branch": _git(project_root, "branch", "--show-current"),
        "dirty": status is None or bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "origin": _sanitize_remote_url(origin),
    }


def _sanitize_remote_url(url: Optional[str]) -> Optional[str]:
    """Remove URL userinfo while preserving ordinary SSH/scp-style remotes."""

    if not url or "://" not in url:
        return url
    parsed = urlsplit(url)
    if parsed.hostname is None:
        return url
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _upstream_revisions(project_root: Path) -> Dict[str, str]:
    registry_path = project_root / "third_party" / "UPSTREAMS.yaml"
    if not registry_path.exists():
        return dict(PINNED_UPSTREAM_REVISIONS)

    registry = yaml.safe_load(registry_path.read_text()) or {}
    revisions = {}
    for source in registry.get("sources", {}).values():
        repository = source.get("repository")
        commit = source.get("commit")
        if repository and commit:
            revisions[repository.removeprefix("https://github.com/")] = commit
    return revisions


def _collect_seeds(value: Any, prefix: str = "") -> Dict[str, Any]:
    seeds: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower().endswith(("seed", "seeds")):
                seeds[path] = child
            else:
                seeds.update(_collect_seeds(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            seeds.update(_collect_seeds(child, f"{prefix}[{index}]"))
    return seeds


def _package_versions() -> Dict[str, Optional[str]]:
    packages = ("torch", "torchvision", "lightning", "hydra-core", "omegaconf", "wandb", "numpy")
    versions: Dict[str, Optional[str]] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _runtime_record() -> Dict[str, Any]:
    selected_environment = {
        key: os.environ[key]
        for key in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK", "SLURM_JOB_ID")
        if key in os.environ
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "working_directory": str(Path.cwd()),
        "environment": selected_environment,
        "packages": _package_versions(),
    }


def _atomic_create_files(contents: Mapping[Path, str]) -> None:
    """Atomically publish new files and refuse to replace an existing record."""

    existing = [path for path in contents if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to replace existing run record(s): {names}")

    temporary_files: Dict[Path, Path] = {}
    created = []
    try:
        for path, content in contents.items():
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_files[path] = Path(temporary.name)

        # Hard-link publication is atomic and fails if a concurrent writer created
        # the destination. It therefore never replaces an existing run record.
        for path, temporary in temporary_files.items():
            os.link(temporary, path)
            created.append(path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_files.values():
            temporary.unlink(missing_ok=True)


def write_run_records(cfg: DictConfig, output_dir: Path, project_root: Path) -> Dict[str, Any]:
    """Write resolved configuration and provenance, returning the provenance record."""

    output_dir = Path(output_dir)
    project_root = Path(project_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    resolved_container = OmegaConf.to_container(cfg, resolve=True)
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_record(project_root),
        "upstream_revisions": _upstream_revisions(project_root),
        "runtime": _runtime_record(),
        "seeds": _collect_seeds(resolved_container),
    }

    _atomic_create_files(
        {
            output_dir / "resolved_config.yaml": resolved_yaml,
            output_dir / "provenance.json": json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        }
    )
    return provenance
