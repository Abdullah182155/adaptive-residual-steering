"""Create integrity-checked archives of a Session 3 run directory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable
import zipfile


_EXCLUDED_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_session3_bundle(
    run_dir: str | Path,
    archive_path: str | Path,
    *,
    required_files: Iterable[str] = (),
) -> dict:
    """Zip a run's artifacts and record checksums; retain partial runs for diagnosis."""
    source = Path(run_dir).resolve()
    destination = Path(archive_path).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Session 3 run directory does not exist: {source}")
    if destination == source or source in destination.parents:
        raise ValueError("artifact archive must be outside the run directory")

    files = sorted(
        path for path in source.rglob("*")
        if path.is_file()
        and path != source / "artifact_manifest.json"
        and not any(part in _EXCLUDED_PARTS for part in path.relative_to(source).parts)
        and path.suffix.lower() != ".zip"
        and not path.name.endswith(".tmp")
    )
    required = sorted({Path(name).as_posix() for name in required_files})
    present = {path.relative_to(source).as_posix() for path in files}
    missing = [name for name in required if name not in present]
    manifest_path = source / "artifact_manifest.json"
    entries = [
        {
            "path": path.relative_to(source).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files if path != manifest_path
    ]
    manifest = {
        "format": "ars_session3_artifact_bundle_v1",
        "run_id": source.name,
        "complete": not missing,
        "required_files": required,
        "missing_required_files": missing,
        "manifest_excludes_self": True,
        "files": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        with zipfile.ZipFile(
            temp_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(source).as_posix())
            archive.write(manifest_path, arcname=manifest_path.name)
        with zipfile.ZipFile(temp_path, mode="r") as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise zipfile.BadZipFile(f"archive verification failed at {corrupt_member}")
            archived_names = set(archive.namelist())
        required_present = [name for name in required if name in present]
        absent_from_archive = sorted(name for name in required_present if name not in archived_names)
        if absent_from_archive:
            raise zipfile.BadZipFile(f"required files were not archived: {absent_from_archive}")
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {
        **manifest,
        "archive_path": str(destination),
        "archive_size_bytes": destination.stat().st_size,
        "archive_sha256": _sha256(destination),
        "archive_verified": True,
    }
