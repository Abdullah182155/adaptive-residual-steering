"""Shared file, subprocess, and result helpers for the paper notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TRAINING_ARTIFACTS = (
    "steernet_v2.pt",
    "session3_steering_bank.pt",
    "session3_router.pt",
    "norms_v2_fresh.json",
    "candidate_norms_p99.json",
    "session3_static_bank_action.json",
    "manifest.json",
)


def atomic_write_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_named_file(roots: Iterable[str | Path], name: str) -> Path | None:
    matches: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.name == name:
            matches.append(root)
        elif root.is_dir():
            matches.extend(path for path in root.rglob(name) if path.is_file())
    return sorted(set(matches), key=lambda item: (len(item.parts), str(item)))[0] if matches else None


def extract_zip_once(archive: str | Path, destination: str | Path) -> Path:
    archive_path = Path(archive)
    target = Path(destination)
    marker = target / ".source_sha256"
    digest = sha256_file(archive_path)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as bundle:
        base = target.resolve()
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if resolved != base and base not in resolved.parents:
                raise ValueError(f"unsafe path in archive: {member.filename}")
        bundle.extractall(target)
    marker.write_text(digest + "\n", encoding="utf-8")
    return target


def merge_result_bundle(archive: str | Path, destination: str | Path) -> Path:
    """Merge a previous result archive into a workspace without nesting bundles."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    incoming = extract_zip_once(archive, target / "_incoming_bundle")
    for source in sorted(incoming.rglob("*")):
        if not source.is_file() or source.name == ".source_sha256":
            continue
        relative = source.relative_to(incoming)
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if sha256_file(output) != sha256_file(source):
                raise FileExistsError(
                    f"result bundle conflicts with an existing file: {relative}; "
                    "keep the runs separate or choose a new results folder"
                )
            continue
        shutil.copy2(source, output)
    shutil.rmtree(incoming)
    return target


def resolve_artifacts(root: str | Path, names: Sequence[str]) -> dict[str, Path]:
    base = Path(root)
    manifest_path = locate_named_file([base], "manifest.json")
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") == "ars_training_bundle_v1":
            if not manifest.get("training_ready", False):
                raise ValueError(
                    "training bundle manifest is not marked training_ready; "
                    f"phase status: {manifest.get('required_phase_status', {})}"
                )
            for section in ("artifacts", "code"):
                for relative_name, metadata in manifest.get(section, {}).items():
                    path = locate_named_file([base], Path(relative_name).name)
                    expected = metadata.get("sha256") if isinstance(metadata, dict) else None
                    if path is None or not expected or sha256_file(path) != expected:
                        raise ValueError(
                            f"training bundle {section} checksum is missing or invalid: {relative_name}"
                        )
    resolved = {name: locate_named_file([base], name) for name in names}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        found = [name for name, path in resolved.items() if path is not None]
        raise FileNotFoundError(f"missing required artifacts: {missing}; found: {found}")
    return {name: path for name, path in resolved.items() if path is not None}


def pack_files(files: Mapping[str | Path, str], destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source, archive_name in files.items():
            path = Path(source)
            if path.is_file():
                bundle.write(path, arcname=archive_name)
    temporary.replace(output)
    return output


def run_live(command: Sequence[str], *, cwd: str | Path, log_path: str | Path, env: Mapping[str, str] | None = None) -> int:
    """Stream a child process to the notebook and persist its output."""
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("$", " ".join(map(str, command)), flush=True)
    with log.open("a", encoding="utf-8") as saved:
        process = subprocess.Popen(
            list(map(str, command)), cwd=str(cwd), env=merged_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            saved.write(line)
            saved.flush()
        return int(process.wait())


def run_parallel_live(jobs, *, log_dir: str | Path, cwd: str | Path) -> list[tuple[str, int]]:
    """Run (name, command, env) jobs concurrently while keeping logs separate."""
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    output: list[tuple[str, int]] = []
    lock = threading.Lock()

    def worker(job):
        name, command, env = job
        path = logs / f"{name.replace('/', '_')}.log"
        code = run_live(command, cwd=cwd, log_path=path, env=env)
        return name, code

    with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            output.append(future.result())
    return sorted(output)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def kaggle_input_roots() -> list[Path]:
    roots = [Path("/kaggle/input")]
    roots = [path for path in roots if path.exists()]
    return roots
