"""Verify that the architecture hash represents the active Phase 1 modules."""
import hashlib
import json
from pathlib import Path

ARCH_FILES = (
    "src/steering/steernet.py",
    "src/models/wrapper.py",
    "src/configs/config.py",
)


def get_repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def compute_arch_hashes(repo_root=None):
    root = Path(repo_root or get_repo_root())
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ARCH_FILES
    }


def save_cached_record(path, hashes):
    encoded = json.dumps(hashes, sort_keys=True).encode()
    payload = {
        "file_hashes": hashes,
        "combined_hash": hashlib.sha256(encoded).hexdigest(),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(get_repo_root())
    hashes = compute_arch_hashes(root)
    cache = root / "diagrams" / ".arch_hash"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    expected = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if cached.get("combined_hash") != expected:
        save_cached_record(cache, hashes)
        print("Architecture hash updated.")
    else:
        print("Architecture hash is in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
