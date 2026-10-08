import os
import sys
import json
import hashlib
import argparse
from typing import Dict, Any, Tuple

ARCH_FILES = [
    "src/steering/steernet.py",
    "src/gating/gate.py",
    "src/routing/router.py",
    "src/models/wrapper.py",
]

DIAGRAM_FILES = [
    "diagrams/diagram_steernet.png",
    "diagrams/diagram_steernet.drawio",
    "diagrams/diagram_gate.png",
    "diagrams/diagram_gate.drawio",
    "diagrams/diagram_router.png",
    "diagrams/diagram_router.drawio",
    "diagrams/diagram_ars_full_system.png",
    "diagrams/diagram_ars_full_system.drawio",
]


def get_repo_root() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_arch_hashes(repo_root: str = None) -> Dict[str, str]:
    """Computes SHA-256 hashes for all architectural core source files."""
    root = repo_root or get_repo_root()
    hashes = {}
    for rel_path in ARCH_FILES:
        full_path = os.path.join(root, rel_path)
        if os.path.exists(full_path):
            hashes[rel_path] = compute_file_hash(full_path)
        else:
            hashes[rel_path] = "missing"
    return hashes


def compute_combined_hash(hashes: Dict[str, str]) -> str:
    """Computes a single combined SHA-256 hash from sorted file hashes."""
    combined = hashlib.sha256()
    for k in sorted(hashes.keys()):
        combined.update(f"{k}:{hashes[k]}\n".encode("utf-8"))
    return combined.hexdigest()


def get_cached_record(hash_file_path: str) -> Dict[str, Any]:
    if not os.path.exists(hash_file_path):
        return {}
    try:
        with open(hash_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cached_record(hash_file_path: str, hashes: Dict[str, str]):
    os.makedirs(os.path.dirname(hash_file_path), exist_ok=True)
    combined = compute_combined_hash(hashes)
    record = {
        "combined_hash": combined,
        "file_hashes": hashes,
    }
    with open(hash_file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def check_integrity(repo_root: str = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Checks whether the architectural source code matches cached hash and diagrams exist."""
    root = repo_root or get_repo_root()
    hash_file_path = os.path.join(root, "diagrams", ".arch_hash")
    cached = get_cached_record(hash_file_path)

    current_hashes = compute_arch_hashes(root)
    current_combined = compute_combined_hash(current_hashes)

    # Check if all diagram files exist
    missing_diagrams = []
    for diag in DIAGRAM_FILES:
        if not os.path.exists(os.path.join(root, diag)):
            missing_diagrams.append(diag)

    if missing_diagrams:
        return False, f"Missing {len(missing_diagrams)} diagram files: {missing_diagrams}", {
            "current_combined": current_combined,
            "cached_combined": cached.get("combined_hash"),
            "missing_diagrams": missing_diagrams,
        }

    if not cached or "combined_hash" not in cached:
        return False, "No cached architectural hash found at diagrams/.arch_hash", {
            "current_combined": current_combined,
            "cached_combined": None,
        }

    if cached["combined_hash"] != current_combined:
        diff_files = [
            f for f, h in current_hashes.items()
            if cached.get("file_hashes", {}).get(f) != h
        ]
        return False, f"Architectural source modified in {len(diff_files)} files: {diff_files}", {
            "current_combined": current_combined,
            "cached_combined": cached["combined_hash"],
            "modified_files": diff_files,
        }

    return True, "Architectural code and diagram cache are in sync.", {
        "current_combined": current_combined,
        "cached_combined": cached["combined_hash"],
    }


def run_verifier(force: bool = False, generate_if_stale: bool = True) -> int:
    root = get_repo_root()
    hash_file_path = os.path.join(root, "diagrams", ".arch_hash")
    is_valid, reason, info = check_integrity(root)

    print("=" * 80)
    print("  Adaptive Residual Steering (ARS) - Architecture Diagram Verifier")
    print("=" * 80)

    if is_valid and not force:
        print(f"[CACHED] {reason}")
        print(f"   Combined Architecture SHA-256: {info['current_combined'][:16]}...")
        print("   All 4 Draw.io diagrams and publication PNGs are up to date.")
        print("=" * 80)
        return 0

    if force:
        print("[FORCE] Force flag provided. Triggering regeneration...")
    else:
        print(f"[SYNC] STALE OR OUT OF SYNC: {reason}")

    if generate_if_stale:
        print("\n>>> Re-generating all architecture diagrams...")
        try:
            from scripts.generate_architecture_diagrams import generate_all_diagrams
            out_paths = generate_all_diagrams(os.path.join(root, "diagrams"))
            print(f"[OK] Generated {len(out_paths)} diagram artifacts in diagrams/")

            # Update cache
            current_hashes = compute_arch_hashes(root)
            save_cached_record(hash_file_path, current_hashes)
            new_combined = compute_combined_hash(current_hashes)
            print(f"[OK] Updated cache diagrams/.arch_hash with SHA-256: {new_combined[:16]}...")
            print("=" * 80)
            return 0
        except Exception as e:
            print(f"[ERROR] Error generating diagrams: {e}")
            return 1

    else:
        print("Generation skipped. Exiting with non-zero code.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and update architecture diagrams integrity.")
    parser.add_argument("--force", action="store_true", help="Force re-generation of diagrams.")
    parser.add_argument("--check-only", action="store_true", help="Only check integrity without regenerating.")
    args = parser.parse_args()

    exit_code = run_verifier(force=args.force, generate_if_stale=not args.check_only)
    sys.exit(exit_code)
