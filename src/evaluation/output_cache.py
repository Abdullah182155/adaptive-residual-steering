"""Persistent generation cache keyed by (prompt_text, route_key).

Any prompt whose predicted route is empty canonicalises to EMPTY_ROUTE_KEY,
so the baseline arm and every arm that abstains on that prompt share exactly
one generation.  Non-empty routes are additionally namespaced
(e.g. "guards_on"/"guards_off") because the cached entry also records whether
the guards tripped, which only exists when guards ran.  The empty key ignores
the namespace.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

EMPTY_ROUTE_KEY = ((), "none", None)


def canonical_route_key(
    layers: Optional[Sequence[int]],
    gate_mode: str = "const",
    scale: Optional[float] = None,
):
    ls = tuple(sorted(int(l) for l in (layers or ())))
    if not ls:
        return EMPTY_ROUTE_KEY
    return (ls, str(gate_mode), None if scale is None else round(float(scale), 6))


def normalize_route_key(key):
    layers, mode, scale = key
    return canonical_route_key(layers, mode, scale)


def key_to_str(key) -> str:
    k = normalize_route_key(key)
    return json.dumps([list(k[0]), k[1], k[2]], separators=(",", ":"))


class OutputCache:
    def __init__(self, cache_dir: str, fingerprint: str = "default"):
        os.makedirs(cache_dir, exist_ok=True)
        safe      = "".join(c if (c.isalnum() or c in "-_") else "_" for c in fingerprint)[:64]
        self.path = os.path.join(cache_dir, f"output_cache_{safe}.jsonl")
        self._data: Dict[str, Dict[str, Any]] = {}
        self.hits   = 0
        self.misses = 0
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._data[rec["k"]] = rec["v"]
                except (ValueError, KeyError):
                    continue  # tolerate a truncated final line after a crash

    @staticmethod
    def make_key(prompt: str, route_key, namespace: str = "") -> str:
        k  = normalize_route_key(route_key)
        ns = "" if k == EMPTY_ROUTE_KEY else namespace
        raw = json.dumps([prompt, key_to_str(k), ns])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt: str, route_key, namespace: str = "") -> Optional[Dict[str, Any]]:
        e = self._data.get(self.make_key(prompt, route_key, namespace))
        if e is None:
            self.misses += 1
        else:
            self.hits += 1
        return e

    def put(self, prompt: str, route_key, namespace: str, entry: Dict[str, Any]):
        h             = self.make_key(prompt, route_key, namespace)
        self._data[h] = entry
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": h, "route": key_to_str(route_key), "v": entry}) + "\n")
            f.flush()

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> Dict[str, Any]:
        tot = self.hits + self.misses
        return {
            "size":     len(self._data),
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": (self.hits / tot) if tot else 0.0,
            "path":     self.path,
        }
