from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Sequence, Tuple
import torch

@dataclass(frozen=True)
class Catalog:
    layers: Tuple[int, ...]
    max_k: int
    subsets: Tuple[Tuple[int, ...], ...]          # index 0 == (); then by (len, lexicographic)

    @classmethod
    def build(cls, layers: Sequence[int], max_k: int = 3) -> "Catalog":
        """Build catalog with index 0 == () and all combinations of size 1..max_k sorted."""
        layers_tuple = tuple(sorted(layers))
        all_subsets = [()]
        for k in range(1, max_k + 1):
            for comb in itertools.combinations(layers_tuple, k):
                all_subsets.append(comb)
        return cls(layers=layers_tuple, max_k=max_k, subsets=tuple(all_subsets))

    def __len__(self) -> int:
        return len(self.subsets)

    def index_of(self, subset: Sequence[int]) -> int:
        normalized = tuple(sorted(subset))
        return self.subsets.index(normalized)

    def sizes(self) -> torch.LongTensor:
        """Returns (C,) tensor of subset sizes."""
        return torch.tensor([len(s) for s in self.subsets], dtype=torch.long)

    def membership(self) -> torch.BoolTensor:
        """Returns (C, len(layers)) boolean membership table."""
        c = len(self.subsets)
        nl = len(self.layers)
        layer_to_idx = {l: i for i, l in enumerate(self.layers)}
        mem = torch.zeros(c, nl, dtype=torch.bool)
        for i, s in enumerate(self.subsets):
            for l in s:
                mem[i, layer_to_idx[l]] = True
        return mem

    def to_candidate_matrix(self, class_idx: torch.LongTensor, target_layers: Sequence[int]) -> torch.BoolTensor:
        """Convert a batch of catalog class indices (B,) into (B, n_candidates) boolean matrix."""
        B = class_idx.shape[0]
        nc = len(target_layers)
        layer_to_cand = {l: i for i, l in enumerate(target_layers)}
        mat = torch.zeros(B, nc, dtype=torch.bool, device=class_idx.device)
        for b in range(B):
            idx = int(class_idx[b].item())
            subset = self.subsets[idx]
            for lid in subset:
                if lid in layer_to_cand:
                    mat[b, layer_to_cand[lid]] = True
        return mat
