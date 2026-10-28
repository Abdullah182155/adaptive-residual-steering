"""
Decontaminate new exemplars against GSM8K test set using 8-gram overlap.

Phase 0.7: Ensure new exemplars don't overlap with test set to avoid testing memorized content.

Usage:
    python scripts/decontaminate_exemplars.py --exemplar_set v5
"""

import argparse
import json
from typing import List, Dict, Set
from collections import defaultdict
from datasets import load_dataset
from src.data.prompts import V5_EXEMPLARS, LEGACY_GSM8K_EXEMPLARS


def extract_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extract n-grams from text (normalized to lowercase)."""
    words = text.lower().split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.add(ngram)
    return ngrams


def check_overlap(
    exemplars: List[Dict[str, str]],
    test_set_ngrams: Set[str],
    n_gram: int = 8,
) -> Dict[str, any]:
    """Check 8-gram overlap between exemplars and test set."""
    overlaps = []
    
    for i, ex in enumerate(exemplars):
        q_ngrams = extract_ngrams(ex["q"], n_gram)
        a_ngrams = extract_ngrams(ex["a"], n_gram)
        
        q_overlap = q_ngrams & test_set_ngrams
        a_overlap = a_ngrams & test_set_ngrams
        
        overlaps.append({
            "index": i,
            "question": ex["q"],
            "answer": ex["a"],
            "question_ngram_overlap": len(q_overlap),
            "answer_ngram_overlap": len(a_overlap),
            "total_overlap": len(q_overlap | a_overlap),
        })
    
    total_overlaps = sum(o["total_overlap"] for o in overlaps)
    max_overlap = max(o["total_overlap"] for o in overlaps) if overlaps else 0
    
    return {
        "n_gram": n_gram,
        "total_exemplars": len(exemplars),
        "exemplars_with_overlap": sum(1 for o in overlaps if o["total_overlap"] > 0),
        "total_overlapping_ngrams": total_overlaps,
        "max_overlap_per_exemplar": max_overlap,
        "details": overlaps,
    }


def main():
    parser = argparse.ArgumentParser(description="Decontaminate exemplars against test set")
    parser.add_argument("--exemplar_set", choices=["v5", "legacy"], default="v5", help="Exemplar set to check")
    parser.add_argument("--n_gram", type=int, default=8, help="N-gram size for overlap detection")
    parser.add_argument("--output", default="decontamination_report.json", help="Output JSON path")
    args = parser.parse_args()
    
    # Load exemplars
    if args.exemplar_set == "v5":
        exemplars = V5_EXEMPLARS
        print(f"Checking V5 exemplars ({len(exemplars)} problems)")
    else:
        exemplars = LEGACY_GSM8K_EXEMPLARS
        print(f"Checking legacy exemplars ({len(exemplars)} problems)")
    
    # Load GSM8K test set
    print("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    
    # Build n-gram index from test set
    print(f"Building {args.n_gram}-gram index from test set...")
    test_ngrams = set()
    for ex in ds:
        q_ngrams = extract_ngrams(ex["question"], args.n_gram)
        a_ngrams = extract_ngrams(ex["answer"], args.n_gram)
        test_ngrams.update(q_ngrams)
        test_ngrams.update(a_ngrams)
    
    print(f"Test set contains {len(test_ngrams)} unique {args.n_gram}-grams")
    
    # Check overlap
    print("Checking exemplar overlap...")
    report = check_overlap(exemplars, test_ngrams, args.n_gram)
    
    # Print summary
    print(f"\n=== Decontamination Report ({args.exemplar_set}) ===")
    print(f"N-gram size: {args.n_gram}")
    print(f"Total exemplars: {report['total_exemplars']}")
    print(f"Exemplars with overlap: {report['exemplars_with_overlap']}")
    print(f"Total overlapping n-grams: {report['total_overlapping_ngrams']}")
    print(f"Max overlap per exemplar: {report['max_overlap_per_exemplar']}")
    
    if report['exemplars_with_overlap'] > 0:
        print("\nExemplars with overlap:")
        for o in report['details']:
            if o['total_overlap'] > 0:
                print(f"  [{o['index']}] Q: {o['question'][:50]}... (overlap: {o['total_overlap']})")
    else:
        print("\n✅ No overlaps found - exemplars are decontaminated!")
    
    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
