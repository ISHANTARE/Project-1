"""
LAPS Human Evaluation Export
-----------------------------
Exports the gold corpus as a structured CSV for Likert-scale annotation
by bilingual (Hindi-English) human raters.

Output columns:
    ID, Sentence_ID, Sentence, Register, Token, LAPS_Score, P_Preserve,
    LAPS_Decision, Reference, Human_Label_Gold, Rater1, Rater2, Rater3, Notes

Usage:
    python scripts/human_eval_export.py
    python scripts/human_eval_export.py --output results/human_eval.csv
    python scripts/human_eval_export.py --register business
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CORPUS_PATH  = os.path.join(ROOT, "data", "evaluation", "gold_corpus.json")
DEFAULT_OUT  = os.path.join(ROOT, "results", "human_eval_export.csv")
WEIGHTS_PATH = os.path.join(ROOT, "data", "weights", "w_star_business.json")


# ---------------------------------------------------------------------------
# LAPS Score & Sigmoid (standalone, no pipeline import needed)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "w1": 0.20,  # Borrowing Frequency
    "w2": 0.20,  # Domain Specificity
    "w3": 0.20,  # Semantic Importance
    "w4": 0.15,  # Context Consistency
    "w5": 0.15,  # Audience Familiarity
    "w6": 0.10,  # Translation Benefit (subtracted)
}

REGISTER_THRESHOLDS = {
    "casual":   0.35,
    "social":   0.40,
    "business": 0.50,
    "academic": 0.65,
}

SIGMOID_LAMBDA = 10.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_score(features: dict, weights: dict) -> float:
    score = (
        weights["w1"] * features.get("BF", 0)
        + weights["w2"] * features.get("DS", 0)
        + weights["w3"] * features.get("SI", 0)
        + weights["w4"] * features.get("CC", 0)
        + weights["w5"] * features.get("AF", 0)
        - weights["w6"] * features.get("TB", 0)
    )
    return round(max(0.0, min(1.0, score)), 4)


def compute_p_preserve(score: float, theta: float) -> float:
    return round(_sigmoid(SIGMOID_LAMBDA * (score - theta)), 4)


def decide(p: float) -> str:
    if p >= 0.70:
        return "PRESERVE"
    elif p >= 0.50:
        return "UNCERTAIN"
    return "REWRITE"


def load_weights(register: str) -> dict:
    """Load optimized weights for register if available, else use defaults."""
    weights_file = os.path.join(ROOT, "data", "weights", f"w_star_{register}.json")
    if os.path.isfile(weights_file):
        with open(weights_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        w = data.get("optimised_weights") or data.get("weights", {})
        if len(w) == 6:
            print(f"  Loaded optimized weights from: w_star_{register}.json")
            return w
    print(f"  Using default equal weights (no w_star_{register}.json found).")
    return dict(DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(corpus_path: str, output_path: str, register_filter: str | None = None) -> None:
    # Load corpus
    if not os.path.isfile(corpus_path):
        print(f"ERROR: Corpus not found at: {corpus_path}")
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    sentences = corpus.get("sentences", [])
    print(f"\nLAPS Human Evaluation Export")
    print(f"=============================")
    print(f"Corpus       : {corpus_path}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Output       : {output_path}\n")

    # Pre-load weights per register
    weight_cache: dict[str, dict] = {}
    registers_present = {s["register"] for s in sentences}
    for reg in registers_present:
        weight_cache[reg] = load_weights(reg)

    # Build rows
    rows = []
    token_id = 1
    stats = {"PRESERVE": 0, "UNCERTAIN": 0, "REWRITE": 0, "total": 0}

    for sent in sentences:
        sid      = sent["id"]
        sentence = sent["sentence"]
        register = sent["register"]
        reference = sent.get("reference", "")

        # Skip if register filter is active
        if register_filter and register != register_filter:
            continue

        theta   = REGISTER_THRESHOLDS.get(register, 0.50)
        weights = weight_cache[register]

        for tok in sent.get("tokens", []):
            token      = tok["token"]
            gold_label = tok.get("label", 1)
            features   = tok.get("features", {})

            laps_score  = compute_score(features, weights)
            p_preserve  = compute_p_preserve(laps_score, theta)
            laps_decision = decide(p_preserve)

            stats[laps_decision] = stats.get(laps_decision, 0) + 1
            stats["total"] += 1

            rows.append({
                "ID":               token_id,
                "Sentence_ID":      sid,
                "Sentence":         sentence,
                "Register":         register,
                "Token":            token,
                "LAPS_Score":       laps_score,
                "P_Preserve":       p_preserve,
                "LAPS_Decision":    laps_decision,
                "Reference":        reference,
                "Human_Label_Gold": gold_label,
                "Rater1":           "",
                "Rater2":           "",
                "Rater3":           "",
                "Notes":            "",
            })
            token_id += 1

    if not rows:
        print("No matching rows found. Check your --register filter.")
        sys.exit(1)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "ID", "Sentence_ID", "Sentence", "Register", "Token",
        "LAPS_Score", "P_Preserve", "LAPS_Decision",
        "Reference", "Human_Label_Gold",
        "Rater1", "Rater2", "Rater3", "Notes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"Export complete.")
    print(f"  Total token rows : {stats['total']}")
    print(f"  PRESERVE         : {stats['PRESERVE']:>4}  ({100*stats['PRESERVE']/max(stats['total'],1):.1f}%)")
    print(f"  UNCERTAIN        : {stats['UNCERTAIN']:>4}  ({100*stats['UNCERTAIN']/max(stats['total'],1):.1f}%)")
    print(f"  REWRITE          : {stats['REWRITE']:>4}  ({100*stats['REWRITE']/max(stats['total'],1):.1f}%)")
    print(f"\nDistributed to raters: {output_path}")
    print(f"  Rater columns (Rater1, Rater2, Rater3) are blank for annotation.")
    print(f"  Instructions: Each rater fills 1=Preserve / 0=Rewrite for each token.")
    print(f"  Compute Fleiss' kappa across Rater1/Rater2/Rater3 after collection.\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export LAPS gold corpus as a human evaluation CSV."
    )
    parser.add_argument(
        "--corpus",
        default=CORPUS_PATH,
        help=f"Path to gold corpus JSON (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--register",
        default=None,
        choices=list(REGISTER_THRESHOLDS.keys()),
        help="Filter to a single register (default: export all).",
    )
    args = parser.parse_args()
    export(args.corpus, args.output, args.register)


if __name__ == "__main__":
    main()
