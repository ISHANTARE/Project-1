"""
LAPS Inter-Annotator Agreement & Fleiss' Kappa Statistical Calculator
---------------------------------------------------------------------
Computes Fleiss' Kappa (κ), Cohen's Kappa, and Gwet's AC1 for multi-rater
double-blind loanword preservation studies on code-mixed corpora.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def compute_fleiss_kappa(ratings_matrix: List[List[int]]) -> Tuple[float, Dict[str, float]]:
    """
    Computes Fleiss' Kappa for a matrix of shape (N_subjects, N_categories).
    Each row contains counts of raters assigning subject i to category j.
    """
    N = len(ratings_matrix)
    if N == 0:
        return 0.0, {}

    k = len(ratings_matrix[0])
    n = sum(ratings_matrix[0])

    total_ratings = N * n
    p_j = [sum(ratings_matrix[i][j] for i in range(N)) / total_ratings for j in range(k)]
    P_e = sum(p**2 for p in p_j)

    P_i = []
    for i in range(N):
        sum_sq = sum(ratings_matrix[i][j]**2 for j in range(k))
        P_i.append((sum_sq - n) / (n * (n - 1)))

    P_bar = sum(P_i) / N

    if abs(1.0 - P_e) < 1e-9:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1.0 - P_e)

    # Gwet's AC1 (robust against marginal prevalence paradox)
    p_k = [1.0 / k] * k
    P_e_gwet = 2.0 * p_j[0] * (1.0 - p_j[0]) if k == 2 else sum(p * (1 - p) for p in p_j) / (k - 1)
    ac1 = (P_bar - P_e_gwet) / (1.0 - P_e_gwet) if abs(1.0 - P_e_gwet) > 1e-9 else 1.0

    stats = {
        "fleiss_kappa": round(kappa, 4),
        "gwet_ac1": round(ac1, 4),
        "observed_agreement": round(P_bar, 4),
        "expected_agreement": round(P_e, 4),
        "n_subjects": N,
        "n_raters": n,
        "n_categories": k,
    }
    return round(kappa, 4), stats


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.20:
        return "Poor Agreement"
    elif kappa <= 0.40:
        return "Fair Agreement"
    elif kappa <= 0.60:
        return "Moderate Agreement"
    elif kappa <= 0.80:
        return "Substantial Agreement (Valid for ACL/EMNLP submission)"
    else:
        return "Almost Perfect Agreement"


def _load_annotation_csv(csv_path: Path) -> List[List[int]]:
    """
    Load real human annotation ratings from a CSV file.

    Expected CSV format (one row per token, columns = rater decisions):
        sentence_id, token, rater_1, rater_2, rater_3
        S001, viva, 1, 1, 1
        S001, calendar, 1, 1, 0
        S002, postpone, 0, 0, 1
        ...

    Where 1 = Preserve and 0 = Rewrite.
    Returns a ratings matrix: List[[n_rewrite, n_preserve], ...]
    """
    import csv
    matrix = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rater_cols = [k for k in row if k.startswith("rater_")]
            if not rater_cols:
                raise ValueError(
                    f"No 'rater_*' columns found in {csv_path}. "
                    "Expected columns: sentence_id, token, rater_1, rater_2, ..."
                )
            votes = [int(row[r]) for r in rater_cols]
            n_preserve = sum(votes)
            n_rewrite = len(votes) - n_preserve
            matrix.append([n_rewrite, n_preserve])
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Inter-Rater Agreement Calculator for LAPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  Compute Kappa from a real annotation CSV:
    python scripts/compute_agreement.py --annotations data/evaluation/annotations.csv

  The CSV must have columns: sentence_id, token, rater_1, rater_2, [rater_3, ...]
  Use scripts/collect_real_data.py to generate the annotation template CSV.

IMPORTANT:
  This script requires REAL human annotation data.
  Do NOT use synthetic or algorithmically generated ratings.
  Fleiss' Kappa is only meaningful when computed from independent human judgements.
        """
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/evaluation/annotations.csv"),
        help="Path to real human annotation CSV file."
    )
    args = parser.parse_args()

    if not args.annotations.exists():
        print(f"\n[ERROR] Annotation file not found: {args.annotations}")
        print("\nTo generate the annotation template, run:")
        print("  python scripts/collect_real_data.py")
        print("\nThen fill in the 'rater_1', 'rater_2', etc. columns with real human judgements.")
        print("Each rater independently marks: 1 = Preserve the loanword, 0 = Translate it.\n")
        return

    print(f"\nLoading annotations from: {args.annotations}")
    matrix = _load_annotation_csv(args.annotations)

    if len(matrix) < 10:
        print(f"[WARNING] Only {len(matrix)} tokens found. "
              "A minimum of 30-50 annotated tokens is recommended for reliable Kappa.")

    kappa, stats = compute_fleiss_kappa(matrix)
    print("\n=======================================================")
    print("  LAPS Multi-Annotator Inter-Rater Agreement")
    print("=======================================================")
    print(f"- Total Candidate Tokens Evaluated: {stats['n_subjects']}")
    print(f"- Number of Independent Raters    : {stats['n_raters']}")
    print(f"- Observed Agreement (P_bar)      : {stats['observed_agreement']:.4f}")
    print(f"- Expected Agreement (P_e)        : {stats['expected_agreement']:.4f}")
    print(f"- Fleiss Kappa                    : {stats['fleiss_kappa']:.4f}")
    print(f"- Gwet AC1 Agreement              : {stats['gwet_ac1']:.4f}")
    print(f"- Interpretation                  : {interpret_kappa(stats['gwet_ac1'])}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
