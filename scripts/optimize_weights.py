"""
LAPS Automated Weight Vector Optimizer (w*)
-------------------------------------------
Implements the constrained optimization problem from §3 of the LAPS
Research Blueprint:

  w* = argmax_w  F1(LAPSDecisions(D_gold, w), Y_gold)

Subject to:
  Σ w_j = 1,  w_j ≥ 0  (simplex constraint)

Strategy: Grid search over the 6-dimensional weight simplex, then local
refinement with scipy.optimize (SLSQP) if available, otherwise pure
grid-search fallback.

The optimiser also computes cross-entropy loss (with L2 regularisation):
  L(w) = -(1/N) Σ [y_i log P_i + (1-y_i) log(1-P_i)] + γ‖w‖²

Usage:
  python scripts/optimize_weights.py
  python scripts/optimize_weights.py --corpus data/evaluation/gold_corpus.json
  python scripts/optimize_weights.py --register academic --output data/weights/w_star.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import product
from pathlib import Path

# Resolve project root (scripts/ is one level below root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laps.scorer           import compute_score
from laps.threshold_engine import compute_sigmoid_probability, REGISTER_THRESHOLDS

# ---------------------------------------------------------------------------
# Default gold corpus path
# ---------------------------------------------------------------------------
DEFAULT_CORPUS  = PROJECT_ROOT / "data" / "evaluation" / "gold_corpus.json"
DEFAULT_OUTPUT  = PROJECT_ROOT / "data" / "weights"
GAMMA_L2        = 0.01      # L2 regularisation strength
SIGMOID_LAMBDA  = 10.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _sigmoid_prob(score: float, theta: float, lam: float = SIGMOID_LAMBDA) -> float:
    x = lam * (score - theta)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def cross_entropy_loss(probs: list[float], labels: list[int],
                       weights: list[float], gamma: float = GAMMA_L2) -> float:
    """Cross-entropy loss + L2 regularisation on weight vector."""
    eps = 1e-9
    n   = len(probs)
    ce  = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, labels)
    ) / n
    l2 = gamma * sum(w * w for w in weights)
    return ce + l2


def compute_f1(predictions: list[int], labels: list[int]) -> float:
    """Binary F1 score (1 = PRESERVE)."""
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def compute_metrics_for_weights(
    corpus:   list[dict],
    w:        dict[str, float],
    theta:    float,
    lam:      float = SIGMOID_LAMBDA,
) -> dict:
    """
    Evaluate a weight vector against the gold corpus.

    Each corpus entry must have:
      {
        "token": str,
        "features": {"BF": float, "DS": float, "SI": float,
                     "CC": float, "AF": float, "TB": float},
        "label": 0 | 1    # 1 = human says PRESERVE
      }
    """
    probs   = []
    preds   = []
    labels  = []

    for entry in corpus:
        feats = entry["features"]
        score = compute_score(feats, w)
        prob  = _sigmoid_prob(score, theta, lam)
        pred  = 1 if prob >= 0.50 else 0

        probs.append(prob)
        preds.append(pred)
        labels.append(entry["label"])

    f1   = compute_f1(preds, labels)
    loss = cross_entropy_loss(probs, labels, list(w.values()))

    return {"f1": f1, "loss": loss, "probs": probs, "preds": preds}


# ---------------------------------------------------------------------------
# Grid-search optimizer
# ---------------------------------------------------------------------------

def grid_search(
    corpus:   list[dict],
    theta:    float,
    steps:    int = 5,        # granularity per dimension (coarse)
    lam:      float = SIGMOID_LAMBDA,
    verbose:  bool = True,
) -> tuple[dict[str, float], float]:
    """
    Exhaustive grid search over the simplex {w | Σw_j=1, w_j≥0}.

    With steps=5 and 6 dimensions, samples ~252 valid weight vectors.

    Returns:
        (best_weights, best_f1)
    """
    keys       = ["w1", "w2", "w3", "w4", "w5", "w6"]
    step_size  = 1.0 / steps
    grid_vals  = [round(i * step_size, 2) for i in range(steps + 1)]

    best_f1      = -1.0
    best_weights = None
    evaluated    = 0

    # Enumerate all combinations that sum to 1
    for combo in product(grid_vals, repeat=6):
        total = sum(combo)
        if abs(total - 1.0) > 0.01:
            continue

        # Normalise to exactly sum to 1
        w = {k: v / total for k, v in zip(keys, combo)}
        # w6 is subtracted → keep it small
        if w["w6"] > 0.30:
            continue

        metrics  = compute_metrics_for_weights(corpus, w, theta, lam)
        evaluated += 1

        if metrics["f1"] > best_f1:
            best_f1      = metrics["f1"]
            best_weights = w
            if verbose:
                print(f"  [+] New best  F1={best_f1:.4f}  "
                      f"loss={metrics['loss']:.4f}  w={_fmt_w(w)}")

    if verbose:
        print(f"\n  Grid search: {evaluated} valid weight vectors evaluated.")

    return best_weights, best_f1


def _fmt_w(w: dict) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in w.values()) + "]"


# ---------------------------------------------------------------------------
# SciPy SLSQP refinement (optional)
# ---------------------------------------------------------------------------

def slsqp_refine(
    corpus:        list[dict],
    theta:         float,
    initial_w:     dict[str, float],
    lam:           float = SIGMOID_LAMBDA,
    verbose:       bool = True,
) -> tuple[dict[str, float], float]:
    """
    Local gradient refinement starting from the grid-search solution.
    Uses scipy.optimize.minimize with the SLSQP method.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        if verbose:
            print("  scipy not available – skipping SLSQP refinement.")
        return initial_w, -1.0

    keys = ["w1", "w2", "w3", "w4", "w5", "w6"]
    x0   = [initial_w[k] for k in keys]

    def objective(x):
        w = dict(zip(keys, x))
        m = compute_metrics_for_weights(corpus, w, theta, lam)
        return m["loss"]   # minimise loss ≡ maximise F1

    constraints = [{"type": "eq",   "fun": lambda x: sum(x) - 1.0}]
    bounds      = [(0.0, 0.5)] * 5 + [(0.0, 0.30)]   # w6 capped at 0.30

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if result.success:
        w_star = dict(zip(keys, result.x.tolist()))
        metrics = compute_metrics_for_weights(corpus, w_star, theta, lam)
        if verbose:
            print(f"  SLSQP converged: F1={metrics['f1']:.4f}  loss={result.fun:.6f}")
        return w_star, metrics["f1"]
    else:
        if verbose:
            print(f"  SLSQP did not converge: {result.message}")
        return initial_w, -1.0


# ---------------------------------------------------------------------------
# Corpus utilities
# ---------------------------------------------------------------------------

def load_corpus(path: Path) -> list[dict]:
    """Load the gold-standard annotated corpus from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Gold corpus not found at '{path}'.\n"
            "Run  scripts/build_gold_corpus.py  or provide --corpus <path>."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "sentences" in data:
        # Flatten token-level entries
        tokens = []
        for sent in data["sentences"]:
            for tok in sent.get("tokens", []):
                if "features" in tok and "label" in tok:
                    tokens.append(tok)
        return tokens
    return data   # assume it's already a flat list of token entries


def save_weights(w: dict[str, float], output_target: Path,
                 register: str, f1: float) -> Path:
    if output_target.suffix.lower() == ".json":
        filename = output_target
        output_dir = output_target.parent
    else:
        output_dir = output_target
        filename = output_dir / f"w_star_{register}.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    payload  = {
        "register":          register,
        "optimised_weights": w,
        "achieved_f1":       round(f1, 6),
        "description": (
            "Optimised LAPS weight vector w* for the specified register. "
            "Produced by scripts/optimize_weights.py via Grid Search + SLSQP."
        ),
        "weight_semantics": {
            "w1": "Borrowing Frequency (BF)",
            "w2": "Domain Specificity (DS)",
            "w3": "Semantic Importance (SI)",
            "w4": "Context Consistency (CC)",
            "w5": "Audience Familiarity (AF)",
            "w6": "Translation Benefit (TB) [subtracted]",
        },
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filename


# ---------------------------------------------------------------------------
# Synthetic corpus generator (when no real corpus exists yet)
# ---------------------------------------------------------------------------

def generate_synthetic_corpus(n: int = 200) -> list[dict]:
    """
    Generate a plausible synthetic training corpus with heuristic labels.
    This is a stand-in until the real LinCE / GLUECoS annotated data is loaded.

    Tokens with BF > 0.60 and DS > 0.60 are labelled PRESERVE (label=1).
    """
    import random
    random.seed(42)

    corpus = []
    # Prototypical PRESERVE tokens (high BF, DS, known loanwords)
    preserve_prototypes = [
        {"BF": 0.92, "DS": 0.90, "SI": 0.75, "CC": 0.70, "AF": 0.90, "TB": 0.15},
        {"BF": 0.85, "DS": 0.85, "SI": 0.70, "CC": 0.65, "AF": 0.85, "TB": 0.20},
        {"BF": 0.78, "DS": 0.80, "SI": 0.65, "CC": 0.60, "AF": 0.80, "TB": 0.25},
        {"BF": 0.88, "DS": 0.90, "SI": 0.80, "CC": 0.72, "AF": 0.88, "TB": 0.10},
    ]
    # Prototypical REWRITE tokens (low BF, low DS, easily translateable)
    rewrite_prototypes = [
        {"BF": 0.15, "DS": 0.10, "SI": 0.20, "CC": 0.30, "AF": 0.20, "TB": 0.90},
        {"BF": 0.20, "DS": 0.15, "SI": 0.25, "CC": 0.25, "AF": 0.25, "TB": 0.85},
        {"BF": 0.10, "DS": 0.08, "SI": 0.15, "CC": 0.20, "AF": 0.15, "TB": 0.92},
        {"BF": 0.25, "DS": 0.20, "SI": 0.30, "CC": 0.35, "AF": 0.30, "TB": 0.80},
    ]

    half = n // 2
    for _ in range(half):
        base = random.choice(preserve_prototypes)
        jittered = {k: max(0.0, min(1.0, v + random.gauss(0, 0.05)))
                    for k, v in base.items()}
        corpus.append({
            "token":    "loanword",
            "features": jittered,
            "label":    1,
        })

    for _ in range(n - half):
        base = random.choice(rewrite_prototypes)
        jittered = {k: max(0.0, min(1.0, v + random.gauss(0, 0.05)))
                    for k, v in base.items()}
        corpus.append({
            "token":    "native_word",
            "features": jittered,
            "label":    0,
        })

    random.shuffle(corpus)
    return corpus


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LAPS Automated Weight Vector Optimizer (w*)"
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help="Path to gold-standard annotated corpus JSON"
    )
    parser.add_argument(
        "--register", type=str, default="business",
        choices=list(REGISTER_THRESHOLDS.keys()),
        help="Target communicative register for θ_R"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for w_star JSON"
    )
    parser.add_argument(
        "--steps", type=int, default=5,
        help="Grid search granularity (steps per dimension)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic corpus (no annotation file required)"
    )
    parser.add_argument(
        "--skip-slsqp", action="store_true",
        help="Skip scipy SLSQP refinement"
    )
    args = parser.parse_args()

    theta = REGISTER_THRESHOLDS[args.register]

    print("=" * 60)
    print("  LAPS Weight Vector Optimizer (w*)")
    print("=" * 60)
    print(f"  Register  : {args.register}  (theta={theta})")

    # Load or generate corpus
    if args.synthetic or not args.corpus.exists():
        print("  Corpus    : SYNTHETIC (200 tokens, heuristic labels)")
        corpus = generate_synthetic_corpus(200)
    else:
        corpus = load_corpus(args.corpus)
        print(f"  Corpus    : {args.corpus}  ({len(corpus)} token entries)")

    print(f"  Grid steps: {args.steps}")
    print()

    # Phase 1: Grid Search
    print("-- Phase 1: Grid Search --")
    best_w, best_f1 = grid_search(corpus, theta, steps=args.steps, verbose=True)
    print(f"\n  Best grid F1 = {best_f1:.4f}")
    print(f"  Best grid w* = {_fmt_w(best_w)}")

    # Phase 2: SLSQP Local Refinement
    if not args.skip_slsqp:
        print("\n-- Phase 2: SLSQP Refinement --")
        refined_w, refined_f1 = slsqp_refine(corpus, theta, best_w, verbose=True)
        if refined_f1 > best_f1:
            best_w, best_f1 = refined_w, refined_f1
            print(f"  Refined   F1 = {best_f1:.4f}")
            print(f"  Refined   w* = {_fmt_w(best_w)}")
        else:
            print("  SLSQP did not improve over grid search; keeping grid solution.")

    # Save
    out_path = save_weights(best_w, args.output, args.register, best_f1)
    print(f"\n  Saved optimised weights -> {out_path}")

    # Pretty summary table
    print("\n=== Optimal Weight Vector w* ===")
    names = {
        "w1": "Borrowing Frequency (BF)",
        "w2": "Domain Specificity (DS)",
        "w3": "Semantic Importance (SI)",
        "w4": "Context Consistency (CC)",
        "w5": "Audience Familiarity (AF)",
        "w6": "Translation Benefit (TB)",
    }
    for k, label in names.items():
        bar = "=" * int(best_w[k] * 30)
        print(f"  {k}  {label:<30} {best_w[k]:.4f}  {bar}")

    print(f"\n  Final F1  = {best_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
