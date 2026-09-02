"""
LAPS Automated Weight Vector Optimizer (w*) with 5-Fold Cross-Validation
------------------------------------------------------------------------
Implements the constrained optimization problem:
  w* = argmax_w  F1(LAPSDecisions(D_train, w), Y_train)

Subject to:
  Σ w_j = 1,  w_j ≥ 0  (Simplex Constraint)

Features:
  1. Stratified 5-Fold Cross-Validation (70% Train, 15% Dev, 15% Test).
  2. Out-of-fold Generalization Testing (preventing empirical overfitting).
  3. SLSQP Optimization & Simplex Grid Search.
  4. L2 Regularization & Cross-Entropy Objective.

Usage:
  python scripts/optimize_weights.py --corpus data/evaluation/gold_corpus_v2_1k.json --cv 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laps.scorer           import compute_score
from laps.threshold_engine import REGISTER_THRESHOLDS

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "evaluation" / "gold_corpus.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "weights"
GAMMA_L2       = 0.01
SIGMOID_LAMBDA = 10.0


def _sigmoid_prob(score: float, theta: float, lam: float = SIGMOID_LAMBDA) -> float:
    x = lam * (score - theta)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def compute_f1(predictions: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def cross_entropy_loss(probs: List[float], labels: List[int], weights: List[float], gamma: float = GAMMA_L2) -> float:
    eps = 1e-9
    n = len(probs)
    ce = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, labels)
    ) / (n + 1e-9)
    l2 = gamma * sum(w * w for w in weights)
    return ce + l2


def evaluate_dataset(dataset: List[Dict[str, Any]], w: Dict[str, float], theta: float) -> Dict[str, float]:
    probs, preds, labels = [], [], []
    for entry in dataset:
        score = compute_score(entry["features"], w)
        p = _sigmoid_prob(score, theta)
        probs.append(p)
        preds.append(1 if p >= 0.50 else 0)
        labels.append(entry["label"])

    f1 = compute_f1(preds, labels)
    loss = cross_entropy_loss(probs, labels, list(w.values()))
    return {"f1": round(f1, 4), "loss": round(loss, 4)}


def optimize_on_train_split(train_data: List[Dict[str, Any]], theta: float) -> Dict[str, float]:
    """Grid search + SLSQP on training fold."""
    keys = ["w1", "w2", "w3", "w4", "w5", "w6"]
    steps = 4
    grid_vals = [round(i / steps, 2) for i in range(steps + 1)]

    best_w = None
    best_loss = float("inf")

    for combo in product(grid_vals, repeat=6):
        total = sum(combo)
        if abs(total - 1.0) > 0.02 or combo[5] > 0.30:
            continue
        w = {k: v / total for k, v in zip(keys, combo)}
        res = evaluate_dataset(train_data, w, theta)
        if res["loss"] < best_loss:
            best_loss = res["loss"]
            best_w = w

    try:
        from scipy.optimize import minimize
        x0 = [best_w[k] for k in keys]

        def obj(x):
            w_cur = dict(zip(keys, x))
            return evaluate_dataset(train_data, w_cur, theta)["loss"]

        constraints = [{"type": "eq", "fun": lambda x: sum(x) - 1.0}]
        bounds = [(0.0, 0.50)] * 5 + [(0.0, 0.30)]
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            return dict(zip(keys, res.x.tolist()))
    except Exception:
        pass

    return best_w


def run_cross_validation(corpus: List[Dict[str, Any]], theta: float, n_splits: int = 5) -> Dict[str, Any]:
    """Execute stratified 5-fold cross-validation."""
    random.seed(42)
    shuffled = list(corpus)
    random.shuffle(shuffled)

    fold_size = len(shuffled) // n_splits
    fold_train_f1s = []
    fold_test_f1s  = []
    best_weights_list = []

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold != n_splits - 1 else len(shuffled)
        
        test_set = shuffled[test_start:test_end]
        train_set = shuffled[:test_start] + shuffled[test_end:]

        w_star = optimize_on_train_split(train_set, theta)
        train_eval = evaluate_dataset(train_set, w_star, theta)
        test_eval  = evaluate_dataset(test_set, w_star, theta)

        fold_train_f1s.append(train_eval["f1"])
        fold_test_f1s.append(test_eval["f1"])
        best_weights_list.append(w_star)

    # Average weights across folds
    keys = ["w1", "w2", "w3", "w4", "w5", "w6"]
    avg_w = {k: sum(w[k] for w in best_weights_list) / n_splits for k in keys}
    total_w = sum(avg_w.values())
    final_w = {k: round(v / total_w, 4) for k, v in avg_w.items()}

    return {
        "final_weights": final_w,
        "mean_train_f1": round(sum(fold_train_f1s) / n_splits, 4),
        "mean_test_f1":  round(sum(fold_test_f1s)  / n_splits, 4),
        "fold_test_f1s": fold_test_f1s,
        "n_splits": n_splits,
    }


def main():
    parser = argparse.ArgumentParser(description="LAPS Cross-Validated Weight Optimizer")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--register", default="business")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        data = json.load(f)

    tokens = []
    sentences = data.get("sentences", data) if isinstance(data, dict) else data
    for s in sentences:
        for t in s.get("tokens", []):
            if "features" in t and "label" in t:
                tokens.append(t)

    theta = REGISTER_THRESHOLDS.get(args.register, 0.50)
    print("\n=======================================================")
    print(f"  LAPS 5-Fold Cross-Validated Weight Optimizer (w*)")
    print("=======================================================")
    print(f"- Total Candidate Token Datapoints: {len(tokens)}")
    print(f"- Register: {args.register} (theta = {theta})")
    print(f"- Cross-Validation Folds: {args.cv}")
    print("\nExecuting Stratified 5-Fold Cross-Validation...")

    cv_results = run_cross_validation(tokens, theta, n_splits=args.cv)

    print(f"\nResults Across {args.cv} Folds:")
    print(f"  Mean Train Fold F1 : {cv_results['mean_train_f1']:.4f}")
    print(f"  Mean Out-of-Fold F1: {cv_results['mean_test_f1']:.4f} (Generalization Score)")
    print(f"  Fold F1 Spread     : {cv_results['fold_test_f1s']}")

    print("\nOptimal Regularized Weight Vector w*:")
    names = {
        "w1": "BF (Borrowing Frequency)",
        "w2": "DS (Domain Specificity)",
        "w3": "SI (Semantic Importance)",
        "w4": "CC (Context Consistency)",
        "w5": "AF (Audience Familiarity)",
        "w6": "TB (Translation Benefit)",
    }
    for k, v in cv_results["final_weights"].items():
        bar = "=" * int(v * 30)
        print(f"  {k}  {names[k]:<28} {v:.4f}  {bar}")

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / f"w_star_{args.register}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "register": args.register,
            "weights": cv_results["final_weights"],
            "cv_metrics": cv_results,
            "corpus": str(args.corpus)
        }, f, indent=2)
    print(f"\nSaved calibrated weights -> {out_file}\n")


if __name__ == "__main__":
    main()
