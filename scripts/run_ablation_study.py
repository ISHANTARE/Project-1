"""
LAPS Comprehensive Feature & Threshold Ablation Study Runner
------------------------------------------------------------
Systematically quantifies the marginal contribution of each feature vector
and architectural component:
  1. Full LAPS (All 6 Features + Soft Sigmoid)
  2. Ablation: Without Borrowing Frequency (-BF)
  3. Ablation: Without Domain Specificity (-DS)
  4. Ablation: Without Semantic Importance (-SI)
  5. Ablation: Without Context Consistency (-CC)
  6. Ablation: Without Audience Familiarity (-AF)
  7. Ablation: Without Translation Benefit (-TB)
  8. Ablation: Hard Step Threshold (replacing Soft Sigmoid)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laps.scorer           import DEFAULT_WEIGHTS
from laps.threshold_engine import compute_sigmoid_probability, REGISTER_THRESHOLDS

CORPUS_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_corpus.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "ablation_results.json"


def compute_f1(preds: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return round(2 * prec * rec / (prec + rec + 1e-9), 4)


def score_custom(feats: Dict[str, float], active_keys: List[str], weights: Dict[str, float]) -> float:
    # Filter features
    f_active = {k: feats[k] for k in active_keys}
    w_active = {k: weights.get(f"w{i+1}", 0.20) for i, k in enumerate(active_keys)}
    total_w = sum(w_active.values()) or 1.0
    
    score = 0.0
    for k in active_keys:
        val = f_active[k]
        w = w_active[k] / total_w
        if k == "TB":
            score -= w * val
        else:
            score += w * val
    return max(0.0, min(1.0, score))


def evaluate_ablation(tokens: List[Dict[str, Any]], active_features: List[str], use_sigmoid: bool = True) -> Dict[str, float]:
    preds = []
    labels = []
    theta = 0.50

    for t in tokens:
        feats = t["features"]
        label = t["label"]
        score = score_custom(feats, active_features, DEFAULT_WEIGHTS)
        
        if use_sigmoid:
            prob = compute_sigmoid_probability(score, theta)
            pred = 1 if prob >= 0.50 else 0
        else:
            # Hard cutoff
            pred = 1 if score >= theta else 0
            
        preds.append(pred)
        labels.append(label)

    f1 = compute_f1(preds, labels)
    return {"f1": f1, "accuracy": round(sum(1 for p, l in zip(preds, labels) if p == l) / len(labels), 4)}


def run_ablation_matrix():
    with open(CORPUS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    sentences = data.get("sentences", data) if isinstance(data, dict) else data
    
    tokens = []
    for s in sentences:
        tokens.extend(s.get("tokens", []))

    all_feats = ["BF", "DS", "SI", "CC", "AF", "TB"]

    ablations = {
        "Full LAPS (All 6 Features + Soft Sigmoid)": (all_feats, True),
        "Ablation: Without BF (-BF)":                ([f for f in all_feats if f != "BF"], True),
        "Ablation: Without DS (-DS)":                ([f for f in all_feats if f != "DS"], True),
        "Ablation: Without SI (-SI)":                ([f for f in all_feats if f != "SI"], True),
        "Ablation: Without CC (-CC)":                ([f for f in all_feats if f != "CC"], True),
        "Ablation: Without AF (-AF)":                ([f for f in all_feats if f != "AF"], True),
        "Ablation: Without TB (-TB)":                ([f for f in all_feats if f != "TB"], True),
        "Ablation: Hard Step Threshold (No Sigmoid)": (all_feats, False),
    }

    results = {}
    print("\n=======================================================")
    print(f"  LAPS Comprehensive Feature Ablation Matrix (N={len(tokens)})")
    print("=======================================================\n")

    headers = ["Configuration", "Token F1", "Accuracy", "Delta F1 (vs Full)"]
    rows = []

    full_f1 = evaluate_ablation(tokens, all_feats, True)["f1"]

    for name, (feats, sig) in ablations.items():
        eval_res = evaluate_ablation(tokens, feats, sig)
        delta = eval_res["f1"] - full_f1
        delta_str = "0.0000" if abs(delta) < 1e-6 else f"{delta:+.4f}"
        results[name] = eval_res
        rows.append([name, f"{eval_res['f1']:.4f}", f"{eval_res['accuracy']:.4f}", delta_str])

    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in col_w)
    print(sep)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, col_w)))
    print(sep + "\n")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved -> {RESULTS_PATH}\n")


if __name__ == "__main__":
    run_ablation_matrix()
