"""
LAPS Comprehensive Feature & Architectural Component Ablation Study Runner
--------------------------------------------------------------------------
Systematically quantifies the marginal contribution of each feature vector
and architectural component under rigorous research standards:

1. Canonical Feature Weight Mapping:
   Each feature maintains its dedicated weight parameter (w1..w6) from DEFAULT_WEIGHTS.
   When a feature is ablated, its weight is set to 0.0 and remaining weights are normalized to sum to 1.0.

2. Register-Aware Thresholding:
   Evaluates against register-specific operational thresholds:
     - academic: 0.40
     - business: 0.45
     - social:   0.50
     - casual:   0.55

3. Architectural Component Ablations:
   - Full LAPS (All 6 Features + Soft Sigmoid + KG Enrichment)
   - Feature Ablations: -BF, -DS, -SI, -CC, -AF, -TB
   - Architectural Ablations:
     - Without Register Thresholds (Fixed global theta = 0.50)
     - Without Soft Sigmoid (Hard Step Threshold, no uncertainty buffer)
     - Without Knowledge Graph (Ego-density bonus / naturalness penalty disabled)

Usage:
  python scripts/run_ablation_study.py --corpus data/evaluation/gold_corpus.json
  python scripts/run_ablation_study.py --corpus data/evaluation/val_split.json
  python scripts/run_ablation_study.py --corpus data/evaluation/test_split.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laps.scorer import DEFAULT_WEIGHTS
from laps.threshold_engine import compute_sigmoid_probability, REGISTER_THRESHOLDS, batch_decide
from laps.feature_extractor import extract_features

CORPUS_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_corpus.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "ablation_results_audited.json"

FEATURE_WEIGHT_KEYS = {
    "BF": "w1",
    "DS": "w2",
    "SI": "w3",
    "CC": "w4",
    "AF": "w5",
    "TB": "w6",
}


def compute_token_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    total = len(labels)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def score_custom(
    feats: Dict[str, float],
    active_features: List[str],
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> float:
    """Computes preservation score with dedicated feature weight mapping."""
    active_set = set(active_features)
    raw_weights = {
        feat: (weights.get(FEATURE_WEIGHT_KEYS[feat], 0.20) if feat in active_set else 0.0)
        for feat in FEATURE_WEIGHT_KEYS
    }
    total_w = sum(raw_weights.values()) or 1.0
    norm_w = {k: v / total_w for k, v in raw_weights.items()}

    score = 0.0
    for feat in ["BF", "DS", "SI", "CC", "AF"]:
        score += norm_w[feat] * feats.get(feat, 0.0)
    score -= norm_w["TB"] * feats.get("TB", 0.0)

    return max(0.0, min(1.0, score))


def evaluate_ablation_on_corpus(
    sentences: List[Dict[str, Any]],
    active_features: List[str],
    use_register_thresholds: bool = True,
    use_sigmoid_uncertainty: bool = True,
    use_kg: bool = True,
) -> Dict[str, Any]:
    preds = []
    labels = []

    for sent in sentences:
        reg = sent.get("register", "business")
        theta = REGISTER_THRESHOLDS.get(reg, 0.45) if use_register_thresholds else 0.50

        surface_tokens = [t["token"] for t in sent.get("tokens", [])]

        for idx, tok in enumerate(sent.get("tokens", [])):
            if "label" not in tok:
                continue

            feats = tok.get("features")
            if not feats:
                feats = extract_features(tok["token"], surface_tokens, idx)

            score = score_custom(feats, active_features)

            # KG ego-density bonus (if enabled)
            if use_kg and feats.get("DS", 0.0) >= 0.70:
                score = min(1.0, score + 0.03)

            if use_sigmoid_uncertainty:
                prob = compute_sigmoid_probability(score, theta)
                # In LAPS, both PRESERVE (prob >= theta) and UNCERTAIN (|prob - theta| < 0.05) are preserved
                is_preserve = (prob >= theta) or (abs(prob - theta) < 0.05)
                pred = 1 if is_preserve else 0
            else:
                # Hard cutoff without uncertainty boundary
                pred = 1 if score >= theta else 0

            preds.append(pred)
            labels.append(tok["label"])

    metrics = compute_token_metrics(preds, labels)
    return metrics


def run_ablation_study(corpus_path: Path, output_path: Path) -> Dict[str, Any]:
    with open(corpus_path, encoding="utf-8") as f:
        data = json.load(f)
    sentences = data.get("sentences", data) if isinstance(data, dict) else data

    total_tokens = sum(len(s.get("tokens", [])) for s in sentences)
    candidate_tokens = sum(1 for s in sentences for t in s.get("tokens", []) if "label" in t)

    all_feats = ["BF", "DS", "SI", "CC", "AF", "TB"]

    ablation_configs = {
        "Full LAPS (All 6 Features + Soft Sigmoid + KG)": {
            "features": all_feats, "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without BF (-BF)": {
            "features": [f for f in all_feats if f != "BF"], "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without DS (-DS)": {
            "features": [f for f in all_feats if f != "DS"], "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without SI (-SI)": {
            "features": [f for f in all_feats if f != "SI"], "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without CC (-CC)": {
            "features": [f for f in all_feats if f != "CC"], "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without AF (-AF)": {
            "features": [f for f in all_feats if f != "AF"], "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without TB (-TB)": {
            "features": [f for f in all_feats if f != "TB"], "reg": True, "sig": True, "kg": True,
        },
        "Ablation: Without Register Thresholds (Fixed theta=0.50)": {
            "features": all_feats, "reg": False, "sig": True, "kg": True,
        },
        "Ablation: Without Soft Sigmoid (Hard Cutoff, No Uncertainty)": {
            "features": all_feats, "reg": True, "sig": False, "kg": True,
        },
        "Ablation: Without Knowledge Graph (No Ego Density Bonus)": {
            "features": all_feats, "reg": True, "sig": True, "kg": False,
        },
    }

    full_res = evaluate_ablation_on_corpus(
        sentences, all_feats, use_register_thresholds=True, use_sigmoid_uncertainty=True, use_kg=True
    )
    full_f1 = full_res["f1"]

    results: Dict[str, Any] = {}
    table_rows = []

    print("\n" + "=" * 80)
    print(f"  LAPS Audited Feature & Component Ablation Matrix (N_cand={candidate_tokens})")
    print(f"  Corpus: {corpus_path.name}")
    print("=" * 80)

    for name, cfg in ablation_configs.items():
        res = evaluate_ablation_on_corpus(
            sentences,
            cfg["features"],
            use_register_thresholds=cfg["reg"],
            use_sigmoid_uncertainty=cfg["sig"],
            use_kg=cfg["kg"],
        )
        delta_f1 = res["f1"] - full_f1
        res["delta_f1"] = round(delta_f1, 4)
        results[name] = res

        delta_str = "0.0000" if abs(delta_f1) < 1e-6 else f"{delta_f1:+.4f}"
        table_rows.append([
            name[:58],
            f"{res['f1']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['accuracy']:.4f}",
            delta_str,
        ])

    headers = ["Configuration", "Token F1", "Precision", "Recall", "Accuracy", "Delta F1"]
    col_w = [max(len(h), max(len(r[i]) for r in table_rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in col_w)

    print(sep)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)
    for r in table_rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, col_w)))
    print(sep + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "corpus": str(corpus_path),
            "n_sentences": len(sentences),
            "n_candidate_tokens": candidate_tokens,
            "full_f1": full_f1,
            "ablations": results,
        }, f, indent=2)
    print(f"Audited ablation results saved -> {output_path}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="LAPS Audited Ablation Study")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    run_ablation_study(args.corpus, args.output)


if __name__ == "__main__":
    main()
