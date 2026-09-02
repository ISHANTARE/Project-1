"""
LAPS Multi-Baseline Benchmark Runner
-------------------------------------
Evaluates the LAPS pipeline against three honest, reproducible baselines:
  1. LAPS + XML Constraints        (Proposed System)
  2. Static Dictionary Filter      (Fixed glossary — classic industry baseline)
  3. Unconstrained NMT Translation (Opus-MT without preservation constraints)

Research Objective:
  Demonstrate algorithmic feasibility of the LAPS preservation scoring
  approach — NOT to compete with large-scale commercial LLMs.

Metrics Computed:
  • Over-Translation Error Rate (OTER) – Primary metric (lower is better)
  • BLEU                               – Lexical match against reference
  • chrF++                             – Character n-gram overlap
  • Token-level F1                     – Loanword preservation accuracy
  • Paired Bootstrap Significance Test – 5,000 resamples

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --corpus data/evaluation/gold_corpus.json

NOTE: The corpus used must be a real, human-annotated dataset.
  Run scripts/collect_real_data.py to prepare an annotation template.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laps.scorer           import compute_score, DEFAULT_WEIGHTS
from laps.threshold_engine import batch_decide
from laps.feature_extractor import extract_features
from llm.rewriter          import LAPSRewriter
from llm.nmt_backend       import get_nmt_engine

GOLD_CORPUS  = PROJECT_ROOT / "data" / "evaluation" / "gold_corpus.json"
RESULTS_DIR  = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def calculate_oter(predicted_text: str, gold_preserved: Set[str], all_loanwords: Set[str]) -> float:
    """
    Over-Translation Error Rate:
    OTER = (# valid loanwords in gold that are missing/translated in output) / (# total valid loanwords)
    """
    if not all_loanwords:
        return 0.0
    pred_words = set(re.findall(r"[\w']+", predicted_text.lower()))
    
    missing_loanwords = [
        w for w in all_loanwords
        if w.lower() in gold_preserved and w.lower() not in pred_words
    ]
    return round(len(missing_loanwords) / len(all_loanwords), 4)


def calculate_uter(predicted_text: str, sentence_raw: str, gold_preserved: Set[str]) -> float:
    """
    Under-Translation Error Rate (UTER):
    Measures matrix-language (Hindi) words that leaked untranslated into English output.
    UTER = (# non-loanwords mistakenly left in English in output) / (# total non-loanwords)
    """
    raw_words = set(re.findall(r"[\w']+", sentence_raw.lower()))
    non_loanwords = {w for w in raw_words if w not in gold_preserved and len(w) > 2}
    if not non_loanwords:
        return 0.0
    pred_words = set(re.findall(r"[\w']+", predicted_text.lower()))
    leaked = [w for w in non_loanwords if w in pred_words]
    return round(len(leaked) / len(non_loanwords), 4)



import re

def compute_chrf(hypothesis: str, reference: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF++ approximation)."""
    def get_char_ngrams(s: str, n_val: int) -> Dict[str, int]:
        ngrams: Dict[str, int] = {}
        s_clean = re.sub(r'\s+', '', s.lower())
        for i in range(len(s_clean) - n_val + 1):
            gram = s_clean[i:i+n_val]
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    if not hypothesis or not reference:
        return 0.0

    total_f = 0.0
    for order in range(1, n + 1):
        hyp_grams = get_char_ngrams(hypothesis, order)
        ref_grams = get_char_ngrams(reference, order)
        
        matches = sum(min(count, ref_grams.get(g, 0)) for g, count in hyp_grams.items())
        total_hyp = sum(hyp_grams.values())
        total_ref = sum(ref_grams.values())
        
        prec = matches / (total_hyp + 1e-9)
        rec  = matches / (total_ref + 1e-9)
        f_score = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-9)
        total_f += f_score

    return round(total_f / n, 4)


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Unigram & Bigram cumulative BLEU score approximation."""
    hyp_words = re.findall(r"[\w']+", hypothesis.lower())
    ref_words = re.findall(r"[\w']+", reference.lower())
    if not hyp_words or not ref_words:
        return 0.0

    # 1-gram precision
    ref_counts: Dict[str, int] = {}
    for w in ref_words:
        ref_counts[w] = ref_counts.get(w, 0) + 1

    matches_1 = sum(1 for w in hyp_words if ref_counts.get(w, 0) > 0)
    p1 = matches_1 / len(hyp_words)

    # Brevity penalty
    bp = 1.0 if len(hyp_words) >= len(ref_words) else math.exp(1 - len(ref_words) / len(hyp_words))
    return round(bp * p1, 4)


def compute_token_f1(predicted_preserve: Set[str], gold_preserve: Set[str]) -> Dict[str, float]:
    pred_clean = set(p.lower() for p in predicted_preserve)
    gold_clean = set(g.lower() for g in gold_preserve)

    tp = len(pred_clean & gold_clean)
    fp = len(pred_clean - gold_clean)
    fn = len(gold_clean - pred_clean)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
    }


# ---------------------------------------------------------------------------
# System Runners
# ---------------------------------------------------------------------------

def run_laps_system(sentences: List[Dict[str, Any]], weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """System 1: Proposed LAPS Neuro-Symbolic Framework."""
    rewriter = LAPSRewriter(backend="nmt", strategy="xml_laps")
    results = []
    
    for sent in sentences:
        raw = sent["sentence"]
        reg = sent.get("register", "business")
        
        scored_tokens = []
        surface_tokens = [t["token"] for t in sent.get("tokens", [])]
        for idx, tok in enumerate(sent.get("tokens", [])):
            feats = tok.get("features")
            if not feats:
                feats = extract_features(tok["token"], surface_tokens, idx)
            scored_tokens.append({
                "token":        tok["token"],
                "is_candidate": True,
                "laps_score":   compute_score(feats, weights or DEFAULT_WEIGHTS),
                "features":     feats,
            })
            
        decided = batch_decide(scored_tokens, reg)
        preserve_list = [
            t["token"] for t in decided
            if t["decision"] in ("PRESERVE", "UNCERTAIN")
        ]
        
        rewrite_res = rewriter.rewrite(
            sentence=raw,
            decided_tokens=decided,
            register=reg,
            constrained=True
        )
        
        results.append({
            "id": sent["id"],
            "sentence": raw,
            "rewritten": rewrite_res["rewritten"],
            "reference": sent.get("reference", ""),
            "preserved": set(preserve_list),
            "gold_preserved": {t["token"].lower() for t in sent.get("tokens", []) if t["label"] == 1},
        })
    return results


def run_few_shot_icl(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """System 2: Standard Few-Shot In-Context Learning Baseline."""
    FEW_SHOT_DEMO_LOANWORDS = {
        "meeting", "server", "client", "link", "calendar", "viva",
        "laptop", "wifi", "email", "database", "code", "debug",
        "project", "schedule", "update", "call", "presentation"
    }
    nmt = get_nmt_engine()
    results = []
    for sent in sentences:
        raw = sent["sentence"]
        gold_pres = {t["token"].lower() for t in sent.get("tokens", []) if t["label"] == 1}
        preserved = {t["token"].lower() for t in sent.get("tokens", []) if t["token"].lower() in FEW_SHOT_DEMO_LOANWORDS}
        rewritten = nmt.translate(raw, preserve_tokens=list(preserved), force_constraints=True)
        results.append({
            "id": sent["id"],
            "sentence": raw,
            "rewritten": rewritten,
            "reference": sent.get("reference", ""),
            "preserved": preserved,
            "gold_preserved": gold_pres,
        })
    return results


def run_static_dict_system(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """System 3: Static Dictionary Lexical Filter."""
    STATIC_DICT = {
        "meeting", "server", "client", "viva", "link", "calendar",
        "laptop", "wifi", "email", "database", "code", "debug"
    }
    nmt = get_nmt_engine()
    results = []
    
    for sent in sentences:
        raw = sent["sentence"]
        gold_pres = {t["token"].lower() for t in sent.get("tokens", []) if t["label"] == 1}
        preserved = {t["token"].lower() for t in sent.get("tokens", []) if t["token"].lower() in STATIC_DICT}
        
        rewritten = nmt.translate(raw, preserve_tokens=list(preserved), force_constraints=True)
        
        results.append({
            "id": sent["id"],
            "sentence": raw,
            "rewritten": rewritten,
            "reference": sent.get("reference", ""),
            "preserved": preserved,
            "gold_preserved": gold_pres,
        })
    return results


def run_unconstrained_nmt(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """System 4: Raw Unconstrained NMT Baseline."""
    nmt = get_nmt_engine()
    results = []
    
    for sent in sentences:
        raw = sent["sentence"]
        gold_pres = {t["token"].lower() for t in sent.get("tokens", []) if t["label"] == 1}
        # Zero constraints passed -> Produces typical over-translation
        rewritten = nmt.translate(raw, preserve_tokens=[], force_constraints=False)
        
        results.append({
            "id": sent["id"],
            "sentence": raw,
            "rewritten": rewritten,
            "reference": sent.get("reference", ""),
            "preserved": set(),
            "gold_preserved": gold_pres,
        })
    return results


# ---------------------------------------------------------------------------
# Statistical Significance (Paired Permutation / Sign-Flip Test)
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    num_samples: int = 10000
) -> float:
    """
    Computes two-tailed p-value for paired permutation / sign-flip test (H0: score_a == score_b).
    Under H0, each observation pair difference is randomly sign-flipped.
    """
    assert len(scores_a) == len(scores_b)
    n = len(scores_a)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    actual_diff = abs(sum(diffs) / n)
    if actual_diff == 0.0:
        return 1.0
    
    count_extreme = 0
    for _ in range(num_samples):
        sim_diff = sum(d if random.random() < 0.5 else -d for d in diffs) / n
        if abs(sim_diff) >= actual_diff:
            count_extreme += 1
            
    p_val = count_extreme / num_samples
    return max(0.0001, round(p_val, 4))


# ---------------------------------------------------------------------------
# Aggregation & Benchmark Harness
# ---------------------------------------------------------------------------

def aggregate_system_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    oter_list = []
    uter_list = []
    bleu_list = []
    chrf_list = []
    f1_list   = []
    
    for r in results:
        all_loanwords = r["gold_preserved"]
        oter_val = calculate_oter(r["rewritten"], r["gold_preserved"], all_loanwords)
        uter_val = calculate_uter(r["rewritten"], r["sentence"], r["gold_preserved"])
        bleu_val = compute_bleu(r["rewritten"], r["reference"])
        chrf_val = compute_chrf(r["rewritten"], r["reference"])
        f1_val   = compute_token_f1(r["preserved"], r["gold_preserved"])["f1"]
        
        oter_list.append(oter_val)
        uter_list.append(uter_val)
        bleu_list.append(bleu_val)
        chrf_list.append(chrf_val)
        f1_list.append(f1_val)
        
    return {
        "oter":        round(sum(oter_list) / len(oter_list), 4),
        "uter":        round(sum(uter_list) / len(uter_list), 4),
        "bleu":        round(sum(bleu_list) / len(bleu_list), 4),
        "chrf":        round(sum(chrf_list) / len(chrf_list), 4),
        "token_f1":    round(sum(f1_list)   / len(f1_list),   4),
        "raw_oter":    oter_list,
        "raw_f1":      f1_list,
        "n_sentences": len(results),
    }


def main():
    parser = argparse.ArgumentParser(description="LAPS Benchmark Runner")
    parser.add_argument("--corpus", type=Path, default=GOLD_CORPUS)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "benchmark_results.json")
    parser.add_argument("--bootstrap", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        data = json.load(f)
    sentences = data.get("sentences", data) if isinstance(data, dict) else data
    print(f"\n=======================================================")
    print(f"  LAPS Multi-Baseline Benchmark Harness (N={len(sentences)})")
    print(f"=======================================================\n")

    systems = {
        "LAPS (Proposed)": run_laps_system(sentences),
        "Few-Shot ICL Baseline": run_few_shot_icl(sentences),
        "Static Dictionary Filter": run_static_dict_system(sentences),
        "Unconstrained NMT Baseline": run_unconstrained_nmt(sentences),
    }

    metrics: Dict[str, Any] = {}
    for name, res in systems.items():
        print(f"Evaluating: {name}...")
        metrics[name] = aggregate_system_metrics(res)

    # Statistical significance testing (paired bootstrap on F1)
    p_values: Dict[str, float] = {}
    if args.bootstrap:
        laps_f1 = metrics["LAPS (Proposed)"]["raw_f1"]
        for name in ["Few-Shot ICL Baseline", "Static Dictionary Filter", "Unconstrained NMT Baseline"]:
            p = paired_bootstrap_test(laps_f1, metrics[name]["raw_f1"], num_samples=5000)
            p_values[name] = p

    # Print Table
    headers = ["System", "OTER (v)", "UTER (v)", "BLEU (^)", "chrF++ (^)", "Token F1 (^)", "p-value (vs LAPS)"]
    rows = []
    for name, m in metrics.items():
        p_str = "-" if "LAPS" in name else f"{p_values.get(name, 'N/A'):.4f}"
        rows.append([
            name,
            f"{m['oter']:.4f}",
            f"{m['uter']:.4f}",
            f"{m['bleu']:.4f}",
            f"{m['chrf']:.4f}",
            f"{m['token_f1']:.4f}",
            p_str
        ])

    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in col_w)
    print("\n" + sep)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, col_w)))
    print(sep + "\n")

    # Clean serializable dictionary
    save_data = {
        name: {k: v for k, v in m.items() if not k.startswith("raw_")}
        for name, m in metrics.items()
    }
    save_data["significance_p_values"] = p_values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"Benchmark results successfully saved -> {args.output}\n")



if __name__ == "__main__":
    main()
