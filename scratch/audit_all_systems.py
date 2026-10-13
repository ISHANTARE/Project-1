import json
import sys
from pathlib import Path
import sacrebleu

PROJECT_ROOT = Path("d:/Project-1-main/Project-1-main")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.benchmark import (
    run_laps_system,
    run_few_shot_icl,
    run_static_dict_system,
    run_unconstrained_nmt,
    evaluate_output_preservation,
    paired_permutation_test
)

with open(PROJECT_ROOT / "data/evaluation/gold_corpus.json", encoding="utf-8") as f:
    sents = json.load(f)["sentences"]

systems = {
    "LAPS (Proposed)": run_laps_system(sents),
    "Few-Shot Vocabulary Filter (Demonstration Heuristic)": run_few_shot_icl(sents),
    "Static Dictionary Filter": run_static_dict_system(sents),
    "Unconstrained NMT Baseline": run_unconstrained_nmt(sents),
}

refs = [s["reference"] for s in sents]

results = {}

for name, res in systems.items():
    hyps = [r["rewritten"] for r in res]
    
    # Correct SacreBLEU: [refs] is a list containing 1 reference stream of N sentences
    c_bleu = sacrebleu.corpus_bleu(hyps, [refs])
    c_chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    s_bleus = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)]
    s_chrfs = [sacrebleu.sentence_chrf(h, [r], word_order=2).score for h, r in zip(hyps, refs)]
    
    # Output retention
    out_evals = [evaluate_output_preservation(r["rewritten"], r["tokens"]) for r in res]
    oters = [oe["oter"] for oe in out_evals]
    out_f1s = [oe["output_f1"] for oe in out_evals]
    
    results[name] = {
        "corpus_bleu": round(c_bleu.score, 2),
        "corpus_chrf": round(c_chrf.score, 2),
        "sentence_bleu_mean": round(sum(s_bleus)/len(s_bleus), 2),
        "sentence_chrf_mean": round(sum(s_chrfs)/len(s_chrfs), 2),
        "mean_oter": round(sum(oters)/len(oters), 4),
        "mean_output_f1": round(sum(out_f1s)/len(out_f1s), 4),
        "raw_output_f1": out_f1s,
        "raw_oter": oters,
    }

print("=== CORRECTED SACREBLEU & CHRF++ RESULTS ===")
for name, data in results.items():
    print(f"\nSystem: {name}")
    print(f"  Corpus BLEU: {data['corpus_bleu']} | Mean Sent BLEU: {data['sentence_bleu_mean']}")
    print(f"  Corpus chrF++: {data['corpus_chrf']} | Mean Sent chrF++: {data['sentence_chrf_mean']}")
    print(f"  Mean OTER: {data['mean_oter']} | Mean Output F1: {data['mean_output_f1']}")

# Statistical testing
print("\n=== PAIRED PERMUTATION TESTS ON OUTPUT F1 ===")
laps_f1 = results["LAPS (Proposed)"]["raw_output_f1"]
for name in [
    "Few-Shot Vocabulary Filter (Demonstration Heuristic)",
    "Static Dictionary Filter",
    "Unconstrained NMT Baseline"
]:
    pt = paired_permutation_test(laps_f1, results[name]["raw_output_f1"], num_samples=5000, seed=42)
    print(f"LAPS vs {name}:")
    print(f"  Delta: {pt['delta']} (95% CI: [{pt['ci_lower']}, {pt['ci_upper']}])")
    print(f"  p-value: {pt['p_value']}")
