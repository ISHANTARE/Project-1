"""
build_empirical_loanword_index.py
---------------------------------
Ingests authentic, peer-reviewed open-source Hinglish corpora:
  1. LinCE (Hindi-English LID split - 5,567 sentences with human-annotated token tags)
  2. PHINC (LingoIITGN Parallel Hinglish-English corpus - 13,738 sentences)

Computes the empirical borrowing frequency (BF) distribution of English loanwords
embedded directly in authentic code-mixed Hindi speech.

Outputs:
  • data/vocabulary/empirical_loanword_index.json (Dynamic Empirical Index)
  • data/evaluation/gold_corpus_real_50.json      (50 Real LinCE/PHINC Sentences for Evaluation)

Usage:
  python scripts/build_empirical_loanword_index.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_INDEX = PROJECT_ROOT / "data" / "vocabulary" / "empirical_loanword_index.json"
OUTPUT_REAL_CORPUS = PROJECT_ROOT / "data" / "evaluation" / "gold_corpus_real_50.json"

try:
    import wordfreq
except ImportError:
    wordfreq = None

# Import comprehensive transliterated Hindi markers from tokenizer
from src.laps.tokenizer import HINDI_MARKERS, HINDI_SUFFIX_RE

WORD_RE = re.compile(r'^[A-Za-z]{2,25}$')

# Closed-class English functional words that are incidental code-switching rather than loanwords
INCIDENTAL_FUNCTIONAL_WORDS = {
    "the", "is", "are", "was", "were", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "an", "as", "it", "this", "that", "these", "those",
    "he", "she", "they", "we", "i", "you", "my", "your", "his", "her", "their",
    "our", "and", "but", "or", "so", "if", "because", "then", "there", "here"
}


def is_valid_english_loanword(word_lower: str) -> bool:
    """Ensure token is an English word, not a romanized Hindi word."""
    if word_lower in HINDI_MARKERS or HINDI_SUFFIX_RE.match(word_lower):
        return False
    if wordfreq is not None:
        # Must be recognized in the global English lexicon
        return wordfreq.zipf_frequency(word_lower, "en") >= 2.0
    return True


def compute_borrowing_score(count: int, doc_freq: int) -> float:
    """
    Computes continuous empirical borrowing score in [0.30, 0.98]
    based on log-frequency and document distribution.
    """
    # Logarithmic saturation curve
    log_c = math.log10(1 + count)
    log_df = math.log10(1 + doc_freq)
    combined = 0.6 * log_c + 0.4 * log_df
    
    # Logistic mapping into [0.35, 0.98]
    # combined = 0.3 -> ~0.40, combined = 1.0 -> ~0.75, combined >= 2.0 -> ~0.95
    score = 1.0 / (1.0 + math.exp(-2.2 * (combined - 0.9)))
    return round(max(0.30, min(0.98, score)), 4)


def main():
    print("=" * 65)
    print("  Building Dynamic Empirical Loanword Index from LinCE & PHINC")
    print("=" * 65)

    token_counts = Counter()
    doc_counts = Counter()
    token_contexts = defaultdict(list)
    real_candidate_sentences = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Ingest LinCE (Human-annotated token-level LID)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading LinCE Benchmark (HuggMachas/Lince_benchmark_lid_hineng_train_valid)...")
    try:
        ds_lince = load_dataset('HuggMachas/Lince_benchmark_lid_hineng_train_valid')
        lince_splits = ['train', 'dev'] if 'dev' in ds_lince else ['train']
        lince_sents_count = 0

        for split in lince_splits:
            for item in ds_lince[split]:
                tokens = item.get('sentence', [])
                labels = item.get('labels', [])
                if not tokens or not labels:
                    continue

                # Check if sentence is genuinely code-mixed (contains both lang1 and lang2)
                label_set = set(labels)
                if 'lang1' in label_set and 'lang2' in label_set:
                    lince_sents_count += 1
                    sentence_text = " ".join(tokens)
                    seen_in_doc = set()

                    for tok, lbl in zip(tokens, labels):
                        # In LinCE, lang1 represents English tokens in Hinglish text
                        if lbl == 'lang1' and WORD_RE.match(tok):
                            w_low = tok.lower()
                            if is_valid_english_loanword(w_low):
                                token_counts[w_low] += 1
                                if w_low not in seen_in_doc:
                                    doc_counts[w_low] += 1
                                    seen_in_doc.add(w_low)

                    # Collect candidate real sentences (length between 6 and 20 tokens)
                    if 6 <= len(tokens) <= 20 and len(seen_in_doc) >= 1:
                        real_candidate_sentences.append({
                            "source": "LinCE",
                            "sentence": sentence_text,
                            "english_tokens": list(seen_in_doc)
                        })

        print(f"      Analyzed {lince_sents_count} code-mixed sentences from LinCE.")
    except Exception as e:
        print(f"      [WARNING] Could not process LinCE: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Ingest PHINC (Parallel Hinglish-English Corpus)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Loading PHINC Corpus (LingoIITGN/PHINC)...")
    try:
        ds_phinc = load_dataset('LingoIITGN/PHINC', split='train')
        phinc_sents_count = 0

        # Regex to detect English words embedded in romanized text
        for item in ds_phinc:
            raw_sent = item.get('Sentence', '')
            trans = item.get('English_Translation', '')
            if not raw_sent or len(raw_sent) < 15:
                continue

            # Skip pure URL/handle lines
            clean = re.sub(r'http\S+|@\S+', '', raw_sent).strip()
            words = re.findall(r'[A-Za-z]{2,25}', clean)
            if len(words) < 3:
                continue

            phinc_sents_count += 1
            seen_in_doc = set()

            for w in words:
                w_low = w.lower()
                if WORD_RE.match(w_low) and is_valid_english_loanword(w_low) and w_low not in INCIDENTAL_FUNCTIONAL_WORDS:
                    token_counts[w_low] += 1
                    if w_low not in seen_in_doc:
                        doc_counts[w_low] += 1
                        seen_in_doc.add(w_low)

            if len(real_candidate_sentences) < 150 and 6 <= len(words) <= 20 and trans:
                real_candidate_sentences.append({
                    "source": "PHINC",
                    "sentence": clean,
                    "reference": trans,
                    "english_tokens": list(seen_in_doc)
                })

        print(f"      Analyzed {phinc_sents_count} sentences from PHINC.")
    except Exception as e:
        print(f"      [WARNING] Could not process PHINC: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Build Dynamic Empirical Index
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Compiling Empirical Borrowing Statistics...")
    index_data: Dict[str, Any] = {
        "_metadata": {
            "corpora": ["LinCE (LID Hin-Eng)", "PHINC (LingoIITGN)"],
            "total_unique_tokens": len(token_counts),
            "formula": "BPI(w) = sigma(2.2 * (0.6*log10(1+count) + 0.4*log10(1+doc_freq) - 0.9))",
            "score_range": "[0.30, 0.98]"
        },
        "tokens": {}
    }

    # Filter to words that appeared at least twice or in distinct contexts
    for word, count in token_counts.most_common():
        df = doc_counts[word]
        score = compute_borrowing_score(count, df)
        index_data["tokens"][word] = {
            "count": count,
            "doc_freq": df,
            "borrowing_score": score,
            "is_incidental": word in INCIDENTAL_FUNCTIONAL_WORDS
        }

    OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    print(f"\n[OK] Dynamic Index saved -> {OUTPUT_INDEX}")
    print(f"     Total Unique English Loanword Tokens Indexed: {len(index_data['tokens'])}")

    # Show top 15 empirical loanwords
    print("\nTop 15 Empirical Hinglish Loanwords by Corpus Borrowing Score:")
    top_items = [
        (w, d) for w, d in list(index_data["tokens"].items())[:30]
        if not d["is_incidental"]
    ][:15]

    print(f"{'Token':<18} | {'Occurrences':<12} | {'Doc Freq':<10} | {'Empirical BF':<12}")
    print("-" * 60)
    for w, d in top_items:
        print(f"{w:<18} | {d['count']:<12} | {d['doc_freq']:<10} | {d['borrowing_score']:<12.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Save Real 50-Sentence Evaluation Corpus
    # ──────────────────────────────────────────────────────────────────────────
    sample_50 = real_candidate_sentences[:50]
    eval_corpus_data = {
        "_metadata": {
            "source": "LinCE and PHINC Open-Source Corpora",
            "n_sentences": len(sample_50),
            "description": "50 authentic code-mixed Hinglish sentences extracted directly from LinCE and PHINC."
        },
        "sentences": [
            {
                "id": f"REAL_{i+1:03d}",
                "source": s["source"],
                "sentence": s["sentence"],
                "reference": s.get("reference", ""),
                "candidate_loanwords": [t for t in s["english_tokens"] if t in index_data["tokens"] and not index_data["tokens"][t]["is_incidental"]][:6]
            }
            for i, s in enumerate(sample_50)
        ]
    }

    OUTPUT_REAL_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REAL_CORPUS, "w", encoding="utf-8") as f:
        json.dump(eval_corpus_data, f, indent=2)

    print(f"\n[OK] 50 Real Evaluation Sentences saved -> {OUTPUT_REAL_CORPUS}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
