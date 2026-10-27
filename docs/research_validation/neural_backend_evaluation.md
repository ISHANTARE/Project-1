# LAPS Real Neural Machine Translation Backend Evaluation

**Document Status**: COMPLETED AND EMPIRICALLY VERIFIED  
**Experiment Execution Date**: October 13, 2026  
**Hardware Platform**: NVIDIA GeForce RTX 4050 Laptop GPU (5.996 GB VRAM, CUDA 12.8)  
**Primary Neural Model**: `Helsinki-NLP/opus-mt-hi-en` (MarianMT Seq2Seq Transformer)  
**Decoding Framework**: Lexically Constrained Beam Search (`force_words_ids`)  

---

## 1. Technical Diagnosis & Engineering Remediation

### 1.1 The Transformers `force_words_ids` Compatibility Issue
In earlier iterations of this project, calling `model.generate(force_words_ids=...)` under modern HuggingFace Transformers (v5.15.1 / late v4.x) caused an immediate runtime exception:

```
[transformers] Constrained Beam Search was moved to a custom_generate repo:
https://hf.co/transformers-community/constrained-beam-search.
OSError: Constrained Beam Search requires trust_remote_code=True in your generate call...
```

When previous implementations caught this exception, they fell back to `_generate_realistic_nmt_translation` (a deterministic regex/template surrogate), recording `backend_execution_mode: DETERMINISTIC_SURROGATE_FALLBACK`. No actual neural tokens were decoded on GPU.

### 1.2 Engineering Solution
1. **Dynamic Generation Loading**:
   Explicitly passed `custom_generate="transformers-community/constrained-beam-search"` and `trust_remote_code=True` to `model.generate()`.
2. **Cache Poisoning Resolution**:
   Cleared poisoned `.no_exist` HuggingFace cache entries and downloaded the required support scripts (`beam_constraints.py`, `beam_search.py`, `generate.py`) into the snapshot cache.
3. **Beam Search Space Scaling**:
   Scaled the beam count dynamically to $\min(16, \max(6, 2 \times |\text{constraints}|))$ to provide sufficient hypothesis exploration for constrained beam search.
4. **Repetition Penalization**:
   Added `no_repeat_ngram_size=3` and `max_new_tokens=64` to prevent degenerate repetition loops in MarianMT decoder outputs.
5. **Token Deduplication**:
   Sanitized and deduplicated constraint lists in `_build_force_words_ids()` before mapping to subword token IDs.

---

## 2. Experimental Setup & Paired Protocol

To ensure strict scientific integrity (Non-Negotiable Rule 3.3), the exact same input utterances were passed through four experimental conditions under identical decoding parameters:

- **Condition A (Unconstrained Neural)**: `Helsinki-NLP/opus-mt-hi-en` with beam search ($B = 4$), zero lexical constraints.
- **Condition B (LAPS-Constrained Neural)**: The same `Helsinki-NLP/opus-mt-hi-en` model with `force_words_ids` injected from the LAPS neuro-symbolic preservation decisions.
- **Condition C (Static Dictionary)**: The same model with constraints forced only for a fixed 12-word technical glossary.
- **Condition D (Few-Shot Demonstration Heuristic)**: The same model with constraints forced only for a fixed 17-word demo vocabulary.

**Traceability**: All generated outputs, input IDs, configuration metadata, and per-sentence metrics were saved to:
- `results/benchmark_real_neural_test_split.json`
- `results/predictions_real_neural_test_split.json`
- `results/benchmark_real_neural_gold_100.json`
- `results/predictions_real_neural_gold_100.json`

---

## 3. Empirical Results: Held-Out Test Split (N = 20)

Evaluated on `data/evaluation/test_split.json` (15 held-out in-distribution sentences + 5 unseen OOV domain sentences):

| System / Condition | Backend Execution Mode | OTER (↓) | Retention Recall (↑) | Output F1 (↑) | BLEU (↑) | chrF++ (↑) | $\Delta$ F1 vs LAPS | $p$-value (perm) |
|---|---|---|---|---|---|---|---|---|
| **LAPS (Proposed)** | `REAL_INFERENCE_CONSTRAINED` | **0.0125** | **0.9875** | **0.9383** | **4.79** | **44.32** | — | — |
| **Few-Shot Heuristic** | `REAL_INFERENCE_CONSTRAINED` | 0.9125 | 0.0875 | 0.1200 | 0.62 | 13.55 | +0.8183 | 0.0002 * |
| **Static Dictionary** | `REAL_INFERENCE_CONSTRAINED` | 0.9375 | 0.0625 | 0.0867 | 0.71 | 12.75 | +0.8516 | 0.0002 * |
| **Unconstrained Neural** | `REAL_INFERENCE_UNCONSTRAINED` | 0.9500 | 0.0500 | 0.0733 | 0.85 | 12.91 | +0.8649 | 0.0002 * |

*\* Statistically significant under Bonferroni correction ($\alpha = 0.05 / 3 = 0.0167$).*

---

## 4. Empirical Results: Full Gold Corpus (N = 100)

Evaluated across the entire primary benchmark corpus (`data/evaluation/gold_corpus.json`, 327 candidate tokens):

| System / Condition | Backend Execution Mode | OTER (↓) | Retention Recall (↑) | Output F1 (↑) | BLEU (↑) | chrF++ (↑) | $\Delta$ F1 vs LAPS | $p$-value (perm) |
|---|---|---|---|---|---|---|---|---|
| **LAPS (Proposed)** | `REAL_INFERENCE_CONSTRAINED` | **0.0108** | **0.9892** | **0.9485** | **4.16** | **38.38** | — | — |
| **Few-Shot Heuristic** | `REAL_INFERENCE_CONSTRAINED` | 0.8657 | 0.1343 | 0.1908 | 0.36 | 14.14 | +0.7577 | 0.0002 * |
| **Static Dictionary** | `REAL_INFERENCE_CONSTRAINED` | 0.8990 | 0.1010 | 0.1445 | 0.46 | 13.36 | +0.8040 | 0.0002 * |
| **Unconstrained Neural** | `REAL_INFERENCE_UNCONSTRAINED` | 0.9240 | 0.0760 | 0.1098 | 0.73 | 12.77 | +0.8387 | 0.0002 * |

---

## 5. Qualitative Linguistic Inspection: Generated Outputs

Below are representative generated outputs from the real neural model on GPU (`predictions_real_neural_test_split.json`):

### Example 1: Technical Vocabulary Retention (Sentence `S005`)
- **Input**: *Presentation ka feedback kal milega*
- **Gold Loanwords**: `presentation`, `feedback`
- **Reference**: *The feedback for the presentation will be shared tomorrow.*
- **Unconstrained Output**: `And then we're going to have a meeting tomorrow.`
  - *Error*: Both *presentation* and *feedback* are completely omitted/over-translated!
- **LAPS-Constrained Output**: `Presentation of the feedback will be available tomorrow.`
  - *Result*: Both *presentation* and *feedback* appear verbatim in the output text!

### Example 2: Medical OOV Domain Term (Sentence `OOV_002`)
- **Input**: *Patient ka MRI scan aur biopsy report normal hai*
- **Gold Loanwords**: `patient`, `mri`, `scan`, `biopsy`, `report`
- **Reference**: *The patient's MRI scan and biopsy report are normal.*
- **Unconstrained Output**: `And then you can see that the report is in good condition.`
  - *Error*: *patient*, *MRI*, *scan*, and *biopsy* are omitted!
- **LAPS-Constrained Output**: `The patient has a normal MRI scan and biopsy report.`
  - *Result*: All 5 domain borrowings are successfully forced and retained!

---

## 6. Scientific Analysis & Discussion

1. **Massive Over-Translation in Unconstrained Models**:
   The unconstrained neural model exhibited an Over-Translation Error Rate (OTER) of **92.4%** across the corpus and **95.0%** on the test split. Raw NMT models trained predominantly on monolingual parallel corpora inherently treat English borrowings in Romanized Hindi as noise or translate them away into unrelated phrasing.
2. **Failure of Static Lexicons**:
   Static industry glossaries (12 words) and few-shot vocabulary heuristics (17 words) only reduced OTER by ~3–6%, because real discourse exhibits high lexical diversity across academic, business, social, and medical registers.
3. **Dynamic Neuro-Symbolic Impact**:
   LAPS reduced OTER from 92.4% to **1.08%** across the corpus and to **1.25%** on the test split—an absolute reduction of over 91 percentage points ($p = 0.0002$).
4. **Model Fluency Caveat**:
   `Helsinki-NLP/opus-mt-hi-en` was originally pre-trained on standard Devanagari Hindi. When fed Romanized Hinglish directly, its base translation quality is limited (BLEU $\approx 4$), though chrF++ reaches 38–44. Applying LAPS constraints dramatically elevates lexical preservation without lowering n-gram overlap.
