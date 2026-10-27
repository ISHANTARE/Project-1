# Multi-Baseline Experimental Comparison & System Audit

**Author:** NLP Research & Systems Evaluation Team  
**Date:** October 2026  
**Status:** Audited & Statistically Validated  
**Associated Benchmark Artifacts:**  
- `results/benchmark_real_neural_test_split.json`
- `results/benchmark_real_neural_gold_100.json`
- `results/predictions_real_neural_test_split.json`
- `results/predictions_real_neural_gold_100.json`

---

## 1. Experimental Overview

To establish whether **LAPS (Linguistic Adaptive Preservation Scoring)** provides statistically defensible improvements in preserving English borrowings during Romanized Hinglish-to-English neural translation, we compare LAPS against three controlled baseline systems.

Each system is evaluated across two distinct experimental paradigms:
1. **Live Neural Inference on GPU (`Helsinki-NLP/opus-mt-hi-en`)**: True neural encoder-decoder translation where lexical constraints are strictly enforced via HuggingFace's constrained beam search.
2. **Deterministic Surrogate Simulation**: Historical rule-based simulation used for unit verification.

---

## 2. Description of Evaluated Systems

### System 1: LAPS (Proposed Framework)
- **Architecture**: Two-stage neuro-symbolic framework.
  - **Stage 1 (Neuro-Symbolic Gating)**: Computes a composite preservation score $S(t) \in [0, 1]$ across lexical, semantic, syntactic, graph, and frequency features, gated by register-adaptive thresholds. Borderline candidates trigger uncertainty fallback.
  - **Stage 2 (Constrained Decoding)**: Preserved tokens are injected as disjunctive lexical constraints into HuggingFace `AutoModelForSeq2SeqLM.generate()` via dynamic beam scaling ($\min(16, \max(6, 2 \times N))$).
- **Classification Status**: Active classifier (Token F1: 0.9567 on gold corpus, 0.8494 on test split).

### System 2: Few-Shot Vocabulary Filter (Demonstration Heuristic)
- **Architecture**: Simulates in-context learning prompts by preserving tokens belonging to a fixed 17-word technical vocabulary extracted from demonstration examples (`meeting`, `server`, `client`, `link`, `calendar`, `viva`, `laptop`, `wifi`, `email`, `database`, `code`, `debug`, `project`, `schedule`, `update`, `call`, `presentation`).
- **Classification Status**: Fixed rule classifier (Low recall: 0.0762; high precision: 1.0000).

### System 3: Static Dictionary Filter (Industry Glossary Baseline)
- **Architecture**: Standard commercial dictionary lookup using a 12-word fixed technical glossary (`meeting`, `server`, `client`, `viva`, `link`, `calendar`, `laptop`, `wifi`, `email`, `database`, `code`, `debug`).
- **Classification Status**: Fixed glossary classifier (Recall: 0.0397; precision: 1.0000).

### System 4: Unconstrained NMT Baseline
- **Architecture**: Standard end-to-end neural machine translation without lexical intervention (`force_constraints=False`).
- **Classification Status**: **N/A (No Internal Classifier)**. This system operates end-to-end; evaluating it on internal token classification is a category error. Output loanword retention is credited strictly from the generated English surface text.

---

## 3. Empirical Results: Live GPU Neural Generation

### Table 1: End-to-End Generative Performance (Gold Corpus, N=100)
*Evaluated on NVIDIA GeForce RTX 4050 GPU using `Helsinki-NLP/opus-mt-hi-en`.*

| System | OTER ($\downarrow$) | Output F1 ($\uparrow$) | Output Recall | SacreBLEU | chrF++ | $\Delta$ F1 vs LAPS | $p$-value (perm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LAPS (Proposed)** | **0.0108** | **0.9485** | **0.9892** | **4.16** | **38.38** | — | — |
| Few-Shot Vocabulary Filter | 0.8657 | 0.1908 | 0.1343 | 0.36 | 14.14 | -0.7577 | $0.0002^*$ |
| Static Dictionary Filter | 0.8990 | 0.1445 | 0.1010 | 0.46 | 13.36 | -0.8040 | $0.0002^*$ |
| Unconstrained NMT Baseline | 0.9240 | 0.1098 | 0.0760 | 0.73 | 12.77 | -0.8387 | $0.0002^*$ |

*\* Statistically significant under Bonferroni-corrected threshold ($\alpha = 0.05 / 3 = 0.0167$).*

---

### Table 2: Held-Out Test Split Performance (N=20: 15 Held-Out + 5 OOV)

| System | OTER ($\downarrow$) | Output F1 ($\uparrow$) | Output Recall | SacreBLEU | chrF++ | $\Delta$ F1 vs LAPS | $p$-value (perm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LAPS (Proposed)** | **0.0125** | **0.9383** | **0.9875** | **4.79** | **44.32** | — | — |
| Few-Shot Vocabulary Filter | 0.9125 | 0.1246 | 0.0875 | 0.48 | 15.65 | -0.8137 | $0.0002^*$ |
| Static Dictionary Filter | 0.9375 | 0.0906 | 0.0625 | 0.61 | 14.73 | -0.8477 | $0.0002^*$ |
| Unconstrained NMT Baseline | 0.9500 | 0.0733 | 0.0500 | 0.94 | 14.13 | -0.8650 | $0.0002^*$ |

---

### Table 3: Internal Token Preservation Classifier Performance (Gold Corpus, N=327 candidate tokens)

| System | Macro F1 | Precision | Recall | Accuracy | Uncertainty Rate | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LAPS (Proposed)** | **0.9567** | **0.9290** | **0.9967** | **0.9266** | **0.1223** | Dynamic multi-feature scoring |
| Few-Shot Filter | 0.0973 | 1.0000 | 0.0762 | 0.1468 | 0.0000 | Limited 17-token coverage |
| Static Dict Filter | 0.0477 | 1.0000 | 0.0397 | 0.1131 | 0.0000 | Limited 12-token coverage |
| Unconstrained NMT | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | No token classifier present |

---

## 4. Key Scientific Insights

### 4.1. The Catastrophic Over-Translation of Unconstrained NMT
In unconstrained neural translation, the MarianMT decoder attempts to translate Romanized Hindi phonetically into standard English. Because Romanized Hinglish lacks standardized orthography, the decoder hallucinates semantic equivalents for English borrowings:
- Input: `"hi! Kisi ke pass viva calender ki link h kya?"`
- Gold Preserved: `["calendar", "link", "viva"]`
- Unconstrained Output: `"What happened to my dad's face?"` (OTER = 100%, 0 borrowings retained).
- LAPS Output: Preserves `calendar`, `link`, and `viva` deterministically in the decoder beam (OTER = 0.0%).

Across the full gold corpus, **Unconstrained NMT loses 92.4% of technical borrowings** (OTER = 0.9240). LAPS reduces this loss to **1.08%** (OTER = 0.0108), representing an absolute reduction of **91.32 percentage points** ($p = 0.0002$).

### 4.2. Dictionary Coverage Bottlenecks
Both the Few-Shot and Static Dictionary baselines exhibit near-zero false positive rates (Precision = 1.0000) but suffer from cripplingly low recall (3.9%–7.6%). Code-mixed Hinglish in professional and technical domains features an open-ended borrowing distribution that cannot be covered by static glossaries. LAPS achieves **99.67% recall** while maintaining **92.90% precision** through its neuro-symbolic feature gating.
