# LAPS Research-Paper Readiness & Scientific Defense Report (v3)

**Document:** `docs/research_validation_v3/research_readiness_report.md`  
**Date:** October 13, 2026  
**Framework:** LAPS (Linguistic Adaptive Preservation Scoring)  
**Primary Execution Commit:** `e66693a` / `876f04d`  

---

## 1. Core Research Question & Novel Contributions

### 1.1 The Research Problem
In Romanized Hinglish rewriting and translation, off-the-shelf Neural Machine Translation (NMT) models suffer from severe **over-translation**: domain-critical English borrowings (`presentation`, `dataset`, `confidence`) are either improperly translated away into unnatural matrix Hindi equivalents or completely erased due to script fragmentation. Conversely, naïve verbatim dictionary copying fails because contextually dispensable loanwords (`try`, `free`) in casual registers sound stilted if forcefully retained.

### 1.2 The LAPS Solution & Theoretical Contributions
1. **Explainable Multi-Feature Vector Scoring:** Evaluates each candidate English borrowing across 6 grounded linguistic dimensions:
   - Borrowing Frequency prior ($w_1 = 0.20$)
   - Domain Specificity via Knowledge Graph ($w_2 = 0.20$)
   - Semantic Importance via MiniLM cosine similarity ($w_3 = 0.20$)
   - Context Window Consistency ($w_4 = 0.15$)
   - Audience Familiarity / Register Prior ($w_5 = 0.15$)
   - Translation Benefit penalty ($w_6 = 0.10$)
2. **Register-Aware Dynamic Thresholding:** Dynamically modulates the decision boundary $\theta \in [0.40, 0.55]$ based on the communicative register (`academic` = 0.40, `business` = 0.45, `social` = 0.50, `casual` = 0.55).
3. **Soft Sigmoid Uncertainty Buffer:** Introduces a probabilistic buffer ($\pm 0.05$) around $\theta$, preserving boundary loanwords to prevent catastrophic technical information loss.
4. **Constrained Neural Decoding Integration:** Compiles preservation decisions into HuggingFace Constrained Beam Search tokens, deterministically enforcing preservation during neural sequence generation without requiring model retraining.

---

## 2. Scientifically Defensible Claims (Supported by Evidence)

The empirical artifacts in `results/research_validation_v2/` support the following claims:

1. **Massive Reduction in Over-Translation Error (OTER):**
   Under identical GPU neural inference conditions on `Helsinki-NLP/opus-mt-hi-en`, LAPS reduces OTER from **0.9500** (unconstrained baseline) down to **0.2450** on the held-out test split, and from **0.9240** down to **0.1603** on the full gold corpus.
2. **Statistically Significant Improvement in Output Retention F1:**
   LAPS achieves an Output Retention F1 of **0.7796** compared to **0.0733** for unconstrained translation on the held-out test split ($\Delta\text{F1} = +0.7063$, $p = 0.0002$ under 5,000-sample Monte Carlo permutation test with Bonferroni correction).
3. **Multi-Feature Ablation Validity:**
   Ablation experiments on the held-out test split demonstrate that removing Borrowing Frequency (-BF), Semantic Importance (-SI), Context Consistency (-CC), Soft Sigmoid, or the Knowledge Graph degrades performance ($\Delta\text{F1}$ drops between $-0.0175$ and $-0.0637$).
4. **High Inter-Annotator Agreement:**
   The human preservation gold standard achieves Cohen's $\kappa = 0.8744$ across independent bilingual raters.
5. **Negligible Scoring Overhead:**
   LAPS feature scoring requires only $\sim 0.01$ ms per token in-memory and $\sim 57$ ms under live uncached extraction.

---

## 3. Explicit Scientific Bounds & Prohibited Claims

To ensure publication defense and academic integrity, the following claims **must NOT be made**:

1. **Do NOT Claim Superior Overall Translation Fluency:**
   - *Empirical Truth:* BLEU scores for both systems remain low ($0.25 - 0.29$ for LAPS vs $0.73 - 0.85$ for unconstrained baseline).
   - *Reason:* `Helsinki-NLP/opus-mt-hi-en` is an NMT checkpoint trained on standard Devanagari script. Feeding it Romanized Latin characters triggers subword fragmentation and low encoder confidence, causing the constrained beam search decoder to emit repetitive filler between the forced tokens.
   - *Paper Framing:* Frame LAPS as a **token preservation and constraint-steering framework** that solves the over-translation problem. Acknowledge that overall fluency requires a Latin-script adapted backbone (e.g., IndicTrans2).
2. **Do NOT Report "Gold UTER":**
   - Full-utterance word-level Language Identification (LID) labels are not available.
   - Report the Hindi matrix leakage metric strictly as an **operational proxy**, not as Gold UTER.
3. **Do NOT Claim Broad Multi-Lingual Generalization Without Additional Corpora:**
   - The verified dataset comprises 100 curated sentences and 20 held-out/OOV sentences in Romanized Hinglish. The paper must frame this as a rigorous proof-of-concept benchmark.

---

## 4. Remaining Submission Blockers & Roadmap to Camera-Ready

| Milestone / Requirement | Current Status | Resolution Plan for Final Submission |
| :--- | :---: | :--- |
| **Strict Real-Neural Inference on GPU** | **COMPLETED & VERIFIED** | Validated on RTX 4050; zero fallbacks recorded. |
| **Paired Baseline Comparison** | **COMPLETED & VERIFIED** | Strictly paired against unconstrained MarianMT. |
| **Ablation Study (Held-Out Split)** | **COMPLETED & VERIFIED** | Verified feature degradations recorded. |
| **Test Suite Passing Rate** | **COMPLETED & VERIFIED** | 55/55 passed (100%). |
| **Evidence Manifest Integrity** | **COMPLETED & VERIFIED** | 100% SHA-256 hash match on all 21 artifacts. |
| **Fluency Adaptation Backbone** | **FUTURE WORK** | Evaluate LAPS on IndicTrans2 or fine-tuned NLLB-200. |
| **Human Evaluation on Generated Text** | **FUTURE WORK** | Run the 50-sentence 4-axis Likert rating protocol. |
| **Expanded Test Corpus ($N \ge 500$)** | **FUTURE WORK** | Scale dataset size for broader dialectal coverage. |
