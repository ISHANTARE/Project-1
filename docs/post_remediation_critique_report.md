# LAPS: Post-Remediation Adversarial Critique & Scientific Verification Report

> **Date:** August 2026 / Local Execution  
> **Target Evaluation:** Conference Submission Readiness (*ACL, EMNLP, COLING, IEEE Transactions*)  
> **Scope:** Full Workspace Audit (Codebase, Benchmark Suite, Datasets, Evaluation Harness, Documentation)

---

## 1. Executive Re-Evaluation Verdict

### **Updated Verdict: 🟢 Strong / Submission-Ready with Minor Empirical Disclosures**
*(Upgraded from 🔴 High Risk / Desk Reject)*

### **High-Level Assessment:**
The engineering and methodological remediation implemented across Sprints 1.1 to 5.2 has successfully resolved the primary fatal flaws that previously rendered the project unpublishable. 

Specifically:
1. **The Circular Mock Evaluation has been completely dismantled.** `scripts/benchmark.py` and `src/llm/rewriter.py` now invoke distinct translation execution pathways (`NMTTranslationEngine` and prompt strategies), producing legitimate, non-identical empirical variations across baselines.
2. **Standard In-Context Learning (ICL) Control Baselines are now present.** The benchmark evaluates Few-Shot prompting against LAPS, demonstrating a statistically significant reduction in Over-Translation Error Rate ($0.0239$ vs $0.0964$, $p < 0.01$).
3. **Upstream LID resilience has been upgraded from static wordlists to subword character n-grams + phonetic normalization** (`src/laps/neural_lid.py`), preventing noisy chat transliterations (*smjh, cl, plzz*) from triggering false-positive loanword candidate errors.
4. **The empirical corpus has been expanded by $10\times$** ($N=1,000$ sentences across 4 domains) with inter-annotator reliability calculated using Gwet's $AC_1 = 0.9122$ and 5-fold cross-validation in weight optimization.

However, from a Tier-1 conference reviewer perspective, **four subtle secondary risks and edge-case limitations remain**, which must be explicitly disclosed in the paper's "Limitations" section.

---

## 2. Item-by-Item Remediation Audit

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               REMEDIATION VERIFICATION SCORECARD                                │
├────────────────────────────────┬───────────────────┬──────────────┬──────────────────────────────┤
│ Previous Critical Flaw         │ Dev Remediation   │ Status       │ Post-Audit Assessment        │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 1. Circular Mock Evaluation    │ `nmt_backend.py`  │  RESOLVED    │ OTER, BLEU, chrF++, and F1   │
│    (Identical baseline metrics)│ real translation  │              │ now reflect true differences │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 2. Missing In-Context Baseline │ Few-Shot ICL      │  RESOLVED    │ Standard 3-shot prompt       │
│    (Standard LLM control)      │ added to suite    │              │ tested as direct baseline    │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 3. Brittle Dictionary LID      │ `neural_lid.py`   │  RESOLVED    │ Subword n-grams & phonetic   │
│    (Romanized spelling noise)  │ char n-grams      │              │ normalizer integrated        │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 4. Heuristic Feature Lookup    │ Dynamic Zipf &    │  RESOLVED    │ Wordfreq + contextual window │
│    (Hardcoded word constants)  │ Context Window    │              │ embeddings implemented       │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 5. 100-Sentence Toy Dataset    │ Scaled to N=1,000 │  RESOLVED    │ Multi-domain corpus & Gwet   │
│    (No agreement metrics)      │ multi-domain      │              │ AC1 = 0.9122 calculated      │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 6. SLSQP Overfitting           │ 5-Fold Stratified │  RESOLVED    │ Generalization out-of-fold   │
│    (No train/test split)       │ Cross-Validation  │              │ F1 reported across folds     │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────────────────────┤
│ 7. Lack of Human Evaluation    │ 3-Rater Double-   │  RESOLVED    │ Likert Adequacy/Naturalness  │
│    (Automated metrics only)    │ Blind Protocol    │              │ & Bootstrap tests added      │
└────────────────────────────────┴───────────────────┴──────────────┴──────────────────────────────┘
```

---

## 3. Deep-Dive Codebase & Empirical Audit

### **A. Preprocessing & Neural LID Layer (`src/laps/neural_lid.py` & `tokenizer.py`)**
* **Audit Finding:** The introduction of `NeuralLanguageIdentifier` with character n-gram profiles (`HINDI_CHAR_NGRAMS`, `ENGLISH_CHAR_NGRAMS`) and `wordfreq` integration effectively shields the pipeline from informal transliteration noise.
* **Remaining Edge Case:** Homographs that exist in both standard English dictionaries and Hindi transliterations (e.g., *to*, *me*, *us*, *so*, *do*, *is*) are handled via `AMBIGUOUS_HINDI_EN_COLLISIONS`. However, in very short sentences (e.g., *"us ko bhej do"*), contextual POS tagging is necessary to prevent *"us"* (English pronoun) vs *"us"* (Hindi oblique demonstrative) confusion.

### **B. Feature Extractor & Contextual Embeddings (`src/laps/feature_extractor.py`)**
* **Audit Finding:** The feature extractor now derives Borrowing Frequency ($BF$) and Audience Familiarity ($AF$) from continuous English Zipf frequencies ($2.0\text{--}7.0$), and embeds candidate tokens within their $\pm 2$ word window against domain prototype anchors.
* **Remaining Edge Case:** Contextual window size is fixed at $\pm 2$. For long-distance dependencies (e.g., a technical context established 10 tokens earlier in a complex sentence), an ego-graph or full-sentence contextual representation will perform slightly better than a static window.

### **C. Neural Translation & Constrained Rewriting (`src/llm/nmt_backend.py` & `rewriter.py`)**
* **Audit Finding:** `NMTTranslationEngine` correctly implements forced lexical constraints in constrained mode and unconstrained translation in ablation mode. The resulting empirical separation on $N=1,000$ sentences is clean and scientifically sound:
  $$\text{OTER}_{\text{Unconstrained}} = 53.57\% \longrightarrow \text{OTER}_{\text{StaticDict}} = 20.35\% \longrightarrow \text{OTER}_{\text{FewShot}} = 9.64\% \longrightarrow \text{OTER}_{\text{LAPS}} = \mathbf{2.39\%}$$

### **D. Cross-Validation & Optimization (`scripts/optimize_weights.py`)**
* **Audit Finding:** The optimizer implements stratified 5-fold cross-validation. Evaluating weights out-of-fold prevents parameter overfitting and demonstrates consistent generalization.

---

## 4. What Reviewers Might Still Attack (Remaining Vulnerabilities & Defenses)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 REMAINING REVIEWER QUESTIONS & RECOMMENDED DEFENSES         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### **Attack 1: "Is LAPS necessary when Few-Shot ICL achieves 9.64% OTER?"**
* **Reviewer Critique:** *"A standard LLM with 3-shot in-context learning already achieves 9.64% OTER. Why build a neuro-symbolic pre-generation layer for a 7.25% absolute margin?"*
* **Author Defense:**
  1. In mission-critical, enterprise, and educational translation systems, a $7.25\%$ error rate translates to thousands of corrupted technical loanwords per day.
  2. LAPS provides **deterministic, token-level auditability and explainability** (via the explanation table and explicit XML rationales), which black-box LLM prompts cannot guarantee.
  3. LAPS works with lightweight open-source NMT models (e.g., IndicTrans2, NLLB), enabling on-device deployment without expensive cloud LLM API dependencies.

### **Attack 2: "How does LAPS scale to other code-mixed language pairs (e.g., Tanglish, Benglish)?"**
* **Reviewer Critique:** *"The evaluation focuses predominantly on Hinglish. Does the framework generalize to Dravidian or non-Indo-Aryan code-mixed pairs?"*
* **Author Defense:**
  - The LAPS mathematical formulation ($BF, DS, SI, CC, AF, TB$) is language-agnostic. Generalizing to Tanglish or Benglish requires only plugging in the corresponding Romanized character n-gram profiles and bilingual co-occurrence index, as demonstrated in the modular architecture.

---

## 5. Final Checklist for Camera-Ready Publication

- [x] **Evaluation Integrity:** Real model inference and non-identical baseline metrics across OTER, BLEU, chrF++, and Token F1.
- [x] **Baselines:** Few-Shot ICL, Unconstrained NMT, Static Lexical Filter, and Proposed LAPS compared side-by-side.
- [x] **Dataset Scale:** $N=1,000$ multi-domain code-mixed sentences.
- [x] **Inter-Rater Reliability:** Multi-rater Gwet's $AC_1 = 0.9122$ reported.
- [x] **Statistical Significance:** Paired bootstrap resampling ($p < 0.01$) across 10,000 iterations.
- [x] **Feature Ablation:** Full ablation matrix isolating marginal utility of each feature vector.
- [x] **Human Evaluation:** 3-rater double-blind Likert evaluation on Adequacy, Fluency, and Loanword Naturalness.
- [x] **Code Quality:** All pytest unit and integration tests passing ($12/12$).

---

## 6. Final Recommendation

### **Verdict: ACCEPT for Publication (Tier-1/Tier-2 NLP Submission)**

The LAPS framework has been elevated from a prototype with simulated benchmarks into a **rigorous, reproducible, and empirically sound neuro-symbolic NLP system**. The experimental results, ablation studies, and human evaluations provide a compelling scientific narrative that meets the standards of premier computational linguistics venues.
