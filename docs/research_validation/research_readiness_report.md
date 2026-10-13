# Research Readiness Report & Scientific Paper Submission Audit

**Author:** NLP Research & Systems Evaluation Team  
**Date:** October 2026  
**Status:** Final Audit Synthesis  

---

## 1. Executive Summary

This report synthesizes the comprehensive scientific validation of **LAPS (Linguistic Adaptive Preservation Scoring)**. Through code repair, neural GPU benchmarking, mathematical ablation restructuring, operational profiling, and rigorous error auditing, the project has been brought to academic submission standard.

---

## 2. Distinction Between Defensible and Non-Defensible Claims

To ensure academic credibility and prevent rejection during peer review, research claims must strictly align with empirical evidence.

### 2.1. Defensible Scientific Claims (Supported by Replicable Evidence)
1. **Catastrophic Over-Translation Mitigation**:
   Standard unconstrained neural translation (`Helsinki-NLP/opus-mt-hi-en`) fails to preserve English borrowings in Romanized Hinglish, suffering an Over-Translation Error Rate (OTER) of **95.0%** on held-out test data. LAPS reduces OTER to **1.25%**, yielding an absolute Output F1 increase of **+0.8649** ($p = 0.0002$, Bonferroni significant).
2. **Superiority Over Static Lexicons**:
   While static dictionaries achieve high precision (1.0000), their recall is severely bounded (3.9%–7.6%) by the open-ended nature of code-mixed technical borrowings. LAPS achieves **99.67% recall** while maintaining **92.90% precision** on candidate borrowings.
3. **Feature Importance Hierarchy**:
   Mathematical feature ablation demonstrates that **Borrowing Frequency** ($\Delta F1 = -0.0107$) and **Knowledge Graph Ego-Density** ($\Delta F1 = -0.0056$) provide the largest unique marginal contributions to preservation accuracy.
4. **Soft Uncertainty Gating Utility**:
   Ablating the uncertainty interval $[T_{\text{lower}}, T_{\text{upper}}]$ in favor of a hard cutoff degrades Macro F1 by $-0.0025$ on the full corpus and $-0.0138$ on validation, confirming the value of conservative borderline preservation.

### 2.2. Non-Defensible Claims (Must Be Corrected or Qualified in Submission)
1. **"0.21 ms Real-Time Latency"**:  
   *Correction:* Must be explicitly clarified as *in-memory dot product arithmetic on pre-cached features*. Real-time live inference requires **73.25 ms** for feature extraction and **1,643.83 ms** for GPU constrained beam search decoding.
2. **"Strict Gold UTER = 0.0%"**:  
   *Correction:* The gold corpus lacks full-utterance word-level LID ground truth for non-candidate Hindi matrix words. Strict Gold UTER must be designated as **`N/A`**, and matrix Hindi leakage reported via an operational proxy ($0.0123$–$0.0157$).
3. **"Universal Cross-Lingual Evaluation"**:  
   *Correction:* Experiments were validated on Romanized Hinglish $\to$ English using `opus-mt-hi-en`. Claims of generalizability to other Indic languages (e.g., Bengali, Tamil) or reverse translation must be qualified as future work.
4. **"Large-Scale Benchmark"**:  
   *Correction:* Characterize the corpus as a targeted pilot evaluation benchmark ($N=100$ sentences, $327$ candidate tokens) rather than an industrial-scale dataset.

---

## 3. Recommended Research Paper Outline (Targeting ACL / EMNLP / NAACL)

1. **Introduction**:
   - The challenge of code-mixing in global NLP: Romanized Hinglish in professional/technical domains.
   - The over-translation problem: why standard NMT decoders mistranslate technical borrowings into nonsensical phonological targets.
   - Core contribution: LAPS, an explainable neuro-symbolic framework coupling multi-feature gating with dynamically constrained decoding.
2. **Related Work**:
   - Code-mixed machine translation and transliteration.
   - Lexically constrained decoding (Grid Beam Search, Neuro-Logic, Disjunctive Constraints).
   - Register adaptation and language identification in low-resource orthographies.
3. **The LAPS Framework**:
   - Feature formulation (lexical, semantic, syntactic, graph ego-density, borrowing frequency).
   - Composite scoring function and register-adaptive thresholding.
   - Integration with Seq2Seq decoders via constrained beam search.
4. **Experimental Setup**:
   - Curated dataset description and 70/15/15 validation/test splits.
   - Evaluation metrics: OTER, Output F1, SacreBLEU 2.6.0, chrF++, and qualification of UTER.
   - Baselines: Few-Shot heuristic, Static Dictionary, Unconstrained NMT.
5. **Results & Discussion**:
   - End-to-end generative translation results on GPU.
   - Token preservation classification results.
   - Statistical significance testing via paired permutation tests ($B=5,000$).
6. **Ablation Studies**:
   - Feature weight ablation with normalized reweighting.
   - Component ablations (KG boost, uncertainty window, register adaptability).
7. **Operational Latency Profiling**:
   - Granular breakdown across preprocessing, embedding extraction, and decoding.
   - Discussion of constrained decoding overhead ($7.1\times$) and deployment optimizations.
8. **Limitations & Threats to Validity**:
   - Explicit discussion of the 15 identified threat dimensions.
9. **Conclusion & Future Work**:
   - Expansion to other Indic languages, speculative constrained decoding, and integration with 70B+ LLMs.

---

## 4. Submission Checklist

- [x] Hugging Face constrained decoding code verified on GPU.
- [x] End-to-end neural outputs generated and evaluated with SacreBLEU 2.6.0.
- [x] Re-weighted mathematical feature ablation study executed.
- [x] Paired permutation significance tests ($B=5,000$) with Bonferroni correction computed.
- [x] Latency profile disentangled into live vs cached components.
- [x] 15 threat dimensions comprehensively catalogued.
- [x] Test suite fully green (53 passing tests).
- [ ] Prepare camera-ready LaTeX manuscript and anonymous code repository.
