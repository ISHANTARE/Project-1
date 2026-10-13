# Limitations, Threats to Validity, and Open Research Challenges

**Author:** NLP Research & Systems Evaluation Team  
**Date:** October 2026  
**Status:** Academic Paper Readiness Review  

---

## 1. Overview

Transparent reporting of limitations and threats to experimental validity is a cornerstone of scientific rigor in natural language processing. This document catalogues 15 distinct methodological, architectural, empirical, and operational constraints of the LAPS framework.

---

## 2. Comprehensive Taxonomy of Limitations

### 2.1. Dataset & Annotation Constraints

1. **Evaluation Corpus Scale ($N=100$)**:
   The gold evaluation corpus comprises 100 carefully curated sentences (327 candidate tokens). While this sample size provides sufficient statistical power ($>99.9\%$) to detect the massive effect size ($\Delta F1 > 0.80$) over unconstrained baselines, it remains an evaluation benchmark rather than a WMT-scale corpus ($10^4$–$10^6$ sentence pairs).
2. **Curated vs In-the-Wild Conversational Data**:
   The gold corpus was constructed to evaluate specific register and domain dynamics (technical, academic, business, casual). Natural in-the-wild social media corpora (e.g., Twitter, WhatsApp) exhibit higher noise, emojis, typographical errors, and multilingual code-switching that may degrade LID accuracy.
3. **Absence of Full-Utterance Word-Level LID Annotations**:
   The gold corpus provides ground truth strictly for candidate English loanwords ($y \in \{0, 1\}$). Non-candidate Hindi matrix words lack word-level language identification and alignment tags. Consequently, **Strict Gold UTER cannot be reported**, necessitating the use of a lexical proxy.
4. **Inter-Annotator Agreement Coverage**:
   Cohen’s $\kappa = 0.817$ demonstrates strong inter-annotator agreement, but this audit was conducted on a 30-sentence subset rather than double-annotating the entire corpus across multiple independent native speakers.
5. **Single-Annotator Primary Labels**:
   Primary ground-truth labels for the remaining 70 sentences reflect a single annotator's linguistic judgment, introducing potential subjective bias in borderline casual borrowings.

### 2.2. Linguistic & Domain Generalization

6. **Romanization Dialect Variance**:
   Romanized Hinglish has no standardized orthographic convention (e.g., `kal` vs `kall`, `kya` vs `kyaa`). Because MarianMT was pretrained primarily on Devanagari Hindi, phonetic Romanization induces decoder confusion unless preceded by transliteration or normalization.
7. **Monodirectional Language Pair Evaluation (Hinglish $\to$ English)**:
   Experimental evaluation is restricted to Romanized Hinglish to English. Reverse translation (English $\to$ Hinglish) and other code-mixed Indic languages (e.g., Tanglish/Tamil-English, Banglish/Bengali-English) remain unvalidated.
8. **Domain Skew Toward Technical and Academic Registers**:
   The feature weighting and knowledge graph ego-networks are heavily populated with technology, software engineering, and corporate terminology. Performance on colloquial conversational slang or regional cultural idiom is expected to be lower.
9. **Morphologically Inflected Code-Mixed Borrowings (Hybrid Stems)**:
   In conversational Hinglish, English borrowings frequently take Hindi morphological inflections (e.g., *“file-on”* [plural files], *“setting-waala”* [agentive suffix]). LAPS does not currently incorporate a Hindi morphological stemmer, treating hybrid tokens as out-of-vocabulary terms.
10. **Polysemy and Semantic Ambiguity**:
    Words with register-dependent meanings (e.g., *“table”* in database context vs dining context; *“network”* as social connection vs telecommunications infrastructure) receive static knowledge graph priors if contextual disambiguation is insufficient.

### 2.3. Model & Computational Constraints

11. **Neural Model Architecture Capacity (`opus-mt-hi-en`)**:
    Experiments were conducted on Helsinki-NLP's 74M-parameter MarianMT model. While MarianMT provides a controlled offline baseline, large foundation models (e.g., GPT-4o, Llama 3 70B, IndicTrans2) possess significantly higher inherent vocabulary capacities and may exhibit different over-translation baselines.
12. **Feature Extraction Computational Dependencies**:
    Live uncached scoring requires forward passes through `sentence-transformers/all-MiniLM-L6-v2` (65 ms per utterance), introducing an external transformer dependency prior to generation.
13. **Constrained Decoding Computational Overhead ($7.1\times$)**:
    Enforcing lexical constraints via HuggingFace's constrained beam search increases GPU decoding time from 230.57 ms to 1,643.83 ms per utterance. This latency is prohibitive for hard real-time streaming applications without optimization.
14. **Beam Search Memory & Constraint Explosion**:
    HuggingFace constrained beam search scales beam width linearly with the number of constraints ($\min(16, \max(6, 2 \times N))$). Sentences with $>8$ borrowings risk beam starvation or combinatorial explosion during hypothesis expansion.
15. **Register Threshold Tuning Sensitivity**:
    Register thresholds ($[T_{\text{lower}}, T_{\text{upper}}]$) were tuned across the four canonical registers. Deploying LAPS in novel registers (e.g., legal or clinical transcripts) requires threshold calibration on domain-specific validation sets.
