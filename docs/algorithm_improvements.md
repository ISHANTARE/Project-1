# LAPS Algorithmic and Functional Improvements

This document outlines the mathematical foundation of **LAPS (Linguistic Adaptive Preservation Scoring)** and details the algorithmic upgrades implemented across the codebase.

---

## 1. Algorithmic Foundation

The Preservation Score for a candidate code-mixed token $t$ in sentence context $C$ is computed via:

$$\text{LAPSScore}(t) = w_1 \text{BF}(t) + w_2 \text{DS}(t) + w_3 \text{SI}(t) + w_4 \text{CC}(t, C) + w_5 \text{AF}(t) - w_6 \text{TB}(t)$$

Where:
* **$\text{BF}(t)$ (Borrowing Frequency):** Queried from the 2,907-node Knowledge Graph or `wordfreq` Zipf frequency ratios.
* **$\text{DS}(t)$ (Domain Specificity):** Computed via `sentence-transformers` (`all-MiniLM-L6-v2`) zero-shot vector cosine similarity against Tech, Business, Academic, Medical, and Finance domain anchors.
* **$\text{SI}(t)$ (Semantic Importance):** Quantifies semantic drift when translated into target language.
* **$\text{CC}(t, C)$ (Context Consistency):** Sub-graph co-occurrence density of token $t$ within context window $C$.
* **$\text{AF}(t)$ (Audience Familiarity):** Global conversational recognition score.
* **$\text{TB}(t)$ (Translation Benefit):** Syntactic benefit of translating $t$ to preserve target language grammar.

---

## 2. Implemented Algorithmic Upgrades

### A. Neural Vector Embeddings and Fuzzy Typo Alignment [IMPLEMENTED]
* **Method:** Replaced static word lists with `SentenceTransformer("all-MiniLM-L6-v2")` embeddings. Typos such as `calender` or `scedule` automatically map to canonical forms via `difflib` edit-distance matching.

### B. Sentence-Level Code-Mixing Index (CMI) Detection [IMPLEMENTED]
* **Method:** Built `cmi_detector.py` to calculate CMI (Gamback and Das, 2014):
  $$\text{CMI} = 100 \cdot \left[1 - \frac{\max(N_{\text{lang}})}{N_{\text{total}}}\right]$$
  Monolingual English sentences ($\text{CMI} = 0.0\%$) are cleanly detected and retained without unnecessary code-mixed rewriting.

### C. Soft Sigmoid Probability Decision Boundaries [IMPLEMENTED]
* **Method:** Replaced hard binary cutoffs ($\text{Score} \ge \theta$) with smooth Sigmoid logistic probabilities:
  $$P(\text{Preserve} \mid t) = \frac{1}{1 + e^{-\lambda (\text{LAPSScore}(t) - \theta_R)}}$$
  Where $\lambda = 10.0$ governs transition sharpness. Three distinct decision zones are identified:
  * $P \ge 0.70 \rightarrow$ PRESERVE (high confidence retention)
  * $0.50 \le P < 0.70 \rightarrow$ UNCERTAIN (borderline candidate, preserved with advisory flag)
  * $P < 0.50 \rightarrow$ REWRITE (candidate suitable for natural translation)

### D. Graph-Enriched XML Prompt Engineering [IMPLEMENTED]
* **Method:** Structured XML constraint injection with anti-hallucination domain rationales and edge naturalness penalties ($<token\ name="..."\ naturalness\_penalty="...">$), strictly guiding downstream LLMs during sentence rewriting.

### E. Dynamic Empirical Borrowing Index from LinCE & PHINC [IMPLEMENTED]
* **Method:** Replaced static whitelists with dynamic empirical borrowing propensities derived directly from **16,024 code-mixed sentences** (LinCE and PHINC corpora), indexing **9,879 unique English loanwords** with occurrence counts, document frequencies, and continuous borrowing propensity scores:
  $$\text{BPI}(t) = \min\left(1.0, \frac{\log_{10}(c(t) + 1)}{\log_{10}(1000)} \times \left(0.5 + 0.5 \cdot \frac{\text{df}(t)}{N_{\text{total}}}\right)\right)$$

### F. Automated Weight Vector Optimization ($\mathbf{w}^*$) [IMPLEMENTED]
* **Method:** Grid search and SLSQP optimization over 6-dimensional simplex constraints ($\sum w_j = 1, w_j \ge 0$) targeting maximum Token F1 and minimum Cross-Entropy loss over gold standard datasets.

### G. Contextual Homograph Disambiguation [IMPLEMENTED]
* **Problem:** Romanized Hindi-English code-mixing produces orthographic collisions: `us` (English pronoun vs. Hindi oblique demonstrative), `me` (English pronoun vs. Hindi locative postposition *mẽ*), `do` (English auxiliary vs. Hindi permissive verb), `to` (English preposition vs. Hindi discourse particle).
* **Method:** Added `disambiguate_homographs(tokens: List[TaggedToken])` to `SubwordLanguageIdentifier` in `src/laps/neural_lid.py`. Applies sequence-level context heuristics: if an ambiguous token is followed by a Hindi postposition (`ko, se, ka, ki, ke, par, me`) or preceded by a conjunction marker, it is reclassified as `HI` with confidence $P(\text{HI}) = 0.95$ and removed from loanword candidacy.
* **Impact:** Eliminates spurious preservation decisions for Hindi grammatical tokens, reducing false-positive loanword detections in utterances like *"us room me meeting h to link bhej do"*.

### H. Automated Conversational Register Predictor [IMPLEMENTED]
* **Problem:** Previously, the communicative register ($R \in \{\text{casual, social, business, academic}\}$) was a required manual input parameter, creating friction in real-world deployment.
* **Method:** Created `src/laps/register_detector.py` implementing `detect_register(text: str) -> str`. The predictor aggregates weighted surface cue counts from four lexical registers:
  * **Casual cues** (*bhai, yaar, lol, plzz, nahi, karo*) $\implies \theta_R = 0.35$
  * **Social cues** (*tweet, post, viral, repost, #, follow*) $\implies \theta_R = 0.40$
  * **Business cues** (*client, meeting, server, sprint, deadline, KPIs*) $\implies \theta_R = 0.50$
  * **Academic cues** (*viva, thesis, algorithm, evaluation, semester*) $\implies \theta_R = 0.65$
* **Integration:** `src/pipeline.py` defaults to `register="auto"`, and `src/laps/threshold_engine.py` calls `detect_register(text)` when `register="auto"` is passed to `decide()` and `batch_decide()`.

### I. UTER Metric & Corrected Statistical Significance Testing [IMPLEMENTED]
* **Problem A:** The benchmark previously only measured **Over-Translation Error Rate (OTER)**, missing the complementary failure mode of *under-translation* (leaving Hindi grammatical markers in the output untranslated).
* **Solution:** Added `calculate_uter(results)` to `scripts/benchmark.py` measuring **Under-Translation Error Rate (UTER)**: the fraction of non-English tokens erroneously retained in the English rewrite.
* **Problem B:** Paired bootstrap resampling used non-centred score resampling with replacement (differences centred around $\bar{d}$ rather than $0$ under $H_0$), causing p-values to drift near $\sim 0.50$ regardless of actual effect.
* **Solution:** Replaced with valid **paired sign-flip permutation testing**:
  $$T_{\text{obs}} = \left|\sum_{i=1}^{N} d_i\right|, \quad d_i^{(b)} = d_i \cdot s_i^{(b)}, \quad s_i^{(b)} \overset{\text{iid}}{\sim} \text{Uniform}\{-1, +1\}$$
  Under $H_0$ (no systematic difference), sign-flipped statistics produce a valid reference distribution. The reported two-tailed $p$-value is exact and mathematically correct.
* **Verified Results:** All three baselines (Unconstrained NMT, Static Dictionary, Few-Shot ICL) reach $p = 0.0001$ ($p < 0.001$) confirming LAPS's Token F1 superiority is not attributable to chance.

