# Metric Validation & Mathematical Specification Audit (v3)

**Document:** `docs/research_validation_v3/metric_validation.md`  
**Date:** October 13, 2026  
**Scope:** Mathematical verification and implementation audit of evaluation metrics  

---

## 1. Metric Mathematical Formulations & Implementations

### 1.1 Over-Translation Error Rate (OTER)

**Definition:**  
OTER measures the proportion of gold-preserved loanwords that fail to appear in the generated output text (i.e., loanwords that were improperly translated away into matrix Hindi or dropped):

$$\text{OTER}(s) = 1.0 - \frac{|G_{\text{preserve}}(s) \cap Y(s)|}{|G_{\text{preserve}}(s)|}$$

Where:
- $G_{\text{preserve}}(s)$ is the set of gold preservation tokens for sentence $s$.
- $Y(s)$ is the bag of lowercased, word-tokenized surface strings in the hypothesis output.

**Edge Cases & Handling Policies:**
1. **Empty Gold Set ($|G_{\text{preserve}}(s)| = 0$):** If an utterance contains no required English borrowings, OTER is defined as $0.0000$ (no preservation failure possible).
2. **Casing & Punctuation:** All candidate matching is performed on case-normalized (`.lower()`) alphanumeric tokens stripped of leading/trailing punctuation (`re.findall(r"\b\w+\b", ...)`).
3. **Repeated Tokens:** Set intersection is computed over distinct lexical types; partial subword fragments do not count towards whole-word retention.
4. **Output Retention Recall:** Explicitly defined as:
   $$\text{Output Retention Recall}(s) = 1.0 - \text{OTER}(s) = \frac{|G_{\text{preserve}}(s) \cap Y(s)|}{|G_{\text{preserve}}(s)|}$$

---

### 1.2 Output-Level Retention Precision, Recall, and F1

To avoid rewarding trivial trivial copying of all English vocabulary, Output F1 evaluates both precision and recall of retained English tokens in the output string:

$$\text{Precision}_{\text{output}}(s) = \frac{|E(Y(s)) \cap G_{\text{preserve}}(s)|}{|E(Y(s))|}$$

$$\text{Recall}_{\text{output}}(s) = \frac{|E(Y(s)) \cap G_{\text{preserve}}(s)|}{|G_{\text{preserve}}(s)|} = 1.0 - \text{OTER}(s)$$

$$\text{F1}_{\text{output}}(s) = \frac{2 \cdot \text{Precision}_{\text{output}}(s) \cdot \text{Recall}_{\text{output}}(s)}{\text{Precision}_{\text{output}}(s) + \text{Recall}_{\text{output}}(s)}$$

Where $E(Y(s))$ is the set of English words present in the hypothesis.
If both precision and recall are $0$, $\text{F1}_{\text{output}} = 0.0$. If $|G_{\text{preserve}}| = 0$ and $|E(Y)| = 0$, $\text{F1}_{\text{output}} = 1.0$.

---

### 1.3 Token Preservation Classifier Metrics vs Generation Metrics

A fundamental methodological distinction is maintained:

| Metric Category | Target Evaluation | Applied To | Baselines Without Classifier |
| :--- | :--- | :--- | :--- |
| **Classifier Metrics** | Internal token decision: $d \in \{\text{PRESERVE}, \text{TRANSLATE}\}$ vs $l \in \{1, 0\}$ | LAPS, Few-Shot Heuristic, Static Dictionary | **N/A** (End-to-End systems have no classifier) |
| **Output Retention Metrics** | String presence of gold loanwords in final translated text | All systems (LAPS, Few-Shot, Static Dict, Unconstrained NMT) | Fully evaluated and credited |

**Classifier Macro F1:**
Calculated as the unweighted mean of per-sentence token classification F1 scores across candidate loanword tokens.

**Uncertainty Buffer Rate:**
$$\text{Uncertainty Rate} = \frac{|\{t \in T_{\text{cand}} \mid |P(\text{preserve}(t)) - \theta(\text{reg})| < 0.05\}|}{|T_{\text{cand}}|}$$
In LAPS, uncertain tokens are conservatively assigned to `PRESERVE` to prevent catastrophic information loss.

---

### 1.4 Machine Translation Metrics: SacreBLEU & chrF++

All generation metrics are computed using official `sacrebleu` library bindings (version 2.6.0):

- **Metric Library:** `sacrebleu` v2.6.0
- **Corpus BLEU:** `sacrebleu.corpus_bleu(hypotheses, [references], smooth_method="exp")`
  - Scale: $0 - 100$
  - Tokenizer: `13a` (standard academic benchmark tokenizer)
- **Corpus chrF++:** `sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)`
  - Scale: $0 - 100$
  - Character n-gram length: 6, Word n-gram order: 2
- **Methodological Qualification:** BLEU and chrF++ evaluate overall n-gram overlap against human reference translations. Because `opus-mt-hi-en` struggles with Romanized Latin input, BLEU scores are low across all systems ($0.25 - 0.85$). They are reported for academic completeness, but lexical preservation is evaluated via Output F1 and OTER.

---

### 1.5 Under-Translation Error Rate (UTER) & Hindi Leakage Proxy

**Gold UTER Status: N/A (Lacks Full-Utterance LID).**  
Strict Gold UTER requires word-level Language Identification (LID) labels for every Hindi matrix word in the source utterance to measure whether Hindi words were mistakenly left untranslated.

**Operational Leakage Proxy:**
Because full manual LID is unavailable, LAPS implements an operational proxy:
$$\text{Proxy}_{\text{leakage}}(s) = \frac{|\{w \in Y(s) \mid w \text{ is Romanized Hindi} \land w \notin G_{\text{preserve}}(s)\}|}{|Y(s)|}$$
Identified via Romanized Hindi stopword lexicons and character n-gram heuristics.

> **Directive:** The paper must never label this proxy as "Gold UTER". It is strictly an operational heuristic proxy.

---

## 2. Hand-Verified Test Suite Implementations

In `tests/test_laps.py`, the following unit tests verify these metric definitions against known ground-truth matrices:
- `test_preservation_classifier_hand_crafted_matrix`: Verifies TP, FP, FN, TN, precision, recall, and F1 on synthetic confusion matrices.
- `test_output_preservation_hand_crafted_example`: Verifies OTER, output precision, recall, and F1 on hand-crafted sentences with known missing and spurious English words.
