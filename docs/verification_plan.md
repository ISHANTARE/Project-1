# LAPS Verification & Testing Plan

This document details the verification protocols used to validate the accuracy, performance, and novelty of the **LAPS (Linguistic Adaptive Preservation Scoring)** framework.

---

## 🧪 1. Unit & Functional Verification

We use `pytest` to verify the execution of individual Python components:

| Component | Target File | Verification Criteria |
| :--- | :--- | :--- |
| **Tokenizer** | `tokenizer.py` | Validates split boundaries, handling of punctuation, and Latin/Devanagari scripts. |
| **Feature Extractor** | `feature_extractor.py` | Asserts score calculations fall strictly within the $[0.0, 1.0]$ range. |
| **Scorer Engine** | `scorer.py` | Asserts LAPS equations compute correctly given predefined weights. |
| **Threshold Engine** | `threshold_engine.py` | Verifies dynamic mapping of thresholds ($\theta$) based on register configurations. |
| **Prompt Builder** | `prompt_builder.py` | Asserts hard token constraints are cleanly parsed into prompt templates. |

---

## 📈 2. Automated Performance Metrics

LAPS rewrites will be validated quantitatively against reference translations on the test benchmark:

1. **Semantic Similarity (BERTScore):** Asserts the preserved rewrite does not lose semantic grounding relative to reference sentences.
2. **Translation Quality (COMET):** Evaluates overall sentence fluency using cross-lingual embeddings.
3. **BLEU & ChrF++:** Measures structural overlap. Note that since LAPS intentionally preserves specific source-language loanwords, BLEU may decrease slightly compared to fully translated targets, but semantic and readability indices should increase.
4. **Over-Translation Error Rate (OTER):** Quantifies how many naturally borrowed terms were incorrectly translated:
   $$\text{OTER} = \frac{\text{Number of loanwords translated}}{\text{Total loanwords in source sentence}}$$

---

## 👥 3. Human Evaluation Protocol

To prove the practical value of LAPS, a human evaluation survey will be conducted with 15-20 bilingual (e.g., Hinglish-English) speakers:

* **Readability (Scale 1-5):** How natural and flowing is the rewritten text?
* **Meaning Preservation (Scale 1-5):** Does the rewrite capture the exact intent and domain nuance of the original source?
* **Explanation Acceptability (Scale 1-5):** How logical, clear, and helpful are the generated token explanations?

---

## 🏁 4. Verification Checklists for Academic Reviews

### **Review 1 Checklist**
* [ ] Verify folder structures and virtual environments compile cleanly.
* [ ] Run basic checks on character parsing and vocabulary lexicons.
* [ ] Launch Streamlit UI framework with mock inputs.

### **Review 2 Checklist**
* [ ] Execute standard token-level classification tests.
* [ ] Verify feature scoring runs successfully on input sentences.
* [ ] Verify prompt compilation contains exact preservation markers.

### **Review 3 & Final Checklist**
* [ ] Generate experimental charts and data comparison matrices.
* [ ] Complete automated evaluation runs.
* [ ] Verify LaTeX draft formatting matches IEEE guidelines.
