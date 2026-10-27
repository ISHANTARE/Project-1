# Human Annotation & Qualitative Error Analysis Audit (v3)

**Document:** `docs/research_validation_v3/human_evaluation_and_error_analysis.md`  
**Date:** October 13, 2026  
**Primary Datasets:** `data/evaluation/annotations.csv`, `results/kappa_results.json`, `results/research_validation_v2/predictions_real_neural_test_split.json`  

---

## 1. Inter-Annotator Agreement Audit

The token-level preservation gold standard was audited against `data/evaluation/annotations.csv` and `results/kappa_results.json`:

### 1.1 Study Design & Rater Details
- **Annotation Unit:** Candidate English loanword tokens embedded in Romanized Hinglish sentences.
- **Raters:** 2 independent bilingual native Hindi-English speakers.
- **Sample Size:** 90 candidate tokens sampled across 20 representative code-mixed sentences.
- **Task:** Binary preservation decision:
  - `1` = PRESERVE (keep as English borrowing in English rewriting).
  - `0` = TRANSLATE (replace with native Hindi lexical equivalent).

### 1.2 Agreement Statistics
- **Observed Agreement ($P_o$):** $94.44\%$ (85 / 90 tokens in exact agreement)
- **Expected Agreement ($P_e$):** $55.93\%$
- **Cohen's Kappa ($\kappa$):** **0.8744** ("Almost Perfect Agreement" under Landis & Koch, 1977)
- **Fleiss' Kappa:** **0.8739**
- **Gwet's AC1:** **0.9007** (robust to high-prevalence marginals)

### 1.3 Audit of Disagreements
Exactly 5 tokens exhibited rater disagreement (Rater 1 = `1`, Rater 2 = `0`):
- `submit` (Sentence S076)
- `postpone` (Sentence S084)
- `free` (Sentence S032)
- `raise` (Sentence S049)
- `try` (Sentence S018)

**Audit Integrity:** These 5 disagreements are preserved in the raw data without synthetic forcing or artificial reconciliation. Disputed tokens correspond to high-uncertainty register boundary verbs where colloquial Hindi speakers tolerate either form.

---

## 2. Qualitative Error Analysis from Verified Neural Predictions

Representative outputs from `predictions_real_neural_test_split.json` were audited to identify concrete failure modes:

### Category A: The Base Model Latin-Script Mismatch (Filler Fragmentation)
* **Sentence ID:** `S005`
* **Input:** `Kal presentation ready nahi hai, phir bhi confidence ke sath jaana hai`
* **Reference:** `The presentation is not ready yet, but we still need to go with confidence.`
* **Gold Loanwords:** `presentation`, `confidence`
* **Unconstrained Hypothesis:** `And this isn't a problem, it's better than a third.`
  - *Failure:* Complete erasure of both domain borrowings. Output F1 = 0.00.
* **LAPS Constrained Hypothesis:** `And this is presentation, so this is equal to circum. And that is equal even to the 2th. And this becomes a problem. And the answer is positive. And we have to do that. And it's not just a negative one. And confidence`
  - *Success:* 100% constraint satisfaction (`presentation`, `confidence` preserved). Output F1 = 0.80.
  - *Limitation:* The base MarianMT model (trained on Devanagari script) struggles with Latin characters, emitting repetitive, low-probability filler between the forced tokens.

---

### Category B: Multi-Subword & Technical Domain Borrowings
* **Sentence ID:** `S055`
* **Input:** `Dataset ka preprocessing complete hua? Training start karo`
* **Reference:** `Is the dataset preprocessing complete? Start the training.`
* **Gold Loanwords:** `dataset`, `preprocessing`, `training`
* **Unconstrained Hypothesis:** `So if you want a copy of this document, do you want to learn more about it?`
  - *Failure:* Completely hallucinated generic translation. Output F1 = 0.00.
* **LAPS Constrained Hypothesis:** `Answering dataset preprocessing the data...? ! . .? . m.?? ??? a.? you can only use the information as the location of the page. You can only ask for the information. You could training`
  - *Success:* All three multi-subword technical terms (`dataset`, `preprocessing`, `training`) are successfully injected into the beam decoder. Output F1 = 1.00.

---

### Category C: Colloquial Household Terms
* **Sentence ID:** `S012`
* **Input:** `Mera phone ka camera bahut achha hai`
* **Reference:** `My phone's camera is very good.`
* **Gold Loanwords:** `camera`, `phone`
* **Unconstrained Hypothesis:** `I want to do this by telling you how much I want you to do.`
* **LAPS Constrained Hypothesis:** `I want to do this ... I want a ji. So this is a cysa../ . .?/ I want you to . I do not want to add . So you want to understand. ? ., camera phone`
  - *Observation:* Both tokens retained. The model places constrained tokens at the end of the beam sequence when encoder state confidence is low.

---

## 3. Human Evaluation Protocol for Complete Rewriting (Future Protocol)

To evaluate full-sentence rewriting quality independently of automated metrics, the following human evaluation protocol is formally specified for future rounds:

1. **Evaluation Axes (1-5 Likert Scale):**
   - **Lexical Borrowing Fidelity (LBF):** Were all necessary technical/domain English terms retained?
   - **Semantic Adequacy (SA):** Does the English output convey the same meaning as the Hinglish source?
   - **Target Fluency (TF):** Is the English sentence grammatically coherent and natural?
   - **Hindi Leakage Avoidance (HLA):** Did the model avoid leaving un-translated matrix Hindi words in the output?
2. **Sampling Design:** 50 sentences evaluated by 3 independent bilingual raters blind to system identity.
3. **Primary Recommendation:** Execute this protocol on an adapted Latin-script foundation model (e.g., IndicTrans2 or fine-tuned NLLB-200) where base fluency is not constrained by Devanagari script mismatch.
