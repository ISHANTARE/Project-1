# Data Provenance & Leakage Audit Report (v3)

**Document:** `docs/research_validation_v3/data_provenance_and_leakage_audit.md`  
**Date:** October 13, 2026  
**Audit Scope:** Dataset splits, provenance of external lexical assets, feature cache insulation, and contamination verification  

---

## 1. Dataset Partition Integrity & Disjointness Verification

The evaluation framework operates on four distinct corpora stored under `data/evaluation/`:

| Dataset Artifact | Sentence Count | Candidate Tokens | Role in Research Protocol | Disjointness Status |
| :--- | :---: | :---: | :--- | :--- |
| `train_split.json` | 70 | 227 | Hyperparameter exploration & register prior tuning | $Train \cap (Val \cup Test) = \emptyset$ |
| `val_split.json` | 15 | 52 | Validation threshold calibration ($\theta$) | $Val \cap (Train \cup Test) = \emptyset$ |
| `test_split.json` | 20 | 78 | Held-out evaluation (15 held-out + 5 unseen OOV) | $Test \cap (Train \cup Val) = \emptyset$ |
| `gold_corpus.json` | 100 | 327 | Full curated reference corpus (Train + Val + 15 held-out) | Contains all non-OOV sentences |

### 1.1 Disjointness Verification Test Results
A rigorous programmatic audit across all four dataset artifacts confirmed:
1. **Sentence ID Disjointness:**
   - $\text{IDs}(\text{train}) \cap \text{IDs}(\text{val}) = \emptyset$ (0 overlap)
   - $\text{IDs}(\text{train}) \cap \text{IDs}(\text{test}) = \emptyset$ (0 overlap)
   - $\text{IDs}(\text{val}) \cap \text{IDs}(\text{test}) = \emptyset$ (0 overlap)
2. **Normalized Text Disjointness:**
   - Lowercased, whitespace-normalized sentence text comparison confirms **zero identical sentences** between train, val, and test splits.
3. **Test Split Composition:**
   - Exactly 15 sentences are drawn from `gold_corpus.json` (`S004`, `S005`, `S007`, `S012`, `S018`, `S025`, `S032`, `S036`, `S041`, `S049`, `S055`, `S063`, `S070`, `S076`, `S084`).
   - Exactly 5 sentences are newly constructed Out-Of-Vocabulary sentences (`OOV_001` through `OOV_005`), which **do not exist** in `gold_corpus.json` or `train_split.json`.

---

## 2. Definition & Audit of Out-of-Vocabulary (OOV) Records

### 2.1 Formal Definition of OOV in LAPS
An OOV evaluation record is defined as an utterance containing English loanwords that:
1. Do not appear anywhere in the training split (`train_split.json`).
2. Do not have pre-computed feature vectors in the cached feature database.
3. Require the live feature extractor (`extract_features()`) to compute all 6 features on the fly using runtime MiniLM sentence embeddings, live SQLite Knowledge Graph queries, and live WordNet lookups.

### 2.2 Schema & Feature Cache Insulation
The 5 OOV records (`OOV_001` - `OOV_005`) contain tokens:
- `fintech`, `agile`, `blockchain`, `latency`, `hyperparameter`, `microservices`, `kubernetes`, `gradient`, `backpropagation`, `zero-shot`.

**Audit Outcome:**
- Confirmed: None of these 5 sentences contain stored `features` dictionaries in their JSON files.
- Confirmed: During inference, `is_candidate = True` triggers the live pipeline (`extract_features()`).
- All 5 OOV sentences were successfully processed by LAPS under strict neural inference without error.

---

## 3. Provenance of Feature Resources & Lexical Assets

To prevent data snooping, the origin of all background data resources was audited:

### 3.1 Borrowing Frequency Prior (BF)
- **Source:** Pre-compiled frequency index derived from Romanized Hinglish conversational corpora and general Indian English web text.
- **Leakage Status: CLEAN.** The dictionary maps word types to empirical frequency log-scores. It does not contain test-set sentence IDs, sentence-level co-occurrences, or human gold labels.

### 3.2 Domain Specificity & Knowledge Graph (DS & KG)
- **Source:** SQLite database (`data/laps_kg.db`) structured around Wikipedia and Wikidata technical domain taxonomies (IT, medical, academic, business).
- **Leakage Status: CLEAN.** KG edges represent semantic relationships between concepts (e.g., `kubernetes` $\rightarrow$ `containerization`). The KG was not built from the evaluation corpus.

### 3.3 Semantic Importance & Context Consistency (SI & CC)
- **Source:** Pre-trained dense encoder `sentence-transformers/all-MiniLM-L6-v2`.
- **Leakage Status: CLEAN.** Standard frozen weights from HuggingFace. Zero fine-tuning on the evaluation data.

### 3.4 Audience Familiarity (AF)
- **Source:** Register prior lookup conditioned on the utterance register tag (`academic`, `business`, `social`, `casual`).
- **Leakage Status: CLEAN.** Fixed scalar prior table.

### 3.5 Translation Benefit (TB)
- **Source:** Python `nltk.corpus.wordnet` synsets mapped against Hindi WordNet lexical sets.
- **Leakage Status: CLEAN.** Standard external linguistic resource.

---

## 4. Distinction Between Shared Vocabulary and Data Leakage

In cross-lingual NLP, the presence of common English loanwords (e.g., `phone`, `camera`, `meeting`) in both background frequency tables and test utterances is **legitimate linguistic knowledge**, not data leakage:
- **Legitimate Prior Knowledge:** General-domain word frequency tables and Knowledge Graph concepts exist independently of test utterances.
- **Prohibited Contamination (Audited & Absent):**
  - No gold binary labels ($l \in \{1, 0\}$) were accessible to the feature extractor or scorer.
  - Operational thresholds ($\theta_{\text{reg}}$) were calibrated strictly on `val_split.json` and fixed before running `test_split.json`.
  - Feature weights ($w_1 \dots w_6$) were defined by mathematical specification, not tuned by supervised gradient descent on the test set.
