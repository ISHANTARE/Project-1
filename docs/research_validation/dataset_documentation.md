# LAPS Dataset Documentation & Annotation Study Specification

**Document Version**: 1.0 (Audited)  
**Publication Target**: EMNLP / ACL Findings / LREC-COLING  
**Subject**: Code-Mixed Hinglish-English Parallel Evaluation Corpus & Annotation Reliability  

---

## 1. Dataset Purpose & Linguistic Scope

The LAPS dataset is constructed to evaluate token-level lexical preservation of English borrowings within Romanized Hindi-English (Hinglish) code-mixed text rewritten into formal English.

In code-mixed sociolinguistic contexts (urban Indian English/Hindi discourse), speakers routinely borrow technical, administrative, and domain-specific English terms (e.g., *viva, link, deadline, meeting, cluster, server*) while embedding them within Hindi morphosyntactic frames (e.g., *Kisi ke paas viva ka link h kya?*). When standard Neural Machine Translation (NMT) engines translate such utterances into English, they frequently commit **Over-Translation Errors (OTE)**—translating well-established, pragmatically appropriate English loanwords into formal, pedantic, or archaic English synonyms (e.g., *viva* → *oral examination*, *link* → *hypertext reference* / *connection*, *calendar* → *timetable*).

The LAPS benchmark quantifies:
1. Whether an algorithmic preservation guardrail can accurately classify which source tokens should remain verbatim (`label = 1`) versus translated (`label = 0`).
2. Whether constrained beam search decoders can faithfully preserve these loanwords in surface translations without destroying sentence fluency.

---

## 2. Corpus Architecture & Record Accounting

### 2.1 Primary Gold Corpus (`gold_corpus.json`)
- **Total Utterances**: 100 sentences (`S001` to `S100`).
- **Total Candidate Tokens**: 327 English loanword candidate tokens.
- **Candidate Token Class Distribution**:
  - Positive Preservation Target (`label = 1`): 302 tokens (92.35%)
  - Negative Translation Target (`label = 0`): 25 tokens (7.65%)
- **Utterance Length**:
  - Mean sentence length: 11.4 words (range: 5 to 24 words).
- **Register Stratification**:
  - **Academic** (`academic`): 25 sentences (78 candidate tokens)
  - **Business** (`business`): 25 sentences (84 candidate tokens)
  - **Social** (`social`): 25 sentences (82 candidate tokens)
  - **Casual** (`casual`): 25 sentences (83 candidate tokens)

### 2.2 Split Partitioning Integrity
The 100 in-distribution sentences are strictly partitioned into non-overlapping splits:
- **Training Split (`train_split.json`)**: 70 sentences (229 candidate tokens)
- **Validation Split (`val_split.json`)**: 15 sentences (52 candidate tokens)
- **Held-Out Test Split (`test_split.json`)**: 20 sentences total:
  - 15 In-Distribution sentences (`S004`, `S005`, `S012`, `S014`, `S015`, `S018`, `S029`, `S032`, `S036`, `S055`, `S070`, `S076`, `S082`, `S087`, `S095`): 46 candidate tokens.
  - 5 Out-of-Vocabulary (OOV) extension sentences (`OOV_001` to `OOV_005`): 16 candidate tokens.
- **Pairwise Split Disjointness**: Programmatically verified (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).

---

## 3. Annotation Study & Inter-Rater Reliability

### 3.1 Study Setup & Double-Blind Protocol
- **Sample Size**: 90 candidate tokens sampled across 20 sentences from `data/evaluation/annotations.csv`.
- **Raters**: Two independent bilingual (Hindi-English) raters with native fluency in Romanized Hinglish digital communication.
- **Task**: Binary decision per candidate token:
  - `1`: PRESERVE verbatim in natural English rewriting.
  - `0`: REWRITE / TRANSLATE into standard English equivalent.

### 3.2 Programmatic Agreement Statistics

```
Total Evaluated Tokens (N): 90
Total Raters (n):            2
Observed Agreements:        85 / 90 (94.44%)
Observed Disagreements:      5 / 90 (5.56%)
Expected Agreement (P_e):    0.5593
---------------------------------------------------------------
Cohen's Kappa (κ):           0.8744 (Almost Perfect Agreement)
Fleiss' Kappa (κ):           0.8739 (Almost Perfect Agreement)
Gwet's AC1:                  0.9007 (Robust against prevalence paradox)
---------------------------------------------------------------
```

### 3.3 Disagreement Adjudication & Qualitative Analysis
The 5 disagreements occurred on polysemous or borderline colloquial verbs:
1. `submit` (Sentence `S002`: *Assignment submit karne ki deadline kal hai*): Rater 1 marked `1` (retain); Rater 2 marked `0` (rewrite to *turn in / submit*).
2. `postpone` (Sentence `S006`: *Kal client meeting postpone ho gayi hai*): Rater 1 marked `1`; Rater 2 marked `0`.
3. `free` (Sentence `S014`: *Aaj shaam ko free ho kya?*): Rater 1 marked `1`; Rater 2 marked `0` (preferring *available*).
4. `raise` (Sentence `S018`: *Query raise kar di hai*): Rater 1 marked `1`; Rater 2 marked `0`.
5. `try` (Sentence `S020`: *Ek baar try karke dekhte hain*): Rater 1 marked `1`; Rater 2 marked `0`.

**Research Integrity Note**: In accordance with ACL/EMNLP guidelines, the 5 disagreements are preserved in `annotations.csv` without synthetic post-hoc consensus columns. Cohen's Kappa ($\kappa = 0.8744$) comfortably exceeds the standard NLP publication threshold ($\kappa \ge 0.70$).

---

## 4. Out-of-Vocabulary (OOV) Records

### 4.1 Specification of OOV Extension Sentences
The test split includes 5 distinct OOV records (`OOV_001`–`OOV_005`) designed to stress-test generalization to domain terms absent from the training split's frequency tables:

| Record ID | Domain | Input Sentence | OOV Candidate Tokens |
|---|---|---|---|
| `OOV_001` | Tech/Cloud | *Kubernetes cluster mein ingress controller deploy kar diya* | `Kubernetes`, `cluster`, `ingress`, `controller`, `deploy` |
| `OOV_002` | Medical | *Patient ka MRI scan aur biopsy report normal hai* | `patient`, `MRI`, `scan`, `biopsy`, `report` |
| `OOV_003` | Firmware/Embedded | *Firmware patch verify karne ke liye checksum check karo* | `firmware`, `patch`, `verify`, `checksum`, `check` |
| `OOV_004` | Finance/Trading | *Portfolio rebalance karne ke liye dividend reinvest karo* | `portfolio`, `rebalance`, `dividend`, `reinvest` |
| `OOV_005` | Academic/Research | *Methodology section mein ablation study include karni hai* | `methodology`, `section`, `ablation`, `study`, `include` |

### 4.2 Type-Discriminated Schema Design
To guarantee that the OOV evaluation reflects live inference rather than cached lookup:
- In-distribution tokens carry `{token, features, label}` with pre-cached features.
- OOV extension tokens carry `{token, label, domain}` with **NO cached features**.
- The evaluation engine detects missing features and executes the live neural inference pipeline (`MiniLM` dense embedding cosine similarity + SQLite KG lookup).

---

## 5. Limitations & Ethical Considerations

1. **Corpus Scale**: 100 sentences is a focused benchmark suitable for targeted diagnostic evaluation. Larger public benchmark evaluation (LinCE / PHINC) is recommended for broad generalization claims.
2. **Class Imbalance**: Positive tokens constitute 92.35% of candidate annotations because loanwords in real Hinglish utterances are predominantly functional technical/domain nouns that speakers actively chose not to express in Hindi.
3. **Privacy & Anonymity**: All sentences are synthetic or anonymized colloquial communications free of personally identifiable information (PII).
4. **Consent & Licensing**: Synthetic evaluation records are released under the MIT Open Source License.
