# Baseline & Environment Audit Report (v3)

**Document:** `docs/research_validation_v3/baseline_audit.md`  
**Date:** October 13, 2026  
**System Under Test:** LAPS (Linguistic Adaptive Preservation Scoring)  
**Primary Execution Commit:** `e66693a` / `876f04d`  

---

## 1. Execution Environment & Hardware Specifications

The audit and experimental evaluations were executed directly on the user's primary GPU workstation:

- **Operating System:** Windows 11 Home (64-bit)
- **Host Python:** Python 3.10.9 (C:\Users\aryan\AppData\Local\Programs\Python\Python310\python.exe)
- **Deep Learning Framework:** PyTorch 2.11.0+cu128 (`torch.cuda.is_available() == True`)
- **GPU Accelerator:** NVIDIA GeForce RTX 4050 Laptop GPU
  - **Total VRAM:** 6,141.00 MB (6.0 GB)
  - **CUDA Runtime:** CUDA 12.8
  - **Driver Capabilities:** SM 8.9 (Ada Lovelace)
- **NLP Libraries:**
  - `transformers`: 5.15.1
  - `sacrebleu`: 2.6.0
  - `sentence-transformers`: 5.3.0
  - `torchvision`: 0.26.0+cu128

---

## 2. Forensic Investigation of Historical Discrepancies

### 2.1 Silent Surrogate Fallback (Finding 1)
- **Historical Evidence:** All historical outputs in `results/predictions_real_neural_gold_100.json` and `results/predictions_real_neural_test_split.json` had recorded `backend_execution_mode = "DETERMINISTIC_SURROGATE_FALLBACK"`.
- **Root Cause Identified:** Dual module registration in Python's `sys.modules`. In `scripts/benchmark.py`, `sys.path.insert(0, str(PROJECT_ROOT / "src"))` caused `llm.nmt_backend` to be registered. `benchmark.py` loaded the MarianMT model onto GPU inside that singleton (`_NMT_INSTANCE`). However, `src/llm/rewriter.py` used `from src.llm.nmt_backend import get_nmt_engine`, creating a second module entry (`src.llm.nmt_backend`) with an uninitialized instance (`_is_loaded = False`). When `LAPSRewriter` invoked `translate()`, it called the second uninitialized instance, immediately caught the exception, and executed `_generate_realistic_nmt_translation()` (regex surrogate).
- **Repair Status: FIXED.**
  1. Synchronized singleton across all aliased keys in `sys.modules`.
  2. Refactored `LAPSRewriter.__init__` to accept explicit `nmt_engine` injection.
  3. Added strict `--require-real-neural` flag that raises `RuntimeError` rather than silently catching exceptions.

### 2.2 Baseline Asymmetry (Finding 2)
- **Historical Problem:** The historical comparison compared surrogate LAPS (which retained words by string replacement) against real neural unconstrained generation, creating an invalid causal claim.
- **Repair Status: FIXED.** Both systems were rerun under strictly paired conditions on the exact same checkpoint (`Helsinki-NLP/opus-mt-hi-en`) on GPU.

### 2.3 Evidence Manifest Hash Discrepancies (Finding 3)
- **Historical Problem:** `docs/research_validation/evidence_manifest.json` was formatted as a flat dictionary pointing to files in `results/`, where several files had been modified by ad-hoc diagnostic scripts after manifest creation.
- **Repair Status: FIXED.** All historical files were backed up in `results/historical_audit_backup/`. The verified v2 bundle in `results/research_validation_v2/` was audited and confirmed to match 100% (21/21 files) against its manifest.

### 2.4 Test Suite Failures (Finding 4)
- **Historical Problem:** Earlier test runs showed 2 failures (empty `preserve_list` expectation mismatch in mock pipeline, and missing `transformers` in CPU environments).
- **Repair Status: FIXED.** The test suite was expanded to 55 tests, including strict neural rejection tests. `pytest tests/ -v` passes **55/55 tests** (100%).

---

## 3. Verified Benchmark Outputs (Strict Real Neural on GPU)

### 3.1 Held-Out Test Split ($N=20$, including 5 OOV sentences)
- **LAPS (Proposed):**
  - Execution Mode: `REAL_INFERENCE_CONSTRAINED` (100%)
  - OTER: **0.2450** (Output Retention Recall: **0.7550**)
  - Output F1: **0.7796**
  - Token Classifier Macro F1: **0.8494** (Precision: 0.9138, Recall: 0.7794)
  - BLEU: 0.29 | chrF++: 26.06 | UTER Proxy: 0.0646
- **Unconstrained NMT Baseline:**
  - Execution Mode: `REAL_INFERENCE_UNCONSTRAINED` (100%)
  - OTER: 0.9500 (Output Retention Recall: 0.0500)
  - Output F1: **0.0733**
  - Token Classifier Macro F1: **N/A** (End-to-End sequence generation)
  - BLEU: 0.85 | chrF++: 12.91 | UTER Proxy: 0.0000
- **Paired Significance:** $\Delta\text{F1} = +0.7063$, Monte Carlo Permutation $p = 0.0002$ ($B=5,000$, seed=42, Bonferroni-corrected $\alpha = 0.0167$).

### 3.2 Full Gold Corpus ($N=100$, 327 Candidate Loanwords)
- **LAPS (Proposed):** Output F1 = **0.8570**, OTER = **0.1603**, Classifier F1 = **0.9567**.
- **Unconstrained NMT Baseline:** Output F1 = **0.1098**, OTER = **0.9240**, Classifier F1 = **N/A**.
- **Paired Significance:** $\Delta\text{F1} = +0.7472$, Monte Carlo Permutation $p = 0.0002$.

---

## 4. Current Status of Research Claims & Unresolved Questions

1. **Token Preservation Claim:** Validated. LAPS Constrained Beam Search significantly out-preserves unconstrained translation on Romanized Hinglish ($F_1 = 0.7796$ vs $0.0733$, $p < 0.001$).
2. **Translation Fluency Qualification:** BLEU remains low ($\sim 0.29$) across both systems because `opus-mt-hi-en` is trained on Devanagari Hindi, not Romanized Latin script. The paper must not claim that LAPS solves overall rewriting fluency; it specifically solves the lexical borrowing preservation problem.
3. **Gold UTER Qualification:** Strict Gold UTER remains qualified as **N/A** because word-level language identification labels for every matrix Hindi word in the source utterances are not available.
