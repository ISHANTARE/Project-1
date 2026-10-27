# LAPS Baseline Research & Experimental Audit

**Audit Date**: October 13, 2026  
**Repository**: LAPS — Linguistic Adaptive Preservation Scoring  
**Git Branch**: `main`  
**Git Commit**: `5870c0c4cd66562881767153d537cb33fb6f9a2f` (`docs: preserve LAPS defence evidence and preparation guide`)  
**Working Tree Status**: Clean (no uncommitted changes at baseline audit start)

---

## 1. Environment & Computational Hardware

| Attribute | Detected Value | Verification Method |
|---|---|---|
| **OS** | Windows 11 AMD64 (build platform win32) | `sys.platform`, `os.uname` |
| **Python** | 3.10.9 (tags/v3.10.9:1dd9be6, Dec 6 2022) | `sys.version` |
| **PyTorch** | 2.11.0+cu128 | `torch.__version__` |
| **CUDA Available** | `True` (Device count: 1) | `torch.cuda.is_available()` |
| **GPU Hardware** | NVIDIA GeForce RTX 4050 Laptop GPU (5.996 GB VRAM) | `torch.cuda.get_device_name(0)` |
| **Transformers** | 5.15.1 | `transformers.__version__` |
| **SacreBLEU** | 2.6.0 | `sacrebleu.__version__` |
| **Pytest** | 9.1.1 (pluggy 1.6.0, asyncio 1.4.0, cov 7.1.0) | `pytest --version` |

---

## 2. Existing Test Results (Baseline)

- **Test Suite Command**: `pytest -v`
- **Total Test Count**: 45 collected items
- **Passed**: 45
- **Failed**: 0
- **Skipped**: 0
- **Warnings**: 16 deprecation warnings from `fpdf2` (`ln` parameter deprecated in `exporter.py`)
- **Execution Time**: 30.35 seconds
- **Exit Code**: 0

The 45 existing unit tests pass cleanly and verify CMI detection, tokenization, neural char LID, feature caching/extraction, threshold engines, export utilities, SacreBLEU metrics, paired permutation significance tests, data split disjointness, and surrogate reference independence.

---

## 3. Existing Benchmark Results & Artifacts

- **Primary Command**: `python scripts/benchmark.py`
- **Corpus**: `data/evaluation/gold_corpus.json` (N = 100 sentences, 327 candidate tokens)
- **Historical Benchmark Results**: `results/benchmark_results_v2.json`, `results/benchmark_results.json`
- **Reported Historical Metrics (from `benchmark_results_v2.json`)**:
  - **LAPS (Proposed)**:
    - OTER: 0.0108 (Output retention recall: 0.9892)
    - Output F1: 0.9485
    - Corpus BLEU: 4.16, Corpus chrF++: 38.38
    - Preservation Classifier F1 (macro): 0.9567 (micro: 0.9617, prec: 0.9290, rec: 0.9967)
    - Latency (Scoring on pre-cached features): 0.21 ms mean total
    - **Execution Mode**: `DETERMINISTIC_SURROGATE_FALLBACK`
  - **Few-Shot Demonstration Baseline**:
    - OTER: 0.0485, Output F1: 0.9269, BLEU: 3.99, chrF++: 36.31
    - Classifier macro F1: 0.0973 (fixed 17-word lexicon)
    - Execution Mode: `DETERMINISTIC_SURROGATE_FALLBACK`
  - **Static Dictionary Baseline**:
    - OTER: 0.0852, Output F1: 0.9041, BLEU: 3.99, chrF++: 36.29
    - Classifier macro F1: 0.0477 (fixed 12-word lexicon)
    - Execution Mode: `DETERMINISTIC_SURROGATE_FALLBACK`
  - **Unconstrained NMT Baseline**:
    - OTER: 0.1193, Output F1: 0.8748, BLEU: 2.10, chrF++: 35.02
    - Classifier macro F1: `N/A` (no internal classifier)
    - Execution Mode: `DETERMINISTIC_SURROGATE_FALLBACK`

---

## 4. Available Datasets and Model Checkpoints

### 4.1 Evaluation Datasets
1. `data/evaluation/gold_corpus.json`: 100 sentences (S001–S100), 327 candidate tokens across 4 registers (`academic`: 25, `casual`: 25, `business`: 25, `social`: 25).
2. `data/evaluation/train_split.json`: 70 sentences (in-distribution split).
3. `data/evaluation/val_split.json`: 15 sentences (in-distribution validation split).
4. `data/evaluation/test_split.json`: 20 sentences (15 in-distribution held-out sentences + 5 OOV extension records: `OOV_001`–`OOV_005`).
5. `data/evaluation/annotations.csv`: 90 candidate tokens across 20 sentences, evaluated by 2 independent bilingual raters (85 agreements, 5 disagreements).
6. `data/evaluation/gold_corpus_real_50.json`: Ingested subset of 50 examples.

### 4.2 Local / Cached Model Checkpoints
1. `data/models/char_lid.pt`: 100.9 KB character-level CNN/BiLSTM for Romanized Hindi/English language identification.
2. `~/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-hi-en`: Local cached Seq2Seq MarianMT model (~300 MB).
3. `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`: Local cached embedding model for semantic feature extraction.
4. `~/.cache/huggingface/hub/models--transformers-community--constrained-beam-search`: Constrained beam search custom generate repo.

---

## 5. Neural Backend Root Cause Analysis

### 5.1 The Transformers `force_words_ids` Compatibility Issue
In earlier benchmark runs, the neural generation path failed and fell back to `_generate_realistic_nmt_translation` (regex/template dictionary surrogate).

**Diagnostic Investigation Results**:
1. When `transformers` upgraded to version 5.x / late 4.x, the constrained beam search algorithm was decoupled from core `transformers.generation` and moved to a remote Hub module (`transformers-community/constrained-beam-search`).
2. Calling `model.generate(force_words_ids=...)` under this setup triggered an exception:
   `OSError: Constrained Beam Search requires trust_remote_code=True in your generate call, since it loads https://hf.co/transformers-community/constrained-beam-search.`
3. Furthermore, when `trust_remote_code=True` was passed without specifying `custom_generate="transformers-community/constrained-beam-search"` and without ensuring that `custom_generate/beam_constraints.py` and `beam_search.py` were locally present, HuggingFace's dynamic module loader encountered an `OSError: Could not locate custom_generate\beam_constraints.py` due to a poisoned `.no_exist` cache entry created during a prior unauthenticated network attempt.
4. The exception in `nmt_backend.py:translate()` was caught by a general `except Exception as e:` block, which set `self.last_execution_mode = "DETERMINISTIC_SURROGATE_FALLBACK"` and executed the regex/template surrogate.
5. In addition, `NMTTranslationEngine.load_model()` was never called automatically by `benchmark.py` or `LAPSRewriter.__init__()`, so `self._is_loaded` remained `False` during standard benchmark script runs.

### 5.2 Resolution Strategy
- Clean the `.no_exist` cache directory.
- Ensure `beam_constraints.py`, `beam_search.py`, and `generate.py` are explicitly downloaded and available in the snapshot.
- Pass `custom_generate="transformers-community/constrained-beam-search"` and `trust_remote_code=True` in `model.generate()`.
- Add proper generation decoding constraints (`no_repeat_ngram_size=3`, `max_new_tokens=48`, dynamic beams) to prevent repetition loops in MarianMT.
- Add an explicit `--backend-mode` flag or `load_model()` invocation to allow paired execution of:
  - Real unconstrained neural model (`Helsinki-NLP/opus-mt-hi-en`)
  - Real LAPS-constrained neural model (`Helsinki-NLP/opus-mt-hi-en` + `force_words_ids`)
  - Deterministic surrogate baseline (labeled clearly as surrogate)

---

## 6. Material Gaps & Research Integrity Risks Discovered

1. **Surrogate Generation Labeled as Neural Baseline**: Historical benchmark results used the deterministic surrogate fallback (`OVER_TRANSLATION_MAP` + regex templates), although documentation correctly recorded `backend_execution_mode: DETERMINISTIC_SURROGATE_FALLBACK`. No actual neural generation had run end-to-end on the corpus.
2. **Feature Ablation Script Flaw**: In `scripts/run_ablation_study.py`, feature weights were assigned by index `f"w{i+1}"` over remaining features rather than matching feature identities, and a fixed cutoff `theta = 0.50` rendered the soft-sigmoid test algebraically identical to the hard cutoff test.
3. **Corpus Size**: The primary evaluation corpus contains 100 sentences with 327 candidate tokens. While useful for proof of concept and course project evaluation, claims of broad statistical power must be framed with appropriate confidence intervals and bootstrap resampling.
4. **Proxy Language Identification (UTER)**: The lack of token-level gold LID for matrix Hindi means UTER is currently a heuristic leakage proxy rather than verified human LID.
5. **Pre-cached Features vs Live Latency**: Latency of 0.21 ms reported in Table 1 measures dot-product scoring of pre-cached dictionary values, whereas live inference (MiniLM embeddings + SQLite KG queries) takes 60–150 ms.
