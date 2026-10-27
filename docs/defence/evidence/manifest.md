# LAPS Defence Evidence Manifest

**Date of Preservation:** 2026-10-13  
**Preservation Directory:** `docs/defence/evidence/`  
**Git Baseline Commit:** `6ba1523` (`main` branch)  
**Repository Remote:** `https://github.com/aryan-chopra-dev/LAPS-LINGUISTIC-ADAPTIVE-PRESERVATION-SCORING.git`

---

## 1. Preserved Artifacts Inventory

| File Name | Artifact Purpose | Type | Status / Provenance | SHA-256 Hash |
|---|---|---|---|---|
| `pytest_execution_log.txt` | Full test suite console execution log | Terminal Log | Freshly captured run (45 passed, 16 warnings, 27.51s, exit code 0) | `C780AA2F015E804D10F3878DFE76BFC4E48FD01C4EB6E71DF75C4F54A6DCBDAE` |
| `benchmark_console_output.txt` | Console log of reproduced benchmark execution | Terminal Log | Captured from executed run (`python scripts/benchmark.py --corpus data/evaluation/gold_corpus.json`) | `8877BEF592BAF202A1C6CFF3132B918C42290F71749A5F998B1F5F5F43333729` |
| `benchmark_results_v2.json` | Complete benchmark metric records (OTER, F1, BLEU, chrF++, significance) | JSON Artifact | Verified copy of `results/benchmark_results_v2.json` (executed 2026-10-13 14:58:37) | `59C013595A87185DDCFB6E442D53D1F384A74CBDD8A22B23A5D30A20C007EDD0` |
| `latency_results.json` | CPU latency profile records across N=100 gold sentences | JSON Artifact | Verified copy of `results/latency_results.json` (0.21 ms mean, 0.37 ms P95 cached scoring) | `45969D15DBA496B9EA3EA0236C38979E71779041A7D7A2159AFC5D10696A508A` |
| `kappa_results.json` | Two-rater inter-annotator agreement metrics | JSON Artifact | Verified copy of `results/kappa_results.json` (2 raters, 90 tokens, 85/90 agreements, Fleiss κ = 0.8739, Cohen κ = 0.8744) | `0DE33DE73DE0393F0EF17DBE6D28ADA0BF1E9195F16708A53F433812D2A3CDE2` |
| `dataset_integrity_audit.json` | Dataset split integrity and OOV schema validation audit | JSON Artifact | Verified copy of `results/dataset_integrity_audit.json` (70/15/15 disjoint + 5 OOV = 105 entries) | `2147D861002C8E3739AE6B4098F5EDAB0DF42A0317174EFF16478D1F95CABAD9` |
| `environment_metadata.json` | Execution environment, dependency versions, Git status, and commands | Metadata JSON | Newly generated metadata record | `6E3C2FD5A1009C7C7864E046CD9B88585A4096200C9F936972228351B6B81041` |

---

## 2. Execution Parameters & Reproducibility Scope

1. **PyTest Execution**:
   - Command: `python -m pytest -v`
   - Exit Code: `0`
   - Test Count: 45 passed, 0 failed, 0 skipped, 16 deprecation warnings (from `fpdf2` in `test_multi_format_exports`)
   - Duration: 27.51 seconds

2. **Benchmark Execution**:
   - Command: `python scripts/benchmark.py --corpus data/evaluation/gold_corpus.json`
   - Exit Code: `0`
   - Corpus Size: 100 sentences (327 candidate tokens)
   - Execution Mode: `DETERMINISTIC_SURROGATE_FALLBACK`
   - Pre-Cached Features: Active (all candidate tokens in `gold_corpus.json` contain pre-extracted feature vectors)
   - Real Neural Evaluation: Pending (neural backend `laps/nmt_backend.py` is implemented and unit-tested; benchmark evaluated in deterministic surrogate mode)
   - Statistical Test: Monte Carlo Paired Permutation Test (5,000 sign-flip resamples, `seed=42`, $p = (b+1)/(B+1)$, 2,000 bootstrap CI samples)
   - Multiple Comparison Control: Bonferroni correction ($\alpha = 0.05 / 3 = 0.0167$)

3. **Software Environment**:
   - OS: Windows 11 (win32)
   - Python: 3.10.9
   - PyTest: 9.1.1
   - SacreBLEU: 2.6.0
   - PyTorch: 2.11.0+cu128
   - Transformers: 5.15.1
