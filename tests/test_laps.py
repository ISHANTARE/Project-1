"""
Unit and Integration Tests for LAPS Framework
==============================================
Tests all modular components, neural LID, corpus-grounded features,
and end-to-end pipeline.
Run: pytest tests/
"""

import os
import json
import pytest
from pathlib import Path

from src.laps.cmi_detector import detect_sentence_language
from src.laps.tokenizer import tokenize
from src.laps.feature_extractor import extract_features
from src.laps.neural_lid import get_neural_lid
from src.laps.graph_engine import get_graph_engine
from src.laps.scorer import compute_score, score_all_tokens
from src.laps.threshold_engine import decide, batch_decide, compute_sigmoid_probability
from src.llm.rewriter import LAPSRewriter, build_xml_prompt
from src.llm.explainer import explain_token, explain_all
from src.pipeline import run


class TestCMIDetector:
    def test_monolingual_english(self):
        res = detect_sentence_language("Hi! Does anyone have the link for the viva calendar?")
        assert res["mode"] == "MONOLINGUAL_EN"
        assert res["cmi"] == 0.0

    def test_code_mixed_hinglish(self):
        res = detect_sentence_language("hi! Kisi ke pass viva calender ki link h kya?")
        assert res["mode"] == "CODE_MIXED"
        assert res["cmi"] > 0.0


class TestTokenizer:
    def test_tokenization_and_lid(self):
        tokens = tokenize("Kal meeting postpone ho gayi")
        assert len(tokens) > 0
        token_texts = [t["token"] for t in tokens]
        assert "meeting" in token_texts
        meeting_tok = next(t for t in tokens if t["token"] == "meeting")
        assert meeting_tok["is_candidate"] is True


class TestNeuralLanguageIdentifier:
    def test_neural_char_lid_inference(self):
        lid = get_neural_lid()
        assert lid is not None

        # Test common Hinglish chat markers vs English loanwords
        en_res = lid.predict_token_language("meeting")
        assert en_res["language"] == "EN"
        assert en_res["p_en"] > 0.50

        hi_res = lid.predict_token_language("karna")
        assert hi_res["language"] == "HI"
        assert hi_res["p_hi"] > 0.50

        # Verify probability normalization
        assert abs((en_res["p_en"] + en_res["p_hi"]) - 1.0) < 0.01
        assert abs((hi_res["p_en"] + hi_res["p_hi"]) - 1.0) < 0.01


class TestCorpusGroundedFeatures:
    def test_extract_features_oov_word(self):
        # Test completely unseen OOV technical term
        tokens = ["Kubernetes", "cluster", "deploy", "karo"]
        f = extract_features("Kubernetes", tokens, 0)
        assert 0.0 <= f["BF"] <= 1.0
        assert 0.0 <= f["DS"] <= 1.0
        assert 0.0 <= f["SI"] <= 1.0
        assert 0.0 <= f["CC"] <= 1.0
        assert 0.0 <= f["AF"] <= 1.0
        assert 0.0 <= f["TB"] <= 1.0

    def test_tb_syntactic_differentiation(self):
        # Core nouns/entities should have lower translation benefit (preservation preferred)
        # Functional modifiers/verbs should have higher translation benefit
        tokens = ["meeting", "postpone", "karo"]
        f_noun = extract_features("meeting", tokens, 0)
        f_verb = extract_features("postpone", tokens, 1)
        assert f_verb["TB"] > f_noun["TB"], f"Expected verb TB ({f_verb['TB']}) > noun TB ({f_noun['TB']})"


class TestKnowledgeGraphEngine:
    def test_graph_engine_singleton(self):
        engine = get_graph_engine()
        assert engine is not None
        attrs = engine.get_node_attributes("meeting")
        assert isinstance(attrs, dict)


class TestScorerAndSigmoid:
    def test_compute_score_bounds(self):
        feats = {"BF": 0.90, "DS": 0.85, "SI": 0.80, "CC": 0.75, "AF": 0.88, "TB": 0.15}
        score = compute_score(feats)
        assert 0.0 <= score <= 1.0

    def test_sigmoid_probabilities(self):
        p_high = compute_sigmoid_probability(score=0.90, theta=0.50)
        p_low  = compute_sigmoid_probability(score=0.20, theta=0.50)
        assert p_high > 0.70
        assert p_low < 0.50

    def test_decide_zones(self):
        d_pres = decide(token="meeting", score=0.85, register="business")
        assert d_pres.decision == "PRESERVE"

        d_rew = decide(token="baithak", score=0.15, register="business")
        assert d_rew.decision == "REWRITE"


class TestRewriterAndExplainer:
    def test_xml_prompt_construction(self):
        preserved = {
            "viva": {"action": "PRESERVE", "domain": "Academic", "naturalness_penalty": 0.85},
            "link": {"action": "PRESERVE", "domain": "Tech", "naturalness_penalty": 0.90}
        }
        xml = build_xml_prompt(
            sentence="Kisi ke pass viva link hai kya?",
            preserved_map=preserved,
            register="academic"
        )
        assert "<rewrite_task" in xml
        assert "<preservation_constraints>" in xml
        assert "viva" in xml
        assert "link" in xml

    def test_offline_rewriter(self):
        rw = LAPSRewriter(backend="mock")
        res = rw.rewrite(
            sentence="Kal client meeting postpone ho gayi, kya hum phir se schedule karein?",
            preserve_tokens=["client", "meeting", "schedule"],
            register="business"
        )
        assert "meeting" in res["rewritten"]
        assert "client" in res["rewritten"]


class TestEndToEndPipeline:
    def test_pipeline_execution(self):
        res = run(
            sentence="hi! Kisi ke pass viva calender ki link h kya?",
            register="academic",
            llm_backend="mock"
        )
        assert "preserve_list" in res
        assert "viva" in res["preserve_list"] or "calender" in res["preserve_list"] or "link" in res["preserve_list"]
        assert res["rewrite_result"]["rewritten"] != ""
        assert len(res["explanations"]) > 0

    def test_monolingual_passthrough_no_fake_scores(self):
        res = run(
            sentence="The server has crashed and needs to be restarted immediately.",
            register="business",
            llm_backend="mock"
        )
        assert res["is_monolingual"] is True
        assert res["cmi_info"]["mode"] == "MONOLINGUAL_EN"
        # Verify NO fake scores are injected
        for t in res["tokens"]:
            assert t["decision"] == "PASSTHROUGH"
            assert t["laps_score"] is None
            assert t["probability"] is None
            assert t["features"] == {}


class TestBenchmarkSplitsIntegrity:
    def test_splits_exist_and_non_circular(self):
        root = Path(__file__).resolve().parent.parent
        train_p = root / "data" / "evaluation" / "train_split.json"
        val_p   = root / "data" / "evaluation" / "val_split.json"
        test_p  = root / "data" / "evaluation" / "test_split.json"

        assert train_p.exists(), "Train split missing"
        assert val_p.exists(), "Validation split missing"
        assert test_p.exists(), "Test split missing"

        with open(test_p, encoding="utf-8") as f:
            test_data = json.load(f)

        test_sentences = test_data.get("sentences", [])
        assert len(test_sentences) >= 20
        # Verify presence of OOV tokens in test split
        has_oov = any(s.get("id", "").startswith("OOV") for s in test_sentences)
        assert has_oov is True, "Held-out test set must contain authentic OOV test cases"


class TestHomographDisambiguation:
    def test_short_utterance_homographs(self):
        tokens = tokenize("us room me meeting h to link bhej do")
        token_map = {t["token"].lower(): t for t in tokens}

        # Colliding homographs must be identified as Hindi grammatical particles/verbs
        assert token_map["us"]["language"] == "HI"
        assert token_map["us"]["is_candidate"] is False

        assert token_map["me"]["language"] == "HI"
        assert token_map["me"]["is_candidate"] is False

        assert token_map["to"]["language"] == "HI"
        assert token_map["to"]["is_candidate"] is False

        assert token_map["do"]["language"] == "HI"
        assert token_map["do"]["is_candidate"] is False

        # True loanwords must remain English candidates
        assert token_map["meeting"]["language"] == "EN"
        assert token_map["meeting"]["is_candidate"] is True

        assert token_map["link"]["language"] == "EN"
        assert token_map["link"]["is_candidate"] is True


class TestRegisterDetector:
    def test_auto_register_detection(self):
        from src.laps.register_detector import detect_register

        casual_sample = "bhai kal ka ppt share kr dena plzz!"
        res_c = detect_register(casual_sample)
        assert res_c["predicted_register"] == "casual"
        assert res_c["threshold"] == 0.35

        business_sample = "The client requested an update on the server deployment project."
        res_b = detect_register(business_sample)
        assert res_b["predicted_register"] == "business"
        assert res_b["threshold"] == 0.50

        academic_sample = "The viva evaluation and thesis presentation are scheduled tomorrow."
        res_a = detect_register(academic_sample)
        assert res_a["predicted_register"] == "academic"
        assert res_a["threshold"] == 0.65

    def test_pipeline_with_auto_register(self):
        res = run(
            sentence="bhai client meeting ki link share kr dena",
            register="auto",
            llm_backend="mock"
        )
        assert "preserve_list" in res
        assert "meeting" in res["preserve_list"] or "link" in res["preserve_list"]


class TestAuditReportExporter:
    def test_multi_format_exports(self):
        from src.laps.exporter import format_audit_record, export_to_json, export_to_csv, export_to_pdf

        res = run(sentence="Kal client meeting postpone ho gayi", register="business", llm_backend="mock")
        record = format_audit_record(res, backend_info="mock test")

        # 1. JSON Export verification
        json_str = export_to_json(record)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["framework"].startswith("LAPS")
        assert "candidate_tokens" in parsed
        assert len(parsed["candidate_tokens"]) > 0

        # 2. CSV Export verification
        csv_str = export_to_csv(record)
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2  # Header + at least 1 candidate row
        assert "Token,Language,Domain,BF,DS,SI,CC,AF,TB,LAPS_Score" in lines[0]

        # 3. PDF Export verification
        pdf_bytes = export_to_pdf(record)
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert len(pdf_bytes) > 500  # Valid PDF size
        assert pdf_bytes.startswith(b"%PDF")  # Standard PDF magic header


class TestNMTBackendAndConstraints:
    def test_model_family_detection(self):
        from src.llm.nmt_backend import NMTTranslationEngine

        eng_opus = NMTTranslationEngine("Helsinki-NLP/opus-mt-hi-en")
        assert eng_opus._model_family == "opus"

        eng_indic = NMTTranslationEngine("ai4bharat/indictrans2-indic-en-1B")
        assert eng_indic._model_family == "indictrans2"

        eng_nllb = NMTTranslationEngine("facebook/nllb-200-distilled-600M")
        assert eng_nllb._model_family == "nllb"

    def test_constrained_translation_preservation(self):
        from src.llm.nmt_backend import NMTTranslationEngine

        engine = NMTTranslationEngine()
        sentence = "Kal client meeting postpone ho gayi"

        # Unconstrained translation should suffer from over-translation (e.g. customer/conference)
        unconstrained_out = engine.translate(sentence, preserve_tokens=None, force_constraints=False)
        assert isinstance(unconstrained_out, str)

        # Constrained translation with preserved loanwords must preserve them verbatim
        constrained_out = engine.translate(sentence, preserve_tokens=["meeting", "client"], force_constraints=True)
        assert "meeting" in constrained_out.lower()
        assert "client" in constrained_out.lower()


class TestEdgeCasesAndEmptyInput:
    def test_empty_and_whitespace_input(self):
        # Empty input string
        res_empty = run(sentence="", register="business", llm_backend="mock")
        assert res_empty is not None
        assert "tokens" in res_empty

        # Whitespace-only input
        res_ws = run(sentence="   \t\n  ", register="casual", llm_backend="mock")
        assert res_ws is not None
        assert "tokens" in res_ws

    def test_numeric_and_punct_input(self):
        res_punct = run(sentence="123 456 !@#$%", register="business", llm_backend="mock")
        assert res_punct is not None


class TestPreservationClassifierEvaluation:
    def test_classifier_evaluation_standard(self):
        from scripts.benchmark import evaluate_preservation_classifier

        gold_tokens = [
            {"token": "meeting", "label": 1},
            {"token": "calendar", "label": 1},
            {"token": "karna", "label": 0},
        ]
        # System correctly predicts "meeting", misses "calendar", falsely predicts "karna"
        pred = {"meeting", "karna"}
        res = evaluate_preservation_classifier(pred, gold_tokens, is_applicable=True)

        assert res["is_applicable"] is True
        assert res["tp"] == 1
        assert res["fp"] == 1
        assert res["fn"] == 1
        assert res["tn"] == 0
        assert res["precision"] == 0.5
        assert res["recall"] == 0.5
        assert res["f1"] == 0.5

    def test_unconstrained_baseline_reports_na_not_zero(self):
        from scripts.benchmark import evaluate_preservation_classifier

        gold_tokens = [{"token": "meeting", "label": 1}]
        # Baseline with NO classifier
        res = evaluate_preservation_classifier(None, gold_tokens, is_applicable=False)

        assert res["is_applicable"] is False
        assert res["f1"] is None
        assert res["precision"] is None
        assert res["recall"] is None
        assert "N/A" in res["note"]


class TestOutputPreservationEvaluation:
    def test_output_credit_for_unconstrained_baseline(self):
        from scripts.benchmark import evaluate_output_preservation

        # Even if unconstrained baseline has no classifier, if "calendar" appears in output, it gets credit
        hyp = "Please check the calendar for updates."
        gold_tokens = [
            {"token": "calendar", "label": 1},
            {"token": "meeting", "label": 1},
        ]
        res = evaluate_output_preservation(hyp, gold_tokens)

        assert "calendar" in res["retained_loanwords"]
        assert "meeting" in res["missing_loanwords"]
        assert res["output_recall"] == 0.5  # 1 out of 2 retained
        assert res["oter"] == 0.5          # 1 out of 2 missing
        assert res["output_f1"] > 0.0

    def test_output_preservation_case_and_punct(self):
        from scripts.benchmark import evaluate_output_preservation

        hyp = "Tomorrow's TEAM MEETING! (postponed)"
        gold_tokens = [{"token": "meeting", "label": 1}]
        res = evaluate_output_preservation(hyp, gold_tokens)

        assert "meeting" in res["retained_loanwords"]
        assert res["output_recall"] == 1.0
        assert res["oter"] == 0.0


class TestSacreBLEUAndChrFMetrics:
    def test_identical_hypothesis_and_reference(self):
        from scripts.benchmark import compute_sentence_bleu, compute_sentence_chrf

        text = "Tomorrow's client meeting was postponed."
        bleu = compute_sentence_bleu(text, text)
        chrf = compute_sentence_chrf(text, text)
        assert bleu == 100.0
        assert chrf == 100.0

    def test_disjoint_text(self):
        from scripts.benchmark import compute_sentence_bleu

        hyp = "Completely different words."
        ref = "Unrelated sentence here."
        bleu = compute_sentence_bleu(hyp, ref)
        assert bleu < 20.0

    def test_empty_input_graceful_handling(self):
        from scripts.benchmark import compute_sentence_bleu, compute_sentence_chrf

        assert compute_sentence_bleu("", "Reference") == 0.0
        assert compute_sentence_chrf("Hypothesis", "") == 0.0

    def test_corpus_level_multi_sentence_metric_alignment(self):
        from scripts.benchmark import compute_corpus_bleu, compute_corpus_chrf

        hyps = [
            "Hi! Does anyone have the link for the viva calendar?",
            "Tomorrow client meeting delay ho gayi, what we phir from schedule karein?",
            "Report ki deadline today hai, submission portal on upload karo",
        ]
        refs = [
            "Hi! Does anyone have the link for the viva calendar?",
            "Tomorrow's client meeting was postponed. Should we reschedule it?",
            "The report deadline is today; please upload it to the submission portal.",
        ]
        # Multi-sentence corpus BLEU must be strictly less than 100 when non-initial sentences differ
        c_bleu = compute_corpus_bleu(hyps, refs)
        c_chrf = compute_corpus_chrf(hyps, refs)
        assert 0.0 < c_bleu < 100.0, f"Expected 0 < BLEU < 100, got {c_bleu}"
        assert 0.0 < c_chrf < 100.0, f"Expected 0 < chrF++ < 100, got {c_chrf}"


class TestPairedPermutationSignificance:
    def test_identical_series_yields_null(self):
        from scripts.benchmark import paired_permutation_test

        series = [0.8, 0.9, 0.7, 0.85]
        res = paired_permutation_test(series, series, num_samples=1000, seed=42)
        assert res["p_value"] == 1.0
        assert res["delta"] == 0.0

    def test_large_difference_finite_p_value(self):
        from scripts.benchmark import paired_permutation_test

        series_a = [1.0] * 50
        series_b = [0.0] * 50
        num_samples = 1000
        res = paired_permutation_test(series_a, series_b, num_samples=num_samples, seed=42)
        # Finite sample formula p = (b + 1) / (B + 1)
        expected_min = 1.0 / (num_samples + 1)
        assert res["p_value"] == pytest.approx(expected_min, rel=1e-3)
        assert res["p_value"] > 0.0  # Never identically 0.0
        assert res["delta"] == 1.0

    def test_reproducibility_under_seed(self):
        from scripts.benchmark import paired_permutation_test

        a = [0.8, 0.6, 0.9, 0.7, 0.85]
        b = [0.5, 0.4, 0.6, 0.5, 0.55]
        r1 = paired_permutation_test(a, b, num_samples=500, seed=42)
        r2 = paired_permutation_test(a, b, num_samples=500, seed=42)
        assert r1["p_value"] == r2["p_value"]
        assert r1["ci_lower"] == r2["ci_lower"]


class TestUTERAnnotationLimitationAndProxy:
    def test_uter_status_and_matrix_leakage(self):
        from scripts.benchmark import calculate_uter_status

        # Sentence with Hindi matrix words and 1 preserved loanword
        raw = "Kal client meeting postpone ho gayi"
        gold_pres = {"client", "meeting"}
        # If output leaves Hindi word "kal" untranslated
        hyp_with_leak = "Kal client meeting was delayed"
        stat = calculate_uter_status(hyp_with_leak, raw, gold_pres)

        assert stat["gold_uter"] is None
        assert "N/A" in stat["status"]
        assert stat["eligible_hindi_tokens"] > 0
        assert "kal" in stat["leaked_words"]
        assert stat["matrix_leakage_proxy"] > 0.0


class TestBackendExecutionModeTelemetry:
    def test_telemetry_reports_execution_mode(self):
        from src.llm.nmt_backend import get_nmt_engine

        engine = get_nmt_engine()
        engine.translate("Kal meeting postpone ho gayi", preserve_tokens=["meeting"], force_constraints=True)
        status = engine.execution_status
        assert "last_execution_mode" in status
        assert status["last_execution_mode"] in (
            "REAL_INFERENCE_CONSTRAINED",
            "REAL_INFERENCE_UNCONSTRAINED",
            "DETERMINISTIC_SURROGATE_FALLBACK",
        )


class TestDataSplitIntegrityAndLeakage:
    def test_split_disjointness(self):
        train_path = Path("data/evaluation/train_split.json")
        val_path = Path("data/evaluation/val_split.json")
        test_path = Path("data/evaluation/test_split.json")

        if train_path.exists() and val_path.exists() and test_path.exists():
            with open(train_path, encoding="utf-8") as f:
                train_ids = {s["id"] for s in json.load(f)["sentences"]}
            with open(val_path, encoding="utf-8") as f:
                val_ids = {s["id"] for s in json.load(f)["sentences"]}
            with open(test_path, encoding="utf-8") as f:
                test_ids = {s["id"] for s in json.load(f)["sentences"]}

            assert len(train_ids & val_ids) == 0, "Train and Val split have overlapping sentence IDs!"
            assert len(train_ids & test_ids) == 0, "Train and Test split have overlapping sentence IDs!"
            assert len(val_ids & test_ids) == 0, "Val and Test split have overlapping sentence IDs!"


class TestSurrogateReferenceIndependenceAndExactMatchAudit:
    def test_surrogate_has_no_reference_lookup_or_id_access(self):
        """
        Verify that the surrogate generator signature only accepts sentence, preserve_set, force_constraints.
        It does not accept sentence_id, reference, gold corpus, or memorized lookup tables.
        """
        import inspect
        from src.llm.nmt_backend import get_nmt_engine

        engine = get_nmt_engine()
        sig = inspect.signature(engine._generate_realistic_nmt_translation)
        params = list(sig.parameters.keys())
        assert "sentence" in params
        assert "preserve_set" in params
        assert "force_constraints" in params
        assert "reference" not in params
        assert "id" not in params
        assert "sentence_id" not in params

    def test_s001_exact_match_mechanism_and_coincidence(self):
        """
        Verify that S001's exact match is due to generic regex idiom matching
        'kisi ke pass ... ki link h kya?' on the input string, not memorization.
        """
        from src.llm.nmt_backend import get_nmt_engine

        engine = get_nmt_engine()
        # Same regex template triggered on a novel sentence with preserved terms
        out_preserved = engine._generate_realistic_nmt_translation(
            sentence="hi! Kisi ke pass project guidelines ki link h kya?",
            preserve_set={"project", "guidelines", "link"},
            force_constraints=True
        )
        assert out_preserved == "Hi! Does anyone have the link for the project guidelines?"

        # When project is NOT in preserve_set, OVER_TRANSLATION_MAP converts project -> assignment
        out_overtranslated = engine._generate_realistic_nmt_translation(
            sentence="hi! Kisi ke pass project guidelines ki link h kya?",
            preserve_set={"guidelines", "link"},
            force_constraints=True
        )
        assert out_overtranslated == "Hi! Does anyone have the link for the assignment guidelines?"

    def test_exact_match_rate_across_gold_corpus(self):
        """
        Verify empirical finding from audit: exactly 1/100 sentences (S001) produces
        an exact string match with reference in the surrogate, and 99/100 differ.
        """
        from src.llm.nmt_backend import get_nmt_engine

        gold_path = Path("data/evaluation/gold_corpus.json")
        if not gold_path.exists():
            pytest.skip("gold_corpus.json not found")

        with open(gold_path, encoding="utf-8") as f:
            gold = json.load(f)

        engine = get_nmt_engine()
        exact_matches = []
        for s in gold["sentences"]:
            preserve_tokens = {tok["token"].lower() for tok in s.get("tokens", []) if tok.get("label") == 1}
            hyp = engine.translate(s["sentence"], preserve_tokens=list(preserve_tokens), force_constraints=True)
            if hyp.strip().lower() == s["reference"].strip().lower():
                exact_matches.append(s["id"])

        assert len(exact_matches) == 1, f"Expected exactly 1 exact match (S001), got: {exact_matches}"
        assert exact_matches == ["S001"]



class TestOOVRecordSchemaValidation:
    """
    Regression tests for the two-schema design:
      IN_DIST records: {token, features, label}
      OOV_EXT records: {token, label, domain}  — no cached features by design.

    These tests verify:
      1. All 5 OOV records have the correct OOV_EXT token schema.
      2. No OOV token accidentally carries a 'features' cache (which would corrupt the OOV test).
      3. Live extract_features() works correctly for OOV tokens and returns valid [0,1] floats.
    """
    OOV_IDS = {"OOV_001", "OOV_002", "OOV_003", "OOV_004", "OOV_005"}
    OOV_TOKEN_REQUIRED = {"token", "label", "domain"}
    OOV_TOKEN_FORBIDDEN = {"features"}

    def _load_oov_records(self):
        test_path = Path("data/evaluation/test_split.json")
        if not test_path.exists():
            pytest.skip("test_split.json not found")
        with open(test_path, encoding="utf-8") as f:
            data = json.load(f)
        return [s for s in data["sentences"] if s["id"] in self.OOV_IDS]

    def test_all_five_oov_records_present(self):
        recs = self._load_oov_records()
        found_ids = {r["id"] for r in recs}
        assert found_ids == self.OOV_IDS, (
            f"Expected OOV IDs {self.OOV_IDS}, found {found_ids}"
        )

    def test_oov_records_have_required_top_level_fields(self):
        recs = self._load_oov_records()
        required = {"id", "sentence", "register", "reference", "tokens"}
        for rec in recs:
            missing = required - rec.keys()
            assert not missing, (
                f"OOV record {rec['id']} missing top-level fields: {missing}"
            )
            assert isinstance(rec["sentence"], str) and rec["sentence"].strip()
            assert isinstance(rec["tokens"], list) and len(rec["tokens"]) > 0

    def test_oov_token_schema_correct_and_no_features_cache(self):
        """
        Each token in an OOV record must have {token, label, domain} and must NOT have
        'features', because the purpose of OOV records is to exercise live feature
        extraction via extract_features().
        """
        recs = self._load_oov_records()
        for rec in recs:
            for tok in rec["tokens"]:
                missing = self.OOV_TOKEN_REQUIRED - tok.keys()
                assert not missing, (
                    f"OOV record {rec['id']} token '{tok.get('token')}' "
                    f"missing required fields: {missing}"
                )
                assert isinstance(tok["token"], str) and tok["token"].strip()
                assert isinstance(tok["label"], int) and tok["label"] in (0, 1)
                assert isinstance(tok["domain"], str) and tok["domain"].strip()
                # Must not have cached features (defeats the OOV test purpose)
                forbidden_present = self.OOV_TOKEN_FORBIDDEN & tok.keys()
                assert not forbidden_present, (
                    f"OOV record {rec['id']} token '{tok.get('token')}' "
                    f"has forbidden fields {forbidden_present} "
                    f"(cached features corrupt the OOV evaluation)"
                )

    def test_live_feature_extraction_for_oov_tokens(self):
        """
        The pipeline's live fallback path (extract_features) must produce valid [0,1]
        float feature vectors for OOV tokens that have no cached features.
        Tests three representative OOV tokens covering different domains.
        """
        oov_samples = [
            # (token, context_tokens, index)
            ("Kubernetes", ["Kubernetes", "cluster", "ingress", "controller"], 0),
            ("MRI", ["patient", "MRI", "scan", "blood"], 1),
            ("firmware", ["firmware", "patch", "checksum", "verify"], 0),
        ]
        feature_keys = {"BF", "DS", "SI", "CC", "AF", "TB"}
        for token, ctx, idx in oov_samples:
            feats = extract_features(token, ctx, idx)
            assert feats.keys() == feature_keys, (
                f"extract_features returned unexpected keys for '{token}': {feats.keys()}"
            )
            for key, val in feats.items():
                assert isinstance(val, float), (
                    f"Feature {key} for '{token}' is not float: {type(val)}"
                )
                assert 0.0 <= val <= 1.0, (
                    f"Feature {key}={val:.4f} for '{token}' out of [0,1] range"
                )


class TestDatasetIntegrityAuditorSchemaResult:
    """
    End-to-end regression test for the type-discriminated schema auditor.
    Runs the full audit_dataset_integrity.audit_splits() function and asserts
    that is_schema_valid=True when both IN_DIST and OOV_EXT records are correctly typed.
    """

    def test_schema_valid_with_type_discriminated_validator(self):
        """
        The type-discriminated validator must report is_schema_valid=True:
         - IN_DIST records (100 gold) validated against {token, features, label}.
         - OOV_EXT records (5) validated against {token, label, domain}.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "audit_dataset_integrity",
            Path("scripts/audit_dataset_integrity.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        results = mod.audit_splits()
        sv = results["schema_validation"]

        assert sv["is_schema_valid"] is True, (
            f"Expected schema_valid=True but got errors:\n" +
            "\n".join(sv["errors"][:20])
        )
        assert sv["total_errors"] == 0, (
            f"Expected 0 schema errors, got {sv['total_errors']}"
        )
        assert sv["in_dist_records_validated"] == 100, (
            f"Expected 100 in-dist records validated, got {sv['in_dist_records_validated']}"
        )
        assert sv["oov_ext_records_validated"] == 5, (
            f"Expected 5 OOV records validated, got {sv['oov_ext_records_validated']}"
        )


class TestNeuralBackendRealInference:
    """
    Tests live execution of the neural translation engine with real model checkpoints.
    Validates constraint enforcement, unconstrained vs constrained telemetry,
    and verbatim token retention.
    """

    def test_neural_model_loads_successfully(self):
        from src.llm.nmt_backend import get_nmt_engine
        engine = get_nmt_engine()
        success = engine.load_model()
        assert success is True, "Expected neural NMT model to load successfully"
        assert engine._is_loaded is True
        assert engine._model is not None
        assert engine._tokenizer is not None

    def test_real_constrained_generation_enforces_tokens(self):
        from src.llm.nmt_backend import get_nmt_engine
        engine = get_nmt_engine()
        if not engine.load_model():
            pytest.skip("Neural NMT model not available")

        sentence = "Assignment submit karne ki deadline kal hai"
        forced = ["assignment", "deadline"]
        output = engine.translate(sentence, preserve_tokens=forced, force_constraints=True)

        assert engine.last_execution_mode == "REAL_INFERENCE_CONSTRAINED"
        out_lower = output.lower()
        for token in forced:
            assert token in out_lower, (
                f"Forced token '{token}' must appear verbatim in constrained output: '{output}'"
            )

    def test_real_unconstrained_generation_telemetry(self):
        from src.llm.nmt_backend import get_nmt_engine
        engine = get_nmt_engine()
        if not engine.load_model():
            pytest.skip("Neural NMT model not available")

        sentence = "Assignment submit karne ki deadline kal hai"
        output = engine.translate(sentence, preserve_tokens=[], force_constraints=False)

        assert engine.last_execution_mode == "REAL_INFERENCE_UNCONSTRAINED"
        assert isinstance(output, str) and len(output.strip()) > 0


class TestAuditedFeatureAblationStudy:
    """
    Validates the mathematical correctness and schema integrity of the audited
    feature ablation framework.
    """

    def test_score_custom_canonical_weight_mapping(self):
        from scripts.run_ablation_study import score_custom, FEATURE_WEIGHT_KEYS

        # Test case: perfect features (positive features=1.0, TB=0.0)
        # Formula: 0.20(1) + 0.20(1) + 0.20(1) + 0.15(1) + 0.15(1) - 0.10(0) = 0.90
        feats = {"BF": 1.0, "DS": 1.0, "SI": 1.0, "CC": 1.0, "AF": 1.0, "TB": 0.0}
        full_score = score_custom(feats, ["BF", "DS", "SI", "CC", "AF", "TB"])
        assert round(full_score, 4) == 0.90, f"Expected 0.90, got {full_score}"

        # Test ablating BF: remaining positive weights sum to 0.70, total weight 0.80 -> 0.70/0.80 = 0.875
        no_bf_score = score_custom(feats, ["DS", "SI", "CC", "AF", "TB"])
        assert round(no_bf_score, 4) == 0.875, f"Expected 0.875 after normalization, got {no_bf_score}"

        # Test penalizing TB: when TB is 1.0, score must drop by 0.10 to 0.80
        feats_high_tb = {"BF": 1.0, "DS": 1.0, "SI": 1.0, "CC": 1.0, "AF": 1.0, "TB": 1.0}
        penalized_score = score_custom(feats_high_tb, ["BF", "DS", "SI", "CC", "AF", "TB"])
        assert round(penalized_score, 4) == 0.80, f"Expected 0.80 with TB penalty, got {penalized_score}"

    def test_ablation_matrix_runner_produces_valid_schema(self):
        from scripts.run_ablation_study import run_ablation_study
        val_path = Path("data/evaluation/val_split.json")
        out_test = Path("results/test_ablation_check.json")

        res = run_ablation_study(val_path, out_test)
        assert "Full LAPS (All 6 Features + Soft Sigmoid + KG)" in res
        assert "Ablation: Without BF (-BF)" in res
        assert "Ablation: Without Knowledge Graph (No Ego Density Bonus)" in res

        for name, metrics in res.items():
            assert "f1" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "accuracy" in metrics
            assert "delta_f1" in metrics
            assert 0.0 <= metrics["f1"] <= 1.0

        if out_test.exists():
            out_test.unlink()


class TestHandVerifiedPreservationMetrics:
    """
    Analytically tests precision, recall, F1, and OTER metrics against hand-verified
    gold test cases.
    """

    def test_preservation_classifier_hand_crafted_matrix(self):
        from scripts.benchmark import evaluate_preservation_classifier

        # Hand-crafted tokens: 2 positive, 2 negative
        gold_tokens = [
            {"token": "viva", "label": 1},
            {"token": "link", "label": 1},
            {"token": "karne", "label": 0},
            {"token": "hai", "label": 0},
        ]
        # Predicted preserve: "viva" (TP) and "karne" (FP) -> TP=1, FP=1, FN=1, TN=1
        pred_preserve = {"viva", "karne"}
        res = evaluate_preservation_classifier(pred_preserve, gold_tokens)

        assert res["tp"] == 1
        assert res["fp"] == 1
        assert res["fn"] == 1
        assert res["tn"] == 1
        assert res["precision"] == 0.50
        assert res["recall"] == 0.50
        assert res["f1"] == 0.50
        assert res["accuracy"] == 0.50

    def test_output_preservation_hand_crafted_example(self):
        from scripts.benchmark import evaluate_output_preservation

        gold_tokens = [
            {"token": "meeting", "label": 1},
            {"token": "link", "label": 1},
            {"token": "postpone", "label": 0},
        ]
        # Hypothesis preserves "meeting" but misses "link", and includes verbatim candidate "postpone" (label=0)
        hyp = "The meeting was postpone today."
        res = evaluate_output_preservation(hyp, gold_tokens)

        assert "meeting" in res["retained_loanwords"]
        assert "link" in res["missing_loanwords"]
        assert "postpone" in res["unintended_retentions"]
        assert res["oter"] == 0.50  # 1 of 2 gold loanwords missing
        assert res["output_recall"] == 0.50  # 1 of 2 retained
        assert res["output_precision"] == 0.50  # 1 TP / (1 TP + 1 FP)
        assert res["output_f1"] == 0.50


class TestOperationalLatencyProfileSchema:
    """
    Verifies that operational latency measurements exist and record realistic
    mean, median, and P95 numbers across all pipeline stages.
    """

    def test_profile_results_exist_and_conform_to_schema(self):
        profile_path = Path("results/operational_latency_profile.json")
        assert profile_path.exists(), "Operational latency profile must exist"

        with open(profile_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "stages" in data
        assert "scientific_interpretation" in data

        required_stages = [
            "cmi_detection",
            "tokenization_and_lid",
            "live_feature_extraction_minilm_kg",
            "preservation_scoring",
            "thresholding_batch_decide",
            "neural_nmt_unconstrained",
            "neural_nmt_constrained",
            "end_to_end_pipeline_live",
            "pre_cached_scoring_alone",
        ]
        for st in required_stages:
            assert st in data["stages"], f"Missing stage {st} in latency profile"
            stat = data["stages"][st]
            assert "mean_ms" in stat and stat["mean_ms"] >= 0.0
            assert "median_ms" in stat and stat["median_ms"] >= 0.0
            assert "p95_ms" in stat and stat["p95_ms"] >= 0.0

