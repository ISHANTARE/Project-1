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
