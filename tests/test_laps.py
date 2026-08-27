"""
Unit and Integration Tests for LAPS Framework
==============================================
Tests all modular components and end-to-end pipeline.
Run: pytest tests/
"""

import os
import json
import pytest
from pathlib import Path

from src.laps.cmi_detector import detect_sentence_language
from src.laps.tokenizer import tokenize
from src.laps.feature_extractor import extract_features
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
        # Meeting should be marked as candidate loanword
        meeting_tok = next(t for t in tokens if t["token"] == "meeting")
        assert meeting_tok["is_candidate"] is True


class TestKnowledgeGraphEngine:
    def test_graph_engine_singleton(self):
        engine = get_graph_engine()
        assert engine is not None
        # Check node lookup for a common technical term
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

    def test_monolingual_passthrough(self):
        res = run(
            sentence="The server has crashed and needs to be restarted immediately.",
            register="business",
            llm_backend="mock"
        )
        assert res["is_monolingual"] is True
        assert res["cmi_info"]["mode"] == "MONOLINGUAL_EN"


class TestGoldCorpusIntegrity:
    def test_corpus_size_and_schema(self):
        root = Path(__file__).resolve().parent.parent
        corpus_file = root / "data" / "evaluation" / "gold_corpus.json"
        assert corpus_file.exists()
        
        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sentences = data.get("sentences", [])
        assert len(sentences) == 100, f"Expected 100 sentences in gold corpus, found {len(sentences)}"
        
        for s in sentences:
            assert "id" in s
            assert "sentence" in s
            assert "register" in s
            assert "tokens" in s
            for t in s["tokens"]:
                assert "token" in t
                assert "label" in t
                assert "features" in t
