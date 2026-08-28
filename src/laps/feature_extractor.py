"""
LAPS Context-Aware Neural Feature Extractor & Lexical Graph Query Engine
-------------------------------------------------------------------------
Computes 6 multi-dimensional linguistic feature vectors:
  1. BF (Borrowing Frequency)   : Empirical Zipf frequency ratio & corpus borrowing prevalence.
  2. DS (Domain Specificity)    : Context-aware neural similarity to domain prototype vectors.
  3. SI (Semantic Importance)   : Information density & semantic drift risk under translation.
  4. CC (Context Consistency)   : Sub-graph co-occurrence density & bilingual context window PMI.
  5. AF (Audience Familiarity)  : Universal conversational recognition index.
  6. TB (Translation Benefit)   : Dynamic syntactic role benefit (high for auxiliary/function words, low for core nouns).
"""

from __future__ import annotations

import json
import difflib
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import wordfreq
except ImportError:
    wordfreq = None

from sentence_transformers import SentenceTransformer, util

try:
    from src.laps.graph_engine import get_graph_engine
except ModuleNotFoundError:
    from laps.graph_engine import get_graph_engine

# ── Vocabulary & Graph Paths ──────────────────────────────────────────────────
VOCAB_DIR = Path(__file__).resolve().parents[2] / "data" / "vocabulary"
BF_PATH   = VOCAB_DIR / "borrowing_frequency.json"

# Load SentenceTransformer model once (cached singleton)
_MODEL: Optional[SentenceTransformer] = None

def _get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

# ── Domain Vector Prototype Anchors ──────────────────────────────────────────
DOMAIN_ANCHORS = {
    "tech":     "software code server database debug api link url repository cloud deployment github bug patch algorithm backend frontend architecture",
    "business": "meeting client project deadline schedule manager sprint pitch feedback presentation contract proposal email office corporate workflow",
    "academic": "viva calendar exam test quiz assignment syllabus lecture lab grade mark result degree diploma course subject semester thesis dissertation",
    "medical":  "doctor hospital medicine clinic patient fever symptom prescription emergency surgery pharmacy scan diagnosis treatment",
    "finance":  "tax bill cash card pay cost price bank transaction loan credit invoice market budget salary revenue interest",
    "social":   "chat message post reel video photo friend party weekend trend viral follower update"
}

_DOMAIN_EMBEDDINGS: Optional[Dict[str, Any]] = None

def _get_domain_embeddings() -> Dict[str, Any]:
    global _DOMAIN_EMBEDDINGS
    if _DOMAIN_EMBEDDINGS is None:
        model = _get_embedding_model()
        _DOMAIN_EMBEDDINGS = {
            dom: model.encode(text, convert_to_tensor=True)
            for dom, text in DOMAIN_ANCHORS.items()
        }
    return _DOMAIN_EMBEDDINGS


# Canonical Loanwords Knowledge Fallback
WELL_KNOWN_LOANWORDS: Dict[str, float] = {
    "calendar": 0.90, "calender": 0.90, "viva": 0.95, "link": 0.95, "meeting": 0.95,
    "client": 0.92, "server": 0.90, "project": 0.93, "postpone": 0.85, "deadline": 0.88,
    "schedule": 0.86, "manager": 0.91, "code": 0.94, "design": 0.87, "debug": 0.80,
    "database": 0.89, "deploy": 0.84, "application": 0.83, "interface": 0.76,
    "product": 0.85, "release": 0.82, "testing": 0.88, "feedback": 0.84, "backup": 0.86,
    "network": 0.89, "system": 0.87, "update": 0.91, "upload": 0.93, "download": 0.92,
    "install": 0.90, "cancel": 0.88, "confirm": 0.85, "email": 0.96, "chat": 0.95,
    "call": 0.94, "message": 0.93, "login": 0.95, "logout": 0.92, "password": 0.94,
    "data": 0.89, "file": 0.91, "folder": 0.88, "image": 0.87, "video": 0.89,
    "app": 0.95, "zoom": 0.92, "teams": 0.90, "assignment": 0.89, "lab": 0.88,
    "exam": 0.92, "test": 0.89, "class": 0.93, "lecture": 0.88, "course": 0.89,
    "hi": 0.90, "hello": 0.92, "hey": 0.88, "ok": 0.97, "okay": 0.97, "ppt": 0.92,
    "pdf": 0.95, "doc": 0.91, "docx": 0.91, "wifi": 0.96, "laptop": 0.95
}


def _load_bf_dict() -> Dict[str, float]:
    if BF_PATH.exists():
        try:
            with open(BF_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return WELL_KNOWN_LOANWORDS


def _fuzzy_loanword_match(token_lower: str, bf_dict: Dict[str, float]) -> Tuple[str, float]:
    """Find closest canonical loanword match for phonetic typos (e.g. 'calender' -> 'calendar')."""
    if token_lower in bf_dict:
        return token_lower, bf_dict[token_lower]
    if token_lower in WELL_KNOWN_LOANWORDS:
        return token_lower, WELL_KNOWN_LOANWORDS[token_lower]

    matches = difflib.get_close_matches(token_lower, list(WELL_KNOWN_LOANWORDS.keys()), n=1, cutoff=0.75)
    if matches:
        matched = matches[0]
        return matched, WELL_KNOWN_LOANWORDS[matched]
    return token_lower, 0.40


# ── Feature Vector Derivations ────────────────────────────────────────────────

def _compute_bf_dynamic(token_lower: str, bf_dict: Dict[str, float]) -> float:
    """
    1. BF: Dynamic Borrowing Frequency derived from Zipf frequency & loanword index.
    """
    canonical, base_score = _fuzzy_loanword_match(token_lower, bf_dict)
    if wordfreq is not None:
        zipf = wordfreq.zipf_frequency(canonical, "en")
        # Map Zipf score (range ~2.0 to 7.0) to [0.2, 0.98]
        if zipf > 0:
            zipf_norm = min(0.98, max(0.20, (zipf - 1.5) / 5.5))
            return round((base_score + zipf_norm) / 2.0, 4)
    return round(base_score, 4)


def _compute_ds_contextual(token_lower: str, context_window: List[str]) -> float:
    """
    2. DS: Context-Aware Neural Domain Specificity.
    Measures cosine similarity of token embedded within its local context against domain prototypes.
    """
    try:
        model = _get_embedding_model()
        dom_embeds = _get_domain_embeddings()
        
        # Embed token with contextual window
        context_str = " ".join(context_window) if context_window else token_lower
        tok_embed = model.encode(f"{token_lower} in context: {context_str}", convert_to_tensor=True)

        max_sim = 0.0
        for dom, dom_vec in dom_embeds.items():
            sim = util.cos_sim(tok_embed, dom_vec).item()
            if sim > max_sim:
                max_sim = sim

        # Normalize score into [0.15, 0.98]
        return round(max(0.15, min(0.98, max_sim)), 4)
    except Exception:
        return 0.75 if token_lower in WELL_KNOWN_LOANWORDS else 0.35


def _compute_si_dynamic(token_lower: str, bf_score: float) -> float:
    """
    3. SI: Semantic Importance / Information Density.
    Measures information retention need based on token specificity and semantic drift risk.
    """
    canonical, _ = _fuzzy_loanword_match(token_lower, {})
    # Core nouns and entities carry high semantic load
    entity_stems = {"viva", "calendar", "link", "meeting", "client", "server", "code", "debug",
                    "database", "deploy", "project", "assignment", "lab", "test", "exam", "quiz",
                    "pdf", "doc", "ppt", "zoom", "teams", "api", "url", "portal", "system", "wifi"}
    
    if canonical in entity_stems:
        return 0.90
    
    # Information content scaling with length & frequency
    length_weight = min(1.0, len(token_lower) / 10.0)
    return round(max(0.40, min(0.95, (bf_score * 0.6) + (length_weight * 0.4))), 4)


def _compute_cc_dynamic(token_lower: str, context_tokens: List[str], bf_dict: Dict[str, float], kg_engine) -> float:
    """
    4. CC: Context Consistency.
    Measures surrounding loanword density and graph co-occurrence PMI.
    """
    if not context_tokens:
        return 0.50
    
    # Graph co-occurrence
    cc_graph = kg_engine.get_context_cooccurrence(token_lower, context_tokens) if kg_engine else 0.50
    
    # Window loanword density
    matches = sum(1 for t in context_tokens if t.lower() in WELL_KNOWN_LOANWORDS or t.lower() in bf_dict)
    cc_window = max(0.30, min(1.0, matches / len(context_tokens) + 0.3))
    
    return round((cc_graph + cc_window) / 2.0, 4)


def _compute_af_dynamic(token_lower: str) -> float:
    """
    5. AF: Audience Familiarity.
    Measures universal recognition across bilingual audiences.
    """
    canonical, score = _fuzzy_loanword_match(token_lower, {})
    if wordfreq is not None:
        zipf = wordfreq.zipf_frequency(canonical, "en")
        if zipf > 4.5:
            score = max(score, 0.92)
    return round(score, 4)


def _compute_tb_dynamic(token_lower: str, is_verb_or_function: bool = False) -> float:
    """
    6. TB: Translation Benefit.
    Measures grammatical/syntactic benefit of translating the token.
    High for functional verbs/adverbs/particles; Low for technical/domain nouns.
    """
    verbs_modifiers = {
        "postpone", "reschedule", "arrange", "continue", "prefer", "require",
        "delay", "start", "stop", "finish", "complete", "cancel", "verify"
    }
    if token_lower in verbs_modifiers or is_verb_or_function:
        return 0.85
    
    # Technical nouns and entities have low translation benefit (high preservation need)
    return 0.15


# ── Master Feature Extractor Function ─────────────────────────────────────────

def extract_features(token: str, all_tokens: List[str], index: int) -> Dict[str, float]:
    """
    Extract 6 feature vectors for a candidate token by combining contextual
    neural embeddings, corpus frequency statistics, and Lexical Graph topology.
    """
    kg = get_graph_engine()
    tok_low = token.lower()
    bf_dict = _load_bf_dict()

    # Contextual window (±2 tokens)
    window = [
        all_tokens[i] for i in range(max(0, index - 2), min(len(all_tokens), index + 3))
        if i != index
    ]

    # Query Knowledge Graph Node & Edges
    node_attrs = kg.get_node_attributes(tok_low) if kg else {}
    trans_edge = kg.get_translation_edge(tok_low) if kg else {}

    # 1. Borrowing Frequency (BF)
    bf = node_attrs.get("f_borrow", _compute_bf_dynamic(tok_low, bf_dict))

    # 2. Domain Specificity (DS)
    ds_neural = _compute_ds_contextual(tok_low, window)
    ds_graph  = kg.get_domain_specificity(tok_low) if kg else 0.50
    ds = max(ds_neural, ds_graph)

    # 3. Semantic Importance (SI)
    si_drift = trans_edge.get("meaning_drift", 0.50)
    si = max(si_drift, _compute_si_dynamic(tok_low, bf))

    # 4. Translation Benefit (TB)
    tb_penalty = trans_edge.get("naturalness_penalty", 0.50)
    if tb_penalty > 0.50:
        tb = max(0.10, 1.0 - tb_penalty)
    else:
        tb = _compute_tb_dynamic(tok_low)

    # 5. Context Consistency (CC)
    cc = _compute_cc_dynamic(tok_low, window, bf_dict, kg)

    # 6. Audience Familiarity (AF)
    af = node_attrs.get("f_mono_en", _compute_af_dynamic(tok_low))

    return {
        "BF": round(bf, 4),
        "DS": round(ds, 4),
        "SI": round(si, 4),
        "CC": round(cc, 4),
        "AF": round(af, 4),
        "TB": round(tb, 4),
    }


if __name__ == "__main__":
    test_tokens = ["hi", "viva", "calender", "link", "postpone"]
    for i, t in enumerate(test_tokens):
        f = extract_features(t, test_tokens, i)
        print(f"{t:10s} -> BF:{f['BF']:.2f} DS:{f['DS']:.2f} SI:{f['SI']:.2f} CC:{f['CC']:.2f} AF:{f['AF']:.2f} TB:{f['TB']:.2f}")
