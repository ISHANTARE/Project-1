"""
LAPS Advanced Feature Extractor with Neural Vector Embeddings & Fuzzy Matching
-----------------------------------------------------------------------------
Computes 6 scoring vectors:
  BF: Borrowing Frequency & English Zipf / Lexicon Frequency
  DS: Domain Specificity via Sentence-Transformer Cosine Similarity
  SI: Semantic Importance (semantic drift / information retention)
  CC: Context Consistency (surrounding loanword ratio)
  AF: Audience Familiarity (universal conversational recognition)
  TB: Translation Benefit (grammatical necessity of translation)
"""

import json
import difflib
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# ── Paths & Vocabulary ─────────────────────────────────────────────────────────
VOCAB_DIR = Path(__file__).resolve().parents[2] / "data" / "vocabulary"
BF_PATH   = VOCAB_DIR / "borrowing_frequency.json"

# Load SentenceTransformer model once (cached)
_MODEL = None

def _get_embedding_model():
    global _MODEL
    if _MODEL is None:
        # Fast, lightweight embedding model (~80MB)
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL

# ── Domain Vector Anchors ───────────────────────────────────────────────────────
DOMAIN_ANCHORS = {
    "tech":     "software code server database debug api link url repository cloud deployment github bug patch algorithm",
    "business": "meeting client project deadline schedule manager sprint pitch feedback presentation contract proposal email office",
    "academic": "viva calendar exam test quiz assignment syllabus lecture lab grade mark result degree diploma course subject",
    "medical":  "doctor hospital medicine clinic patient fever symptom prescription emergency surgery pharmacy scan",
    "finance":  "tax bill cash card pay cost price bank transaction loan credit invoice market budget salary"
}

_DOMAIN_EMBEDDINGS = None

def _get_domain_embeddings():
    global _DOMAIN_EMBEDDINGS
    if _DOMAIN_EMBEDDINGS is None:
        model = _get_embedding_model()
        _DOMAIN_EMBEDDINGS = {
            dom: model.encode(text, convert_to_tensor=True)
            for dom, text in DOMAIN_ANCHORS.items()
        }
    return _DOMAIN_EMBEDDINGS

# Common English vocabulary dictionary for fuzzy matching typos
WELL_KNOWN_LOANWORDS = {
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
    "hi": 0.90, "hello": 0.92, "hey": 0.88, "ok": 0.97, "okay": 0.97
}


def _load_bf_dict() -> dict:
    if BF_PATH.exists():
        with open(BF_PATH, encoding="utf-8") as f:
            return json.load(f)
    return WELL_KNOWN_LOANWORDS


def _fuzzy_loanword_match(token_lower: str, bf_dict: dict) -> tuple[str, float]:
    """Find closest word match in loanword database to handle typos like 'calender' -> 'calendar'."""
    if token_lower in bf_dict:
        return token_lower, bf_dict[token_lower]
    if token_lower in WELL_KNOWN_LOANWORDS:
        return token_lower, WELL_KNOWN_LOANWORDS[token_lower]

    matches = difflib.get_close_matches(token_lower, list(WELL_KNOWN_LOANWORDS.keys()), n=1, cutoff=0.75)
    if matches:
        matched = matches[0]
        return matched, WELL_KNOWN_LOANWORDS[matched]
    return token_lower, 0.40


def _bf_score(token_lower: str, bf_dict: dict) -> float:
    _, score = _fuzzy_loanword_match(token_lower, bf_dict)
    return score


def _ds_score_neural(token_lower: str) -> float:
    """Compute neural vector similarity between token and domain anchors."""
    try:
        model = _get_embedding_model()
        dom_embeds = _get_domain_embeddings()
        tok_embed = model.encode(token_lower, convert_to_tensor=True)

        max_sim = 0.0
        for dom, dom_vec in dom_embeds.items():
            sim = util.cos_sim(tok_embed, dom_vec).item()
            if sim > max_sim:
                max_sim = sim

        # Normalize score into [0.1, 0.95]
        return round(max(0.1, min(0.95, max_sim)), 4)
    except Exception:
        # Fallback if sentence-transformers unavailable
        return 0.70 if token_lower in WELL_KNOWN_LOANWORDS else 0.30


def _si_score(token_lower: str) -> float:
    """Semantic Importance: high score for nouns/concepts whose replacement alters meaning."""
    canonical, _ = _fuzzy_loanword_match(token_lower, {})
    high_si = {
        "viva", "calendar", "link", "meeting", "client", "server", "code", "debug",
        "database", "deploy", "project", "assignment", "lab", "test", "exam", "quiz",
        "pdf", "doc", "ppt", "zoom", "teams", "api", "url"
    }
    return 0.90 if canonical in high_si else 0.50


def _cc_score(token_lower: str, context_tokens: list[str], bf_dict: dict) -> float:
    """Context Consistency: proportion of surrounding tokens that are also English/loanwords."""
    if not context_tokens:
        return 0.50
    matches = sum(1 for t in context_tokens if t.lower() in WELL_KNOWN_LOANWORDS or t.lower() in bf_dict)
    return round(max(0.30, min(1.0, matches / len(context_tokens) + 0.3)), 4)


def _af_score(token_lower: str) -> float:
    """Audience Familiarity score."""
    canonical, score = _fuzzy_loanword_match(token_lower, {})
    return round(score, 4)


def _tb_score(token_lower: str) -> float:
    """Translation Benefit: high if word is a pure verb/modifier where translation helps flow."""
    verbs_modifiers = {"postpone", "reschedule", "arrange", "continue", "prefer", "require"}
    return 0.85 if token_lower in verbs_modifiers else 0.15


def extract_features(token: str, all_tokens: list[str], index: int) -> dict:
    """
    Extract 6 feature vectors for a candidate token using neural similarity & loanword lexicons.
    """
    bf_dict = _load_bf_dict()
    tok_low = token.lower()

    # Context window
    window = [all_tokens[i] for i in range(max(0, index - 2), min(len(all_tokens), index + 3)) if i != index]

    return {
        "BF": _bf_score(tok_low, bf_dict),
        "DS": _ds_score_neural(tok_low),
        "SI": _si_score(tok_low),
        "CC": _cc_score(tok_low, window, bf_dict),
        "AF": _af_score(tok_low),
        "TB": _tb_score(tok_low),
    }


if __name__ == "__main__":
    test_tokens = ["hi", "viva", "calender", "link"]
    for i, t in enumerate(test_tokens):
        f = extract_features(t, test_tokens, i)
        print(f"{t:10s} -> BF:{f['BF']:.2f} DS:{f['DS']:.2f} SI:{f['SI']:.2f} CC:{f['CC']:.2f} AF:{f['AF']:.2f} TB:{f['TB']:.2f}")
