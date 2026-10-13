"""
LAPS Automated Conversational Register Predictor
------------------------------------------------
Analyzes lexical cues, syntactic markers, informal chat contractions,
and domain vocabulary to dynamically infer the conversational register
(Casual, Social, Business, Academic) and assign the corresponding
Sigmoid threshold θ_R without requiring manual user toggling.

Registers:
  • Casual   (θ_R = 0.35): Informal peer chat, WhatsApp, high slang / contraction density
  • Social   (θ_R = 0.40): Public social commentary, tweets, informal discussions
  • Business (θ_R = 0.50): Professional workplace, Slack/Teams, engineering & client communication
  • Academic (θ_R = 0.65): Formal academic, viva, thesis, research, examination
"""

from __future__ import annotations

import re
from typing import Dict, Any

# Register Threshold Table
REGISTER_THRESHOLDS: Dict[str, float] = {
    "casual":   0.35,
    "social":   0.40,
    "business": 0.50,
    "academic": 0.65,
}

DEFAULT_REGISTER: str = "business"

# Lexical Marker Profiles
CASUAL_MARKERS = {
    "bhai", "yaar", "bro", "plz", "plzz", "pls", "thx", "thanx", "mast",
    "scene", "chill", "arre", "oye", "oy", "sun", "apna", "apun", "tere",
    "mera", "kya", "kyu", "kaisa", "lol", "lmao", "haha", "hahaha", "dude",
    "buddy", "waah", "bindaas", "jugaad", "chal", "chalo", "bata"
}

SOCIAL_MARKERS = {
    "trend", "viral", "post", "tweet", "thread", "insta", "story", "reels",
    "followers", "likes", "subscribers", "feed", "vlog", "channel", "meme"
}

BUSINESS_MARKERS = {
    "client", "meeting", "server", "database", "deploy", "deployment", "sprint",
    "sync", "update", "report", "deadline", "project", "budget", "email", "ticket",
    "issue", "fix", "production", "prod", "staging", "infra", "backend", "frontend",
    "api", "endpoint", "review", "deliverable", "kpi", "metrics", "stakeholder",
    "workflow", "agenda", "call", "schedule", "reschedule", "postpone", "manager"
}

ACADEMIC_MARKERS = {
    "viva", "thesis", "dissertation", "hypothesis", "methodology", "algorithm",
    "evaluation", "experiment", "academic", "professor", "faculty", "lecture",
    "examination", "exam", "syllabus", "citation", "paper", "research",
    "analysis", "benchmark", "corpus", "rubric", "presentation", "lab"
}


def detect_register(text: str) -> Dict[str, Any]:
    """
    Infers the communicative register from text surface characteristics.
    Returns:
      - predicted_register: 'casual' | 'social' | 'business' | 'academic'
      - threshold: float (θ_R)
      - confidence: float (0.0 to 1.0)
      - cue_counts: breakdown of matched cue counts
    """
    cleaned = text.lower().strip()
    words = set(re.findall(r"[\w']+", cleaned))
    
    # 1. Cue Matching
    casual_hits   = len(words & CASUAL_MARKERS)
    social_hits   = len(words & SOCIAL_MARKERS)
    business_hits = len(words & BUSINESS_MARKERS)
    academic_hits = len(words & ACADEMIC_MARKERS)

    # 2. Structural & Orthographic Cues
    # Exclamation or trailing punctuation typical of informal chat
    has_chat_punctuation = bool(re.search(r'[!?]{2,}|\.{3,}', cleaned))
    if has_chat_punctuation:
        casual_hits += 1

    # Hashtags or mentions
    if re.search(r'[#@]\w+', text):
        social_hits += 2

    # All-lowercase with no terminal punctuation often signifies informal chat
    if text == text.lower() and not text.endswith(('.', '?', '!')) and len(words) <= 7:
        casual_hits += 1

    cue_counts = {
        "casual": casual_hits,
        "social": social_hits,
        "business": business_hits,
        "academic": academic_hits,
    }

    # 3. Decision Logic
    # Priority weighting: Academic and Business are higher specificity than Casual
    scores = {
        "academic": academic_hits * 1.5,
        "business": business_hits * 1.2,
        "social":   social_hits * 1.1,
        "casual":   casual_hits * 1.0,
    }

    best_reg = max(scores, key=scores.get)
    max_score = scores[best_reg]

    if max_score == 0:
        # Default fallback if zero specific markers are present:
        if len(words) <= 5:
            best_reg = "casual"
            confidence = 0.55
        else:
            best_reg = DEFAULT_REGISTER
            confidence = 0.60
    else:
        total_score = sum(scores.values()) + 1e-9
        confidence = min(0.98, max(0.65, round(max_score / total_score, 2)))

    return {
        "predicted_register": best_reg,
        "threshold": REGISTER_THRESHOLDS[best_reg],
        "confidence": confidence,
        "cue_counts": cue_counts,
    }


def get_register_threshold(register_name: str, text: str = "") -> float:
    """
    Get register threshold. If register_name == 'auto', dynamically detect from text.
    """
    reg_clean = (register_name or "auto").lower().strip()
    if reg_clean == "auto":
        if not text:
            return REGISTER_THRESHOLDS[DEFAULT_REGISTER]
        return detect_register(text)["threshold"]
    return REGISTER_THRESHOLDS.get(reg_clean, REGISTER_THRESHOLDS[DEFAULT_REGISTER])
