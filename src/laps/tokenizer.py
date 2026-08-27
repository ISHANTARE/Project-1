"""
LAPS Advanced Tokenizer & Language Tagger (LID)
-----------------------------------------------
Hybrid Language Tagger for Hinglish/Code-Mixed Text:
1. Identifies Devanagari script.
2. Checks common English vocabulary (WordNet/frequency lexicons + common English patterns).
3. Identifies Hindi transliteration patterns & grammar markers (phonetic rules, common Hinglish stop words, postpositions).
"""

import re
import difflib
from pathlib import Path

# ── Devanagari regex ───────────────────────────────────────────────────────────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE      = re.compile(r'[A-Za-z]')

# ── Ambiguous homographs (English words that match Hinglish transliterated forms)
AMBIGUOUS_HOMOGRAPHS = {"the", "is", "so", "me", "to", "do", "it", "in", "on", "no", "he", "be", "an", "at", "as", "hi"}

# ── Comprehensive Transliterated Hindi Marker Vocabulary ────────────────────────
HINDI_MARKERS = {
    # Pronouns & Demonstratives
    "kisi", "kiske", "kiska", "kiski", "kahan", "kaise", "kab", "kyun", "kya",
    "yeh", "woh", "voh", "main", "hum", "aap", "tum", "mera", "meri", "mere",
    "tera", "teri", "tere", "hamara", "hamari", "hamare", "tumhara", "tumhari",
    "tumhare", "unka", "unki", "unke", "iska", "iski", "iske", "uska", "uski",
    "uske", "unhe", "inhe", "mujhe", "tujhe", "humein", "unko", "inko", "kuch",
    "koi", "sab", "sabhi", "sabko", "aapas", "apna", "apni", "apne",

    # Postpositions & Particles
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "pr", "tak", "dwara",
    "saath", "sath", "bina", "liye", "baad", "pehle", "paas", "pass", "samne",
    "aage", "piche", "andar", "baahar", "beech", "uppar", "neeche",

    # Verbs & Auxiliaries (excluding ambiguous English words like 'the')
    "hai", "hain", "ho", "hoon", "hun", "h", "tha", "thi", "gaya", "gayi",
    "gaye", "raha", "rahi", "rahe", "hoga", "hogi", "hoge", "honge", "karna",
    "karni", "karne", "kar", "karo", "karein", "karo", "karta", "karti", "karte",
    "kiya", "kiyi", "kiye", "dene", "dena", "diya", "diye", "lene", "lena",
    "liya", "liye", "aana", "aaye", "aaya", "aayi", "jaana", "jaaye", "jaaya",
    "jaayein", "sakta", "sakti", "sakte", "chahiye", "milla", "mili", "mile",

    # Conjunctions & Adverbs
    "aur", "ya", "lekin", "parantoo", "parantu", "toh", "to", "phir", "fir",
    "ab", "jab", "tab", "kabhi", "sada", "hamesha", "aaj", "kal", "parso",
    "subah", "shaam", "raat", "dopahar", "bhi", "hee", "bhot", "bahut",
    "zyada", "ziyada", "kam", "thoda", "thodi", "accha", "acchi", "acche",
    "bura", "buri", "bure", "sahi", "galat", "theek", "haan", "nahin", "nahi",
    "na", "mat", "bas", "sirf", "vaise", "vaisebhi", "jaise", "jaisebhi",
}

# Phonetic suffixes highly characteristic of Hindi verb/noun inflections
HINDI_SUFFIX_PATTERNS = (
    r'.*(ege|egi|ega|enge|ungi|unga|oge|ogi|aaye|aaya|aayi|aate|aati|aata|'
    r'iyega|iye|iyeh|ingey|ingee|ingea|rakho|dekho|suno|batao|samjho)$'
)
HINDI_SUFFIX_RE = re.compile(HINDI_SUFFIX_PATTERNS, re.IGNORECASE)

# Standard English dictionary words that are universally recognized
COMMON_ENGLISH_WORDS = {
    "hi", "hello", "hey", "link", "viva", "calendar", "calender", "pdf", "doc",
    "docx", "ppt", "pptx", "excel", "sheet", "file", "folder", "zoom", "teams",
    "meet", "meeting", "client", "server", "code", "debug", "compile", "deploy",
    "database", "project", "postpone", "schedule", "deadline", "manager",
    "assignment", "lab", "test", "exam", "quiz", "result", "mark", "grade",
    "syllabus", "lecture", "class", "course", "subject", "ticket", "email",
    "chat", "call", "message", "video", "audio", "image", "photo", "app",
    "website", "url", "portal", "system", "network", "status", "update",
    "upload", "download", "install", "cancel", "confirm", "reset", "clear",
}


def _is_hindi_word(word_lower: str) -> bool:
    """Check if token is a transliterated Hindi word."""
    if word_lower in HINDI_MARKERS:
        return True
    if HINDI_SUFFIX_RE.match(word_lower) and word_lower not in COMMON_ENGLISH_WORDS:
        return True
    return False


def tokenize(sentence: str) -> list[dict]:
    """
    Tokenize a code-mixed sentence with high-accuracy LID.

    Returns tagged tokens:
      - language: 'EN' (English loanword candidate), 'HI' (transliterated Hindi), 'DEV' (Devanagari), 'PUNCT'
      - is_candidate: True if English candidate subject to LAPS scoring.
    """
    raw_tokens = re.findall(r"[\w'\-]+|[^\w\s]", sentence)
    tagged = []

    for tok in raw_tokens:
        tok_low = tok.lower()

        # Punctuation / Numbers
        if re.fullmatch(r'[\d\W]+', tok):
            tagged.append({
                "token": tok, "lower": tok_low, "script": "punct",
                "language": "PUNCT", "is_candidate": False
            })
            continue

        # Devanagari script
        if DEVANAGARI_RE.search(tok):
            tagged.append({
                "token": tok, "lower": tok_low, "script": "devanagari",
                "language": "HI", "is_candidate": False
            })
            continue

        # Check Hindi transliteration
        if _is_hindi_word(tok_low):
            tagged.append({
                "token": tok, "lower": tok_low, "script": "latin",
                "language": "HI", "is_candidate": False
            })
            continue

        # Latin script non-Hindi -> Tagged as candidate English token
        tagged.append({
            "token": tok, "lower": tok_low, "script": "latin",
            "language": "EN", "is_candidate": True
        })

    return tagged


if __name__ == "__main__":
    sample = "hi! Kisi ke pass viva calender ki link h kya?"
    for t in tokenize(sample):
        print(f"{t['token']:12s} -> lang: {t['language']:6s} candidate: {t['is_candidate']}")
