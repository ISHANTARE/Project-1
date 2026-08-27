"""
LAPS Bilingual Code-Mixed Knowledge Graph Builder
--------------------------------------------------
Builds a rich, multi-relational Knowledge Graph (KG) combining:
1. Real Zipf monolingual English word frequencies via `wordfreq`.
2. Empirical Hinglish code-mixed borrowing frequencies.
3. Domain mapping edges (Tech, Business, Academic, Medical, Finance).
4. Cross-lingual translation edges with naturalness penalties and semantic drift scores.
5. Word co-occurrence edges.

Exports:
  - data/vocabulary/laps_knowledge_graph.json
"""

import json
import os
from pathlib import Path
import networkx as nx

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "vocabulary"
OUTPUT_JSON = OUTPUT_DIR / "laps_knowledge_graph.json"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Raw Lexicon Seed Data with Translations & Naturalness Penalties
# ─────────────────────────────────────────────────────────────────────────────
LEXICON_SEED = [
    # Academic & University
    {"word": "viva", "hi_trans": "moukhik_pariksha", "domain": "Academic", "penalty": 0.85, "drift": 0.75, "borrow_prob": 0.95},
    {"word": "calendar", "hi_trans": "panchang", "domain": "Academic", "penalty": 0.80, "drift": 0.70, "borrow_prob": 0.90},
    {"word": "calender", "hi_trans": "panchang", "domain": "Academic", "penalty": 0.80, "drift": 0.70, "borrow_prob": 0.90},
    {"word": "assignment", "hi_trans": "karya", "domain": "Academic", "penalty": 0.75, "drift": 0.65, "borrow_prob": 0.92},
    {"word": "syllabus", "hi_trans": "pathyakram", "domain": "Academic", "penalty": 0.70, "drift": 0.60, "borrow_prob": 0.88},
    {"word": "lecture", "hi_trans": "vyakhyan", "domain": "Academic", "penalty": 0.75, "drift": 0.60, "borrow_prob": 0.86},
    {"word": "exam", "hi_trans": "pariksha", "domain": "Academic", "penalty": 0.40, "drift": 0.30, "borrow_prob": 0.90},
    {"word": "test", "hi_trans": "janch", "domain": "Academic", "penalty": 0.45, "drift": 0.35, "borrow_prob": 0.89},
    {"word": "grade", "hi_trans": "sreni", "domain": "Academic", "penalty": 0.60, "drift": 0.50, "borrow_prob": 0.85},
    {"word": "mark", "hi_trans": "ank", "domain": "Academic", "penalty": 0.50, "drift": 0.40, "borrow_prob": 0.88},
    {"word": "degree", "hi_trans": "upadhi", "domain": "Academic", "penalty": 0.70, "drift": 0.60, "borrow_prob": 0.85},

    # Tech & Software
    {"word": "link", "hi_trans": "kadi", "domain": "Tech", "penalty": 0.90, "drift": 0.80, "borrow_prob": 0.96},
    {"word": "server", "hi_trans": "sewak", "domain": "Tech", "penalty": 0.95, "drift": 0.85, "borrow_prob": 0.94},
    {"word": "code", "hi_trans": "koot", "domain": "Tech", "penalty": 0.92, "drift": 0.80, "borrow_prob": 0.95},
    {"word": "debug", "hi_trans": "truti_sudhar", "domain": "Tech", "penalty": 0.90, "drift": 0.85, "borrow_prob": 0.91},
    {"word": "deploy", "hi_trans": "tainat", "domain": "Tech", "penalty": 0.85, "drift": 0.75, "borrow_prob": 0.89},
    {"word": "database", "hi_trans": "tathya_sangrah", "domain": "Tech", "penalty": 0.90, "drift": 0.80, "borrow_prob": 0.92},
    {"word": "api", "hi_trans": "anuprayog_dwar", "domain": "Tech", "penalty": 0.98, "drift": 0.90, "borrow_prob": 0.97},
    {"word": "software", "hi_trans": "softwear", "domain": "Tech", "penalty": 0.85, "drift": 0.70, "borrow_prob": 0.96},
    {"word": "app", "hi_trans": "anuprayog", "domain": "Tech", "penalty": 0.80, "drift": 0.60, "borrow_prob": 0.95},
    {"word": "website", "hi_trans": "jalsthal", "domain": "Tech", "penalty": 0.90, "drift": 0.75, "borrow_prob": 0.93},

    # Business & Workplace
    {"word": "meeting", "hi_trans": "baithak", "domain": "Business", "penalty": 0.85, "drift": 0.70, "borrow_prob": 0.95},
    {"word": "client", "hi_trans": "grahak", "domain": "Business", "penalty": 0.80, "drift": 0.65, "borrow_prob": 0.93},
    {"word": "project", "hi_trans": "pariyojana", "domain": "Business", "penalty": 0.75, "drift": 0.60, "borrow_prob": 0.94},
    {"word": "postpone", "hi_trans": "sthagit", "domain": "Business", "penalty": 0.50, "drift": 0.40, "borrow_prob": 0.85},
    {"word": "schedule", "hi_trans": "anusuchee", "domain": "Business", "penalty": 0.60, "drift": 0.50, "borrow_prob": 0.88},
    {"word": "deadline", "hi_trans": "antim_tithi", "domain": "Business", "penalty": 0.85, "drift": 0.75, "borrow_prob": 0.90},
    {"word": "manager", "hi_trans": "prabandhak", "domain": "Business", "penalty": 0.70, "drift": 0.55, "borrow_prob": 0.92},
    {"word": "presentation", "hi_trans": "prastutikaran", "domain": "Business", "penalty": 0.75, "drift": 0.60, "borrow_prob": 0.89},

    # General & Greetings
    {"word": "hi", "hi_trans": "namaste", "domain": "General", "penalty": 0.30, "drift": 0.20, "borrow_prob": 0.90},
    {"word": "hello", "hi_trans": "namaste", "domain": "General", "penalty": 0.30, "drift": 0.20, "borrow_prob": 0.92},
    {"word": "thanks", "hi_trans": "dhanyawad", "domain": "General", "penalty": 0.40, "drift": 0.30, "borrow_prob": 0.88},
    {"word": "ok", "hi_trans": "theek_hai", "domain": "General", "penalty": 0.20, "drift": 0.10, "borrow_prob": 0.98},
]


def build_graph():
    G = nx.DiGraph()

    # 1. Add Domain Category Nodes
    domains = ["Domain_Tech", "Domain_Business", "Domain_Academic", "Domain_Medical", "Domain_Finance", "Domain_General"]
    for dom in domains:
        G.add_node(dom, type="DOMAIN", label=dom.replace("Domain_", ""))

    # 2. Add Word & Translation Nodes + Edges
    for entry in LEXICON_SEED:
        w_en = entry["word"].lower()
        w_hi = entry["hi_trans"].lower()
        dom_node = f"Domain_{entry['domain']}"

        # Get monolingual English zipf frequency
        if HAS_WORDFREQ:
            zipf_en = wordfreq.zipf_frequency(w_en, 'en')
            # Zipf scale is 0 to 7 -> normalize to [0, 1]
            f_mono = round(zipf_en / 7.0, 4)
        else:
            f_mono = 0.75

        f_borrow = entry["borrow_prob"]
        borrow_ratio = round(f_borrow / max(0.1, f_mono), 3)

        # English Node
        G.add_node(
            w_en,
            type="ENGLISH_LOANWORD",
            language="EN",
            f_mono_en=f_mono,
            f_borrow=f_borrow,
            borrowing_ratio=borrow_ratio,
        )

        # Hindi Node
        G.add_node(
            w_hi,
            type="NATIVE_HINDI",
            language="HI",
        )

        # Edge: BELONGS_TO_DOMAIN
        G.add_edge(w_en, dom_node, relation="BELONGS_TO", weight=0.90)

        # Edge: TRANSLATES_TO
        G.add_edge(
            w_en, w_hi,
            relation="TRANSLATES_TO",
            equivalence=round(1.0 - entry["drift"], 2),
            naturalness_penalty=entry["penalty"],
            meaning_drift=entry["drift"],
        )

    # 3. Add Co-occurrence Edges
    co_occurs = [
        ("viva", "calendar"), ("viva", "link"), ("viva", "assignment"),
        ("client", "meeting"), ("meeting", "schedule"), ("meeting", "postpone"),
        ("server", "debug"), ("code", "deploy"), ("database", "api")
    ]
    for w1, w2 in co_occurs:
        if G.has_node(w1) and G.has_node(w2):
            G.add_edge(w1, w2, relation="CO_OCCURS_WITH", weight=0.85)

    # Export to Node-Link JSON format
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = nx.node_link_data(G)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Successfully built LAPS Knowledge Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    build_graph()
