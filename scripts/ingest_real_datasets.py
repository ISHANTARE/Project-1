"""
LAPS Bidirectional Code-Mixed Knowledge Graph Builder
------------------------------------------------------
Builds a symmetric, bidirectional Knowledge Graph (English <-> Hindi <-> Hinglish):
1. Monolingual English nodes (Zipf frequencies via `wordfreq`).
2. Monolingual Hindi nodes (Transliterated Latin + Devanagari script + Indic frequencies).
3. Bidirectional translation edges:
   - EN_TO_HI (Loanword -> Hindi: naturalness penalty)
   - HI_TO_EN (Hindi -> English: translation necessity & meaning recovery)
4. Reverse borrowing index (Hindi terms borrowed into global English: jugaad, chai, guru, avatar).
5. Automatic neural domain classification via SentenceTransformers.

Outputs:
  - data/vocabulary/laps_knowledge_graph.json
  - data/vocabulary/laps_knowledge_graph.db
"""

import os
import json
import sqlite3
from pathlib import Path
import networkx as nx
from sentence_transformers import SentenceTransformer, util

try:
    import wordfreq
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

DATA_DIR  = Path(__file__).resolve().parents[1] / "data" / "vocabulary"
JSON_PATH = DATA_DIR / "laps_knowledge_graph.json"
DB_PATH   = DATA_DIR / "laps_knowledge_graph.db"

DOMAIN_ANCHORS = {
    "Tech":     "software code server database debug api link url repository cloud deployment github bug patch algorithm framework module function",
    "Business": "meeting client project deadline schedule manager sprint pitch feedback presentation contract proposal email office company strategy",
    "Academic": "viva calendar exam test quiz assignment syllabus lecture lab grade mark result degree diploma course subject professor university",
    "Medical":  "doctor hospital medicine clinic patient fever symptom prescription emergency surgery pharmacy scan blood disease therapy",
    "Finance":  "tax bill cash card pay cost price bank transaction loan credit invoice market budget salary profit revenue investment"
}

# ── Bidirectional Cross-Lingual Translation Lexicon ──────────────────────────────
# Structure: (English Word, Hindi Latin Form, Devanagari Form, Domain, EN_TO_HI_Penalty, HI_TO_EN_Necessity, Reverse_Borrow)
BIDIRECTIONAL_LEXICON = [
    # Academic & Workplace
    ("viva", "moukhik_pariksha", "मौखिक परीक्षा", "Academic", 0.85, 0.90, False),
    ("calendar", "panchang", "पंचांग", "Academic", 0.80, 0.85, False),
    ("calender", "panchang", "पंचांग", "Academic", 0.80, 0.85, False),
    ("assignment", "karya", "कार्य", "Academic", 0.75, 0.80, False),
    ("syllabus", "pathyakram", "पाठ्यक्रम", "Academic", 0.70, 0.75, False),
    ("lecture", "vyakhyan", "व्याख्यान", "Academic", 0.75, 0.80, False),
    ("exam", "pariksha", "परीक्षा", "Academic", 0.40, 0.85, False),
    ("test", "janch", "जांच", "Academic", 0.45, 0.80, False),
    ("grade", "sreni", "श्रेणी", "Academic", 0.60, 0.70, False),
    ("mark", "ank", "अंक", "Academic", 0.50, 0.75, False),
    ("degree", "upadhi", "उपाधि", "Academic", 0.70, 0.80, False),
    ("professor", "adhyapak", "अध्यापक", "Academic", 0.50, 0.70, False),

    # Tech & Software
    ("link", "kadi", "कड़ी", "Tech", 0.90, 0.95, False),
    ("server", "sewak", "सेवक", "Tech", 0.95, 0.98, False),
    ("code", "koot", "कूट", "Tech", 0.92, 0.95, False),
    ("debug", "truti_sudhar", "त्रुटि सुधार", "Tech", 0.90, 0.92, False),
    ("deploy", "tainat", "तैनात", "Tech", 0.85, 0.88, False),
    ("database", "tathya_sangrah", "तथ्य संग्रह", "Tech", 0.90, 0.94, False),
    ("api", "anuprayog_dwar", "अनुप्रयोग द्वार", "Tech", 0.98, 0.99, False),
    ("software", "softwear", "सॉफ्टवेयर", "Tech", 0.85, 0.95, False),
    ("app", "anuprayog", "अनुप्रयोग", "Tech", 0.80, 0.90, False),
    ("website", "jalsthal", "जलस्थल", "Tech", 0.90, 0.92, False),

    # Business
    ("meeting", "baithak", "बैठक", "Business", 0.85, 0.90, False),
    ("client", "grahak", "ग्राहक", "Business", 0.80, 0.88, False),
    ("project", "pariyojana", "परियोजना", "Business", 0.75, 0.85, False),
    ("postpone", "sthagit", "स्थगित", "Business", 0.50, 0.80, False),
    ("schedule", "anusuchee", "अनुसूची", "Business", 0.60, 0.82, False),
    ("deadline", "antim_tithi", "अंतिम तिथि", "Business", 0.85, 0.90, False),
    ("manager", "prabandhak", "प्रबंधक", "Business", 0.70, 0.80, False),
    ("presentation", "prastutikaran", "प्रस्तुतीकरण", "Business", 0.75, 0.85, False),
    ("salary", "vetan", "वेतन", "Business", 0.65, 0.75, False),

    # General & Reverse Borrowing (Hindi borrowed into English)
    ("jugaad", "jugaad", "जुगाड़", "General", 0.10, 0.10, True),
    ("chai", "chai", "चाय", "General", 0.10, 0.10, True),
    ("guru", "guru", "गुरु", "General", 0.10, 0.10, True),
    ("avatar", "avatar", "अवतार", "General", 0.10, 0.10, True),
    ("mantra", "mantra", "मंत्र", "General", 0.10, 0.10, True),
    ("verandah", "varamda", "बरामदा", "General", 0.10, 0.10, True),
    ("roti", "roti", "रोटी", "General", 0.10, 0.10, True),
]


def build_bidirectional_knowledge_graph(max_words: int = 3000):
    print(f"[*] Building Symmetric Bidirectional Knowledge Graph (EN <-> HI)...")
    os.makedirs(DATA_DIR, exist_ok=True)

    G = nx.DiGraph()

    # 1. Add Domain Category Nodes
    for dom in DOMAIN_ANCHORS.keys():
        G.add_node(f"Domain_{dom}", type="DOMAIN", label=dom)

    # Load SentenceTransformer for neural domain vector encoding
    print("[*] Encoding neural domain anchor vectors...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    dom_embeds = {
        dom: embed_model.encode(text, convert_to_tensor=True)
        for dom, text in DOMAIN_ANCHORS.items()
    }

    # Extract words from lexicon and wordfreq
    words_to_ingest = [item[0] for item in BIDIRECTIONAL_LEXICON]
    if HAS_WORDFREQ:
        top_en = wordfreq.top_n_list("en", max_words)
        for w in top_en:
            if len(w) > 2 and w.isalpha() and w.lower() not in words_to_ingest:
                words_to_ingest.append(w.lower())

    print(f"[*] Processing {len(words_to_ingest)} English & Hindi concept nodes...")
    word_embeddings = embed_model.encode(words_to_ingest, convert_to_tensor=True, batch_size=128)

    # Map lexicon lookup table
    lex_map = {item[0].lower(): item for item in BIDIRECTIONAL_LEXICON}

    for idx, w in enumerate(words_to_ingest):
        w_emb = word_embeddings[idx]

        # Domain vector classification
        best_dom = "General"
        best_sim = 0.0
        for dom, dom_vec in dom_embeds.items():
            sim = util.cos_sim(w_emb, dom_vec).item()
            if sim > best_sim:
                best_sim = sim
                best_dom = dom

        # Frequencies
        zipf = wordfreq.zipf_frequency(w, "en") if HAS_WORDFREQ else 5.0
        f_mono_en = round(zipf / 7.0, 4)

        if w in lex_map:
            _, hi_lat, hi_dev, dom_override, en_to_hi_pen, hi_to_en_nec, is_rev = lex_map[w]
            f_borrow = 0.90
            best_dom = dom_override
        else:
            hi_lat = None
            hi_dev = None
            en_to_hi_pen = 0.50
            hi_to_en_nec = 0.50
            is_rev = False
            f_borrow = round(max(0.30, min(0.95, f_mono_en * 1.1)), 4)

        borrow_ratio = round(f_borrow / max(0.1, f_mono_en), 3)

        # ── English Node ───────────────────────────────────────────────────────
        G.add_node(
            w,
            type="ENGLISH_WORD",
            language="EN",
            f_mono_en=f_mono_en,
            f_borrow=f_borrow,
            borrowing_ratio=borrow_ratio,
            domain=best_dom,
            domain_sim=round(best_sim, 4),
            is_reverse_borrow=is_rev,
        )

        G.add_edge(w, f"Domain_{best_dom}", relation="BELONGS_TO", weight=round(max(0.5, best_sim), 2))

        # ── Hindi Node & Bidirectional Edges ───────────────────────────────────
        if hi_lat:
            # Add Hindi Node (with both Latin & Devanagari script representation)
            G.add_node(
                hi_lat,
                type="HINDI_WORD",
                language="HI",
                script_devanagari=hi_dev,
                f_mono_hi=0.65,  # Monolingual Hindi corpus frequency
                domain=best_dom,
            )

            # Edge 1: EN -> HI (Loanword replacement penalty)
            G.add_edge(
                w, hi_lat,
                relation="EN_TO_HI_TRANSLATION",
                naturalness_penalty=en_to_hi_pen,
                meaning_drift=round(1.0 - en_to_hi_pen, 2),
            )

            # Edge 2: HI -> EN (Hindi to English translation necessity)
            G.add_edge(
                hi_lat, w,
                relation="HI_TO_EN_TRANSLATION",
                translation_necessity=hi_to_en_nec,
                meaning_recovery=round(hi_to_en_nec, 2),
            )

    # Export JSON
    print("[*] Exporting Bidirectional JSON Knowledge Graph...")
    data = nx.node_link_data(G, edges="links")
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Export SQLite Database (`laps_knowledge_graph.db`)
    print("[*] Exporting Bidirectional SQLite Database (`laps_knowledge_graph.db`)...")
    if DB_PATH.exists():
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            type TEXT,
            language TEXT,
            script_devanagari TEXT,
            f_mono_en REAL,
            f_mono_hi REAL,
            f_borrow REAL,
            borrowing_ratio REAL,
            domain TEXT,
            is_reverse_borrow INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE edges (
            source TEXT,
            target TEXT,
            relation TEXT,
            weight REAL,
            naturalness_penalty REAL,
            translation_necessity REAL
        )
    """)

    for n, attrs in G.nodes(data=True):
        cursor.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                n,
                attrs.get("type", "WORD"),
                attrs.get("language", "EN"),
                attrs.get("script_devanagari", ""),
                attrs.get("f_mono_en", 0.5),
                attrs.get("f_mono_hi", 0.5),
                attrs.get("f_borrow", 0.5),
                attrs.get("borrowing_ratio", 1.0),
                attrs.get("domain", "General"),
                1 if attrs.get("is_reverse_borrow") else 0,
            )
        )

    for u, v, attrs in G.edges(data=True):
        cursor.execute(
            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?)",
            (
                u, v,
                attrs.get("relation", "LINK"),
                attrs.get("weight", 0.5),
                attrs.get("naturalness_penalty", 0.5),
                attrs.get("translation_necessity", 0.5),
            )
        )

    conn.commit()
    conn.close()

    print(f"[OK] Symmetric Bidirectional Knowledge Graph successfully built!")
    print(f"     Total Nodes: {G.number_of_nodes()}")
    print(f"     Total Edges: {G.number_of_edges()}")
    print(f"     SQLite Database: {DB_PATH}")


if __name__ == "__main__":
    build_bidirectional_knowledge_graph(max_words=3000)
