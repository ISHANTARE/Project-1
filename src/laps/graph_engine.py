"""
LAPS Symmetric Bidirectional SQLite Graph Engine
------------------------------------------------
Queries the Bidirectional Knowledge Graph Database (`laps_knowledge_graph.db`) for:
1. English -> Hindi translation penalties & loanword borrowing ratios.
2. Hindi -> English translation necessity & Devanagari script lookups.
3. Reverse borrowing index (Hindi terms in global English).
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "vocabulary" / "laps_knowledge_graph.db"


class KnowledgeGraphEngine:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def has_node(self, token: str) -> bool:
        if not self.db_path.exists():
            return False
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM nodes WHERE id = ?", (token.lower(),))
            return cur.fetchone() is not None

    def get_node_attributes(self, token: str) -> dict:
        """Fetch node attributes (frequencies, script_devanagari, borrowing ratios, domain)."""
        if not self.db_path.exists():
            return {}
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, type, language, script_devanagari, f_mono_en, f_mono_hi, f_borrow, borrowing_ratio, domain, is_reverse_borrow
                FROM nodes WHERE id = ?
            """, (token.lower(),))
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "type": row[1],
                    "language": row[2],
                    "script_devanagari": row[3],
                    "f_mono_en": row[4],
                    "f_mono_hi": row[5],
                    "f_borrow": row[6],
                    "borrowing_ratio": row[7],
                    "domain": row[8],
                    "is_reverse_borrow": bool(row[9]),
                }
        return {}

    def get_domain_specificity(self, token: str) -> float:
        """Retrieve domain specificity weight from SQLite nodes/domain."""
        attrs = self.get_node_attributes(token)
        if not attrs:
            return 0.30
        dom = attrs.get("domain", "General")
        return 0.85 if dom != "General" else 0.40

    def get_en_to_hi_edge(self, token: str) -> dict:
        """Query EN -> HI translation edge (penalty & meaning drift)."""
        if not self.db_path.exists():
            return {"naturalness_penalty": 0.50, "meaning_drift": 0.50}

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT naturalness_penalty, weight FROM edges
                WHERE source = ? AND relation = 'EN_TO_HI_TRANSLATION'
            """, (token.lower(),))
            row = cur.fetchone()
            if row:
                return {
                    "naturalness_penalty": row[0],
                    "meaning_drift": round(1.0 - row[0], 2),
                }

        return {"naturalness_penalty": 0.50, "meaning_drift": 0.50}

    def get_translation_edge(self, token: str) -> dict:
        """Alias for get_en_to_hi_edge."""
        return self.get_en_to_hi_edge(token)

    def get_hi_to_en_edge(self, token: str) -> dict:
        """Query HI -> EN translation edge (necessity & meaning recovery)."""
        if not self.db_path.exists():
            return {"translation_necessity": 0.50, "meaning_recovery": 0.50}

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT translation_necessity, weight FROM edges
                WHERE source = ? AND relation = 'HI_TO_EN_TRANSLATION'
            """, (token.lower(),))
            row = cur.fetchone()
            if row:
                return {
                    "translation_necessity": row[0],
                    "meaning_recovery": round(row[0], 2),
                }

        return {"translation_necessity": 0.50, "meaning_recovery": 0.50}

    def get_context_cooccurrence(self, token: str, context_window: list[str]) -> float:
        """Assess context co-occurrence density via SQL edge queries."""
        if not context_window or not self.db_path.exists():
            return 0.50

        tok = token.lower()
        with self._get_connection() as conn:
            cur = conn.cursor()
            placeholders = ",".join(["?"] * len(context_window))
            query = f"SELECT COUNT(*) FROM edges WHERE source = ? AND target IN ({placeholders})"
            params = [tok] + [c.lower() for c in context_window]
            cur.execute(query, params)
            count = cur.fetchone()[0]

        return round(min(1.0, 0.50 + (count * 0.20)), 2)

    def is_reverse_borrowed(self, token: str) -> bool:
        """Check if token is a Hindi word borrowed into global English (e.g. jugaad, chai, guru)."""
        attrs = self.get_node_attributes(token)
        return attrs.get("is_reverse_borrow", False)


_KG_ENGINE = None

def get_graph_engine() -> KnowledgeGraphEngine:
    global _KG_ENGINE
    if _KG_ENGINE is None:
        _KG_ENGINE = KnowledgeGraphEngine()
    return _KG_ENGINE


if __name__ == "__main__":
    kg = get_graph_engine()
    print("=== ENGLISH LOANWORD NODE ===")
    print("meeting:", kg.get_node_attributes("meeting"))
    print("EN->HI Edge:", kg.get_en_to_hi_edge("meeting"))

    print("\n=== HINDI NATIVE NODE ===")
    b_attrs = kg.get_node_attributes("baithak")
    print(f"baithak ID: {b_attrs.get('id')} | lang: {b_attrs.get('language')} | domain: {b_attrs.get('domain')}")
    print("HI->EN Edge:", kg.get_hi_to_en_edge("baithak"))

    print("\n=== REVERSE BORROWING NODE ===")
    print("jugaad reverse borrow?", kg.is_reverse_borrowed("jugaad"))
