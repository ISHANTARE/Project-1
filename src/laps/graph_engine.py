"""
LAPS Knowledge Graph Engine
---------------------------
Queries the Bilingual Code-Mixed Knowledge Graph (`data/vocabulary/laps_knowledge_graph.json`)
using graph traversal & sub-graph neighborhood analysis.
"""

import json
from pathlib import Path
import networkx as nx

GRAPH_PATH = Path(__file__).resolve().parents[2] / "data" / "vocabulary" / "laps_knowledge_graph.json"


class KnowledgeGraphEngine:
    def __init__(self, graph_path: Path = GRAPH_PATH):
        self.graph_path = graph_path
        self.G = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        if not self.graph_path.exists():
            # Return empty graph fallback
            return nx.DiGraph()

        with open(self.graph_path, encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data, edges="links")

    def has_node(self, token: str) -> bool:
        return self.G.has_node(token.lower())

    def get_node_attributes(self, token: str) -> dict:
        """Fetch node attributes (frequencies, borrowing ratios, POS)."""
        tok = token.lower()
        if self.G.has_node(tok):
            return dict(self.G.nodes[tok])
        return {}

    def get_borrowing_frequency(self, token: str) -> float:
        """Retrieve borrowing frequency ratio from graph node."""
        attrs = self.get_node_attributes(token)
        return attrs.get("f_borrow", 0.70)

    def get_domain_specificity(self, token: str) -> float:
        """Find graph distance/weight to domain category nodes."""
        tok = token.lower()
        if not self.G.has_node(tok):
            return 0.30

        # Check direct BELONGS_TO edges
        for nbr in self.G.neighbors(tok):
            if nbr.startswith("Domain_"):
                edge_data = self.G.get_edge_data(tok, nbr)
                return edge_data.get("weight", 0.85)

        return 0.40

    def get_translation_edge(self, token: str) -> dict:
        """Fetch TRANSLATES_TO edge parameters (equivalence, naturalness penalty, meaning drift)."""
        tok = token.lower()
        if not self.G.has_node(tok):
            return {"equivalence": 0.50, "naturalness_penalty": 0.50, "meaning_drift": 0.50}

        for nbr in self.G.neighbors(tok):
            edge_data = self.G.get_edge_data(tok, nbr)
            if edge_data.get("relation") == "TRANSLATES_TO":
                return edge_data

        return {"equivalence": 0.50, "naturalness_penalty": 0.50, "meaning_drift": 0.50}

    def get_context_cooccurrence(self, token: str, context_window: list[str]) -> float:
        """Assess context graph neighborhood density."""
        tok = token.lower()
        if not self.G.has_node(tok) or not context_window:
            return 0.50

        neighbors = set(self.G.neighbors(tok))
        matches = sum(1 for c in context_window if c.lower() in neighbors)
        return round(0.50 + (matches * 0.25), 2)


# Singleton graph instance
_KG_ENGINE = None

def get_graph_engine() -> KnowledgeGraphEngine:
    global _KG_ENGINE
    if _KG_ENGINE is None:
        _KG_ENGINE = KnowledgeGraphEngine()
    return _KG_ENGINE


if __name__ == "__main__":
    kg = get_graph_engine()
    print("Node 'meeting':", kg.get_node_attributes("meeting"))
    print("Translation Edge 'meeting':", kg.get_translation_edge("meeting"))
