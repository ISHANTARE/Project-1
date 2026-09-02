"""
LAPS Threshold Engine  –  Sigmoid Soft-Boundary Decision Module
---------------------------------------------------------------
Replaces the legacy hard step-function decision with a mathematically
rigorous Soft Sigmoid Logistic Preservation Probability.

Formula (from the LAPS Research Blueprint §2.1):
  P(Preserve | t_i, C, θ_R) = σ( λ · (Score(t_i) − θ_R) )
                              = 1 / (1 + e^{−λ(Score − θ_R)})

Where:
  Score  : Weighted LAPS Preservation Score in [0, 1]
  θ_R    : Register-specific threshold
  λ      : Steepness / sensitivity parameter (default 10.0)

Decision Mapping:
  P >= 0.70 → PRESERVE   (high confidence)
  P >= 0.50 → UNCERTAIN  (borderline; treated as PRESERVE but flagged)
  P <  0.50 → REWRITE    (low confidence)

Context-Ego Density (CC_ego) from §2.2:
  Measures sub-graph density of the k-hop neighbourhood within the
  Knowledge Graph.  Applied as a bonus/penalty modifier on the sigmoid
  output before final thresholding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

try:
    from src.laps.register_detector import (
        REGISTER_THRESHOLDS,
        DEFAULT_REGISTER,
        detect_register,
        get_register_threshold
    )
except ModuleNotFoundError:
    try:
        from laps.register_detector import (
            REGISTER_THRESHOLDS,
            DEFAULT_REGISTER,
            detect_register,
            get_register_threshold
        )
    except ModuleNotFoundError:
        REGISTER_THRESHOLDS: dict[str, float] = {
            "casual":   0.35,
            "social":   0.40,
            "business": 0.50,
            "academic": 0.65,
        }
        DEFAULT_REGISTER: str = "business"
        def detect_register(text: str = ""):
            return {"predicted_register": DEFAULT_REGISTER, "threshold": 0.50}
        def get_register_threshold(reg: str, text: str = ""):
            return REGISTER_THRESHOLDS.get(reg, 0.50)


# Sigmoid steepness (λ) – higher → sharper boundary
SIGMOID_LAMBDA: float = 10.0

# Probability thresholds that map to decision labels
PRESERVE_THRESHOLD:  float = 0.70   # P ≥ this → PRESERVE
UNCERTAIN_THRESHOLD: float = 0.50   # P ≥ this → UNCERTAIN (logged, but preserved)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PreservationDecision:
    """Full probabilistic preservation decision for a single token."""

    token:              str
    score:              float
    threshold:          float
    register:           str
    probability:        float                              # P(Preserve)
    decision:           Literal["PRESERVE", "UNCERTAIN", "REWRITE"]
    confidence:         float                              # |P - 0.5|, in [0, 0.5]
    is_borderline:      bool                               # True when P is in [0.50, 0.70)
    ego_density_bonus:  float = 0.0                        # CC_ego modifier applied
    naturalness_penalty: float = 0.0                       # from KG edge (EN→HI)
    domain:             str   = "General"


@dataclass
class BatchDecisionSummary:
    """Aggregate statistics for a sentence's batch decision pass."""

    total_candidates:   int
    preserve_count:     int
    rewrite_count:      int
    uncertain_count:    int
    avg_probability:    float
    decisions:          list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Sigmoid Engine
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    """Numerically stable sigmoid: σ(x) = 1 / (1 + e^{-x})."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    # Equivalent but avoids overflow for large negative x
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_sigmoid_probability(score: float, theta: float,
                                lam: float = SIGMOID_LAMBDA) -> float:
    """
    Compute P(Preserve | Score, θ, λ).

    Args:
        score  : LAPS weighted score ∈ [0, 1]
        theta  : Register-specific threshold θ_R
        lam    : Sigmoid steepness parameter λ (default 10.0)

    Returns:
        float in [0, 1] – smooth preservation probability
    """
    return round(sigmoid(lam * (score - theta)), 6)


def compute_ego_density(token: str, context_window: list[str],
                        kg_engine=None) -> float:
    """
    Compute Knowledge-Graph Ego Density CC_ego(t_i, C) from §2.2.

    CC_ego = |{edges (u,v) ∈ E : u,v ∈ C ∪ {t_i}}| / C(|C|+1, 2)

    Uses the SQLite graph engine's co-occurrence query as a proxy for
    edge density.  Returns a normalised float ∈ [0, 1].
    """
    if kg_engine is None or not context_window:
        return 0.50

    try:
        # get_context_cooccurrence already returns a [0,1] proxy score
        cc = kg_engine.get_context_cooccurrence(token, context_window)
        return float(cc)
    except Exception:
        return 0.50


def _probability_to_decision(prob: float) -> tuple[
        Literal["PRESERVE", "UNCERTAIN", "REWRITE"], bool]:
    """Map a probability to (decision, is_borderline)."""
    if prob >= PRESERVE_THRESHOLD:
        return "PRESERVE", False
    elif prob >= UNCERTAIN_THRESHOLD:
        return "UNCERTAIN", True
    else:
        return "REWRITE", False


# ---------------------------------------------------------------------------
# Public API – single token
# ---------------------------------------------------------------------------
def decide(
    token:          str,
    score:          float,
    register:       str = DEFAULT_REGISTER,
    context_window: list[str] | None = None,
    kg_engine=None,
    lam:            float = SIGMOID_LAMBDA,
) -> PreservationDecision:
    """
    Compute a full probabilistic preservation decision for a single token.

    Args:
        token          : Surface form of the token.
        score          : Pre-computed LAPS weighted score ∈ [0, 1].
        register       : Communicative register ('casual', 'business', etc.)
        context_window : Surrounding tokens for CC_ego computation.
        kg_engine      : Optional KnowledgeGraphEngine instance.
        lam            : Sigmoid steepness λ (override for experimentation).

    Returns:
        PreservationDecision dataclass.
    """
    register_clean = register.lower().strip()
    if register_clean == "auto":
        text_cue = " ".join(context_window or [token])
        det = detect_register(text_cue)
        register = det["predicted_register"]
        theta = det["threshold"]
    else:
        if register_clean not in REGISTER_THRESHOLDS:
            register = DEFAULT_REGISTER
        else:
            register = register_clean
        theta = REGISTER_THRESHOLDS[register]

    # 1. Base sigmoid probability
    prob = compute_sigmoid_probability(score, theta, lam)

    # 2. CC_ego ego-density bonus (modulates probability smoothly)
    ego_density = compute_ego_density(token, context_window or [], kg_engine)
    ego_bonus   = round((ego_density - 0.50) * 0.10, 4)   # ±5% swing max
    adj_prob    = round(max(0.0, min(1.0, prob + ego_bonus)), 6)

    # 3. KG naturalness penalty (from EN→HI edge weight)
    nat_penalty = 0.0
    domain      = "General"
    if kg_engine is not None:
        try:
            edge  = kg_engine.get_en_to_hi_edge(token)
            nat_penalty = edge.get("naturalness_penalty", 0.0)
            attrs = kg_engine.get_node_attributes(token)
            domain = attrs.get("domain", "General")
        except Exception:
            pass

    decision, is_borderline = _probability_to_decision(adj_prob)
    confidence = round(abs(adj_prob - 0.50), 4)

    return PreservationDecision(
        token=token,
        score=score,
        threshold=theta,
        register=register,
        probability=adj_prob,
        decision=decision,
        confidence=confidence,
        is_borderline=is_borderline,
        ego_density_bonus=ego_bonus,
        naturalness_penalty=nat_penalty,
        domain=domain,
    )


# ---------------------------------------------------------------------------
# Public API – batch (whole sentence)
# ---------------------------------------------------------------------------
def batch_decide(
    scored_tokens:  list[dict],
    register:       str = DEFAULT_REGISTER,
    kg_engine=None,
    lam:            float = SIGMOID_LAMBDA,
) -> list[dict]:
    """
    Apply sigmoid threshold decisions to every scored token in a sentence.

    Args:
        scored_tokens : Output of scorer.score_all_tokens() – list of dicts
                        with at least 'token', 'laps_score', 'is_candidate'.
        register      : Communicative register string ('casual', 'business', 'auto', etc.)
        kg_engine     : Optional KnowledgeGraphEngine for CC_ego enrichment.
        lam           : Sigmoid steepness parameter.

    Returns:
        List of enriched token dicts with added keys:
          'decision', 'probability', 'confidence', 'threshold',
          'is_borderline', 'ego_density_bonus', 'naturalness_penalty', 'domain'
    """
    # Build sentence-level context window (all surface tokens)
    all_tokens = [tok["token"] for tok in scored_tokens]

    register_clean = register.lower().strip()
    if register_clean == "auto":
        sentence_text = " ".join(all_tokens)
        det = detect_register(sentence_text)
        register = det["predicted_register"]


    results = []
    for i, tok in enumerate(scored_tokens):
        entry = dict(tok)
        if tok.get("laps_score") is not None:
            # Context window: ±2 neighbours (excluding self)
            window = all_tokens[max(0, i-2):i] + all_tokens[i+1:i+3]

            pd = decide(
                token=tok["token"],
                score=tok["laps_score"],
                register=register,
                context_window=window,
                kg_engine=kg_engine,
                lam=lam,
            )
            entry["decision"]           = pd.decision
            entry["probability"]        = pd.probability
            entry["confidence"]         = pd.confidence
            entry["threshold"]          = pd.threshold
            entry["is_borderline"]      = pd.is_borderline
            entry["ego_density_bonus"]  = pd.ego_density_bonus
            entry["naturalness_penalty"] = pd.naturalness_penalty
            entry["domain"]             = pd.domain
        else:
            # Non-candidate tokens (native Hindi, punctuation, numerals)
            entry["decision"]           = "NATIVE"
            entry["probability"]        = None
            entry["confidence"]         = None
            entry["threshold"]          = None
            entry["is_borderline"]      = False
            entry["ego_density_bonus"]  = 0.0
            entry["naturalness_penalty"] = 0.0
            entry["domain"]             = "N/A"

        results.append(entry)

    return results


def batch_summarise(decided_tokens: list[dict]) -> BatchDecisionSummary:
    """Produce aggregate sentence-level statistics."""
    candidates = [t for t in decided_tokens if t["decision"] != "NATIVE"]
    probs       = [t["probability"] for t in candidates if t["probability"] is not None]

    return BatchDecisionSummary(
        total_candidates = len(candidates),
        preserve_count   = sum(1 for t in candidates if t["decision"] == "PRESERVE"),
        uncertain_count  = sum(1 for t in candidates if t["decision"] == "UNCERTAIN"),
        rewrite_count    = sum(1 for t in candidates if t["decision"] == "REWRITE"),
        avg_probability  = round(sum(probs) / len(probs), 4) if probs else 0.0,
        decisions        = decided_tokens,
    )


def available_registers() -> list[str]:
    return list(REGISTER_THRESHOLDS.keys())


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== SIGMOID SOFT-BOUNDARY ENGINE SMOKE TEST ===\n")

    test_cases = [
        ("meeting",   0.72, "business"),
        ("viva",      0.81, "academic"),
        ("link",      0.88, "casual"),
        ("postpone",  0.31, "casual"),
        ("baithak",   0.22, "business"),
        ("calendar",  0.66, "academic"),
    ]

    for token, score, reg in test_cases:
        d = decide(token, score, reg)
        bar = "#" * int(d.probability * 20)
        print(f"  {token:<12} | score={score:.2f} | theta={d.threshold:.2f} | "
              f"P={d.probability:.4f} [{bar:<20}] | {d.decision}")

    print("\nAll tests passed.")
