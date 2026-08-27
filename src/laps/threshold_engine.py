"""
LAPS Threshold Engine
---------------------
Manages context-register-aware thresholds (θ).

Registers:
  casual    → θ = 0.35   (WhatsApp, informal chat)
  business  → θ = 0.50   (professional emails, Slack)
  academic  → θ = 0.65   (reports, papers)
  social    → θ = 0.40   (Twitter, Instagram captions)

Given a LAPS score and a register, returns a PreservationDecision:
  PRESERVE  – keep the token in its original (English) form
  REWRITE   – let the LLM translate / rephrase the token
"""

from dataclasses import dataclass

# ── Threshold table ────────────────────────────────────────────────────────────
REGISTER_THRESHOLDS: dict[str, float] = {
    "casual":   0.35,
    "business": 0.50,
    "academic": 0.65,
    "social":   0.40,
}

DEFAULT_REGISTER = "business"


@dataclass
class PreservationDecision:
    token:     str
    score:     float
    threshold: float
    register:  str
    decision:  str          # "PRESERVE" | "REWRITE"
    confidence: float       # |score - threshold| → how far from the boundary


def decide(token: str, score: float, register: str = DEFAULT_REGISTER) -> PreservationDecision:
    """
    Make a binary preserve/rewrite decision based on the LAPS score and register.

    Args:
        token    : surface form of the token
        score    : computed LAPS preservation score
        register : one of 'casual', 'business', 'academic', 'social'

    Returns:
        PreservationDecision dataclass
    """
    register = register.lower()
    if register not in REGISTER_THRESHOLDS:
        register = DEFAULT_REGISTER

    theta      = REGISTER_THRESHOLDS[register]
    decision   = "PRESERVE" if score >= theta else "REWRITE"
    confidence = round(abs(score - theta), 4)

    return PreservationDecision(
        token=token,
        score=score,
        threshold=theta,
        register=register,
        decision=decision,
        confidence=confidence,
    )


def batch_decide(scored_tokens: list[dict], register: str = DEFAULT_REGISTER) -> list[dict]:
    """
    Apply the threshold decision to every scored token in the sentence.

    Args:
        scored_tokens : output of scorer.score_all_tokens()
        register      : context register string

    Returns:
        list of enriched token dicts with 'decision' and 'confidence' keys added
    """
    results = []
    for tok in scored_tokens:
        entry = dict(tok)
        if tok["laps_score"] is not None:
            pd = decide(tok["token"], tok["laps_score"], register)
            entry["decision"]   = pd.decision
            entry["confidence"] = pd.confidence
            entry["threshold"]  = pd.threshold
        else:
            # Non-candidate tokens (pure Hindi words, punctuation)
            entry["decision"]   = "NATIVE"
            entry["confidence"] = None
            entry["threshold"]  = None
        results.append(entry)
    return results


def available_registers() -> list[str]:
    return list(REGISTER_THRESHOLDS.keys())


if __name__ == "__main__":
    d = decide("meeting", 0.72, "business")
    print(d)
    d2 = decide("postpone", 0.31, "casual")
    print(d2)
