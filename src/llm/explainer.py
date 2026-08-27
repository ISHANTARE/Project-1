"""
LAPS Explanation Generator
--------------------------
Produces human-readable, token-level justifications for every
PRESERVE / REWRITE decision made by the LAPS pipeline.
"""

# ── Explanation templates ──────────────────────────────────────────────────────

_PRESERVE_REASONS = [
    ("BF", 0.80, "Highly frequent borrowed word in code-mixed speech."),
    ("DS", 0.80, "Domain-specific technical or professional term."),
    ("SI", 0.80, "Translating would alter the nuanced meaning."),
    ("AF", 0.80, "Universally familiar to the target audience."),
    ("CC", 0.70, "Professional/English-heavy surrounding context."),
]

_REWRITE_REASONS = [
    ("TB", 0.75, "Native-language equivalent significantly improves readability."),
    ("BF", 0.50, "Word is not commonly borrowed in natural speech."),
    ("DS", 0.30, "Not a recognised domain-specific term."),
    ("AF", 0.40, "Target audience may not be familiar with the English form."),
]


def _primary_reason(features: dict, decision: str) -> str:
    """Pick the single most influential reason for the decision."""
    if decision == "PRESERVE":
        # Highest positive-scoring feature
        top = max(
            [("BF", features.get("BF", 0)),
             ("DS", features.get("DS", 0)),
             ("SI", features.get("SI", 0)),
             ("AF", features.get("AF", 0)),
             ("CC", features.get("CC", 0))],
            key=lambda x: x[1]
        )
        for feat, thresh, msg in _PRESERVE_REASONS:
            if feat == top[0] and top[1] >= thresh:
                return msg
        return "Overall preservation score exceeds the context threshold."
    else:  # REWRITE
        tb = features.get("TB", 0)
        if tb >= 0.75:
            return "Native-language equivalent significantly improves readability."
        return "Insufficient preservation signals for this context register."


def _supporting_features(features: dict, decision: str) -> list[str]:
    """List the top-2 supporting feature observations."""
    items = []
    if decision == "PRESERVE":
        if features.get("BF", 0) >= 0.70:
            items.append(f"BF={features['BF']:.2f} – commonly borrowed word.")
        if features.get("DS", 0) >= 0.70:
            items.append(f"DS={features['DS']:.2f} – matches a technical/business domain.")
        if features.get("SI", 0) >= 0.70:
            items.append(f"SI={features['SI']:.2f} – translation risks semantic drift.")
        if features.get("CC", 0) >= 0.60:
            items.append(f"CC={features['CC']:.2f} – English-dominant surrounding context.")
    else:
        if features.get("TB", 0) >= 0.70:
            items.append(f"TB={features['TB']:.2f} – translation yields higher clarity.")
        if features.get("BF", 0) < 0.50:
            items.append(f"BF={features['BF']:.2f} – word is not commonly code-switched.")
        if features.get("AF", 0) < 0.50:
            items.append(f"AF={features['AF']:.2f} – audience may not recognise the term.")
    return items[:2]   # at most 2


def explain_token(token_result: dict) -> dict:
    """
    Generate a structured explanation for a single token decision.

    Args:
        token_result : a fully-scored token dict from pipeline.run()

    Returns:
        Explanation dict with keys: token, decision, score, threshold,
        probability, is_borderline, ego_density_bonus, naturalness_penalty,
        domain, primary, supporting
    """
    decision      = token_result.get("decision", "NATIVE")
    features      = token_result.get("features", {})
    score         = token_result.get("laps_score")
    thresh        = token_result.get("threshold")
    probability   = token_result.get("probability")
    is_borderline = token_result.get("is_borderline", False)
    ego_bonus     = token_result.get("ego_density_bonus", 0.0)
    nat_penalty   = token_result.get("naturalness_penalty", 0.0)
    domain        = token_result.get("domain", "General")

    if decision == "NATIVE":
        return {
            "token":              token_result["token"],
            "decision":           "NATIVE",
            "score":              None,
            "threshold":          None,
            "probability":        None,
            "is_borderline":      False,
            "ego_density_bonus":  0.0,
            "naturalness_penalty": 0.0,
            "domain":             "N/A",
            "primary":            "Token identified as native Hindi — not subject to LAPS scoring.",
            "supporting":         [],
        }

    # Determine primary reason based on decision type
    if decision == "UNCERTAIN":
        primary = (
            f"Borderline token: P(Preserve)={probability:.3f} falls between "
            f"0.50 and 0.70 for {token_result.get('register','business')} register. "
            "Retained to avoid over-translation."
        ) if probability is not None else "Near-boundary score; decision deferred to preservation."
    else:
        primary = _primary_reason(features, decision)

    return {
        "token":              token_result["token"],
        "decision":           decision,
        "score":              score,
        "threshold":          thresh,
        "probability":        probability,
        "is_borderline":      is_borderline,
        "ego_density_bonus":  ego_bonus,
        "naturalness_penalty": nat_penalty,
        "domain":             domain,
        "primary":            primary,
        "supporting":         _supporting_features(features, decision),
    }


def explain_all(scored_tokens: list[dict]) -> list[dict]:
    """Return explanations for every token in the sentence."""
    return [explain_token(t) for t in scored_tokens]



def format_explanation_table(explanations: list[dict]) -> str:
    """Return a plain-text table for CLI / report display."""
    lines = [
        f"{'Token':<18} {'Decision':<10} {'Score':>6}  {'Primary Reason'}",
        "-" * 80,
    ]
    for ex in explanations:
        score_str = f"{ex['score']:.3f}" if ex["score"] is not None else "  N/A"
        lines.append(
            f"{ex['token']:<18} {ex['decision']:<10} {score_str:>6}  {ex['primary']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sample = [
        {"token": "Kal",      "decision": "NATIVE",   "laps_score": None,  "threshold": None,  "features": {}},
        {"token": "client",   "decision": "PRESERVE",  "laps_score": 0.74,  "threshold": 0.50,  "features": {"BF": 0.92, "DS": 0.9, "SI": 0.3, "CC": 0.67, "AF": 0.9, "TB": 0.2}},
        {"token": "meeting",  "decision": "PRESERVE",  "laps_score": 0.71,  "threshold": 0.50,  "features": {"BF": 0.95, "DS": 0.9, "SI": 0.3, "CC": 0.67, "AF": 0.9, "TB": 0.2}},
        {"token": "postpone", "decision": "REWRITE",   "laps_score": 0.38,  "threshold": 0.50,  "features": {"BF": 0.85, "DS": 0.1, "SI": 0.3, "CC": 0.33, "AF": 0.4, "TB": 0.85}},
    ]
    exps = explain_all(sample)
    print(format_explanation_table(exps))
