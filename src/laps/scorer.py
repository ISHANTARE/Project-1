"""
LAPS Scorer
-----------
Computes the weighted Preservation Score for each token using the formula:

  Score(t) = w1*BF + w2*DS + w3*SI + w4*CC + w5*AF - w6*TB

Default weights are equal (0.2 each).  They can be overridden via a
YAML/JSON config or passed directly.
"""

DEFAULT_WEIGHTS = {
    "w1": 0.20,   # Borrowing Frequency
    "w2": 0.20,   # Domain Specificity
    "w3": 0.20,   # Semantic Importance
    "w4": 0.15,   # Context Consistency
    "w5": 0.15,   # Audience Familiarity
    "w6": 0.10,   # Translation Benefit  (subtracted)
}


def compute_score(features: dict, weights: dict | None = None) -> float:
    """
    Compute the LAPS Preservation Score.

    Args:
        features : dict with keys BF, DS, SI, CC, AF, TB
        weights  : optional weight override dict (keys w1–w6)

    Returns:
        float  – Preservation Score in approximate range [0, 1]
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    score = (
        w["w1"] * features["BF"]
        + w["w2"] * features["DS"]
        + w["w3"] * features["SI"]
        + w["w4"] * features["CC"]
        + w["w5"] * features["AF"]
        - w["w6"] * features["TB"]
    )
    # Clamp to [0, 1]
    return round(max(0.0, min(1.0, score)), 4)


def score_all_tokens(tagged_tokens: list[dict], features_map: dict,
                     weights: dict | None = None) -> list[dict]:
    """
    Attach LAPS scores to every candidate token in the sentence.

    Args:
        tagged_tokens : output of tokenizer.tokenize()
        features_map  : {token_index: feature_dict}
        weights       : optional weight override

    Returns:
        list of dicts – each tagged_token enriched with 'features' and 'laps_score'
    """
    results = []
    for i, tok in enumerate(tagged_tokens):
        entry = dict(tok)
        if tok["is_candidate"] and i in features_map:
            feats = features_map[i]
            entry["features"]   = feats
            entry["laps_score"] = compute_score(feats, weights)
        else:
            entry["features"]   = {}
            entry["laps_score"] = None    # non-candidate tokens are not scored
        results.append(entry)
    return results


if __name__ == "__main__":
    # Quick smoke-test
    feats = {"BF": 0.92, "DS": 0.9, "SI": 0.3, "CC": 0.67, "AF": 0.9, "TB": 0.85}
    print("Score for 'meeting':", compute_score(feats))
