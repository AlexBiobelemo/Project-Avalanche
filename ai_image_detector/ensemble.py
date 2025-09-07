from __future__ import annotations

from typing import Dict, List, Tuple


def weighted_average(scores: List[Dict[str, float]], weights: Tuple[float, ...]) -> Dict[str, float]:
    if not scores:
        return {"real": 0.0, "ai-generated": 1.0}
    
    wsum = float(sum(weights))
    if wsum == 0.0:
        return {"real": 0.0, "ai-generated": 1.0}
    
    # Only use standard prediction keys to avoid propagating extra keys
    standard_keys = ["real", "ai-generated"]
    keys = [k for k in standard_keys if k in scores[0]]
    
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = sum(s.get(k, 0.0) * w for s, w in zip(scores, weights)) / wsum
    return out


