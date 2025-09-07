from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def tta_variants(chw: np.ndarray) -> List[np.ndarray]:
    return [
        chw.copy(),  # original (as copy for immutability)
        np.ascontiguousarray(chw[:, :, ::-1]),  # hflip
        np.ascontiguousarray(chw[:, ::-1, :]),  # vflip
        np.ascontiguousarray(np.transpose(chw, (0, 2, 1))),  # transpose
    ]


def tta_average(chw: np.ndarray, predict_scores: Callable[[np.ndarray], Dict[str, float]]) -> Dict[str, float]:
    outputs = [predict_scores(v) for v in tta_variants(chw)]
    # Only include keys that are present in ALL outputs
    common_keys = set(outputs[0].keys())
    for output in outputs[1:]:
        common_keys &= set(output.keys())
    return {k: float(np.mean([o[k] for o in outputs])) for k in common_keys}


