from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    size: int = 224
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    threshold: float = 0.5
    ensemble_weights: Tuple[float, float] = (0.4, 0.6)  # (heuristic, vit)


