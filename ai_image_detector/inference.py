from __future__ import annotations

from typing import Dict

import numpy as np

from .models import build_detector, Prediction


def run_inference_on_array(img_chw: np.ndarray, *, use_ensemble: bool = True, model_name_or_path: str | None = None, ensemble_weights: tuple[float, float] | None = None) -> Prediction:
    detector = build_detector(prefer_torch=True, use_ensemble=use_ensemble, model_name_or_path=model_name_or_path, ensemble_weights=ensemble_weights)
    return detector.predict(img_chw)


def format_prediction(pred: Prediction) -> str:
    label_nice = "AI-generated" if pred.label.lower().startswith("ai") else "Real"
    confidence_pct = int(round(pred.confidence * 100))
    return f"{label_nice} with {confidence_pct}% confidence"


def prediction_to_dict(pred: Prediction) -> Dict[str, object]:
    return {
        "label": pred.label,
        "confidence": pred.confidence,
        "scores": pred.raw_scores or {},
    }


