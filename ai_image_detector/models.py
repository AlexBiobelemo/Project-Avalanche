from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# Make tta functions available from this module for tests
from .tta import tta_variants, tta_average


@dataclass
class Prediction:
    label: str
    confidence: float
    raw_scores: Optional[Dict[str, float]] = None


class BaseDetector:
    labels: Tuple[str, str] = ("real", "ai-generated")

    def predict(self, batch_chw: np.ndarray) -> Prediction:
        raise NotImplementedError

    def predict_with_tta(self, batch_chw: np.ndarray) -> Prediction:
        from .tta import tta_variants

        scores: List[Dict[str, float]] = []
        variants = tta_variants(batch_chw)
        for arr in variants:
            p = self.predict(arr)
            if p.raw_scores is not None:
                scores.append(p.raw_scores)
        
        if not scores:
            return self.predict(batch_chw)
        
        # Calculate averages using list comprehensions to avoid StopIteration
        real_scores = [s.get("real", 0.0) for s in scores]
        ai_scores = [s.get("ai-generated", 0.0) for s in scores]
        
        real = float(np.mean(real_scores)) if real_scores else 0.0
        ai = float(np.mean(ai_scores)) if ai_scores else 0.0
        
        label = self.labels[1] if ai >= real else self.labels[0]
        confidence = max(ai, real)
        return Prediction(label=label, confidence=confidence, raw_scores={"real": real, "ai-generated": ai})


class HeuristicFrequencyDetector(BaseDetector):
    """
    Lightweight baseline using frequency-domain artifacts and color statistics.
    Returns higher AI score for images with unusually strong high-frequency energy
    and reduced color channel variance.
    """

    def __init__(self) -> None:
        super().__init__()

    def predict(self, batch_chw: np.ndarray) -> Prediction:
        # batch_chw: [C, H, W] normalized to ~[-1, 1]
        # Ensure input is float32 and contiguous for deterministic behavior
        chw = np.ascontiguousarray(batch_chw.astype(np.float32))
        chw = np.clip((chw * 0.5 + 0.5), 0, 1)  # back to [0,1]
        hwc = np.transpose(chw, (1, 2, 0))

        # Use consistent dtype and ensure deterministic calculation
        gray = np.dot(hwc[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

        # Use more stable epsilon and ensure float64 precision for calculations
        eps = np.float64(1e-8)
        low_energy = np.float64(magnitude[mask].mean()) + eps
        high_energy = np.float64(magnitude[~mask].mean()) + eps
        high_ratio = high_energy / (low_energy + eps)

        # Color variance penalty (some generators have smoother color distributions)
        # Use ddof=0 for consistency and float64 precision
        color_var = np.float64(hwc.reshape(-1, 3).var(axis=0, ddof=0).mean())

        # Combine signals with float64 precision to avoid rounding errors
        freq_signal = sigmoid(np.float64(2.0) * (high_ratio - np.float64(1.0)))
        color_signal = sigmoid(np.float64(1.5) * (np.float64(0.12) - color_var))
        ai_score = freq_signal * np.float64(0.7) + color_signal * np.float64(0.3)
        ai_score = float(np.clip(ai_score, 0.0, 1.0))
        real_score = 1.0 - ai_score

        label = self.labels[1] if ai_score >= 0.5 else self.labels[0]
        confidence = max(ai_score, real_score)
        return Prediction(label=label, confidence=confidence, raw_scores={"real": real_score, "ai-generated": ai_score})


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TorchViTDetector(BaseDetector):
    """
    Optional: loads a ViT model via transformers or timm if installed and available locally.
    Uses local_files_only to avoid internet download. Expects binary classification head.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224", local_only: bool = True) -> None:
        self.available = False
        self.model = None
        self.processor = None
        self.labels = ("real", "ai-generated")
        self._init_transformers(model_name, local_only)
        if not self.available:
            self._init_timm(local_only)

    def _init_transformers(self, model_name: str, local_only: bool) -> None:
        try:
            import torch  # noqa
            from transformers import AutoImageProcessor, AutoModelForImageClassification  # type: ignore

            self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_only)
            self.model = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=local_only)
            if hasattr(self.model.config, "id2label") and len(self.model.config.id2label) == 2:
                self.labels = tuple(self.model.config.id2label[i].lower() for i in range(2))  # type: ignore
            self.available = True
        except (ImportError, OSError, Exception):
            # ImportError: transformers not available
            # OSError: model not found locally
            # Exception: other initialization errors
            self.available = False

    def _init_timm(self, local_only: bool) -> None:
        if self.available:
            return
        try:
            import torch  # noqa
            import timm  # type: ignore
            self.model = timm.create_model("vit_base_patch16_224", pretrained=not local_only, num_classes=2)
            self.model.eval()
            self.processor = None
            self.available = True
        except Exception:
            self.available = False

    def predict(self, batch_chw: np.ndarray) -> Prediction:
        if not self.available or self.model is None:
            raise RuntimeError("TorchViTDetector not available")

        try:
            # Convert CHW normalized [-1,1] to tensor
            import torch  # type: ignore

            # Check input size and raise informative error if needed
            c, h, w = batch_chw.shape
            if h != 224 or w != 224:
                raise ValueError(f"Input height ({h}) doesn't match model (224).")

            tensor = torch.from_numpy(batch_chw).float().unsqueeze(0)
            with torch.no_grad():
                if self.processor is not None:
                    # transformers expects PIL; but we already preprocessed; reverse normalization to feed raw
                    # Instead, pass directly to model if it supports tensors
                    outputs = self.model(pixel_values=tensor)
                else:
                    outputs = self.model(tensor)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Map to labels
            if len(self.labels) == 2:
                real_score = float(probs[0])
                ai_score = float(probs[1])
            else:
                real_score = float(probs[0])
                ai_score = 1.0 - real_score

            label = self.labels[1] if ai_score >= real_score else self.labels[0]
            confidence = max(ai_score, real_score)
            return Prediction(label=label, confidence=confidence, raw_scores={self.labels[0]: real_score, self.labels[1]: ai_score})
        
        except Exception as e:
            # If TorchViT fails for any reason, fall back to heuristic
            return HeuristicFrequencyDetector().predict(batch_chw)


class EnsembleDetector(BaseDetector):
    def __init__(self, detectors: List[BaseDetector], weights: Optional[Tuple[float, float]] = None) -> None:
        # Assume consistent labels order across detectors
        self.detectors = detectors
        self.labels = ("real", "ai-generated")
        self.weights = weights or (0.4, 0.6)

    def predict(self, batch_chw: np.ndarray) -> Prediction:
        scores: List[Dict[str, float]] = []
        for det in self.detectors:
            try:
                p = det.predict_with_tta(batch_chw)
                if p.raw_scores:
                    scores.append(p.raw_scores)
            except Exception:
                continue
        if not scores:
            # fallback heuristic
            return HeuristicFrequencyDetector().predict_with_tta(batch_chw)
        from .ensemble import weighted_average
        averaged = weighted_average(scores, self.weights[: len(scores)])
        real = float(averaged.get("real", 0.0))
        ai = float(averaged.get("ai-generated", 0.0))
        label = self.labels[1] if ai >= real else self.labels[0]
        confidence = max(ai, real)
        return Prediction(label=label, confidence=confidence, raw_scores={"real": real, "ai-generated": ai})


def build_detector(prefer_torch: bool = True, use_ensemble: bool = True, model_name_or_path: Optional[str] = None, ensemble_weights: Optional[Tuple[float, float]] = None) -> BaseDetector:
    if prefer_torch:
        try:
            vit = TorchViTDetector(model_name=model_name_or_path or "google/vit-base-patch16-224")
            if getattr(vit, "available", False):
                if use_ensemble:
                    return EnsembleDetector([HeuristicFrequencyDetector(), vit], weights=ensemble_weights)
                return vit
        except Exception:
            # If TorchViT initialization fails, fall back to heuristic
            pass
    return HeuristicFrequencyDetector()


