from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st  # type: ignore
from PIL import Image

try:
    from ai_image_detector.preprocess import preprocess_image, PreprocessConfig
    from ai_image_detector.models import HeuristicFrequencyDetector, TorchViTDetector, BaseDetector
    from ai_image_detector.inference import format_prediction
except ModuleNotFoundError:
    # Fallback for running via `streamlit run ai_image_detector/gui_streamlit.py` without installation
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ai_image_detector.preprocess import preprocess_image, PreprocessConfig
    from ai_image_detector.models import HeuristicFrequencyDetector, TorchViTDetector, BaseDetector
    from ai_image_detector.inference import format_prediction


def _to_pil(uploaded_file) -> Optional[Image.Image]:
    try:
        image_bytes = uploaded_file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _build_detector_cached(backend: str, model_name_or_path: str | None) -> BaseDetector:
    if backend == "Auto":
        # Try Torch ViT, else heuristic
        vit = TorchViTDetector(model_name=model_name_or_path or "google/vit-base-patch16-224")
        if getattr(vit, "available", False):
            return vit
        return HeuristicFrequencyDetector()
    if backend == "Torch ViT":
        vit = TorchViTDetector(model_name=model_name_or_path or "google/vit-base-patch16-224")
        if not getattr(vit, "available", False):
            raise RuntimeError("TorchViTDetector not available. Install torch/transformers/timm and local weights.")
        return vit
    if backend == "Ensemble":
        from ai_image_detector.models import EnsembleDetector
        vit = TorchViTDetector(model_name=model_name_or_path or "google/vit-base-patch16-224")
        if getattr(vit, "available", False):
            return EnsembleDetector([vit, HeuristicFrequencyDetector()])
        return HeuristicFrequencyDetector()
    # Heuristic
    return HeuristicFrequencyDetector()


@st.cache_data(show_spinner=False)
def _preprocess_cached(image_bytes: bytes, size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cfg = PreprocessConfig(target_size=(size, size), normalize_mean=mean, normalize_std=std)
    return preprocess_image(img, cfg)


def _predict(detector: BaseDetector, arr_chw: np.ndarray):
    return detector.predict(arr_chw)


def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # type: ignore


def main() -> None:
    st.set_page_config(page_title="AI Image Detector", layout="wide")
    _init_state()

    st.title("AI Image Detector")
    st.caption("Classify images as Real or AI-generated. Uses local models if available, otherwise a fast heuristic.")

    with st.sidebar:
        st.header("Settings")
        backend = st.selectbox("Backend", ["Auto", "Heuristic", "Torch ViT", "Ensemble"], help="Choose detector backend")
        model_name_or_path = st.text_input("Model name or local path (optional)", value="")
        size = st.slider("Input size", min_value=128, max_value=384, value=224, step=16)
        threshold = st.slider("Decision threshold (AI)", min_value=0.1, max_value=0.9, value=0.5, step=0.01)
        tta_enabled = st.checkbox("Enable TTA", value=True)
        if backend == "Ensemble":
            w_h = st.slider("Heuristic weight", 0.0, 1.0, 0.4, 0.05)
            w_v = 1.0 - w_h
        else:
            w_h, w_v = 0.0, 1.0
        st.caption("AI if AI-score ≥ threshold")

        with st.expander("Normalization", expanded=False):
            mean = st.text_input("Mean (R,G,B)", value="0.5,0.5,0.5")
            std = st.text_input("Std (R,G,B)", value="0.5,0.5,0.5")
        try:
            mean_tuple = tuple(float(x.strip()) for x in mean.split(","))  # type: ignore
            std_tuple = tuple(float(x.strip()) for x in std.split(","))  # type: ignore
            if len(mean_tuple) != 3 or len(std_tuple) != 3:
                raise ValueError
        except Exception:
            st.warning("Invalid mean/std format, using defaults (0.5,0.5,0.5)")
            mean_tuple = (0.5, 0.5, 0.5)
            std_tuple = (0.5, 0.5, 0.5)

        build_btn = st.button("(Re)load Model", use_container_width=True)

    # Build or rebuild detector
    detector_error = None
    if build_btn:
        _build_detector_cached.clear()
    try:
        detector = _build_detector_cached(backend, model_name_or_path or None)
    except Exception as e:
        detector_error = str(e)
        detector = HeuristicFrequencyDetector()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded_files = st.file_uploader(
            "Upload image(s)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        run_btn = st.button("Classify", type="primary")

        if detector_error:
            st.error(detector_error)

        if run_btn and uploaded_files:
            results_batch = []
            progress = st.progress(0, text="Processing...")
            for idx, f in enumerate(uploaded_files):
                progress.progress((idx + 1) / len(uploaded_files), text=f"Processing {f.name}")
                try:
                    bytes_data = f.getvalue()
                    arr = _preprocess_cached(bytes_data, size, mean_tuple, std_tuple)
                    try:
                        pred = detector.predict_with_tta(arr) if tta_enabled else _predict(detector, arr)
                    except Exception:
                        pred = _predict(detector, arr)
                    label_nice = "AI-generated" if pred.label.lower().startswith("ai") else "Real"
                    ai_score = float(pred.raw_scores.get("ai-generated", 1 - pred.confidence) if pred.raw_scores else 1 - pred.confidence)
                    # Apply threshold override and ensemble weighting visualization
                    if backend == "Ensemble" and pred.raw_scores:
                        # Recompute weighted score for display
                        real = float(pred.raw_scores.get("real", 1 - ai_score))
                        ai_score = float(w_v * ai_score + w_h * (1 - real))  # heuristic weight encourages AI if real is low
                        label_nice = "AI-generated" if ai_score >= threshold else "Real"
                    results_batch.append({
                        "name": f.name,
                        "bytes": bytes_data,
                        "pred": pred,
                        "label_nice": label_nice,
                        "ai_score": ai_score,
                    })
                except Exception as e:
                    results_batch.append({
                        "name": f.name,
                        "bytes": None,
                        "error": str(e),
                    })

            st.session_state.history.extend(results_batch)
            progress.empty()

            # Offer CSV export
            try:
                import pandas as pd  # type: ignore
                rows = []
                for r in results_batch:
                    if "error" in r:
                        continue
                    row = {
                        "name": r["name"],
                        "label": r["label_nice"],
                        "confidence": r["pred"].confidence if "pred" in r else None,
                    }
                    if "pred" in r and getattr(r["pred"], "raw_scores", None):
                        row.update(r["pred"].raw_scores or {})
                    rows.append(row)
                df = pd.DataFrame(rows)
                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")
            except Exception:
                pass

        # Display latest results
        if st.session_state.history:
            st.subheader("Results")
            for item in st.session_state.history[::-1][:20]:  # show last 20
                if "error" in item:
                    st.error(f"{item['name']}: {item['error']}")
                    continue
                img = Image.open(io.BytesIO(item["bytes"])) if item["bytes"] else None
                cols = st.columns([1, 1.5])
                with cols[0]:
                    if img is not None:
                        st.image(img, caption=item["name"], use_container_width=True)
                with cols[1]:
                    pred = item["pred"]
                    st.markdown(f"**{item['name']}** — {format_prediction(pred)}")
                    # Confidence/AI-score bars
                    if pred.raw_scores:
                        real_score = float(pred.raw_scores.get("real", 1.0 - item["ai_score"]))
                        ai_score = float(pred.raw_scores.get("ai-generated", item["ai_score"]))
                        st.write("AI score")
                        st.progress(min(max(ai_score, 0.0), 1.0))
                        st.write("Real score")
                        st.progress(min(max(real_score, 0.0), 1.0))
                    with st.expander("Raw scores"):
                        st.json(pred.raw_scores)

    with col_right:
        st.subheader("History")
        if st.session_state.history:
            for idx, item in enumerate(st.session_state.history[::-1][:50]):
                if "error" in item:
                    st.write(f"{item['name']}: error")
                    continue
                label = item["label_nice"]
                conf = int(round(item["pred"].confidence * 100))
                st.write(f"{item['name']}: {label} ({conf}%)")
        clear = st.button("Clear History")
        if clear:
            st.session_state.history = []


if __name__ == "__main__":
    main()


