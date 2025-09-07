from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False
import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    to_rgb: bool = True


class ImageLoadError(Exception):
    pass


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise ImageLoadError(f"File not found: {path}")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(f"Invalid image file: {path}") from exc


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if _HAS_CV2:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    # Fallback: RGB -> BGR by channel flip
    return arr[..., ::-1].copy()


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    if _HAS_CV2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_bgr[..., ::-1]
    return Image.fromarray(rgb)


def center_resize_pad(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if _HAS_CV2:
        # Use OpenCV for potentially faster resize
        arr = np.array(img)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        resized = Image.fromarray(arr)
    else:
        resized = img.resize((new_w, new_h), Image.BICUBIC)
    new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    new_img.paste(resized, (left, top))
    return new_img


def normalize_tensor(img_np: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    img_np = img_np.astype(np.float32) / 255.0
    mean_array = np.array(mean, dtype=np.float32)[None, None, :]
    std_array = np.array(std, dtype=np.float32)[None, None, :]
    
    # Avoid division by zero by replacing zero std values with 1.0
    # This effectively disables normalization for channels with std=0
    std_array = np.where(std_array == 0.0, 1.0, std_array)
    
    img_np = (img_np - mean_array) / std_array
    # HWC -> CHW
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np


def preprocess_image(img: Image.Image, cfg: Optional[PreprocessConfig] = None) -> np.ndarray:
    cfg = cfg or PreprocessConfig()
    if cfg.to_rgb:
        img = img.convert("RGB")
    img = center_resize_pad(img, cfg.target_size)
    arr = np.array(img)
    arr = normalize_tensor(arr, cfg.normalize_mean, cfg.normalize_std)
    return arr


