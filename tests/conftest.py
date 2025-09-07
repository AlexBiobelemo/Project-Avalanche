"""Pytest configuration and shared fixtures for AI Image Detector tests."""

import numpy as np
import pytest
from PIL import Image
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with some pattern
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_file(sample_image):
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        sample_image.save(tmp.name, 'JPEG')
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_image_array():
    """Create a sample numpy array in CHW format for testing."""
    return np.random.rand(3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_prediction():
    """Create a sample Prediction object for testing."""
    from ai_image_detector.models import Prediction
    return Prediction(
        label="ai-generated",
        confidence=0.85,
        raw_scores={"real": 0.15, "ai-generated": 0.85}
    )


@pytest.fixture
def mock_torch_available(monkeypatch):
    """Mock torch availability for testing."""
    def mock_import(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError("torch not available")
        return __import__(name, *args, **kwargs)
    
    monkeypatch.setattr('builtins.__import__', mock_import)


@pytest.fixture
def mock_transformers_available(monkeypatch):
    """Mock transformers availability for testing."""
    def mock_import(name, *args, **kwargs):
        if name == 'transformers':
            raise ImportError("transformers not available")
        return __import__(name, *args, **kwargs)
    
    monkeypatch.setattr('builtins.__import__', mock_import)
