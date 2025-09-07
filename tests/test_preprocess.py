"""Tests for the preprocess module."""

import numpy as np
import pytest
from PIL import Image
import tempfile
import os
from unittest.mock import patch, Mock

from ai_image_detector.preprocess import (
    PreprocessConfig, ImageLoadError, load_image, 
    pil_to_cv2, cv2_to_pil, center_resize_pad, 
    normalize_tensor, preprocess_image
)


class TestPreprocessConfig:
    """Test the PreprocessConfig dataclass."""
    
    def test_preprocess_config_defaults(self):
        """Test default values for PreprocessConfig."""
        config = PreprocessConfig()
        assert config.target_size == (224, 224)
        assert config.normalize_mean == (0.5, 0.5, 0.5)
        assert config.normalize_std == (0.5, 0.5, 0.5)
        assert config.to_rgb is True
    
    def test_preprocess_config_custom_values(self):
        """Test custom values for PreprocessConfig."""
        config = PreprocessConfig(
            target_size=(256, 256),
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225),
            to_rgb=False
        )
        assert config.target_size == (256, 256)
        assert config.normalize_mean == (0.485, 0.456, 0.406)
        assert config.normalize_std == (0.229, 0.224, 0.225)
        assert config.to_rgb is False


class TestImageLoadError:
    """Test the ImageLoadError exception."""
    
    def test_image_load_error_creation(self):
        """Test creating an ImageLoadError."""
        error = ImageLoadError("Test error message")
        assert str(error) == "Test error message"


class TestLoadImage:
    """Test the load_image function."""
    
    def test_load_image_file_not_found(self):
        """Test load_image with non-existent file."""
        with pytest.raises(ImageLoadError, match="File not found"):
            load_image("nonexistent_file.jpg")
    
    def test_load_image_valid_file(self, sample_image_file):
        """Test load_image with valid image file."""
        img = load_image(sample_image_file)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
    
    def test_load_image_invalid_file(self):
        """Test load_image with invalid image file."""
        # Create a temporary file with invalid image data
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"invalid image data")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ImageLoadError, match="Invalid image file"):
                load_image(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_load_image_converts_to_rgb(self, sample_image):
        """Test that load_image converts image to RGB."""
        # Create a grayscale image
        gray_img = sample_image.convert("L")
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            gray_img.save(tmp.name, 'PNG')
            tmp_path = tmp.name
        
        try:
            img = load_image(tmp_path)
            assert img.mode == "RGB"
        finally:
            os.unlink(tmp_path)


class TestPilToCv2:
    """Test the pil_to_cv2 function."""
    
    def test_pil_to_cv2_with_cv2(self, sample_image):
        """Test pil_to_cv2 when cv2 is available."""
        with patch('ai_image_detector.preprocess._HAS_CV2', True):
            with patch('ai_image_detector.preprocess.cv2') as mock_cv2:
                mock_cv2.cvtColor.return_value = np.array(sample_image)[..., ::-1]
                
                result = pil_to_cv2(sample_image)
                
                assert isinstance(result, np.ndarray)
                mock_cv2.cvtColor.assert_called_once()
    
    def test_pil_to_cv2_without_cv2(self, sample_image):
        """Test pil_to_cv2 when cv2 is not available."""
        with patch('ai_image_detector.preprocess._HAS_CV2', False):
            result = pil_to_cv2(sample_image)
            
            assert isinstance(result, np.ndarray)
            # Should flip channels manually
            expected = np.array(sample_image)[..., ::-1]
            np.testing.assert_array_equal(result, expected)


class TestCv2ToPil:
    """Test the cv2_to_pil function."""
    
    def test_cv2_to_pil_with_cv2(self):
        """Test cv2_to_pil when cv2 is available."""
        bgr_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with patch('ai_image_detector.preprocess._HAS_CV2', True):
            with patch('ai_image_detector.preprocess.cv2') as mock_cv2:
                mock_cv2.cvtColor.return_value = bgr_array[..., ::-1]
                
                result = cv2_to_pil(bgr_array)
                
                assert isinstance(result, Image.Image)
                mock_cv2.cvtColor.assert_called_once()
    
    def test_cv2_to_pil_without_cv2(self):
        """Test cv2_to_pil when cv2 is not available."""
        bgr_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with patch('ai_image_detector.preprocess._HAS_CV2', False):
            result = cv2_to_pil(bgr_array)
            
            assert isinstance(result, Image.Image)
            # Should flip channels manually
            expected_rgb = bgr_array[..., ::-1]
            np.testing.assert_array_equal(np.array(result), expected_rgb)


class TestCenterResizePad:
    """Test the center_resize_pad function."""
    
    def test_center_resize_pad_square_image(self):
        """Test center_resize_pad with square image."""
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        result = center_resize_pad(img, (224, 224))
        
        assert result.size == (224, 224)
        assert isinstance(result, Image.Image)
    
    def test_center_resize_pad_rectangular_image(self):
        """Test center_resize_pad with rectangular image."""
        img = Image.new("RGB", (200, 100), (0, 255, 0))
        result = center_resize_pad(img, (224, 224))
        
        assert result.size == (224, 224)
        assert isinstance(result, Image.Image)
    
    def test_center_resize_pad_larger_target(self):
        """Test center_resize_pad when target is larger than image."""
        img = Image.new("RGB", (50, 50), (0, 0, 255))
        result = center_resize_pad(img, (224, 224))
        
        assert result.size == (224, 224)
        assert isinstance(result, Image.Image)
    
    def test_center_resize_pad_with_cv2(self):
        """Test center_resize_pad using cv2 when available."""
        img = Image.new("RGB", (100, 100), (255, 255, 0))
        
        with patch('ai_image_detector.preprocess._HAS_CV2', True):
            with patch('ai_image_detector.preprocess.cv2') as mock_cv2:
                mock_cv2.resize.return_value = np.array(img.resize((224, 224)))
                
                result = center_resize_pad(img, (224, 224))
                
                assert result.size == (224, 224)
                mock_cv2.resize.assert_called_once()
    
    def test_center_resize_pad_without_cv2(self):
        """Test center_resize_pad without cv2."""
        img = Image.new("RGB", (100, 100), (255, 0, 255))
        
        with patch('ai_image_detector.preprocess._HAS_CV2', False):
            result = center_resize_pad(img, (224, 224))
            
            assert result.size == (224, 224)
            assert isinstance(result, Image.Image)


class TestNormalizeTensor:
    """Test the normalize_tensor function."""
    
    def test_normalize_tensor_basic(self):
        """Test basic tensor normalization."""
        # Create a test image array (HWC format)
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = normalize_tensor(img_array, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        assert result.shape == (3, 100, 100)  # Should be CHW
        assert result.dtype == np.float32
        # Values should be normalized
        assert np.all(result >= -1.0) and np.all(result <= 1.0)
    
    def test_normalize_tensor_custom_mean_std(self):
        """Test tensor normalization with custom mean and std."""
        img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        result = normalize_tensor(img_array, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        assert result.shape == (3, 50, 50)
        assert result.dtype == np.float32
    
    def test_normalize_tensor_zero_std(self):
        """Test tensor normalization with zero std (should not crash)."""
        img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        # This should not crash, though mathematically questionable
        result = normalize_tensor(img_array, (0.5, 0.5, 0.5), (0.0, 0.0, 0.0))
        
        assert result.shape == (3, 10, 10)
        assert result.dtype == np.float32


class TestPreprocessImage:
    """Test the preprocess_image function."""
    
    def test_preprocess_image_default_config(self, sample_image):
        """Test preprocess_image with default config."""
        result = preprocess_image(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 224, 224)  # CHW format
        assert result.dtype == np.float32
    
    def test_preprocess_image_custom_config(self, sample_image):
        """Test preprocess_image with custom config."""
        config = PreprocessConfig(
            target_size=(256, 256),
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225)
        )
        
        result = preprocess_image(sample_image, config)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 256, 256)
        assert result.dtype == np.float32
    
    def test_preprocess_image_no_rgb_conversion(self, sample_image):
        """Test preprocess_image without RGB conversion."""
        config = PreprocessConfig(to_rgb=False)
        
        result = preprocess_image(sample_image, config)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 224, 224)
    
    def test_preprocess_image_grayscale_input(self):
        """Test preprocess_image with grayscale input."""
        gray_img = Image.new("L", (100, 100), 128)
        config = PreprocessConfig(to_rgb=True)
        
        result = preprocess_image(gray_img, config)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 224, 224)  # Should be converted to RGB
    
    def test_preprocess_image_none_config(self, sample_image):
        """Test preprocess_image with None config (should use defaults)."""
        result = preprocess_image(sample_image, None)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32
