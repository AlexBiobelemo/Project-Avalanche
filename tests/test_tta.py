"""Tests for the TTA (Test Time Augmentation) module."""

import numpy as np
import pytest
from unittest.mock import Mock

from ai_image_detector.tta import tta_variants, tta_average


class TestTtaVariants:
    """Test the tta_variants function."""
    
    def test_tta_variants_basic(self, sample_image_array):
        """Test basic TTA variants generation."""
        variants = tta_variants(sample_image_array)
        
        assert len(variants) == 4
        assert all(isinstance(v, np.ndarray) for v in variants)
        assert all(v.shape == sample_image_array.shape for v in variants)
    
    def test_tta_variants_original(self, sample_image_array):
        """Test that first variant is the original image."""
        variants = tta_variants(sample_image_array)
        
        np.testing.assert_array_equal(variants[0], sample_image_array)
    
    def test_tta_variants_horizontal_flip(self, sample_image_array):
        """Test horizontal flip variant."""
        variants = tta_variants(sample_image_array)
        
        # Horizontal flip should flip the width dimension (last dimension)
        expected_hflip = sample_image_array[:, :, ::-1]
        np.testing.assert_array_equal(variants[1], expected_hflip)
    
    def test_tta_variants_vertical_flip(self, sample_image_array):
        """Test vertical flip variant."""
        variants = tta_variants(sample_image_array)
        
        # Vertical flip should flip the height dimension (middle dimension)
        expected_vflip = sample_image_array[:, ::-1, :]
        np.testing.assert_array_equal(variants[2], expected_vflip)
    
    def test_tta_variants_transpose(self, sample_image_array):
        """Test transpose variant."""
        variants = tta_variants(sample_image_array)
        
        # Transpose should swap height and width dimensions
        expected_transpose = np.transpose(sample_image_array, (0, 2, 1))
        np.testing.assert_array_equal(variants[3], expected_transpose)
    
    def test_tta_variants_different_shapes(self):
        """Test TTA variants with different input shapes."""
        # Test with different dimensions
        shapes = [(3, 100, 100), (3, 256, 256), (3, 64, 128)]
        
        for shape in shapes:
            img_array = np.random.rand(*shape).astype(np.float32)
            variants = tta_variants(img_array)
            
            assert len(variants) == 4
            # First three variants keep same shape
            assert all(v.shape == shape for v in variants[:3])
            # Transpose swaps height and width
            expected_transpose_shape = (shape[0], shape[2], shape[1])
            assert variants[3].shape == expected_transpose_shape
    
    def test_tta_variants_data_type_preservation(self):
        """Test that TTA variants preserve data type."""
        img_array = np.random.rand(3, 224, 224).astype(np.float32)
        variants = tta_variants(img_array)
        
        assert all(v.dtype == np.float32 for v in variants)
    
    def test_tta_variants_contiguous_arrays(self):
        """Test that TTA variants return contiguous arrays."""
        img_array = np.random.rand(3, 224, 224).astype(np.float32)
        variants = tta_variants(img_array)
        
        # All variants should be contiguous (except possibly the transpose)
        assert variants[0].flags.c_contiguous
        assert variants[1].flags.c_contiguous
        assert variants[2].flags.c_contiguous
        # Transpose might not be contiguous, but should be explicitly made contiguous
        assert variants[3].flags.c_contiguous
    
    def test_tta_variants_immutability(self, sample_image_array):
        """Test that TTA variants don't modify the original array."""
        original = sample_image_array.copy()
        variants = tta_variants(sample_image_array)
        
        # Original should be unchanged after creating variants
        np.testing.assert_array_equal(sample_image_array, original)
        
        # Modifying variants shouldn't affect original
        variants[0][0, 0, 0] = 999.0
        np.testing.assert_array_equal(sample_image_array, original)


class TestTtaAverage:
    """Test the tta_average function."""
    
    def test_tta_average_basic(self, sample_image_array):
        """Test basic TTA averaging functionality."""
        # Mock predict_scores function
        def mock_predict_scores(arr):
            # Return different scores based on array identity
            if np.array_equal(arr, sample_image_array):
                return {"real": 0.8, "ai-generated": 0.2}
            elif np.array_equal(arr, sample_image_array[:, :, ::-1]):  # hflip
                return {"real": 0.7, "ai-generated": 0.3}
            elif np.array_equal(arr, sample_image_array[:, ::-1, :]):  # vflip
                return {"real": 0.6, "ai-generated": 0.4}
            else:  # transpose
                return {"real": 0.5, "ai-generated": 0.5}
        
        result = tta_average(sample_image_array, mock_predict_scores)
        
        # Should average all four predictions
        expected_real = (0.8 + 0.7 + 0.6 + 0.5) / 4  # 0.65
        expected_ai = (0.2 + 0.3 + 0.4 + 0.5) / 4    # 0.35
        
        assert result["real"] == pytest.approx(expected_real, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(expected_ai, rel=1e-6)
    
    def test_tta_average_consistent_scores(self, sample_image_array):
        """Test TTA averaging with consistent scores."""
        def mock_predict_scores(arr):
            return {"real": 0.7, "ai-generated": 0.3}
        
        result = tta_average(sample_image_array, mock_predict_scores)
        
        # Should return the same scores since all variants give same result
        assert result["real"] == pytest.approx(0.7, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(0.3, rel=1e-6)
    
    def test_tta_average_extreme_scores(self, sample_image_array):
        """Test TTA averaging with extreme score differences."""
        def mock_predict_scores(arr):
            if np.array_equal(arr, sample_image_array):
                return {"real": 1.0, "ai-generated": 0.0}
            else:
                return {"real": 0.0, "ai-generated": 1.0}
        
        result = tta_average(sample_image_array, mock_predict_scores)
        
        # Should average to 0.25 and 0.75
        assert result["real"] == pytest.approx(0.25, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(0.75, rel=1e-6)
    
    def test_tta_average_additional_keys(self, sample_image_array):
        """Test TTA averaging with additional score keys."""
        def mock_predict_scores(arr):
            return {
                "real": 0.6,
                "ai-generated": 0.4,
                "uncertainty": 0.1,
                "confidence": 0.9
            }
        
        result = tta_average(sample_image_array, mock_predict_scores)
        
        # Should average all keys
        assert result["real"] == pytest.approx(0.6, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(0.4, rel=1e-6)
        assert result["uncertainty"] == pytest.approx(0.1, rel=1e-6)
        assert result["confidence"] == pytest.approx(0.9, rel=1e-6)
    
    def test_tta_average_missing_keys(self, sample_image_array):
        """Test TTA averaging when some variants have missing keys."""
        def mock_predict_scores(arr):
            if np.array_equal(arr, sample_image_array):
                return {"real": 0.8, "ai-generated": 0.2, "extra": 0.5}
            else:
                return {"real": 0.6, "ai-generated": 0.4}
        
        result = tta_average(sample_image_array, mock_predict_scores)
        
        # Should only include keys present in all variants
        assert "real" in result
        assert "ai-generated" in result
        assert "extra" not in result
        
        assert result["real"] == pytest.approx(0.65, rel=1e-6)  # (0.8 + 0.6 + 0.6 + 0.6) / 4
        assert result["ai-generated"] == pytest.approx(0.35, rel=1e-6)  # (0.2 + 0.4 + 0.4 + 0.4) / 4
    
    def test_tta_average_return_type(self, sample_image_array):
        """Test that tta_average returns correct types."""
        def mock_predict_scores(arr):
            return {"real": 0.5, "ai-generated": 0.5}
        
        result = tta_average(sample_image_array, mock_predict_scores)
        
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())
        assert all(isinstance(k, str) for k in result.keys())
    
    def test_tta_average_function_calls(self, sample_image_array):
        """Test that predict_scores is called the correct number of times."""
        mock_predict_scores = Mock()
        mock_predict_scores.return_value = {"real": 0.5, "ai-generated": 0.5}
        
        tta_average(sample_image_array, mock_predict_scores)
        
        # Should be called 4 times (once for each TTA variant)
        assert mock_predict_scores.call_count == 4
    
    def test_tta_average_different_input_shapes(self):
        """Test tta_average with different input shapes."""
        shapes = [(3, 100, 100), (3, 256, 256), (3, 64, 128)]
        
        for shape in shapes:
            img_array = np.random.rand(*shape).astype(np.float32)
            
            def mock_predict_scores(arr):
                return {"real": 0.6, "ai-generated": 0.4}
            
            result = tta_average(img_array, mock_predict_scores)
            
            assert result["real"] == pytest.approx(0.6, rel=1e-6)
            assert result["ai-generated"] == pytest.approx(0.4, rel=1e-6)
