"""Tests for the ensemble module."""

import pytest
from ai_image_detector.ensemble import weighted_average


class TestWeightedAverage:
    """Test the weighted_average function."""
    
    def test_weighted_average_basic(self):
        """Test basic weighted average calculation."""
        scores = [
            {"real": 0.8, "ai-generated": 0.2},
            {"real": 0.3, "ai-generated": 0.7}
        ]
        weights = (0.6, 0.4)
        
        result = weighted_average(scores, weights)
        
        expected_real = 0.8 * 0.6 + 0.3 * 0.4  # 0.48 + 0.12 = 0.6
        expected_ai = 0.2 * 0.6 + 0.7 * 0.4    # 0.12 + 0.28 = 0.4
        
        assert result["real"] == pytest.approx(expected_real, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(expected_ai, rel=1e-6)
    
    def test_weighted_average_equal_weights(self):
        """Test weighted average with equal weights."""
        scores = [
            {"real": 0.9, "ai-generated": 0.1},
            {"real": 0.1, "ai-generated": 0.9}
        ]
        weights = (0.5, 0.5)
        
        result = weighted_average(scores, weights)
        
        assert result["real"] == pytest.approx(0.5, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(0.5, rel=1e-6)
    
    def test_weighted_average_single_score(self):
        """Test weighted average with single score."""
        scores = [{"real": 0.7, "ai-generated": 0.3}]
        weights = (1.0,)
        
        result = weighted_average(scores, weights)
        
        assert result["real"] == pytest.approx(0.7, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(0.3, rel=1e-6)
    
    def test_weighted_average_three_scores(self):
        """Test weighted average with three scores."""
        scores = [
            {"real": 0.8, "ai-generated": 0.2},
            {"real": 0.5, "ai-generated": 0.5},
            {"real": 0.2, "ai-generated": 0.8}
        ]
        weights = (0.5, 0.3, 0.2)
        
        result = weighted_average(scores, weights)
        
        expected_real = 0.8 * 0.5 + 0.5 * 0.3 + 0.2 * 0.2  # 0.4 + 0.15 + 0.04 = 0.59
        expected_ai = 0.2 * 0.5 + 0.5 * 0.3 + 0.8 * 0.2    # 0.1 + 0.15 + 0.16 = 0.41
        
        assert result["real"] == pytest.approx(expected_real, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(expected_ai, rel=1e-6)
    
    def test_weighted_average_zero_weights(self):
        """Test weighted average with zero weights (should handle gracefully)."""
        scores = [
            {"real": 0.8, "ai-generated": 0.2},
            {"real": 0.3, "ai-generated": 0.7}
        ]
        weights = (0.0, 0.0)
        
        result = weighted_average(scores, weights)
        
        # Should return default values when weights sum to zero
        assert result["real"] == pytest.approx(0.0, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(1.0, rel=1e-6)
    
    def test_weighted_average_empty_scores(self):
        """Test weighted average with empty scores list."""
        scores = []
        weights = ()
        
        result = weighted_average(scores, weights)
        
        # Should return default values
        assert result["real"] == pytest.approx(0.0, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(1.0, rel=1e-6)
    
    def test_weighted_average_missing_keys(self):
        """Test weighted average with scores missing some keys."""
        scores = [
            {"real": 0.8, "ai-generated": 0.2},
            {"real": 0.3}  # Missing "ai-generated" key
        ]
        weights = (0.6, 0.4)
        
        result = weighted_average(scores, weights)
        
        expected_real = 0.8 * 0.6 + 0.3 * 0.4  # 0.48 + 0.12 = 0.6
        expected_ai = 0.2 * 0.6 + 0.0 * 0.4    # 0.12 + 0.0 = 0.12
        
        assert result["real"] == pytest.approx(expected_real, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(expected_ai, rel=1e-6)
    
    def test_weighted_average_extra_keys(self):
        """Test weighted average with scores containing extra keys."""
        scores = [
            {"real": 0.8, "ai-generated": 0.2, "uncertainty": 0.1},
            {"real": 0.3, "ai-generated": 0.7, "confidence": 0.9}
        ]
        weights = (0.5, 0.5)
        
        result = weighted_average(scores, weights)
        
        # Should only average the keys from the first score
        expected_real = 0.8 * 0.5 + 0.3 * 0.5  # 0.4 + 0.15 = 0.55
        expected_ai = 0.2 * 0.5 + 0.7 * 0.5    # 0.1 + 0.35 = 0.45
        
        assert result["real"] == pytest.approx(expected_real, rel=1e-6)
        assert result["ai-generated"] == pytest.approx(expected_ai, rel=1e-6)
        assert "uncertainty" not in result
        assert "confidence" not in result
    
    def test_weighted_average_normalization(self):
        """Test that weights are properly normalized."""
        scores = [
            {"real": 0.8, "ai-generated": 0.2},
            {"real": 0.3, "ai-generated": 0.7}
        ]
        weights = (0.6, 0.4)  # Sum = 1.0
        
        result1 = weighted_average(scores, weights)
        
        # Test with unnormalized weights that should be normalized
        weights_unnormalized = (1.2, 0.8)  # Sum = 2.0, should be normalized to (0.6, 0.4)
        result2 = weighted_average(scores, weights_unnormalized)
        
        assert result1["real"] == pytest.approx(result2["real"], rel=1e-6)
        assert result1["ai-generated"] == pytest.approx(result2["ai-generated"], rel=1e-6)
    
    def test_weighted_average_float_precision(self):
        """Test weighted average with high precision floats."""
        scores = [
            {"real": 0.123456789, "ai-generated": 0.876543211},
            {"real": 0.987654321, "ai-generated": 0.012345679}
        ]
        weights = (0.333333333, 0.666666667)
        
        result = weighted_average(scores, weights)
        
        expected_real = 0.123456789 * 0.333333333 + 0.987654321 * 0.666666667
        expected_ai = 0.876543211 * 0.333333333 + 0.012345679 * 0.666666667
        
        assert result["real"] == pytest.approx(expected_real, rel=1e-8)
        assert result["ai-generated"] == pytest.approx(expected_ai, rel=1e-8)
    
    def test_weighted_average_return_type(self):
        """Test that weighted_average returns the correct type."""
        scores = [{"real": 0.5, "ai-generated": 0.5}]
        weights = (1.0,)
        
        result = weighted_average(scores, weights)
        
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())
        assert all(isinstance(k, str) for k in result.keys())
