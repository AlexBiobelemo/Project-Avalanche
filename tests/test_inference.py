"""Tests for the inference module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from ai_image_detector.inference import (
    run_inference_on_array, format_prediction, prediction_to_dict
)
from ai_image_detector.models import Prediction


class TestRunInferenceOnArray:
    """Test the run_inference_on_array function."""
    
    @patch('ai_image_detector.inference.build_detector')
    def test_run_inference_on_array_basic(self, mock_build_detector, sample_image_array):
        """Test basic inference functionality."""
        # Mock detector
        mock_detector = Mock()
        mock_prediction = Prediction("ai-generated", 0.85, {"real": 0.15, "ai-generated": 0.85})
        mock_detector.predict.return_value = mock_prediction
        mock_build_detector.return_value = mock_detector
        
        result = run_inference_on_array(sample_image_array)
        
        assert result == mock_prediction
        mock_build_detector.assert_called_once_with(
            prefer_torch=True, 
            use_ensemble=True, 
            model_name_or_path=None, 
            ensemble_weights=None
        )
        mock_detector.predict.assert_called_once_with(sample_image_array)
    
    @patch('ai_image_detector.inference.build_detector')
    def test_run_inference_on_array_with_ensemble_false(self, mock_build_detector, sample_image_array):
        """Test inference with ensemble disabled."""
        mock_detector = Mock()
        mock_prediction = Prediction("real", 0.7)
        mock_detector.predict.return_value = mock_prediction
        mock_build_detector.return_value = mock_detector
        
        result = run_inference_on_array(sample_image_array, use_ensemble=False)
        
        assert result == mock_prediction
        mock_build_detector.assert_called_once_with(
            prefer_torch=True, 
            use_ensemble=False, 
            model_name_or_path=None, 
            ensemble_weights=None
        )
    
    @patch('ai_image_detector.inference.build_detector')
    def test_run_inference_on_array_with_model_path(self, mock_build_detector, sample_image_array):
        """Test inference with custom model path."""
        mock_detector = Mock()
        mock_prediction = Prediction("ai-generated", 0.9)
        mock_detector.predict.return_value = mock_prediction
        mock_build_detector.return_value = mock_detector
        
        result = run_inference_on_array(
            sample_image_array, 
            model_name_or_path="/path/to/model"
        )
        
        assert result == mock_prediction
        mock_build_detector.assert_called_once_with(
            prefer_torch=True, 
            use_ensemble=True, 
            model_name_or_path="/path/to/model", 
            ensemble_weights=None
        )
    
    @patch('ai_image_detector.inference.build_detector')
    def test_run_inference_on_array_with_ensemble_weights(self, mock_build_detector, sample_image_array):
        """Test inference with custom ensemble weights."""
        mock_detector = Mock()
        mock_prediction = Prediction("real", 0.6)
        mock_detector.predict.return_value = mock_prediction
        mock_build_detector.return_value = mock_detector
        
        result = run_inference_on_array(
            sample_image_array, 
            ensemble_weights=(0.3, 0.7)
        )
        
        assert result == mock_prediction
        mock_build_detector.assert_called_once_with(
            prefer_torch=True, 
            use_ensemble=True, 
            model_name_or_path=None, 
            ensemble_weights=(0.3, 0.7)
        )
    
    @patch('ai_image_detector.inference.build_detector')
    def test_run_inference_on_array_all_parameters(self, mock_build_detector, sample_image_array):
        """Test inference with all parameters specified."""
        mock_detector = Mock()
        mock_prediction = Prediction("ai-generated", 0.95)
        mock_detector.predict.return_value = mock_prediction
        mock_build_detector.return_value = mock_detector
        
        result = run_inference_on_array(
            sample_image_array,
            use_ensemble=False,
            model_name_or_path="custom/model",
            ensemble_weights=(0.2, 0.8)
        )
        
        assert result == mock_prediction
        mock_build_detector.assert_called_once_with(
            prefer_torch=True, 
            use_ensemble=False, 
            model_name_or_path="custom/model", 
            ensemble_weights=(0.2, 0.8)
        )


class TestFormatPrediction:
    """Test the format_prediction function."""
    
    def test_format_prediction_ai_generated(self, sample_prediction):
        """Test formatting AI-generated prediction."""
        result = format_prediction(sample_prediction)
        
        assert "AI-generated" in result
        assert "85%" in result  # 85% confidence
        assert "confidence" in result
    
    def test_format_prediction_real(self):
        """Test formatting real image prediction."""
        pred = Prediction("real", 0.7, {"real": 0.7, "ai-generated": 0.3})
        result = format_prediction(pred)
        
        assert "Real" in result
        assert "70%" in result
        assert "confidence" in result
    
    def test_format_prediction_ai_label_variations(self):
        """Test formatting with different AI label variations."""
        # Test with "ai" prefix
        pred1 = Prediction("ai", 0.8)
        result1 = format_prediction(pred1)
        assert "AI-generated" in result1
        
        # Test with "AI" prefix
        pred2 = Prediction("AI", 0.8)
        result2 = format_prediction(pred2)
        assert "AI-generated" in result2
        
        # Test with "ai-generated" label
        pred3 = Prediction("ai-generated", 0.8)
        result3 = format_prediction(pred3)
        assert "AI-generated" in result3
    
    def test_format_prediction_real_label_variations(self):
        """Test formatting with different real label variations."""
        # Test with "real" label
        pred1 = Prediction("real", 0.8)
        result1 = format_prediction(pred1)
        assert "Real" in result1
        
        # Test with "Real" label
        pred2 = Prediction("Real", 0.8)
        result2 = format_prediction(pred2)
        assert "Real" in result2
        
        # Test with "human" label (not starting with "ai")
        pred3 = Prediction("human", 0.8)
        result3 = format_prediction(pred3)
        assert "Real" in result3
    
    def test_format_prediction_confidence_rounding(self):
        """Test confidence percentage rounding."""
        # Test rounding up
        pred1 = Prediction("real", 0.756)
        result1 = format_prediction(pred1)
        assert "76%" in result1
        
        # Test rounding down
        pred2 = Prediction("real", 0.754)
        result2 = format_prediction(pred2)
        assert "75%" in result2
        
        # Test exact rounding
        pred3 = Prediction("real", 0.755)
        result3 = format_prediction(pred3)
        assert "76%" in result3  # Should round up at 0.5
    
    def test_format_prediction_edge_cases(self):
        """Test formatting with edge case confidence values."""
        # Test 0% confidence
        pred1 = Prediction("real", 0.0)
        result1 = format_prediction(pred1)
        assert "0%" in result1
        
        # Test 100% confidence
        pred2 = Prediction("ai-generated", 1.0)
        result2 = format_prediction(pred2)
        assert "100%" in result2
        
        # Test very low confidence
        pred3 = Prediction("real", 0.001)
        result3 = format_prediction(pred3)
        assert "0%" in result3
        
        # Test very high confidence
        pred4 = Prediction("ai-generated", 0.999)
        result4 = format_prediction(pred4)
        assert "100%" in result4


class TestPredictionToDict:
    """Test the prediction_to_dict function."""
    
    def test_prediction_to_dict_with_raw_scores(self, sample_prediction):
        """Test converting prediction with raw scores to dict."""
        result = prediction_to_dict(sample_prediction)
        
        expected = {
            "label": "ai-generated",
            "confidence": 0.85,
            "scores": {"real": 0.15, "ai-generated": 0.85}
        }
        assert result == expected
    
    def test_prediction_to_dict_without_raw_scores(self):
        """Test converting prediction without raw scores to dict."""
        pred = Prediction("real", 0.7)
        result = prediction_to_dict(pred)
        
        expected = {
            "label": "real",
            "confidence": 0.7,
            "scores": {}
        }
        assert result == expected
    
    def test_prediction_to_dict_none_raw_scores(self):
        """Test converting prediction with None raw scores to dict."""
        pred = Prediction("ai-generated", 0.9, None)
        result = prediction_to_dict(pred)
        
        expected = {
            "label": "ai-generated",
            "confidence": 0.9,
            "scores": {}
        }
        assert result == expected
    
    def test_prediction_to_dict_empty_raw_scores(self):
        """Test converting prediction with empty raw scores to dict."""
        pred = Prediction("real", 0.6, {})
        result = prediction_to_dict(pred)
        
        expected = {
            "label": "real",
            "confidence": 0.6,
            "scores": {}
        }
        assert result == expected
    
    def test_prediction_to_dict_complex_raw_scores(self):
        """Test converting prediction with complex raw scores to dict."""
        raw_scores = {
            "real": 0.2,
            "ai-generated": 0.8,
            "uncertainty": 0.1,
            "model_confidence": 0.95
        }
        pred = Prediction("ai-generated", 0.8, raw_scores)
        result = prediction_to_dict(pred)
        
        expected = {
            "label": "ai-generated",
            "confidence": 0.8,
            "scores": raw_scores
        }
        assert result == expected
    
    def test_prediction_to_dict_preserves_types(self):
        """Test that prediction_to_dict preserves data types."""
        raw_scores = {
            "real": 0.3,
            "ai-generated": 0.7,
            "metadata": "test"
        }
        pred = Prediction("ai-generated", 0.7, raw_scores)
        result = prediction_to_dict(pred)
        
        assert isinstance(result["label"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["scores"], dict)
        assert isinstance(result["scores"]["real"], float)
        assert isinstance(result["scores"]["metadata"], str)
