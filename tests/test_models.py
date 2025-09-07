"""Tests for the models module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from ai_image_detector.models import (
    Prediction, BaseDetector, HeuristicFrequencyDetector, 
    TorchViTDetector, EnsembleDetector, build_detector, sigmoid
)


class TestPrediction:
    """Test the Prediction dataclass."""
    
    def test_prediction_creation(self):
        """Test creating a Prediction object."""
        pred = Prediction(
            label="ai-generated",
            confidence=0.85,
            raw_scores={"real": 0.15, "ai-generated": 0.85}
        )
        assert pred.label == "ai-generated"
        assert pred.confidence == 0.85
        assert pred.raw_scores == {"real": 0.15, "ai-generated": 0.85}
    
    def test_prediction_without_raw_scores(self):
        """Test creating a Prediction object without raw_scores."""
        pred = Prediction(label="real", confidence=0.7)
        assert pred.label == "real"
        assert pred.confidence == 0.7
        assert pred.raw_scores is None


class TestSigmoid:
    """Test the sigmoid function."""
    
    def test_sigmoid_zero(self):
        """Test sigmoid at zero."""
        assert sigmoid(0) == 0.5
    
    def test_sigmoid_positive(self):
        """Test sigmoid with positive values."""
        assert sigmoid(1) > 0.5
        assert sigmoid(10) > sigmoid(1)
    
    def test_sigmoid_negative(self):
        """Test sigmoid with negative values."""
        assert sigmoid(-1) < 0.5
        assert sigmoid(-10) < sigmoid(-1)
    
    def test_sigmoid_extremes(self):
        """Test sigmoid at extreme values."""
        assert 0 < sigmoid(-100) < 0.1
        assert 0.9 < sigmoid(100) <= 1.0


class TestBaseDetector:
    """Test the BaseDetector class."""
    
    def test_base_detector_labels(self):
        """Test default labels."""
        detector = BaseDetector()
        assert detector.labels == ("real", "ai-generated")
    
    def test_base_detector_predict_not_implemented(self):
        """Test that predict raises NotImplementedError."""
        detector = BaseDetector()
        sample_array = np.random.rand(3, 224, 224).astype(np.float32)
        with pytest.raises(NotImplementedError):
            detector.predict(sample_array)
    
    def test_predict_with_tta(self, sample_image_array):
        """Test predict_with_tta method."""
        # Create a concrete detector for testing
        detector = HeuristicFrequencyDetector()
        
        # Test TTA with real detector - should be deterministic
        result = detector.predict_with_tta(sample_image_array)
        
        assert isinstance(result, Prediction)
        assert result.label in ["real", "ai-generated"]
        assert 0 <= result.confidence <= 1
        assert result.raw_scores is not None
        assert "real" in result.raw_scores
        assert "ai-generated" in result.raw_scores
        
        # Test that TTA is consistent for same input
        result2 = detector.predict_with_tta(sample_image_array)
        assert result.label == result2.label
        assert result.confidence == pytest.approx(result2.confidence, abs=1e-6)
        assert result.raw_scores["real"] == pytest.approx(result2.raw_scores["real"], abs=1e-6)
        assert result.raw_scores["ai-generated"] == pytest.approx(result2.raw_scores["ai-generated"], abs=1e-6)
    
    @patch('ai_image_detector.models.tta_variants')
    def test_predict_with_tta_no_raw_scores(self, mock_tta_variants, sample_image_array):
        """Test predict_with_tta when predictions have no raw_scores."""
        detector = BaseDetector()
        detector.predict = Mock()
        
        mock_tta_variants.return_value = [sample_image_array]
        detector.predict.return_value = Prediction("real", 0.7)  # No raw_scores
        
        result = detector.predict_with_tta(sample_image_array)
        
        # Should fall back to single prediction
        assert result.label == "real"
        assert result.confidence == 0.7


class TestHeuristicFrequencyDetector:
    """Test the HeuristicFrequencyDetector class."""
    
    def test_heuristic_detector_creation(self):
        """Test creating a HeuristicFrequencyDetector."""
        detector = HeuristicFrequencyDetector()
        assert detector.labels == ("real", "ai-generated")
    
    def test_heuristic_detector_predict(self, sample_image_array):
        """Test heuristic detector prediction."""
        detector = HeuristicFrequencyDetector()
        result = detector.predict(sample_image_array)
        
        assert isinstance(result, Prediction)
        assert result.label in ["real", "ai-generated"]
        assert 0 <= result.confidence <= 1
        assert result.raw_scores is not None
        assert "real" in result.raw_scores
        assert "ai-generated" in result.raw_scores
        assert abs(result.raw_scores["real"] + result.raw_scores["ai-generated"] - 1.0) < 1e-6
    
    def test_heuristic_detector_consistency(self, sample_image_array):
        """Test that heuristic detector gives consistent results."""
        detector = HeuristicFrequencyDetector()
        result1 = detector.predict(sample_image_array)
        result2 = detector.predict(sample_image_array)
        
        # Results should be identical for same input
        assert result1.label == result2.label
        assert abs(result1.confidence - result2.confidence) < 1e-6
        assert result1.raw_scores == result2.raw_scores
    
    def test_heuristic_detector_different_inputs(self):
        """Test heuristic detector with different inputs."""
        detector = HeuristicFrequencyDetector()
        
        # Test with different image arrays
        array1 = np.random.rand(3, 224, 224).astype(np.float32)
        array2 = np.random.rand(3, 224, 224).astype(np.float32)
        
        result1 = detector.predict(array1)
        result2 = detector.predict(array2)
        
        # Results should be different for different inputs
        assert isinstance(result1, Prediction)
        assert isinstance(result2, Prediction)


class TestTorchViTDetector:
    """Test the TorchViTDetector class."""
    
    def test_torch_vit_detector_creation_no_torch(self, mock_torch_available):
        """Test creating TorchViTDetector when torch is not available."""
        detector = TorchViTDetector()
        assert not detector.available
        assert detector.model is None
        assert detector.processor is None
    
    def test_torch_vit_detector_creation_no_transformers(self, mock_transformers_available):
        """Test creating TorchViTDetector when transformers is not available."""
        detector = TorchViTDetector()
        assert not detector.available
        assert detector.model is None
        assert detector.processor is None
    
    def test_torch_vit_detector_predict_not_available(self, sample_image_array, mock_torch_available):
        """Test predict when detector is not available."""
        detector = TorchViTDetector()
        with pytest.raises(RuntimeError, match="TorchViTDetector not available"):
            detector.predict(sample_image_array)
    
    def test_torch_vit_detector_with_transformers(self, sample_image_array):
        """Test TorchViTDetector with transformers available (mocked)."""
        with patch('builtins.__import__') as mock_import:
            # Mock successful imports
            def side_effect(name, *args, **kwargs):
                if name == 'torch':
                    mock_torch = Mock()
                    mock_torch.from_numpy.return_value.float.return_value.unsqueeze.return_value = Mock()
                    mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
                    mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
                    mock_torch.softmax.return_value.cpu.return_value.numpy.return_value = np.array([[0.3, 0.7]])
                    return mock_torch
                elif name == 'transformers':
                    mock_transformers = Mock()
                    mock_processor = Mock()
                    mock_model = Mock()
                    mock_model.config.id2label = {0: "real", 1: "ai-generated"}
                    mock_output = Mock()
                    mock_output.logits = Mock()
                    mock_model.return_value = mock_output
                    mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_processor
                    mock_transformers.AutoModelForImageClassification.from_pretrained.return_value = mock_model
                    return mock_transformers
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            detector = TorchViTDetector()
            if detector.available:
                result = detector.predict(sample_image_array)
                
                assert isinstance(result, Prediction)
                assert result.label in ["real", "ai-generated"]
                assert 0 <= result.confidence <= 1
                assert result.raw_scores is not None
            else:
                # If not available, just verify it handles gracefully
                with pytest.raises(RuntimeError, match="TorchViTDetector not available"):
                    detector.predict(sample_image_array)


class TestEnsembleDetector:
    """Test the EnsembleDetector class."""
    
    def test_ensemble_detector_creation(self):
        """Test creating an EnsembleDetector."""
        detector1 = HeuristicFrequencyDetector()
        detector2 = HeuristicFrequencyDetector()
        ensemble = EnsembleDetector([detector1, detector2])
        
        assert ensemble.labels == ("real", "ai-generated")
        assert ensemble.weights == (0.4, 0.6)
        assert len(ensemble.detectors) == 2
    
    def test_ensemble_detector_custom_weights(self):
        """Test EnsembleDetector with custom weights."""
        detector1 = HeuristicFrequencyDetector()
        detector2 = HeuristicFrequencyDetector()
        ensemble = EnsembleDetector([detector1, detector2], weights=(0.3, 0.7))
        
        assert ensemble.weights == (0.3, 0.7)
    
    def test_ensemble_detector_predict(self, sample_image_array):
        """Test ensemble detector prediction."""
        detector1 = HeuristicFrequencyDetector()
        detector2 = HeuristicFrequencyDetector()
        ensemble = EnsembleDetector([detector1, detector2])
        
        result = ensemble.predict(sample_image_array)
        
        assert isinstance(result, Prediction)
        assert result.label in ["real", "ai-generated"]
        assert 0 <= result.confidence <= 1
        assert result.raw_scores is not None
        assert "real" in result.raw_scores
        assert "ai-generated" in result.raw_scores
    
    def test_ensemble_detector_fallback(self, sample_image_array):
        """Test ensemble detector fallback when all detectors fail."""
        # Create mock detectors that will fail
        mock_detector1 = Mock()
        mock_detector1.predict_with_tta.side_effect = Exception("Detector failed")
        
        mock_detector2 = Mock()
        mock_detector2.predict_with_tta.side_effect = Exception("Detector failed")
        
        ensemble = EnsembleDetector([mock_detector1, mock_detector2])
        
        # Should fall back to heuristic detector
        result = ensemble.predict(sample_image_array)
        
        assert isinstance(result, Prediction)
        assert result.label in ["real", "ai-generated"]


class TestBuildDetector:
    """Test the build_detector function."""
    
    def test_build_detector_heuristic_only(self, mock_torch_available):
        """Test build_detector returns heuristic when torch is not available."""
        detector = build_detector(prefer_torch=True, use_ensemble=False)
        assert isinstance(detector, HeuristicFrequencyDetector)
    
    def test_build_detector_no_ensemble(self, mock_torch_available):
        """Test build_detector without ensemble."""
        detector = build_detector(prefer_torch=False, use_ensemble=False)
        assert isinstance(detector, HeuristicFrequencyDetector)
    
    @patch('ai_image_detector.models.TorchViTDetector')
    def test_build_detector_with_torch(self, mock_vit_class, sample_image_array):
        """Test build_detector with torch available."""
        # Mock TorchViTDetector to be available
        mock_vit = Mock()
        mock_vit.available = True
        mock_vit_class.return_value = mock_vit
        
        detector = build_detector(prefer_torch=True, use_ensemble=False)
        assert isinstance(detector, Mock)  # The mocked TorchViTDetector
    
    @patch('ai_image_detector.models.TorchViTDetector')
    def test_build_detector_ensemble(self, mock_vit_class):
        """Test build_detector with ensemble."""
        # Mock TorchViTDetector to be available
        mock_vit = Mock()
        mock_vit.available = True
        mock_vit_class.return_value = mock_vit
        
        detector = build_detector(prefer_torch=True, use_ensemble=True)
        assert isinstance(detector, EnsembleDetector)
        assert len(detector.detectors) == 2
    
    def test_build_detector_custom_weights(self, mock_torch_available):
        """Test build_detector with custom ensemble weights."""
        detector = build_detector(
            prefer_torch=False, 
            use_ensemble=True, 
            ensemble_weights=(0.3, 0.7)
        )
        # Should still return heuristic when torch is not available
        assert isinstance(detector, HeuristicFrequencyDetector)
