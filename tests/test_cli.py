"""Tests for the CLI module."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from io import StringIO
import sys

from ai_image_detector.cli import parse_args, main
from ai_image_detector.preprocess import ImageLoadError


class TestParseArgs:
    """Test the parse_args function."""
    
    def test_parse_args_basic(self):
        """Test basic argument parsing."""
        args = parse_args(["image1.jpg", "image2.png"])
        
        assert args.images == ["image1.jpg", "image2.png"]
        assert args.size == 224
        assert args.mean == (0.5, 0.5, 0.5)
        assert args.std == (0.5, 0.5, 0.5)
        assert args.raw is False
        assert args.no_ensemble is False
        assert args.model is None
        assert args.threshold == 0.5
        assert args.weights is None
    
    def test_parse_args_custom_size(self):
        """Test argument parsing with custom size."""
        args = parse_args(["--size", "256", "image.jpg"])
        
        assert args.size == 256
        assert args.images == ["image.jpg"]
    
    def test_parse_args_custom_mean_std(self):
        """Test argument parsing with custom mean and std."""
        args = parse_args([
            "--mean", "0.485", "0.456", "0.406",
            "--std", "0.229", "0.224", "0.225",
            "image.jpg"
        ])
        
        assert args.mean == (0.485, 0.456, 0.406)
        assert args.std == (0.229, 0.224, 0.225)
    
    def test_parse_args_raw_flag(self):
        """Test argument parsing with raw flag."""
        args = parse_args(["--raw", "image.jpg"])
        
        assert args.raw is True
    
    def test_parse_args_no_ensemble(self):
        """Test argument parsing with no-ensemble flag."""
        args = parse_args(["--no-ensemble", "image.jpg"])
        
        assert args.no_ensemble is True
    
    def test_parse_args_model_path(self):
        """Test argument parsing with model path."""
        args = parse_args(["--model", "/path/to/model", "image.jpg"])
        
        assert args.model == "/path/to/model"
    
    def test_parse_args_threshold(self):
        """Test argument parsing with custom threshold."""
        args = parse_args(["--threshold", "0.7", "image.jpg"])
        
        assert args.threshold == 0.7
    
    def test_parse_args_weights(self):
        """Test argument parsing with ensemble weights."""
        args = parse_args(["--weights", "0.3", "0.7", "image.jpg"])
        
        assert args.weights == (0.3, 0.7)
    
    def test_parse_args_all_options(self):
        """Test argument parsing with all options."""
        args = parse_args([
            "--size", "256",
            "--mean", "0.485", "0.456", "0.406",
            "--std", "0.229", "0.224", "0.225",
            "--raw",
            "--no-ensemble",
            "--model", "/path/to/model",
            "--threshold", "0.8",
            "--weights", "0.2", "0.8",
            "image1.jpg", "image2.png"
        ])
        
        assert args.images == ["image1.jpg", "image2.png"]
        assert args.size == 256
        assert args.mean == (0.485, 0.456, 0.406)
        assert args.std == (0.229, 0.224, 0.225)
        assert args.raw is True
        assert args.no_ensemble is True
        assert args.model == "/path/to/model"
        assert args.threshold == 0.8
        assert args.weights == (0.2, 0.8)
    
    def test_parse_args_no_images(self):
        """Test argument parsing with no images (should fail)."""
        with pytest.raises(SystemExit):
            parse_args([])
    
    def test_parse_args_help(self):
        """Test argument parsing with help flag."""
        with pytest.raises(SystemExit):
            parse_args(["--help"])


class TestMain:
    """Test the main function."""
    
    @patch('ai_image_detector.cli.load_image')
    @patch('ai_image_detector.cli.preprocess_image')
    @patch('ai_image_detector.cli.run_inference_on_array')
    @patch('ai_image_detector.cli.format_prediction')
    def test_main_success(self, mock_format, mock_inference, mock_preprocess, mock_load, sample_image_file, capsys):
        """Test successful main execution."""
        # Mock the functions
        mock_img = Mock()
        mock_load.return_value = mock_img
        
        mock_array = Mock()
        mock_preprocess.return_value = mock_array
        
        mock_pred = Mock()
        mock_pred.raw_scores = {"real": 0.2, "ai-generated": 0.8}
        mock_pred.confidence = 0.8
        mock_inference.return_value = mock_pred
        
        mock_format.return_value = "AI-generated with 80% confidence"
        
        # Run main
        exit_code = main([sample_image_file])
        
        assert exit_code == 0
        
        # Check output
        captured = capsys.readouterr()
        assert "AI-generated" in captured.out
        assert "80% confidence" in captured.out
        
        # Verify function calls
        mock_load.assert_called_once_with(sample_image_file)
        mock_preprocess.assert_called_once()
        mock_inference.assert_called_once()
        mock_format.assert_called_once_with(mock_pred)
    
    @patch('ai_image_detector.cli.load_image')
    @patch('ai_image_detector.cli.preprocess_image')
    @patch('ai_image_detector.cli.run_inference_on_array')
    @patch('ai_image_detector.cli.format_prediction')
    def test_main_with_raw_output(self, mock_format, mock_inference, mock_preprocess, mock_load, sample_image_file, capsys):
        """Test main execution with raw output."""
        # Mock the functions
        mock_img = Mock()
        mock_load.return_value = mock_img
        
        mock_array = Mock()
        mock_preprocess.return_value = mock_array
        
        mock_pred = Mock()
        mock_pred.raw_scores = {"real": 0.3, "ai-generated": 0.7}
        mock_pred.confidence = 0.7
        mock_inference.return_value = mock_pred
        
        mock_format.return_value = "AI-generated with 70% confidence"
        
        # Run main with raw flag
        exit_code = main(["--raw", sample_image_file])
        
        assert exit_code == 0
        
        # Check output includes raw scores
        captured = capsys.readouterr()
        assert "AI-generated" in captured.out
        assert "scores:" in captured.out
        assert "real" in captured.out
        assert "ai-generated" in captured.out
    
    @patch('ai_image_detector.cli.load_image')
    def test_main_image_load_error(self, mock_load, capsys):
        """Test main execution with image load error."""
        mock_load.side_effect = ImageLoadError("File not found")
        
        exit_code = main(["nonexistent.jpg"])
        
        assert exit_code == 2
        
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "File not found" in captured.out
    
    @patch('ai_image_detector.cli.load_image')
    @patch('ai_image_detector.cli.preprocess_image')
    @patch('ai_image_detector.cli.run_inference_on_array')
    def test_main_inference_error(self, mock_inference, mock_preprocess, mock_load, sample_image_file, capsys):
        """Test main execution with inference error."""
        mock_img = Mock()
        mock_load.return_value = mock_img
        
        mock_array = Mock()
        mock_preprocess.return_value = mock_array
        
        mock_inference.side_effect = Exception("Inference failed")
        
        exit_code = main([sample_image_file])
        
        assert exit_code == 3
        
        captured = capsys.readouterr()
        assert "ERROR during inference" in captured.out
        assert "Inference failed" in captured.out
    
    @patch('ai_image_detector.cli.load_image')
    @patch('ai_image_detector.cli.preprocess_image')
    @patch('ai_image_detector.cli.run_inference_on_array')
    @patch('ai_image_detector.cli.format_prediction')
    def test_main_multiple_images(self, mock_format, mock_inference, mock_preprocess, mock_load, capsys):
        """Test main execution with multiple images."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1:
            tmp1.write(b"fake image data")
            tmp1_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
            tmp2.write(b"fake image data")
            tmp2_path = tmp2.name
        
        try:
            # Mock the functions
            mock_img = Mock()
            mock_load.return_value = mock_img
            
            mock_array = Mock()
            mock_preprocess.return_value = mock_array
            
            mock_pred = Mock()
            mock_pred.raw_scores = {"real": 0.5, "ai-generated": 0.5}
            mock_pred.confidence = 0.5
            mock_inference.return_value = mock_pred
            
            mock_format.return_value = "Real with 50% confidence"
            
            # Run main with multiple images
            exit_code = main([tmp1_path, tmp2_path])
            
            assert exit_code == 0
            
            # Verify load_image was called for each image
            assert mock_load.call_count == 2
            
            # Check output contains both images
            captured = capsys.readouterr()
            assert tmp1_path in captured.out
            assert tmp2_path in captured.out
        
        finally:
            os.unlink(tmp1_path)
            os.unlink(tmp2_path)
    
    @patch('ai_image_detector.cli.load_image')
    @patch('ai_image_detector.cli.preprocess_image')
    @patch('ai_image_detector.cli.run_inference_on_array')
    @patch('ai_image_detector.cli.format_prediction')
    def test_main_custom_parameters(self, mock_format, mock_inference, mock_preprocess, mock_load, sample_image_file):
        """Test main execution with custom parameters."""
        # Mock the functions
        mock_img = Mock()
        mock_load.return_value = mock_img
        
        mock_array = Mock()
        mock_preprocess.return_value = mock_array
        
        mock_pred = Mock()
        mock_pred.raw_scores = {"real": 0.4, "ai-generated": 0.6}
        mock_pred.confidence = 0.6
        mock_inference.return_value = mock_pred
        
        mock_format.return_value = "AI-generated with 60% confidence"
        
        # Run main with custom parameters
        exit_code = main([
            "--size", "256",
            "--mean", "0.485", "0.456", "0.406",
            "--std", "0.229", "0.224", "0.225",
            "--no-ensemble",
            "--model", "/path/to/model",
            "--threshold", "0.7",
            "--weights", "0.3", "0.7",
            sample_image_file
        ])
        
        assert exit_code == 0
        
        # Verify run_inference_on_array was called with correct parameters
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        
        assert call_args[0][0] == mock_array  # First positional argument
        assert call_args[1]['use_ensemble'] is False
        assert call_args[1]['model_name_or_path'] == "/path/to/model"
        assert call_args[1]['ensemble_weights'] == (0.3, 0.7)
    
    @patch('ai_image_detector.cli.load_image')
    @patch('ai_image_detector.cli.preprocess_image')
    @patch('ai_image_detector.cli.run_inference_on_array')
    @patch('ai_image_detector.cli.format_prediction')
    def test_main_threshold_override(self, mock_format, mock_inference, mock_preprocess, mock_load, sample_image_file, capsys):
        """Test main execution with threshold override."""
        # Mock the functions
        mock_img = Mock()
        mock_load.return_value = mock_img
        
        mock_array = Mock()
        mock_preprocess.return_value = mock_array
        
        mock_pred = Mock()
        mock_pred.raw_scores = {"real": 0.3, "ai-generated": 0.7}
        mock_pred.confidence = 0.7
        mock_inference.return_value = mock_pred
        
        mock_format.return_value = "AI-generated with 70% confidence"
        
        # Run main with high threshold
        exit_code = main(["--threshold", "0.8", sample_image_file])
        
        assert exit_code == 0
        
        # Check output shows Real (because 0.7 < 0.8 threshold)
        captured = capsys.readouterr()
        assert "Real" in captured.out
    
    def test_main_with_argv_none(self):
        """Test main function with argv=None (uses sys.argv)."""
        with patch('ai_image_detector.cli.sys.argv', ['cli.py', '--help']):
            with pytest.raises(SystemExit):
                main(argv=None)
