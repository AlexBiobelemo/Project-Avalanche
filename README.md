# AI Image Detector

## Overview
A robust Python tool for detecting whether an image is **Real** or **AI-generated/Fake**. This offline-capable detector combines multiple approaches including frequency-domain analysis, color statistics, and optional deep learning models (Vision Transformers) to provide reliable classification with confidence scores.

The tool is designed for production use with comprehensive error handling, test coverage, and support for both single-image analysis and batch processing.

## Features

### 🔍 **Detection Methods**
- **Heuristic Detector**: Fast frequency-domain analysis using FFT and color statistics (no external models required)
- **Vision Transformer Support**: Optional integration with PyTorch and Transformers/TIMM models
- **Ensemble Method**: Weighted combination of multiple detectors for improved accuracy
- **Test-Time Augmentation (TTA)**: Multiple image transformations for robust predictions

### 🖥️ **Interfaces**
- **Command Line Interface (CLI)**: Batch processing with flexible options
- **Streamlit Web GUI**: Interactive interface with drag-and-drop support
- **Python API**: Direct integration into other applications

### 🛠️ **Processing Features**
- Automatic image preprocessing (RGB conversion, center-resize, normalization)
- Support for multiple image formats (JPEG, PNG, WebP, etc.)
- Configurable input sizes and normalization parameters
- CSV export for batch analysis results

### ⚡ **Performance & Reliability**
- Offline operation (no internet required after installation)
- Comprehensive error handling for invalid files
- Cross-platform compatibility (Windows, macOS, Linux)
- Extensive test coverage (140+ tests)

## Known Limitations
- **Model Dependency**: Without fine-tuned models specifically trained on AI vs Real images, accuracy relies on heuristic methods
- **No Auto-Download**: Models must be provided locally; no automatic downloading at runtime
- **Content Variability**: Performance may vary across different content types (photographs, artwork, screenshots)
- **Threshold Sensitivity**: May require calibration of decision thresholds for specific use cases
- **Size Constraints**: Vision Transformer models expect 224x224 input resolution

## Use Cases

### 🏢 **Enterprise & Security**
- Content moderation for social media platforms
- Digital forensics and authentication
- Media verification for news organizations
- Academic research on AI-generated content detection

### 🔒 **Privacy-Focused Applications**
- Offline analysis where data cannot be sent to external APIs
- Local screening in air-gapped environments
- Batch processing of large image datasets

### 🧪 **Development & Research**
- Evaluation framework for custom AI detection models
- Baseline comparison for new detection algorithms
- Integration component for larger AI safety systems

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Quick Install

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-image-detector
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   
   # Windows CMD
   .venv\Scripts\activate.bat
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install core dependencies**
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```
   
   This installs the minimal dependencies: NumPy, Pillow, and OpenCV.

4. **Install the package**
   ```bash
   pip install -e .
   ```

### Optional Components

**For Advanced Model Support:**
```bash
pip install -r requirements-test.txt  # Includes PyTorch, Transformers, TIMM
```

**For Web Interface:**
```bash
pip install streamlit pandas
```

**For Testing/Development:**
```bash
pip install pytest pytest-cov pytest-mock
```

### Verification
Test the installation:
```bash
ai-detector --help
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
ai-detector path/to/image.jpg
```

**Multiple images:**
```bash
ai-detector image1.jpg image2.png image3.webp
```

**With detailed output:**
```bash
ai-detector image.jpg --raw
```

**Example output:**
```
image.jpg: AI-generated (AI-generated with 87% confidence)
  scores: {'real': 0.13, 'ai-generated': 0.87}
```

### Advanced CLI Options

```bash
ai-detector images/*.jpg \
  --size 384 \
  --mean 0.485 0.456 0.406 \
  --std 0.229 0.224 0.225 \
  --threshold 0.7 \
  --weights 0.3 0.7 \
  --raw
```

**Parameter descriptions:**
- `--size`: Input image size (default: 224)
- `--mean`, `--std`: Normalization parameters per RGB channel
- `--threshold`: Decision threshold for AI classification (default: 0.5)
- `--weights`: Ensemble weights for heuristic and ViT models
- `--raw`: Show raw prediction scores
- `--no-ensemble`: Disable ensemble mode (use single model)
- `--model`: Path to custom model (for Transformers)

### Web Interface

**Start the Streamlit GUI:**
```bash
streamlit run ai_image_detector/gui_streamlit.py
```

Features:
- 🖥️ Drag-and-drop image upload
- 📊 Interactive confidence visualization
- 📄 CSV export for batch results
- ⚙️ Configurable detection parameters

### Python API

```python
from ai_image_detector.inference import run_inference_on_array
from ai_image_detector.preprocess import load_image, preprocess_image, PreprocessConfig

# Load and preprocess image
img = load_image("path/to/image.jpg")
config = PreprocessConfig(target_size=(224, 224))
img_array = preprocess_image(img, config)

# Run inference
prediction = run_inference_on_array(img_array, use_ensemble=True)

print(f"Label: {prediction.label}")
print(f"Confidence: {prediction.confidence:.3f}")
print(f"Raw scores: {prediction.raw_scores}")
```

## Technologies

### Core Dependencies
- **Python 3.8+**: Modern Python with type hints and dataclass support
- **NumPy**: Efficient numerical computations and array operations
- **Pillow (PIL)**: Image loading, conversion, and basic preprocessing
- **OpenCV**: Advanced image processing and computer vision operations

### Detection Algorithms
- **FFT (Fast Fourier Transform)**: Frequency domain analysis for artifact detection
- **Statistical Analysis**: Color distribution and variance computations
- **Computer Vision**: Edge detection, texture analysis

### Optional Deep Learning Stack
- **PyTorch**: Deep learning framework for neural network inference
- **Transformers (Hugging Face)**: Pre-trained Vision Transformer models
- **TIMM**: PyTorch image models library with extensive model zoo

### User Interface & Visualization
- **Streamlit**: Interactive web interface with real-time updates
- **Pandas**: Data manipulation and CSV export functionality
- **Matplotlib/Plotly** (via Streamlit): Confidence visualization charts

### Testing & Quality Assurance
- **pytest**: Comprehensive test framework with 140+ test cases
- **pytest-cov**: Code coverage analysis and reporting
- **pytest-mock**: Mock objects for isolated unit testing
- **pytest-xdist**: Parallel test execution for faster CI/CD

### Development & Configuration
- **dataclasses**: Type-safe configuration and prediction objects
- **argparse**: Robust command-line interface with help system
- **typing**: Full type annotation support for better IDE integration

## Project Structure

```
ai-image-detector/
├── ai_image_detector/
│   ├── __init__.py           # Package initialization and version
│   ├── cli.py               # Command-line interface implementation
│   ├── config.py            # Configuration classes and defaults
│   ├── ensemble.py          # Weighted averaging for ensemble methods
│   ├── inference.py         # Main inference pipeline and formatting
│   ├── models.py            # Detection models (Heuristic, ViT, Ensemble)
│   ├── preprocess.py        # Image loading and preprocessing pipeline
│   ├── tta.py              # Test-time augmentation utilities
│   └── gui_streamlit.py     # Web interface (optional)
├── tests/
│   ├── conftest.py          # Pytest fixtures and test configuration
│   ├── test_cli.py         # CLI argument parsing and integration tests
│   ├── test_config.py      # Configuration validation tests
│   ├── test_ensemble.py    # Ensemble method tests
│   ├── test_inference.py   # Inference pipeline tests
│   ├── test_integration.py # End-to-end integration tests
│   ├── test_models.py      # Individual model tests
│   ├── test_preprocess.py  # Image preprocessing tests
│   └── test_tta.py         # Test-time augmentation tests
├── requirements.txt         # Core dependencies
├── requirements-test.txt    # Testing and optional dependencies
├── setup.py                # Package installation configuration
├── README.md               # This documentation
└── LICENSE                 # MIT License
```

## Performance & Benchmarks

### Speed Comparison (on CPU)
- **Heuristic Detector**: ~50-100ms per image
- **Vision Transformer**: ~200-500ms per image (depending on model size)
- **Ensemble + TTA**: ~300-800ms per image (4x TTA variants)

### Memory Usage
- **Base Installation**: ~50MB RAM
- **With PyTorch**: ~500MB-2GB RAM (depending on model)
- **Peak Processing**: +100-300MB per image batch

### Test Coverage
- **Overall**: >95% code coverage
- **Core Modules**: >98% coverage
- **Test Count**: 140+ test cases across all modules
- **CI/CD**: Automated testing on multiple Python versions

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-test.txt`
4. Run tests: `pytest tests/`
5. Submit a pull request with test coverage

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built on top of the excellent PyTorch and Hugging Face ecosystems
- Inspired by research in AI-generated content detection
- Uses computer vision techniques from OpenCV community

---

**Note**: This tool is designed for research and educational purposes. For production use in critical applications, consider fine-tuning with domain-specific datasets and validating performance on your specific use cases.
