# Testing Guide for AI Image Detector

This document describes the testing setup and how to run tests for the AI Image Detector application.

## Test Structure

The test suite is organized into the following modules:

- `test_models.py` - Tests for model classes (HeuristicFrequencyDetector, TorchViTDetector, EnsembleDetector)
- `test_preprocess.py` - Tests for image preprocessing functions
- `test_inference.py` - Tests for inference functions
- `test_ensemble.py` - Tests for ensemble functionality
- `test_tta.py` - Tests for Test Time Augmentation (TTA)
- `test_cli.py` - Tests for CLI functionality
- `test_config.py` - Tests for configuration classes
- `test_integration.py` - End-to-end integration tests

## Setup

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

### Verify Installation

```bash
python -m pytest --version
```

## Running Tests

### Quick Start

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_models.py

# Run specific test class
python -m pytest tests/test_models.py::TestHeuristicFrequencyDetector

# Run specific test method
python -m pytest tests/test_models.py::TestHeuristicFrequencyDetector::test_heuristic_detector_predict
```

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run only integration tests
python run_tests.py --type integration

# Run fast tests (exclude slow tests)
python run_tests.py --type fast

# Run with coverage report
python run_tests.py --type coverage

# Run with HTML coverage report
python run_tests.py --type coverage --coverage-html

# Run tests in parallel
python run_tests.py --parallel

# Run specific test file
python run_tests.py --test-file tests/test_models.py
```

### Test Categories

Tests are marked with categories for selective running:

```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run only fast tests (exclude slow tests)
python -m pytest -m "not slow"

# Run slow tests
python -m pytest -m slow
```

## Test Coverage

### Generate Coverage Report

```bash
# Terminal coverage report
python -m pytest --cov=ai_image_detector --cov-report=term-missing

# XML coverage report (for CI/CD)
python -m pytest --cov=ai_image_detector --cov-report=xml

# HTML coverage report
python -m pytest --cov=ai_image_detector --cov-report=html
```

### View Coverage Report

After generating an HTML coverage report, open `htmlcov/index.html` in your browser to see detailed coverage information.

## Test Fixtures

The test suite includes several fixtures in `conftest.py`:

- `sample_image` - Creates a sample PIL Image for testing
- `sample_image_file` - Creates a temporary image file
- `sample_image_array` - Creates a sample numpy array in CHW format
- `sample_prediction` - Creates a sample Prediction object
- `mock_torch_available` - Mocks torch availability
- `mock_transformers_available` - Mocks transformers availability

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test Structure

```python
import pytest
from ai_image_detector.models import HeuristicFrequencyDetector

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
```

### Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.slow
def test_heavy_computation():
    """This test takes a long time to run."""
    pass

@pytest.mark.integration
def test_end_to_end_workflow():
    """This is an integration test."""
    pass

@pytest.mark.unit
def test_single_function():
    """This is a unit test."""
    pass
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        python -m pytest --cov=ai_image_detector --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the package is installed in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Dependencies**: Install all test dependencies:
   ```bash
   pip install -r requirements-test.txt
   ```

3. **Slow Tests**: Use markers to skip slow tests:
   ```bash
   python -m pytest -m "not slow"
   ```

4. **Memory Issues**: Run tests with limited parallelism:
   ```bash
   python -m pytest -n 2
   ```

### Debug Mode

Run tests with maximum verbosity and no capture:

```bash
python -m pytest -vvv -s
```

### Test Discovery

Check which tests would be run:

```bash
python -m pytest --collect-only
```

## Performance Testing

The test suite includes performance tests that verify:

- Inference speed with heuristic detector
- TTA (Test Time Augmentation) performance
- Memory usage patterns

These tests are marked as `slow` and can be skipped in CI/CD pipelines.

## Mocking

The test suite uses extensive mocking to:

- Mock external dependencies (torch, transformers)
- Mock file I/O operations
- Mock network calls
- Isolate units under test

See individual test files for examples of mocking patterns.
