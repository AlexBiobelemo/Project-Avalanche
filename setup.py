from setuptools import setup, find_packages

setup(
    name="ai-image-detector",
    version="0.1.0",
    description="Classify images as real or AI-generated using local models or heuristic",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "opencv-python>=4.5",
        "Pillow>=9.0",
        # Optional backends; install if available locally
        # "torch>=1.12",  # optional
        # "transformers>=4.40",  # optional
        # "timm>=0.9",  # optional
        # GUI optional
        # "streamlit>=1.22",  # optional
    ],
    entry_points={
        "console_scripts": [
            "ai-detector=ai_image_detector.cli:main",
        ]
    },
    python_requires=">=3.8",
)


