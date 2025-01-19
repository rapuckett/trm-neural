from setuptools import setup, find_packages
import os

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

def get_version():
    """Read version from trm/__init__.py"""
    with open(os.path.join("trm", "__init__.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                # Clean it up
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Version information not found")

install_requires = [
    "torch>=2.0.0",           # Core deep learning functionality
    "transformers>=4.30.0",   # For working with language models
    "numpy>=1.20.0",          # Numerical computing
    "scikit-learn>=1.0.0",    # For metrics and evaluation
    "mlflow>=2.0.0",          # Experiment tracking
    "pandas>=1.3.0",          # Data manipulation
    "tqdm>=4.65.0",           # Progress bars
]

# Development dependencies (for testing, docs, etc.)
dev_requires = [
    "pytest>=7.0.0",         # Testing framework
    "pytest-cov>=4.0.0",     # Test coverage
    "black>=22.0.0",         # Code formatting
    "isort>=5.0.0",          # Import sorting
    "mypy>=1.0.0",           # Type checking
    "sphinx>=4.0.0",         # Documentation
]

setup(
    name="trm-neural",
    version=get_version(),
    author="Richard Puckett",
    author_email="rapuckett@gmail.com",
    description="Thematic Resonance Memory: A neural framework for theme-based text understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rapuckett/trm",
    
    packages=find_packages(include=["trm", "trm.*"]),
    python_requires=">=3.8",
    
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,  # Install with pip install -e ".[dev]"
    },
    
    # Package metadata for PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, machine learning, thematic analysis, language models, neural networks",
)