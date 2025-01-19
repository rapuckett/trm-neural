"""
Thematic Resonance Memory (TRM) is a framework for understanding and processing text
through multiple semantic "frequencies" or themes. Like a sophisticated radio telescope
array that captures different wavelengths of electromagnetic radiation to build a complete
picture of celestial objects, TRM uses multiple specialized neural projections to capture
different aspects of meaning in text.
"""

# Version and metadata
__version__ = "0.1.0"
__author__ = "Richard Puckett"
__email__ = "rapuckett@gmail.com"

# Core components
from .encoder import ThematicEncoder
from .generator import ThematicGenerator
from .analyzer import ThemeAnalyzer

# Convenience functions for common operations
from .encoder.utils import calculate_theme_orthogonality
from .generator.utils import process_embeddings
from .analyzer.metrics import evaluate_theme_quality

from .utils.device import DeviceManager

# Use a global device manager
device_manager = DeviceManager()

__all__ = [
    # Main classes
    'ThematicEncoder',
    'ThematicGenerator',
    'ThemeAnalyzer',
    
    # Utility functions
    'calculate_theme_orthogonality',
    'process_embeddings',
    'evaluate_theme_quality',
    'device_manager'
]