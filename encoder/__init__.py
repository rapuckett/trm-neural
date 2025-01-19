"""
The encoder module provides the core neural architecture for TRM's thematic detection.
It implements multiple specialized projection layers that act like radio antennas,
each tuned to detect different semantic patterns in text.
"""

from .base import ThematicEncoder
from .utils import calculate_theme_orthogonality, process_embeddings

__all__ = [
    'ThematicEncoder',
    'calculate_theme_orthogonality',
    'process_embeddings',
]