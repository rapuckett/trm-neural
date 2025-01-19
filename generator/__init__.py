"""
The generator module handles the processing and analysis of text through TRM's thematic
projections. It measures how strongly different semantic patterns resonate across
our thematic "antennas" and synthesizes these signals into meaningful insights.
"""

from .core import ThematicGenerator
from .utils import process_embeddings

__all__ = [
    'ThematicGenerator',
    'process_embeddings',
]