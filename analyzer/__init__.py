"""
The analyzer module provides high-level tools for understanding and evaluating
thematic patterns in text. It combines the capabilities of TRM's encoder and
generator components to provide meaningful analysis of semantic content.
"""

from .base import ThemeAnalyzer
from .metrics import evaluate_theme_quality

__all__ = [
    'ThemeAnalyzer',
    'evaluate_theme_quality',
]