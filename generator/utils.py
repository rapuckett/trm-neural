import torch
from typing import List, Dict, Tuple, Optional
from torch import Tensor
import numpy as np
from transformers import PreTrainedTokenizer

def prepare_batch_inputs(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Tensor]:
    """
    Prepare a batch of texts for thematic analysis.
    
    This is like preparing multiple signals to be processed by our radio
    telescope array simultaneously.
    
    Args:
        texts: List of input texts
        tokenizer: Tokenizer for converting text to model inputs
        max_length: Maximum sequence length
        device: Device to place tensors on
        
    Returns:
        Dictionary of model inputs
    """
    # Tokenize all texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to appropriate device
    return {k: v.to(device) for k, v in inputs.items()}

def calculate_theme_similarity(
    theme_vectors1: List[Tensor],
    theme_vectors2: List[Tensor]
) -> Tensor:
    """
    Calculate similarity between theme activations of two texts.
    
    This is like comparing how two different signals resonate across
    our array of receivers.
    
    Args:
        theme_vectors1: Theme activations from first text
        theme_vectors2: Theme activations from second text
        
    Returns:
        Tensor of similarities between theme pairs
    """
    similarities = torch.zeros(len(theme_vectors1))
    
    for i, (vec1, vec2) in enumerate(zip(theme_vectors1, theme_vectors2)):
        # Normalize vectors
        vec1_norm = vec1 / (torch.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (torch.norm(vec2) + 1e-6)
        
        # Calculate cosine similarity
        similarities[i] = torch.sum(vec1_norm * vec2_norm)
    
    return similarities

def analyze_theme_distribution(
    theme_strengths: Dict[int, float]
) -> Dict[str, float]:
    """
    Analyze the distribution of theme strengths in a text.
    
    This is like analyzing the spectrum of frequencies our radio
    telescope has detected.
    
    Args:
        theme_strengths: Dictionary mapping theme indices to their strengths
        
    Returns:
        Dictionary of distribution metrics
    """
    strengths = np.array(list(theme_strengths.values()))
    
    return {
        'dominant_theme': max(theme_strengths, key=theme_strengths.get),
        'theme_entropy': float(-np.sum(strengths * np.log(strengths + 1e-10))),
        'strength_std': float(np.std(strengths)),
        'theme_count': sum(strengths > 0.1)  # Number of significant themes
    }

def combine_theme_signals(
    theme_projections: List[Tensor],
    weights: Optional[List[float]] = None
) -> Tensor:
    """
    Combine signals from different thematic "antennas" into a unified
    representation.
    
    This is like combining signals from different radio receivers to
    create a complete picture of what we're observing.
    
    Args:
        theme_projections: List of outputs from different theme projectors
        weights: Optional weights for each theme
        
    Returns:
        Combined thematic representation
    """
    if weights is None:
        weights = [1.0] * len(theme_projections)
    
    # Convert weights to tensor and normalize
    weights = torch.tensor(weights, device=theme_projections[0].device)
    weights = weights / weights.sum()
    
    # Weighted combination of projections
    combined = sum(p * w for p, w in zip(theme_projections, weights))
    
    return combined