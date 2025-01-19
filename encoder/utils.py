import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from torch import Tensor

def calculate_theme_orthogonality(
    projection_matrices: List[Tensor],
    threshold: float = 0.1
) -> float:
    """
    Calculate how independent our thematic "antennas" are from each other.
    
    Just as radio telescopes work best when their receivers capture distinct
    wavelengths, our theme projections should capture distinct semantic patterns.
    
    Args:
        projection_matrices: List of weight matrices from theme projectors
        threshold: Minimum difference to consider themes distinct
        
    Returns:
        Score from 0 to 1, where 1 means perfectly orthogonal themes
    """
    total_pairs = 0
    orthogonal_pairs = 0
    
    for i in range(len(projection_matrices)):
        for j in range(i + 1, len(projection_matrices)):
            # Normalize matrices
            matrix_i = projection_matrices[i] / torch.norm(projection_matrices[i])
            matrix_j = projection_matrices[j] / torch.norm(projection_matrices[j])
            
            # Calculate overlap using Frobenius inner product
            overlap = torch.abs(torch.sum(matrix_i * matrix_j))
            
            if overlap < threshold:
                orthogonal_pairs += 1
            total_pairs += 1
    
    return orthogonal_pairs / total_pairs if total_pairs > 0 else 0.0

def process_embeddings(
    embeddings: Tensor,
    attention_mask: Optional[Tensor] = None
) -> Tensor:
    """
    Process raw embeddings before thematic projection.
    
    This is like calibrating our radio telescope's input signals before
    analyzing them with different receivers.
    
    Args:
        embeddings: Raw embeddings from the base model
        attention_mask: Optional mask for padding tokens
        
    Returns:
        Processed embeddings ready for thematic projection
    """
    if attention_mask is not None:
        # Mask out padding tokens
        embeddings = embeddings * attention_mask.unsqueeze(-1)
    
    # Normalize embeddings
    norm = torch.norm(embeddings, dim=-1, keepdim=True)
    normalized = embeddings / (norm + 1e-6)
    
    return normalized

def analyze_projection_space(
    encoder: 'ThematicEncoder',
    sample_inputs: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, float]:
    """
    Analyze the properties of our thematic projection space.
    
    This is like testing our radio telescope array by analyzing how it
    processes known calibration signals.
    
    Args:
        encoder: The ThematicEncoder instance
        sample_inputs: List of text samples for analysis
        device: Device to run analysis on
        
    Returns:
        Dictionary of metrics about the projection space
    """
    encoder.eval()
    metrics = {}
    
    # Get projections for all samples
    with torch.no_grad():
        projections = []
        for text in sample_inputs:
            inputs = encoder.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(device)
            
            outputs = encoder(**inputs)
            projections.extend([p.cpu() for p in outputs['theme_projections']])
    
    # Calculate coverage of semantic space
    projection_matrices = [proj.weight for proj in encoder.theme_projectors]
    metrics['orthogonality'] = calculate_theme_orthogonality(projection_matrices)
    
    # Calculate theme diversity (how different are theme activations)
    theme_activations = torch.stack(projections)
    theme_corr = torch.corrcoef(theme_activations.flatten(1))
    metrics['theme_diversity'] = 1.0 - torch.mean(torch.abs(theme_corr)).item()
    
    return metrics