import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

class ThemeQualityMetrics:
    """
    A comprehensive suite of metrics for evaluating the quality of our thematic detection system.
    Just as radio astronomers need ways to validate their telescope array's performance,
    we need robust methods to evaluate how well our thematic "antennas" are working.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def evaluate_theme_separation(
        self,
        embeddings: np.ndarray,
        theme_assignments: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate how well our themes separate different semantic patterns.
        This is like measuring how well our radio telescope array distinguishes
        different types of astronomical signals.
        
        Args:
            embeddings: Matrix of text embeddings (n_samples x embedding_dim)
            theme_assignments: Array of theme labels for each embedding
            
        Returns:
            Dictionary of separation metrics
        """
        # Ensure we have enough samples for meaningful metrics
        if len(embeddings) < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_score': 0.0,
                'theme_contrast': 0.0
            }
            
        try:
            # Calculate silhouette score (how well-separated the themes are)
            silhouette = silhouette_score(embeddings, theme_assignments)
            
            # Calculate Calinski-Harabasz score (ratio of between-cluster to within-cluster dispersion)
            calinski = calinski_harabasz_score(embeddings, theme_assignments)
            
            # Calculate theme contrast (average distance between vs within themes)
            distances = squareform(pdist(embeddings))
            within_theme_dist = []
            between_theme_dist = []
            
            for theme in np.unique(theme_assignments):
                mask = theme_assignments == theme
                theme_samples = distances[mask][:, mask]
                other_samples = distances[mask][:, ~mask]
                
                if len(theme_samples) > 0:
                    within_theme_dist.extend(theme_samples[np.triu_indices(len(theme_samples), k=1)])
                if len(other_samples) > 0:
                    between_theme_dist.extend(other_samples.flatten())
            
            within_mean = np.mean(within_theme_dist) if within_theme_dist else 0
            between_mean = np.mean(between_theme_dist) if between_theme_dist else 0
            theme_contrast = (between_mean - within_mean) / (between_mean + within_mean + 1e-6)
            
            return {
                'silhouette_score': float(silhouette),
                'calinski_score': float(calinski),
                'theme_contrast': float(theme_contrast)
            }
            
        except ValueError as e:
            print(f"Error calculating separation metrics: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_score': 0.0,
                'theme_contrast': 0.0
            }
    
    def evaluate_theme_diversity(
        self,
        theme_activations: List[Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate how diverse and independent our themes are.
        This is like measuring whether each antenna in our array is capturing
        unique information rather than redundant signals.
        
        Args:
            theme_activations: List of activation patterns for each theme
            
        Returns:
            Dictionary of diversity metrics
        """
        # Stack activations and move to CPU for numpy operations
        activations = torch.stack(theme_activations).cpu().numpy()
        
        # Calculate activation correlation matrix
        correlations = np.corrcoef(activations.reshape(len(activations), -1))
        
        # Calculate entropy of activation patterns
        activation_entropy = entropy(activations.mean(axis=(1, 2)))
        
        # Calculate average absolute correlation (lower is better)
        avg_correlation = np.mean(np.abs(correlations - np.eye(correlations.shape[0])))
        
        # Calculate "frequency bandwidth" - how much of the possible activation range each theme uses
        activation_ranges = activations.max(axis=(1, 2)) - activations.min(axis=(1, 2))
        avg_bandwidth = np.mean(activation_ranges)
        
        return {
            'theme_independence': 1.0 - avg_correlation,
            'activation_entropy': float(activation_entropy),
            'activation_bandwidth': float(avg_bandwidth)
        }
    
    def evaluate_theme_stability(
        self,
        theme_activations1: List[Tensor],
        theme_activations2: List[Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate how stable our theme detection is across similar inputs.
        This is like measuring how consistently our radio telescope array
        detects the same type of signal under slightly different conditions.
        
        Args:
            theme_activations1: Theme activations for first version of input
            theme_activations2: Theme activations for second version of input
            
        Returns:
            Dictionary of stability metrics
        """
        stability_scores = []
        consistency_scores = []
        
        for act1, act2 in zip(theme_activations1, theme_activations2):
            # Convert to numpy and flatten
            act1_np = act1.cpu().numpy().flatten()
            act2_np = act2.cpu().numpy().flatten()
            
            # Calculate correlation between activations
            correlation = np.corrcoef(act1_np, act2_np)[0, 1]
            stability_scores.append(correlation)
            
            # Calculate consistency of activation patterns
            pattern1 = act1_np > act1_np.mean()
            pattern2 = act2_np > act2_np.mean()
            consistency = np.mean(pattern1 == pattern2)
            consistency_scores.append(consistency)
        
        return {
            'average_stability': float(np.mean(stability_scores)),
            'pattern_consistency': float(np.mean(consistency_scores)),
            'stability_std': float(np.std(stability_scores))
        }
    
    def calculate_overall_quality(
        self,
        separation_metrics: Dict[str, float],
        diversity_metrics: Dict[str, float],
        stability_metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate an overall quality score for our thematic detection system.
        This combines multiple metrics into a single score, like creating an
        overall performance rating for our radio telescope array.
        
        Args:
            separation_metrics: Output from evaluate_theme_separation
            diversity_metrics: Output from evaluate_theme_diversity
            stability_metrics: Output from evaluate_theme_stability
            weights: Optional dictionary of metric weights
            
        Returns:
            Overall quality score between 0 and 1
        """
        if weights is None:
            weights = {
                'silhouette': 0.2,
                'theme_independence': 0.2,
                'activation_entropy': 0.15,
                'average_stability': 0.25,
                'pattern_consistency': 0.2
            }
            
        # Combine metrics using weighted average
        components = {
            'silhouette': separation_metrics['silhouette_score'],
            'theme_independence': diversity_metrics['theme_independence'],
            'activation_entropy': diversity_metrics['activation_entropy'] / np.log(8),  # Normalize by max possible entropy
            'average_stability': stability_metrics['average_stability'],
            'pattern_consistency': stability_metrics['pattern_consistency']
        }
        
        # Calculate weighted sum
        quality_score = sum(
            score * weights.get(metric, 0.2)
            for metric, score in components.items()
        )
        
        return float(quality_score)

def evaluate_theme_quality(
    model: 'ThematicEncoder',
    test_texts: List[str],
    metrics: Optional[ThemeQualityMetrics] = None
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of theme quality using a test dataset.
    
    Args:
        model: The ThematicEncoder model to evaluate
        test_texts: List of texts to use for evaluation
        metrics: Optional ThemeQualityMetrics instance
        
    Returns:
        Nested dictionary of all quality metrics
    """
    if metrics is None:
        metrics = ThemeQualityMetrics()
        
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Get theme activations for all texts
        activations = []
        embeddings = []
        for text in test_texts:
            outputs = model(text)
            activations.append(outputs['theme_projections'])
            embeddings.append(outputs['pooled_output'].cpu().numpy())
            
        # Stack embeddings and determine theme assignments
        embeddings_array = np.vstack(embeddings)
        theme_assignments = model.get_dominant_themes(embeddings_array)
        
        # Calculate all metrics
        results['separation'] = metrics.evaluate_theme_separation(
            embeddings_array,
            theme_assignments
        )
        
        results['diversity'] = metrics.evaluate_theme_diversity(activations)
        
        # Calculate stability using text variations
        variations = [text + " " for text in test_texts]  # Tiny variations
        variation_activations = []
        with torch.no_grad():
            for text in variations:
                outputs = model(text)
                variation_activations.append(outputs['theme_projections'])
                
        results['stability'] = metrics.evaluate_theme_stability(
            activations,
            variation_activations
        )
        
        # Calculate overall quality score
        results['overall_quality'] = metrics.calculate_overall_quality(
            results['separation'],
            results['diversity'],
            results['stability']
        )
        
    return results