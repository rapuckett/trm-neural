import torch
import logging
from pathlib import Path
from typing import Dict, Optional
import mlflow
from .encoder import ThematicEncoder
from .generator import ThematicGenerator
from transformers import AutoTokenizer

class ThemeAnalyzer:
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self._model = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self) -> ThematicEncoder:
        """Load model either from local path or best MLflow run"""
        if self.model_path:
            self.logger.info(f"Loading model from {self.model_path}")
            return torch.load(self.model_path, map_location=self.device)
        
        # Load best model from MLflow
        self.logger.info("Loading best model from MLflow")
        runs = mlflow.search_runs(
            filter_string="params.num_themes = '8'",
            order_by=["metrics.silhouette_score DESC"]
        )
        
        if runs.empty:
            raise ValueError("No suitable model runs found in MLflow!")
            
        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"
        return mlflow.pytorch.load_model(model_uri).to(self.device)
    
    def analyze_text(self, text: str) -> Dict[int, float]:
        """Analyze thematic patterns in text"""
        generator = ThematicGenerator(self.model, self.tokenizer)
        return generator.analyze_themes(text)
        
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Dict[int, float]]:
        """Compare thematic patterns between two texts"""
        themes1 = self.analyze_text(text1)
        themes2 = self.analyze_text(text2)
        
        # Calculate differences
        differences = {
            theme_idx: themes1[theme_idx] - themes2[theme_idx]
            for theme_idx in themes1.keys()
        }
        
        return {
            'text1_themes': themes1,
            'text2_themes': themes2,
            'differences': differences
        }