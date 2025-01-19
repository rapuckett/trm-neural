from dataclasses import dataclass, asdict
from datetime import datetime
import gc
import json
import mlflow
from mlflow import pytorch
import logging
import numpy as np
from pathlib import Path
from safetensors.torch import save_file, load_file
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import Any, Dict, List, Optional
import os

class ThematicEncoder(nn.Module):
    def __init__(
        self, 
        base_model: str = "microsoft/deberta-v3-large",
        projection_dims: int = 384,  # Dimension for each thematic projection
        num_themes: int = 8  # Number of thematic "frequencies" to detect
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        
        # Create multiple "thematic antennas" - each one looking for different patterns
        self.theme_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, projection_dims),
                nn.LayerNorm(projection_dims),
                nn.GELU()
            )
            for _ in range(num_themes)
        ])
        
        # Thematic synthesis layer - combines different frequencies
        self.synthesis = nn.MultiheadAttention(
            embed_dim=projection_dims,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, input_ids, attention_mask):
        # Get base embeddings
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Project into different thematic spaces
        theme_projections = []
        for projector in self.theme_projectors:
            # Each projection captures different semantic aspects
            theme_proj = projector(hidden_states)  # [batch, seq_len, projection_dims]
            theme_projections.append(theme_proj)
            
        # Stack all projections
        all_themes = torch.stack(theme_projections, dim=1)  # [batch, num_themes, seq_len, proj_dims]
        
        # Synthesize thematic representations through attention
        batch_size = all_themes.size(0)
        all_themes_flat = all_themes.view(batch_size, -1, all_themes.size(-1))
        
        # Let themes attend to each other
        synthesized, _ = self.synthesis(
            all_themes_flat, all_themes_flat, all_themes_flat
        )
        
        return {
            'theme_projections': theme_projections,
            'synthesized': synthesized
        }