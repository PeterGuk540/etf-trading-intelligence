#!/usr/bin/env python
"""Train ETF prediction model"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.architectures import MarketTransformer, HybridLSTMGRU
from src.training.trainer import TrainingOrchestrator
from src.data.features import FeatureEngineering
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare training data"""
    data_path = Path("data/raw/combined_etf_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run download_data.py first.")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Simple feature engineering
    features = pd.DataFrame()
    
    # Use returns as features
    for col in df.columns:
        if 'Returns' in col:
            features[col] = df[col]
            # Add moving averages
            features[f"{col}_MA5"] = df[col].rolling(5).mean()
            features[f"{col}_MA20"] = df[col].rolling(20).mean()
            # Add volatility
            features[f"{col}_Vol20"] = df[col].rolling(20).std()
    
    # Forward fill and drop NaN
    features = features.fillna(method='ffill').dropna()
    
    # Create target (next day SPY return)
    target = df['SPY_Returns'].shift(-5)  # 5-day ahead prediction
    target = target.loc[features.index]
    
    return features, target

def main():
    parser = argparse.ArgumentParser(description='Train ETF prediction model')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'lstm_gru'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    args = parser.parse_args()
    
    # Load configuration
    with open('config/model_configs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override epochs
    config['training']['epochs'] = args.epochs
    
    logger.info(f"Training {args.model} model for {args.epochs} epochs")
    
    # Load data
    logger.info("Loading data...")
    features, target = load_data()
    logger.info(f"Data shape: {features.shape}, Target shape: {target.shape}")
    
    # Initialize model
    input_dim = features.shape[1]
    
    if args.model == 'transformer':
        model = MarketTransformer(
            input_dim=input_dim,
            d_model=config['model']['transformer']['d_model'],
            n_heads=config['model']['transformer']['n_heads'],
            n_layers=config['model']['transformer']['n_encoder_layers']
        )
    else:
        model = HybridLSTMGRU(
            input_dim=input_dim,
            lstm_hidden=config['model']['lstm_gru']['lstm_hidden_size'],
            gru_hidden=config['model']['lstm_gru']['gru_hidden_size']
        )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    orchestrator = TrainingOrchestrator(config)
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = orchestrator.prepare_data(
        features, target, 
        sequence_length=config['model']['transformer']['max_sequence_length']
    )
    
    # Train model
    logger.info("Starting training...")
    results = orchestrator.train(model, train_loader, val_loader, test_loader)
    
    logger.info(f"Training completed! Model saved to: {results.get('model_path')}")
    
    # Print test results if available
    if 'test' in results:
        logger.info(f"Test results: {results['test']}")

if __name__ == "__main__":
    main()