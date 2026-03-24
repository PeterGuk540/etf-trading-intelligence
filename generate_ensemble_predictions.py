"""
Ensemble Prediction System for ETF Sector Rotation
Uses 7-model ensemble with sector-specific and VIX regime weighting.

Models: LSTM, TFT, N-BEATS, LSTM-GARCH (PyTorch deep learning)
        LightGBM (gradient boosting regression)
        CatBoost (gradient boosting 3-class classification)
        SARIMAX  (time-series with correlation-based feature selection)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
FRED_API_KEY = "ccf75f3e8501e936dafd9f3e77729525"
SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
PREDICTION_HORIZON = 21  # One month

# Import from existing files
import sys
sys.path.insert(0, '/home/aojie_ju/etf-trading-intelligence')

# ---- nn.Module definitions (kept here for backward compat imports) ----------

class SimpleLSTM(nn.Module):
    """Basic LSTM for baseline"""
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


class SimpleTFT(nn.Module):
    """Simplified TFT with attention"""
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.attention = nn.MultiheadAttention(32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        encoded, _ = self.lstm(x)
        attn_out, _ = self.attention(encoded, encoded, encoded)
        return self.fc(attn_out[:, -1, :]).squeeze()


class SimpleNBeats(nn.Module):
    """Simplified N-BEATS"""
    def __init__(self, input_dim, seq_len=20):
        super().__init__()
        flat_dim = input_dim * seq_len
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze()


class SimpleLSTMGARCH(nn.Module):
    """Simplified LSTM-GARCH"""
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.8))
        self.fc = nn.Linear(33, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        returns = x[:, :, 0]
        volatility = returns.std(dim=1, keepdim=True).clamp(min=1e-6)
        combined = torch.cat([last_hidden, volatility], dim=1)
        return self.fc(combined).squeeze()


# ---- Wrapper imports --------------------------------------------------------

from src.models.pytorch_wrappers import PyTorchModelWrapper
from src.models.lightgbm_wrapper import LightGBMWrapper
from src.models.catboost_wrapper import CatBoostWrapper
from src.models.sarimax_wrapper import SARIMAXWrapper


# ---- 7-Model Ensemble Predictor --------------------------------------------

class EnsemblePredictor:
    """7-Model Ensemble with sector-specific and VIX regime weighting.

    All wrappers receive flat 2D (n_samples, n_features) arrays; each wrapper
    handles its own internal transformation (sliding window, feature selection,
    target binning, etc.).
    """

    def __init__(self, input_dim, feature_names):
        self.feature_names = list(feature_names)
        self.input_dim = input_dim

        # Build 7 model wrappers
        self.wrappers = {
            'lstm':       PyTorchModelWrapper(SimpleLSTM, 'lstm'),
            'tft':        PyTorchModelWrapper(SimpleTFT, 'tft'),
            'nbeats':     PyTorchModelWrapper(SimpleNBeats, 'nbeats'),
            'lstm_garch': PyTorchModelWrapper(SimpleLSTMGARCH, 'lstm_garch'),
            'lightgbm':   LightGBMWrapper(),
            'catboost':   CatBoostWrapper(),
            'sarimax':    SARIMAXWrapper(),
        }

        # Track which models trained successfully
        self._failed_models = set()

        # Sector-specific base weights (expanded for 7 models)
        self.sector_weights = {
            'XLE': {
                'lstm_garch': 0.40, 'lstm': 0.10, 'tft': 0.05, 'nbeats': 0.00,
                'lightgbm': 0.20, 'catboost': 0.15, 'sarimax': 0.10,
            },
            'XLK': {
                'lstm': 0.30, 'nbeats': 0.15, 'tft': 0.05, 'lstm_garch': 0.00,
                'lightgbm': 0.25, 'catboost': 0.15, 'sarimax': 0.10,
            },
            'XLF': {
                'tft': 0.25, 'lstm': 0.15, 'nbeats': 0.10, 'lstm_garch': 0.00,
                'lightgbm': 0.20, 'catboost': 0.15, 'sarimax': 0.15,
            },
            'default': {
                'lstm': 0.15, 'tft': 0.15, 'nbeats': 0.10, 'lstm_garch': 0.10,
                'lightgbm': 0.20, 'catboost': 0.15, 'sarimax': 0.15,
            },
        }

        # VIX regime multipliers (expanded for 7 models)
        self.vix_regime_adjustments = {
            'LOW_VOL': {
                'lstm': 1.2, 'tft': 1.1, 'nbeats': 1.0, 'lstm_garch': 0.8,
                'lightgbm': 1.1, 'catboost': 1.0, 'sarimax': 1.1,
            },
            'MEDIUM_VOL': {
                'lstm': 1.0, 'tft': 1.0, 'nbeats': 1.0, 'lstm_garch': 1.0,
                'lightgbm': 1.0, 'catboost': 1.0, 'sarimax': 1.0,
            },
            'HIGH_VOL': {
                'lstm': 0.8, 'tft': 0.9, 'nbeats': 1.0, 'lstm_garch': 1.3,
                'lightgbm': 0.9, 'catboost': 1.1, 'sarimax': 0.7,
            },
        }

    def train_models(self, X_scaled, y, sector, feature_names=None):
        """Train all 7 wrappers on flat 2D data."""
        if feature_names is None:
            feature_names = self.feature_names
        self._failed_models = set()

        for name, wrapper in self.wrappers.items():
            try:
                print(f"      Training {name}...")
                wrapper.train(X_scaled, y, sector, feature_names)
            except Exception as e:
                print(f"      Warning: {name} training failed: {e}")
                self._failed_models.add(name)

    def classify_vix_regime(self, vix_value):
        """Classify VIX into LOW/MEDIUM/HIGH"""
        if pd.isna(vix_value):
            return 'MEDIUM_VOL'
        elif vix_value < 20:
            return 'LOW_VOL'
        elif vix_value < 30:
            return 'MEDIUM_VOL'
        else:
            return 'HIGH_VOL'

    def predict_ensemble(self, X_scaled, sector, vix_level, feature_names=None):
        """Generate ensemble prediction with sector and VIX weighting.

        Parameters
        ----------
        X_scaled : np.ndarray, shape (n, n_features)
            Flat 2D scaled features (e.g. the last ``seq_len`` rows for
            PyTorch models or a single row for tree models).
        sector : str
        vix_level : float
        feature_names : list[str] or None

        Returns
        -------
        ensemble_pred : float
        predictions : dict[str, float]
        final_weights : dict[str, float]
        vix_regime : str
        """
        if feature_names is None:
            feature_names = self.feature_names

        # Collect point predictions from each model
        predictions = {}
        for name, wrapper in self.wrappers.items():
            if name in self._failed_models:
                predictions[name] = 0.0
                continue
            try:
                preds = wrapper.predict(X_scaled, feature_names)
                # Take last prediction as the point forecast
                predictions[name] = float(preds[-1]) if len(preds) > 0 else 0.0
            except Exception as e:
                print(f"      Warning: {name} prediction error: {e}")
                predictions[name] = 0.0

        # Sector base weights
        base_weights = self.sector_weights.get(sector, self.sector_weights['default'])

        # VIX regime adjustments
        vix_regime = self.classify_vix_regime(vix_level)
        regime_adj = self.vix_regime_adjustments[vix_regime]

        # Compute and normalise final weights (skip failed models)
        final_weights = {}
        total_weight = 0.0
        for model_name in self.wrappers:
            if model_name in self._failed_models:
                final_weights[model_name] = 0.0
                continue
            w = base_weights.get(model_name, 0.0) * regime_adj.get(model_name, 1.0)
            final_weights[model_name] = w
            total_weight += w

        if total_weight > 0:
            for k in final_weights:
                final_weights[k] /= total_weight

        # Weighted ensemble prediction
        ensemble_pred = sum(
            predictions[name] * final_weights[name] for name in self.wrappers
        )

        return ensemble_pred, predictions, final_weights, vix_regime


def generate_ensemble_predictions(month, year, train_end_date, val_start_date, val_end_date):
    """
    Generate ensemble predictions for a specific month

    Args:
        month: Target month name (e.g., "August", "September", "October")
        year: Target year (e.g., 2025)
        train_end_date: Last day of training data (datetime)
        val_start_date: First day of validation data (datetime)
        val_end_date: Last day of validation data (datetime)
    """
    import calendar

    print("="*80)
    print(f"{month.upper()} {year} ENSEMBLE PREDICTIONS")
    print("="*80)
    print(f"\n📅 Configuration:")
    print(f"  Training: 2020-01-01 to {train_end_date.date()}")
    print(f"  Validation: {val_start_date.date()} to {val_end_date.date()}")
    print(f"  Prediction Target: {month} {year}")
    print(f"  Ensemble: LSTM + TFT + N-BEATS + LSTM-GARCH + LightGBM + CatBoost + SARIMAX")
    print("="*80)
    print()

    # Import pipeline for data fetching
    from etf_monthly_prediction_system import MonthlyPredictionPipeline

    # Override dates
    import etf_monthly_prediction_system as eps
    eps.TRAIN_END = train_end_date
    eps.VALIDATION_START = val_start_date
    eps.VALIDATION_END = val_end_date
    try:
        eps.PREDICTION_START = datetime(year, list(calendar.month_name).index(month), 1)
    except ValueError:
        # Non-standard month name (e.g., "Mid_march") — keep existing PREDICTION_START
        pass

    pipeline = MonthlyPredictionPipeline()

    print("📊 Fetching data...")
    market_data, fred_data = pipeline.fetch_all_data()

    print("🔧 Creating features...")
    features = pipeline.create_features(market_data, fred_data)

    print(f"\n🤖 Training 7-Model Ensemble for {month} {year}:")
    print("-" * 80)

    results = {}
    seq_length = 20

    for etf in SECTOR_ETFS:
        print(f"\n{etf} Sector:")
        print("-" * 40)

        df = features[etf].copy()

        # Clean data
        feature_cols = [col for col in df.columns if col != 'target']
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        df = df.dropna(subset=['target'])

        # Split data
        train_data = df[df.index <= train_end_date]
        val_data = df[(df.index >= val_start_date) & (df.index <= val_end_date)]

        print(f"  Train: {len(train_data)} rows, Val: {len(val_data)} rows")

        if len(train_data) < 100:
            print(f"  ⚠️ Insufficient data, skipping")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train = train_data[feature_cols]
        y_train = train_data['target'].values
        X_train_scaled = scaler.fit_transform(X_train)

        # Prepare optional validation data
        val_X_scaled = None
        val_y = None
        if len(val_data) > 20:
            val_X_scaled = scaler.transform(val_data[feature_cols])
            val_y = val_data['target'].values

        # Initialize 7-model ensemble (flat features — wrappers handle transforms)
        ensemble = EnsemblePredictor(X_train_scaled.shape[1], list(feature_cols))

        # Train ensemble
        print(f"  Training 7-model ensemble...")
        ensemble.train_models(X_train_scaled, y_train, etf, list(feature_cols))

        # Get VIX level (21-day lagged)
        vix_col = [c for c in df.columns if 'vix' in c.lower() and 'lag' in c.lower()]
        if vix_col:
            vix_level = df[vix_col[0]].iloc[-1]
        else:
            vix_level = fred_data['vix'].shift(PREDICTION_HORIZON).iloc[-1] if 'vix' in fred_data else 18.0

        # Generate prediction — pass flat scaled features
        all_data = df[df.index <= train_end_date]
        if len(all_data) >= seq_length:
            X_pred = scaler.transform(all_data[feature_cols].iloc[-seq_length:])

            ensemble_pred, model_preds, weights, regime = ensemble.predict_ensemble(
                X_pred, etf, vix_level, list(feature_cols)
            )

            print(f"  ✅ Ensemble Prediction: {ensemble_pred:+.4f} ({ensemble_pred*100:+.2f}%)")
            print(f"     VIX Regime: {regime} (VIX: {vix_level:.1f})")
            print(f"     Model Contributions:")
            for model_name, weight in weights.items():
                if weight > 0.01:
                    print(f"       {model_name}: {model_preds[model_name]:+.4f} (weight: {weight:.1%})")

            results[etf] = float(ensemble_pred)
        else:
            print(f"  ⚠️ Insufficient data for prediction")

    # Save predictions
    output_file = f"{month.lower()}_{year}_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Predictions saved to: {output_file}")
    print(f"   Generated {len(results)} ensemble predictions")

    return results


if __name__ == "__main__":
    # This will be called from separate scripts for each month
    pass
