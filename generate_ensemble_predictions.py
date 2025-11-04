"""
Ensemble Prediction System for ETF Sector Rotation
Uses 4-model ensemble with sector-specific and VIX regime weighting
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
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

# Model architectures from validate_all_models.py
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


class EnsemblePredictor:
    """4-Model Ensemble with sector-specific and VIX regime weighting"""

    def __init__(self, input_dim):
        self.models = {
            'lstm': SimpleLSTM(input_dim),
            'tft': SimpleTFT(input_dim),
            'nbeats': SimpleNBeats(input_dim),
            'lstm_garch': SimpleLSTMGARCH(input_dim)
        }

        # Sector-specific weights (from validation performance)
        self.sector_weights = {
            'XLE': {'lstm_garch': 0.7, 'lstm': 0.2, 'tft': 0.1, 'nbeats': 0.0},
            'XLK': {'lstm': 0.6, 'nbeats': 0.3, 'tft': 0.1, 'lstm_garch': 0.0},
            'XLF': {'tft': 0.5, 'lstm': 0.3, 'nbeats': 0.2, 'lstm_garch': 0.0},
            'default': {'lstm': 0.3, 'tft': 0.3, 'nbeats': 0.2, 'lstm_garch': 0.2}
        }

        # VIX regime adjustments
        self.vix_regime_adjustments = {
            'LOW_VOL': {'lstm': 1.2, 'tft': 1.1, 'nbeats': 1.0, 'lstm_garch': 0.8},
            'MEDIUM_VOL': {'lstm': 1.0, 'tft': 1.0, 'nbeats': 1.0, 'lstm_garch': 1.0},
            'HIGH_VOL': {'lstm': 0.8, 'tft': 0.9, 'nbeats': 1.0, 'lstm_garch': 1.3}
        }

        self.optimizers = {}
        for name, model in self.models.items():
            self.optimizers[name] = None

    def train_models(self, X, y, sector, epochs=50, lr=0.001):
        """Train all 4 models"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Initialize optimizers
        for name, model in self.models.items():
            self.optimizers[name] = torch.optim.Adam(model.parameters(), lr=lr)

        # Train each model
        for epoch in range(epochs):
            for name, model in self.models.items():
                model.train()
                self.optimizers[name].zero_grad()

                try:
                    output = model(X_tensor)
                    # Handle scalar vs array outputs
                    if output.dim() == 0:
                        output = output.unsqueeze(0)
                    loss = nn.MSELoss()(output, y_tensor)
                    loss.backward()
                    self.optimizers[name].step()
                except Exception as e:
                    print(f"      Warning: {name} training error: {e}")
                    continue

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}")

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

    def predict_ensemble(self, X, sector, vix_level):
        """Generate ensemble prediction with sector and VIX weighting"""
        X_tensor = torch.FloatTensor(X)

        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                try:
                    pred = model(X_tensor)
                    if pred.dim() == 0:
                        pred = pred.item()
                    else:
                        pred = pred.numpy()[0] if len(pred) > 0 else pred.item()
                    predictions[name] = float(pred)
                except Exception as e:
                    print(f"      Warning: {name} prediction error: {e}")
                    predictions[name] = 0.0

        # Get sector-specific base weights
        base_weights = self.sector_weights.get(sector, self.sector_weights['default'])

        # Get VIX regime adjustments
        vix_regime = self.classify_vix_regime(vix_level)
        regime_adjustments = self.vix_regime_adjustments[vix_regime]

        # Calculate final weights
        final_weights = {}
        total_weight = 0
        for model_name in self.models.keys():
            weight = base_weights[model_name] * regime_adjustments[model_name]
            final_weights[model_name] = weight
            total_weight += weight

        # Normalize weights
        for model_name in final_weights:
            final_weights[model_name] /= total_weight if total_weight > 0 else 1.0

        # Ensemble prediction
        ensemble_pred = sum(predictions[name] * final_weights[name]
                           for name in self.models.keys())

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
    print(f"  Ensemble: LSTM + TFT + N-BEATS + LSTM-GARCH")
    print("="*80)
    print()

    # Import pipeline for data fetching
    from etf_monthly_prediction_system import MonthlyPredictionPipeline

    # Override dates
    import etf_monthly_prediction_system as eps
    eps.TRAIN_END = train_end_date
    eps.VALIDATION_START = val_start_date
    eps.VALIDATION_END = val_end_date
    eps.PREDICTION_START = datetime(year, list(calendar.month_name).index(month), 1) if month != "October" else datetime(year, 10, 1)

    pipeline = MonthlyPredictionPipeline()

    print("📊 Fetching data...")
    market_data, fred_data = pipeline.fetch_all_data()

    print("🔧 Creating features...")
    features = pipeline.create_features(market_data, fred_data)

    print(f"\n🤖 Training Ensemble Models for {month} {year}:")
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
        y_train = train_data['target']
        X_train_scaled = scaler.fit_transform(X_train)

        # Create sequences
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for i in range(seq_len, len(X)):
                X_seq.append(X[i-seq_len:i])
                y_seq.append(y.iloc[i])
            return np.array(X_seq), np.array(y_seq)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)

        if len(X_train_seq) < 10:
            print(f"  ⚠️ Insufficient sequences, skipping")
            continue

        # Initialize ensemble
        ensemble = EnsemblePredictor(X_train_scaled.shape[1])

        # Train ensemble
        print(f"  Training 4-model ensemble...")
        ensemble.train_models(X_train_seq, y_train_seq, etf, epochs=50, lr=0.001)

        # Get VIX level (21-day lagged)
        vix_col = [c for c in df.columns if 'vix' in c.lower() and 'lag' in c.lower()]
        if vix_col:
            vix_level = df[vix_col[0]].iloc[-1]
        else:
            vix_level = fred_data['vix'].shift(PREDICTION_HORIZON).iloc[-1] if 'vix' in fred_data else 18.0

        # Generate prediction
        all_data = df[df.index <= train_end_date]
        if len(all_data) >= seq_length:
            X_pred = scaler.transform(all_data[feature_cols].iloc[-seq_length:])
            X_pred_seq = X_pred.reshape(1, seq_length, -1)

            ensemble_pred, model_preds, weights, regime = ensemble.predict_ensemble(
                X_pred_seq, etf, vix_level
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
