"""
Multi-Sector ETF Monthly Prediction System
Predicts one-month forward returns with validation on recent data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Configuration
FRED_API_KEY = "ccf75f3e8501e936dafd9f3e77729525"
SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
PREDICTION_HORIZON = 21  # One month (21 trading days)

# Date configuration for validation
TODAY = datetime(2024, 8, 10)  # Current date
TRAIN_END = datetime(2024, 6, 30)  # Training data ends June 30
VALIDATION_START = datetime(2024, 7, 1)  # July for validation
VALIDATION_END = datetime(2024, 7, 31)
PREDICTION_START = datetime(2024, 8, 1)  # August for future predictions

print("="*80)
print("MONTHLY ETF PREDICTION SYSTEM WITH VALIDATION")
print("="*80)
print(f"\n📅 Date Configuration:")
print(f"  Training: 2020-01-01 to {TRAIN_END.date()}")
print(f"  Validation: {VALIDATION_START.date()} to {VALIDATION_END.date()}")
print(f"  Prediction: {PREDICTION_START.date()} onwards")
print(f"  Horizon: {PREDICTION_HORIZON} trading days (1 month)")

class MonthlyPredictionPipeline:
    """Complete pipeline for monthly predictions"""
    
    def __init__(self):
        self.fred_indicators = {
            'DGS10': 'treasury_10y',
            'DGS2': 'treasury_2y', 
            'T5YIE': 'inflation_5y',
            'DEXUSEU': 'usd_eur',
            'DCOILWTICO': 'oil_wti',
            'GOLDAMGBD228NLBM': 'gold',
            'VIXCLS': 'vix',
            'UNRATE': 'unemployment',
            'CPIAUCSL': 'cpi',
            'INDPRO': 'industrial_prod',
            'HOUST': 'housing_starts',
            'UMCSENT': 'consumer_sentiment',
            'BAMLH0A0HYM2': 'high_yield_spread',
            'TEDRATE': 'ted_spread',
            'T10Y2Y': 'yield_curve',
            'NFCI': 'financial_conditions',
            'USSLIND': 'leading_index'
        }
        
    def fetch_all_data(self):
        """Fetch all market and economic data"""
        print("\n📊 FETCHING DATA...")
        
        # Fetch ETF data
        print("Downloading ETF data...")
        all_tickers = SECTOR_ETFS + ['SPY']
        data = yf.download(all_tickers, start='2020-01-01', end=TODAY.strftime('%Y-%m-%d'), auto_adjust=True)
        
        # Flatten multi-index columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        
        # Fetch FRED data
        print("Fetching FRED indicators...")
        fred_data = pd.DataFrame(index=data.index)
        
        for fred_code, name in self.fred_indicators.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': fred_code,
                    'api_key': FRED_API_KEY,
                    'file_type': 'json',
                    'observation_start': '2020-01-01',
                    'observation_end': TODAY.strftime('%Y-%m-%d')
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    fred_json = response.json()
                    if 'observations' in fred_json:
                        fred_df = pd.DataFrame(fred_json['observations'])
                        fred_df['date'] = pd.to_datetime(fred_df['date'])
                        fred_df = fred_df.set_index('date')
                        fred_df[name] = pd.to_numeric(fred_df['value'], errors='coerce')
                        fred_data[name] = fred_df[name]
                        print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        
        # Forward fill FRED data
        fred_data = fred_data.ffill().bfill()
        
        return data, fred_data
    
    def create_features(self, data, fred_data):
        """Create technical and fundamental features"""
        features = {}
        
        for etf in SECTOR_ETFS:
            print(f"\nProcessing {etf}...")
            df = pd.DataFrame(index=data.index)
            
            # Price data
            close_col = f'Close_{etf}'
            volume_col = f'Volume_{etf}'
            
            if close_col not in data.columns:
                close_col = etf  # Try simple column name
                volume_col = f'Volume_{etf}'
            
            # Basic price/volume features
            df['price'] = data[close_col] if close_col in data.columns else data[etf]
            df['volume'] = data[volume_col] if volume_col in data.columns else 0
            df['spy_price'] = data['Close_SPY'] if 'Close_SPY' in data.columns else data['SPY']
            
            # Technical indicators
            # 1. Returns at multiple horizons
            for period in [5, 10, 21, 63]:
                df[f'return_{period}d'] = df['price'].pct_change(period)
                df[f'spy_return_{period}d'] = df['spy_price'].pct_change(period)
                df[f'relative_return_{period}d'] = df[f'return_{period}d'] - df[f'spy_return_{period}d']
            
            # 2. Moving averages
            for period in [10, 20, 50]:
                df[f'sma_{period}'] = df['price'].rolling(period).mean()
                df[f'price_to_sma_{period}'] = df['price'] / df[f'sma_{period}']
            
            # 3. Volatility
            returns = df['price'].pct_change()
            for period in [10, 21, 63]:
                df[f'volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
            
            # 4. RSI
            delta = df['price'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 5. Bollinger Bands
            sma20 = df['price'].rolling(20).mean()
            std20 = df['price'].rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 6. Volume indicators
            df['volume_ratio'] = df['volume'].rolling(20).mean() / df['volume'].rolling(60).mean()
            df['volume_change'] = df['volume'].pct_change(5)
            
            # Add FRED features
            for col in fred_data.columns:
                df[f'fred_{col}'] = fred_data[col]
                # Add changes
                df[f'fred_{col}_change'] = fred_data[col].pct_change(21)
            
            # Create target: 21-day forward relative return
            spy_forward = df['spy_price'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
            etf_forward = df['price'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
            df['target'] = etf_forward - spy_forward
            
            features[etf] = df
            
            print(f"  Created {len(df.columns)-1} features")
        
        return features
    
    def train_and_validate(self, features):
        """Train models and validate on July 2024 data"""
        results = {}
        
        print("\n" + "="*60)
        print("TRAINING AND VALIDATION")
        print("="*60)
        
        for etf in SECTOR_ETFS:
            print(f"\n{etf} Sector:")
            print("-"*40)
            
            df = features[etf].copy()
            
            # Remove any rows with NaN
            df = df.dropna()
            
            # Split data
            train_data = df[df.index <= TRAIN_END]
            val_data = df[(df.index >= VALIDATION_START) & (df.index <= VALIDATION_END)]
            current_data = df[df.index >= PREDICTION_START]
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != 'target']
            X_train = train_data[feature_cols]
            y_train = train_data['target']
            X_val = val_data[feature_cols]
            y_val = val_data['target']
            X_current = current_data[feature_cols]
            
            # Skip if insufficient data
            if len(X_train) < 100 or len(X_val) < 5:
                print(f"  ⚠️  Insufficient data for {etf}")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_current_scaled = scaler.transform(X_current) if len(X_current) > 0 else None
            
            # Use LSTM model (simpler than TFT for monthly predictions)
            class LSTMPredictor(nn.Module):
                def __init__(self, input_dim, hidden_dim=64, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_dim, 1)
                    
                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    lstm_out, _ = self.lstm(x)
                    # Take last timestep
                    last_hidden = lstm_out[:, -1, :]
                    output = self.fc(last_hidden)
                    return output.squeeze()
            
            # Prepare sequences
            seq_length = 20  # Use 20 days of history
            
            def create_sequences(X, y, seq_len):
                X_seq, y_seq = [], []
                for i in range(seq_len, len(X)):
                    X_seq.append(X[i-seq_len:i])
                    y_seq.append(y[i])
                return np.array(X_seq), np.array(y_seq)
            
            X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, seq_length)
            X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, min(seq_length, len(X_val_scaled)-1))
            
            if len(X_train_seq) < 10 or len(X_val_seq) < 1:
                print(f"  ⚠️  Insufficient sequence data for {etf}")
                continue
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_seq)
            y_train_tensor = torch.FloatTensor(y_train_seq)
            X_val_tensor = torch.FloatTensor(X_val_seq) if len(X_val_seq) > 0 else None
            y_val_tensor = torch.FloatTensor(y_val_seq) if len(y_val_seq) > 0 else None
            
            # Create and train model
            model = LSTMPredictor(X_train_scaled.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training
            print(f"  Training LSTM model...")
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/50, Loss: {loss.item():.6f}")
            
            # Validation
            model.eval()
            with torch.no_grad():
                if X_val_tensor is not None and len(X_val_tensor) > 0:
                    val_pred = model(X_val_tensor).numpy()
                else:
                    val_pred = np.array([])
            
            models = {'LSTM': model}
            
            # Calculate validation metrics
            if len(val_pred) > 0 and len(y_val_seq) > 0:
                mse = mean_squared_error(y_val_seq, val_pred)
                mae = mean_absolute_error(y_val_seq, val_pred)
                r2 = r2_score(y_val_seq, val_pred)
                
                # Direction accuracy
                actual_direction = np.sign(y_val_seq)
                pred_direction = np.sign(val_pred)
                direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                
                print(f"\n  LSTM Model Performance:")
                print(f"    MSE: {mse:.6f}")
                print(f"    MAE: {mae:.6f}")
                print(f"    R²: {r2:.4f}")
                print(f"    Direction Accuracy: {direction_accuracy:.1f}%")
                
                best_model = model
                best_score = r2
                best_name = 'LSTM'
            else:
                print(f"  ⚠️  No validation data for {etf}")
                continue
            
            # Generate predictions for next month
            if X_current_scaled is not None and len(X_current_scaled) >= seq_length:
                X_current_seq, _ = create_sequences(X_current_scaled, np.zeros(len(X_current_scaled)), seq_length)
                if len(X_current_seq) > 0:
                    X_current_tensor = torch.FloatTensor(X_current_seq)
                    model.eval()
                    with torch.no_grad():
                        future_pred = model(X_current_tensor).numpy()
                    
                    # Store results
                    results[etf] = {
                        'model': best_name,
                        'validation_r2': best_score,
                        'validation_actual': y_val_seq,
                        'validation_predicted': val_pred,
                        'future_prediction': future_pred[-1] if len(future_pred) > 0 else None,
                        'expected_return': future_pred[-1] * 100 if len(future_pred) > 0 else None
                    }
                else:
                    results[etf] = {
                        'model': best_name,
                        'validation_r2': best_score,
                        'validation_actual': y_val_seq,
                        'validation_predicted': val_pred,
                        'future_prediction': None,
                        'expected_return': None
                    }
                
                print(f"\n  📈 Next Month Prediction:")
                print(f"    Expected Relative Return: {results[etf]['expected_return']:.2f}%")
                print(f"    Model Used: {best_name}")
        
        return results
    
    def generate_portfolio_recommendation(self, results):
        """Generate portfolio allocation based on predictions"""
        print("\n" + "="*60)
        print("PORTFOLIO RECOMMENDATION")
        print("="*60)
        
        # Extract predictions
        predictions = {}
        for etf, res in results.items():
            if res['future_prediction'] is not None:
                predictions[etf] = res['future_prediction']
        
        # Rank sectors by expected return
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        print("\nSector Rankings (by expected 1-month relative return):")
        print("-"*50)
        for i, (etf, return_val) in enumerate(ranked, 1):
            print(f"{i:2}. {etf:4} | Expected Return: {return_val*100:+.2f}%")
        
        # Simple allocation strategy
        print("\n💼 Suggested Portfolio Allocation:")
        print("-"*50)
        
        # Allocate more to positive expected returns
        total_positive = sum(max(0, pred) for pred in predictions.values())
        
        allocations = {}
        for etf, pred in predictions.items():
            if pred > 0:
                weight = (pred / total_positive) * 0.8  # 80% to positive sectors
            else:
                weight = 0.02  # 2% minimum allocation
            allocations[etf] = weight
        
        # Normalize to 100%
        total = sum(allocations.values())
        allocations = {k: v/total for k, v in allocations.items()}
        
        # Sort by allocation
        sorted_alloc = sorted(allocations.items(), key=lambda x: x[1], reverse=True)
        
        for etf, weight in sorted_alloc:
            if weight > 0.01:  # Only show allocations > 1%
                print(f"  {etf:4}: {weight*100:5.1f}%")
        
        return allocations
    
    def create_validation_report(self, results):
        """Create detailed validation report"""
        print("\n" + "="*60)
        print("VALIDATION REPORT (July 2024)")
        print("="*60)
        
        # Calculate aggregate metrics
        all_actual = []
        all_predicted = []
        
        for etf, res in results.items():
            if 'validation_actual' in res:
                all_actual.extend(res['validation_actual'])
                all_predicted.extend(res['validation_predicted'])
        
        if len(all_actual) > 0:
            all_actual = np.array(all_actual)
            all_predicted = np.array(all_predicted)
            
            overall_mse = mean_squared_error(all_actual, all_predicted)
            overall_mae = mean_absolute_error(all_actual, all_predicted)
            overall_r2 = r2_score(all_actual, all_predicted)
            
            # Direction accuracy
            actual_dir = np.sign(all_actual)
            pred_dir = np.sign(all_predicted)
            direction_acc = np.mean(actual_dir == pred_dir) * 100
            
            # Profitable predictions (predicted positive and was positive)
            profitable = np.mean((pred_dir > 0) & (actual_dir > 0)) * 100
            
            print("\n📊 Overall Validation Metrics:")
            print(f"  MSE: {overall_mse:.6f}")
            print(f"  MAE: {overall_mae:.6f}")
            print(f"  R²: {overall_r2:.4f}")
            print(f"  Direction Accuracy: {direction_acc:.1f}%")
            print(f"  Profitable Signal Rate: {profitable:.1f}%")
            
            print("\n📈 Best Performing Models:")
            best_sectors = sorted(results.items(), 
                                key=lambda x: x[1].get('validation_r2', -np.inf), 
                                reverse=True)[:3]
            
            for etf, res in best_sectors:
                print(f"  {etf}: R² = {res['validation_r2']:.4f} ({res['model']})")


# Main execution
def main():
    """Run complete monthly prediction system"""
    
    # Initialize pipeline
    pipeline = MonthlyPredictionPipeline()
    
    # Fetch data
    market_data, fred_data = pipeline.fetch_all_data()
    
    # Create features
    features = pipeline.create_features(market_data, fred_data)
    
    # Train and validate
    results = pipeline.train_and_validate(features)
    
    # Generate recommendations
    allocations = pipeline.generate_portfolio_recommendation(results)
    
    # Create validation report
    pipeline.create_validation_report(results)
    
    # Summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print("\n✅ Models trained on data through June 2024")
    print("✅ Validated on July 2024 actual returns")
    print("✅ Generated predictions for August-September 2024")
    print("\n📊 Key Outputs:")
    print("  • Validation metrics for all sectors")
    print("  • One-month forward return predictions")
    print("  • Portfolio allocation recommendations")
    print("  • Model performance comparison")
    
    return results, allocations


if __name__ == "__main__":
    results, allocations = main()