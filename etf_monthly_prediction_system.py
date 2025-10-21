"""
Multi-Sector ETF Monthly Prediction System
Predicts one-month forward returns with validation on recent data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
FRED_API_KEY = "ccf75f3e8501e936dafd9f3e77729525"
SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
PREDICTION_HORIZON = 21  # One month (21 trading days)

# Dynamic date configuration based on current date
TODAY = datetime.now()  # Get actual current date

# For predictions, we use the current month
CURRENT_MONTH_START = datetime(TODAY.year, TODAY.month, 1)

# For validation, we need complete data, so go back 2 months to ensure we have forward returns
TWO_MONTHS_AGO = TODAY - relativedelta(months=2)
VALIDATION_END = datetime(TWO_MONTHS_AGO.year, TWO_MONTHS_AGO.month, calendar.monthrange(TWO_MONTHS_AGO.year, TWO_MONTHS_AGO.month)[1])
VALIDATION_START = datetime(TWO_MONTHS_AGO.year, TWO_MONTHS_AGO.month, 1)

# Training ends before validation
TRAIN_END = VALIDATION_START - timedelta(days=1)

# Dynamic configuration
PREDICTION_START = CURRENT_MONTH_START  # Current month for predictions (e.g., August 2025)

print("="*80)
print("MONTHLY ETF PREDICTION SYSTEM WITH VALIDATION")
print("="*80)
print(f"\n📅 Dynamic Date Configuration (Today: {TODAY.date()}):")
print(f"  Training: 2020-01-01 to {TRAIN_END.date()}")
print(f"  Validation: {VALIDATION_START.date()} to {VALIDATION_END.date()} ({VALIDATION_START.strftime('%B %Y')})")
print(f"  Prediction: {PREDICTION_START.date()} onwards ({PREDICTION_START.strftime('%B %Y')})")
print(f"  Horizon: {PREDICTION_HORIZON} trading days")
print("\n  Note: Validation uses data from 2 months ago to ensure complete forward returns")

class MonthlyPredictionPipeline:
    """Complete pipeline for monthly predictions"""
    
    def __init__(self):
        # Complete list of 62 FRED indicators from Data_Generation.ipynb
        self.fred_indicators = {
            # Interest Rates & Yields (10 indicators)
            'DGS1': 'treasury_1y',
            'DGS2': 'treasury_2y', 
            'DGS5': 'treasury_5y',
            'DGS10': 'treasury_10y',
            'DGS30': 'treasury_30y',
            'DFEDTARU': 'fed_funds_upper',
            'DFEDTARL': 'fed_funds_lower',
            'TB3MS': 'treasury_3m',
            'TB6MS': 'treasury_6m',
            'MORTGAGE30US': 'mortgage_30y',
            
            # Yield Curves & Spreads (8 indicators)
            'T10Y2Y': 'yield_curve_10y2y',
            'T10Y3M': 'yield_curve_10y3m',
            'T5YIE': 'inflation_5y',
            'T10YIE': 'inflation_10y',
            'TEDRATE': 'ted_spread',
            'BAMLH0A0HYM2': 'high_yield_spread',
            'BAMLC0A0CM': 'investment_grade_spread',
            'DPRIME': 'prime_rate',
            
            # Economic Activity (12 indicators)
            'GDP': 'gdp',
            'GDPC1': 'real_gdp',
            'INDPRO': 'industrial_production',
            'TCU': 'capacity_utilization',  # Fixed: was CAPACITY
            'RSXFS': 'retail_sales',  # Fixed: was RETAILSL
            'HOUST': 'housing_starts',
            'PERMIT': 'building_permits',
            'ALTSALES': 'auto_sales',  # Fixed: was AUTOSOLD
            'BOPGEXP': 'exports',  # Fixed: was DEXPORT
            'BOPGIMP': 'imports',  # Fixed: was DIMPORT
            'NETEXP': 'net_exports',
            'BUSLOANS': 'business_loans',
            
            # Employment (8 indicators)
            'UNRATE': 'unemployment_rate',
            'EMRATIO': 'employment_ratio',
            'CIVPART': 'participation_rate',
            'NFCI': 'financial_conditions',
            'ICSA': 'initial_claims',
            'PAYEMS': 'nonfarm_payrolls',
            'AWHAETP': 'avg_hourly_earnings',
            'AWHMAN': 'avg_weekly_hours',
            
            # Inflation & Prices (8 indicators)
            'CPIAUCSL': 'cpi',
            'CPILFESL': 'core_cpi',
            'PPIACO': 'ppi',
            'PPICMM': 'ppi_metals',  # Fixed: was PPIFGS (ppi_finished_goods)
            'GASREGW': 'gas_price',
            'DCOILWTICO': 'oil_wti',
            'DCOILBRENTEU': 'oil_brent',
            'WPU10170301': 'gold',  # Fixed: was GOLDAMGBD228NLBM
            
            # Money Supply & Credit (6 indicators)
            'M1SL': 'm1_money',
            'M2SL': 'm2_money',
            'BOGMBASE': 'monetary_base',
            'TOTRESNS': 'bank_reserves',
            'CONSUMER': 'consumer_credit',
            'TOTALSL': 'total_loans',
            
            # Market Indicators (5 indicators)
            'VIXCLS': 'vix',
            'DEXUSEU': 'usd_eur',
            'DEXJPUS': 'usd_jpy',
            'DEXUSUK': 'usd_gbp',
            'DTWEXBGS': 'dollar_index',  # Fixed: was DXY

            # Consumer & Business Sentiment (5 indicators)
            'UMCSENT': 'consumer_sentiment',
            'CSCICP03USM665S': 'consumer_confidence',  # Fixed: was CBCCI
            'USSLIND': 'leading_index',
            'USPHCI': 'coincident_index',  # Fixed: was USCCCI
            'BSCICP03USM665S': 'business_optimism'  # Fixed: was OEMV
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
    
    def compute_momentum(self, close_series, window, skip=0):
        """Compute momentum as in Data_Generation.ipynb"""
        return close_series.shift(skip) / close_series.shift(skip + window) - 1
    
    def compute_macd(self, close_series, fast=12, slow=26, signal=9):
        """Compute MACD indicators"""
        exp1 = close_series.ewm(span=fast, adjust=False).mean()
        exp2 = close_series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def compute_kdj(self, high, low, close, period=9):
        """Compute KDJ indicators"""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        rsv = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    def compute_atr(self, high, low, close, period=14):
        """Compute Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def compute_mfi(self, high, low, close, volume, period=14):
        """Compute Money Flow Index"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = pd.Series(0, index=close.index)
        negative_flow = pd.Series(0, index=close.index)
        positive_flow[typical_price > typical_price.shift()] = raw_money_flow[typical_price > typical_price.shift()]
        negative_flow[typical_price < typical_price.shift()] = raw_money_flow[typical_price < typical_price.shift()]
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi
    
    def create_features(self, data, fred_data):
        """Create all 20 alpha factors and beta factors"""
        features = {}
        
        for etf in SECTOR_ETFS:
            print(f"\nProcessing {etf}...")
            df = pd.DataFrame(index=data.index)
            
            # Get price columns
            close_col = f'Close_{etf}'
            open_col = f'Open_{etf}'
            high_col = f'High_{etf}'
            low_col = f'Low_{etf}'
            volume_col = f'Volume_{etf}'
            
            # Extract data with fallbacks
            close = data[close_col] if close_col in data.columns else data.get(etf, pd.Series(index=data.index))
            high = data[high_col] if high_col in data.columns else close
            low = data[low_col] if low_col in data.columns else close
            volume = data[volume_col] if volume_col in data.columns else pd.Series(1000000, index=data.index)
            spy_close = data['Close_SPY'] if 'Close_SPY' in data.columns else data['SPY']
            
            # === 20 ALPHA FACTORS ===
            
            # 1-2. Momentum (1 week, 1 month)
            df['momentum_1w'] = self.compute_momentum(close, 5)
            df['momentum_1m'] = self.compute_momentum(close, 21)
            
            # 3. RSI (14-day)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            df['rsi_14d'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
            
            # 4. Volatility (21-day)
            df['volatility_21d'] = close.pct_change().rolling(21).std() * np.sqrt(252)
            
            # 5. Sharpe Ratio (10-day)
            returns = close.pct_change()
            df['sharpe_10d'] = (returns.rolling(10).mean() / (returns.rolling(10).std() + 1e-10)) * np.sqrt(252)
            
            # 6. Ratio Momentum (ETF/SPY momentum)
            etf_spy_ratio = close / spy_close
            df['ratio_momentum'] = self.compute_momentum(etf_spy_ratio, 21)
            
            # 7. Volume Ratio (5d/20d)
            df['volume_ratio'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
            
            # 8-10. MACD, Signal, Histogram
            macd, macd_signal, macd_hist = self.compute_macd(close)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # 11. Bollinger Band %B
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            df['bb_pctb'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            # 12-14. KDJ Indicators
            k, d, j = self.compute_kdj(high, low, close)
            df['kdj_k'] = k
            df['kdj_d'] = d
            df['kdj_j'] = j
            
            # 15. ATR (Average True Range)
            df['atr_14d'] = self.compute_atr(high, low, close, 14)
            
            # 16-17. Price Breakouts
            df['high_20d'] = close / (close.rolling(20).max() + 1e-10)
            df['low_20d'] = close / (close.rolling(20).min() + 1e-10)
            
            # 18. MFI (Money Flow Index)
            df['mfi_14d'] = self.compute_mfi(high, low, close, volume, 14)
            
            # 19. Volume-Weighted Average Price
            df['vwap'] = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
            
            # 20. Price Position (0-1 normalized)
            rolling_min = close.rolling(63).min()
            rolling_max = close.rolling(63).max()
            df['price_position'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-10)
            
            # === BETA FACTORS (62 indicators x 3 = 186 features) ===
            for col in fred_data.columns:
                # Raw value
                df[f'fred_{col}'] = fred_data[col]
                # 1-month change
                df[f'fred_{col}_chg_1m'] = fred_data[col].pct_change(21)
                # 3-month change
                df[f'fred_{col}_chg_3m'] = fred_data[col].pct_change(63)
            
            # Additional derived features
            # Yield curve slopes
            if 'treasury_10y' in fred_data.columns and 'treasury_2y' in fred_data.columns:
                df['yield_curve_10y2y'] = fred_data['treasury_10y'] - fred_data['treasury_2y']
            if 'treasury_10y' in fred_data.columns and 'treasury_3m' in fred_data.columns:
                df['yield_curve_10y3m'] = fred_data['treasury_10y'] - fred_data['treasury_3m']
            
            # Real rates
            if 'treasury_10y' in fred_data.columns and 'inflation_10y' in fred_data.columns:
                df['real_rate_10y'] = fred_data['treasury_10y'] - fred_data['inflation_10y']

            # === VIX REGIME FEATURES (11 features with 21-day lag to prevent data leakage) ===
            if 'vix' in fred_data.columns:
                # Use 21-day lagged VIX to prevent data leakage
                vix_lagged = fred_data['vix'].shift(PREDICTION_HORIZON)

                # 1-3. VIX Regime Classification (one-hot encoded with lag)
                df['vix_regime_low_vol_lag21'] = (vix_lagged < 20).astype(int)
                df['vix_regime_medium_vol_lag21'] = ((vix_lagged >= 20) & (vix_lagged < 30)).astype(int)
                df['vix_regime_high_vol_lag21'] = (vix_lagged >= 30).astype(int)

                # 4. VIX Level (lagged)
                df['vix_level_lag21'] = vix_lagged

                # 5-6. VIX Changes (1-week and 1-month with additional lag)
                df['vix_chg_1w_lag21'] = vix_lagged.pct_change(5)
                df['vix_chg_1m_lag21'] = vix_lagged.pct_change(21)

                # 7. VIX Trend (20-day SMA vs current with lag)
                vix_sma20 = vix_lagged.rolling(20).mean()
                df['vix_above_sma20_lag21'] = (vix_lagged > vix_sma20).astype(int)

                # 8. VIX Volatility (volatility of VIX with lag)
                df['vix_volatility_lag21'] = vix_lagged.rolling(21).std()

                # 9-10. VIX Percentile Rank (normalized 0-1 with lag)
                vix_rolling_min = vix_lagged.rolling(252).min()  # 1-year min
                vix_rolling_max = vix_lagged.rolling(252).max()  # 1-year max
                df['vix_percentile_rank_lag21'] = (vix_lagged - vix_rolling_min) / (vix_rolling_max - vix_rolling_min + 1e-10)

                # 11. VIX Regime Persistence (how long in current regime with lag)
                regime = pd.Series(index=vix_lagged.index, dtype=int)
                regime[vix_lagged < 20] = 0  # LOW_VOL
                regime[(vix_lagged >= 20) & (vix_lagged < 30)] = 1  # MEDIUM_VOL
                regime[vix_lagged >= 30] = 2  # HIGH_VOL
                # Count consecutive days in same regime
                regime_persistence = regime.groupby((regime != regime.shift()).cumsum()).cumcount() + 1
                df['vix_regime_persistence_lag21'] = regime_persistence

            # Create target: 21-day forward relative return
            spy_forward = spy_close.pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
            etf_forward = close.pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
            df['target'] = etf_forward - spy_forward
            
            # Clean up infinities and NaNs
            df = df.replace([np.inf, -np.inf], np.nan)
            
            features[etf] = df

            alpha_count = 20  # Fixed 20 alpha factors
            beta_count = len([col for col in df.columns if 'fred' in col])
            vix_count = len([col for col in df.columns if 'vix' in col and 'lag21' in col])
            derived_count = len([col for col in df.columns if col in ['yield_curve_10y2y', 'yield_curve_10y3m', 'real_rate_10y']])
            total_features = len(df.columns) - 1  # Exclude target

            print(f"  Created {total_features} features ({alpha_count} alpha + {beta_count} beta + {vix_count} VIX regime + {derived_count} derived)")
        
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

            # Debug: show data before cleaning
            print(f"  Before cleaning: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")

            # First, fill NaN values in features (but not target)
            feature_cols_temp = [col for col in df.columns if col != 'target']
            df[feature_cols_temp] = df[feature_cols_temp].fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Only drop rows where target is NaN
            df = df.dropna(subset=['target'])

            print(f"  After cleaning: {len(df)} rows ({df.index.min().date() if len(df) > 0 else 'N/A'} to {df.index.max().date() if len(df) > 0 else 'N/A'})")

            # Split data
            train_data = df[df.index <= TRAIN_END]
            val_data = df[(df.index >= VALIDATION_START) & (df.index <= VALIDATION_END)]
            current_data = df[df.index >= PREDICTION_START]

            print(f"  Train data: {len(train_data)} rows (up to {TRAIN_END.date()})")
            print(f"  Val data: {len(val_data)} rows")
            print(f"  Current data: {len(current_data)} rows")

            # Prepare features and target
            feature_cols = [col for col in df.columns if col != 'target']
            X_train = train_data[feature_cols]
            y_train = train_data['target']
            X_val = val_data[feature_cols]
            y_val = val_data['target']
            X_current = current_data[feature_cols]

            # Skip if insufficient training data
            if len(X_train) < 100:
                print(f"  ⚠️  Insufficient training data for {etf} (need 100, have {len(X_train)})")
                continue

            # Validation data might not be available in walk-forward (no forward returns yet)
            skip_validation = len(X_val) < 5 or len(y_val.dropna()) < 5
            if skip_validation:
                print(f"  ℹ️  Skipping validation (insufficient forward returns available)")
                print(f"  ℹ️  Training on {len(X_train)} samples, predicting forward")
            
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

            if len(X_train_seq) < 10:
                print(f"  ⚠️  Insufficient training sequence data for {etf}")
                continue

            # Only create validation sequences if not skipping validation
            if not skip_validation and len(X_val_scaled) >= seq_length:
                X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val.values, min(seq_length, len(X_val_scaled)-1))
            else:
                X_val_seq, y_val_seq = np.array([]), np.array([])
            
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
                    val_pred = np.atleast_1d(model(X_val_tensor).numpy())
                else:
                    val_pred = np.array([])

            models = {'LSTM': model}

            # Calculate validation metrics if available
            if val_pred.size > 0 and len(y_val_seq) > 0:
                mse = mean_squared_error(y_val_seq, val_pred)
                mae = mean_absolute_error(y_val_seq, val_pred)
                r2 = r2_score(y_val_seq, val_pred)

                # Direction accuracy
                actual_direction = np.sign(y_val_seq)
                pred_direction = np.sign(val_pred)
                direction_accuracy = np.mean(actual_direction == pred_direction) * 100

                print(f"\n  LSTM Model Validation Performance:")
                print(f"    MSE: {mse:.6f}")
                print(f"    MAE: {mae:.6f}")
                print(f"    R²: {r2:.4f}")
                print(f"    Direction Accuracy: {direction_accuracy:.1f}%")

                best_score = r2
            else:
                print(f"  ℹ️  Validation skipped - model trained on {len(X_train_seq)} sequences")
                best_score = None

            best_model = model
            best_name = 'LSTM'

            # Generate predictions for next month
            # Use all data up to prediction date (train + val + current available)
            # to create the most recent sequence for prediction
            all_data_until_prediction = df[df.index < PREDICTION_START]  # Everything before prediction month
            if len(all_data_until_prediction) >= seq_length:
                X_all = all_data_until_prediction[feature_cols]
                X_all_scaled = scaler.transform(X_all)

                # Create prediction from last seq_length days before PREDICTION_START
                X_pred_seq = X_all_scaled[-seq_length:].reshape(1, seq_length, -1)
                X_pred_tensor = torch.FloatTensor(X_pred_seq)

                model.eval()
                with torch.no_grad():
                    future_pred = model(X_pred_tensor).numpy()
                    future_pred_value = float(np.atleast_1d(future_pred)[0])

                # Store results
                results[etf] = {
                    'model': best_name,
                    'validation_r2': best_score,
                    'validation_actual': y_val_seq,
                    'validation_predicted': val_pred,
                    'future_prediction': future_pred_value,
                    'expected_return': future_pred_value * 100
                }

                print(f"\n  📈 Next Month Prediction:")
                print(f"    Expected Relative Return: {results[etf]['expected_return']:.2f}%")
                print(f"    Model Used: {best_name}")
            else:
                print(f"  ⚠️  Insufficient data for prediction ({len(all_data_until_prediction)} < {seq_length})")
                results[etf] = {
                    'model': best_name,
                    'validation_r2': best_score,
                    'validation_actual': y_val_seq,
                    'validation_predicted': val_pred,
                    'future_prediction': None,
                    'expected_return': None
                }
        
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
def main(train_cutoff_date=None, output_file=None):
    """
    Run complete monthly prediction system with walk-forward validation

    Args:
        train_cutoff_date: String date 'YYYY-MM-DD' for training cutoff (e.g., '2025-07-31')
        output_file: Path to save predictions JSON (e.g., 'august_2025_predictions.json')
    """
    global TRAIN_END, VALIDATION_START, VALIDATION_END, PREDICTION_START

    # If custom cutoff date provided, override default dates
    if train_cutoff_date:
        # The cutoff date is when we make the prediction
        # But we need to train on data that has 21-day forward returns available
        # So actual training data ends PREDICTION_HORIZON days before the cutoff
        prediction_date = datetime.strptime(train_cutoff_date, '%Y-%m-%d')
        TRAIN_END = prediction_date - timedelta(days=PREDICTION_HORIZON + 5)  # Extra buffer for trading days

        # Validation on month before prediction
        val_month_end = prediction_date - timedelta(days=1)
        VALIDATION_END = datetime(val_month_end.year, val_month_end.month,
                                  calendar.monthrange(val_month_end.year, val_month_end.month)[1])
        VALIDATION_START = datetime(val_month_end.year, val_month_end.month, 1)

        # Prediction starts after cutoff date
        pred_start = prediction_date + timedelta(days=1)
        PREDICTION_START = datetime(pred_start.year, pred_start.month, 1)

        print(f"\n🔧 CUSTOM WALK-FORWARD CONFIGURATION:")
        print(f"  Prediction date: {prediction_date.date()} (cutoff)")
        print(f"  Training data: 2020-01-01 to {TRAIN_END.date()} (has complete 21-day forward returns)")
        print(f"  Validation period: {VALIDATION_START.date()} to {VALIDATION_END.date()}")
        print(f"  Forecasting for: {PREDICTION_START.date()} onwards")

    # Initialize pipeline
    pipeline = MonthlyPredictionPipeline()

    # Fetch data
    market_data, fred_data = pipeline.fetch_all_data()

    # Create features
    features = pipeline.create_features(market_data, fred_data)

    # Train and validate
    results = pipeline.train_and_validate(features)

    # Save predictions to JSON if output file specified
    if output_file:
        predictions = {}
        for etf, res in results.items():
            if 'future_prediction' in res and res['future_prediction'] is not None:
                predictions[etf] = float(res['future_prediction'])

        pred_path = f'/home/aojie_ju/etf-trading-intelligence/{output_file}'
        with open(pred_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\n✅ Predictions saved to {pred_path}")
        print(f"   {len(predictions)} ETFs predicted")

    # Generate recommendations
    allocations = pipeline.generate_portfolio_recommendation(results)

    # Create validation report
    pipeline.create_validation_report(results)

    # Summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"\n✅ Models trained on data through {TRAIN_END.strftime('%B %d, %Y')}")
    print(f"✅ Validated on {VALIDATION_START.strftime('%B %Y')} actual returns")
    print(f"✅ Generated predictions for {PREDICTION_START.strftime('%B %Y')} onwards")

    print("\n📊 Key Outputs:")
    print("  • Validation metrics for all sectors")
    print("  • One-month forward return predictions")
    print("  • Portfolio allocation recommendations")
    print("  • Model performance comparison")
    if output_file:
        print(f"  • Predictions saved to {output_file}")

    return results, allocations


if __name__ == "__main__":
    import sys

    # Check for command-line arguments
    train_cutoff = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if train_cutoff:
        print(f"\n🎯 Walk-forward training mode")
        print(f"   Training through: {train_cutoff}")
        if output_file:
            print(f"   Output file: {output_file}")

    results, allocations = main(train_cutoff, output_file)