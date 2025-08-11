"""
Complete Multi-Sector ETF Trading System
With Advanced Models and Stochastic Processes
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data and processing
import requests
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

# Stochastic processes and signal processing
from scipy import stats, signal
from scipy.fft import fft, ifft, fftfreq
from scipy.optimize import minimize
import pywt  # Wavelet transforms
from arch import arch_model  # GARCH models

# Machine Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================== CONFIGURATION ======================

FRED_API_KEY = "ccf75f3e8501e936dafd9f3e77729525"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
SECTOR_NAMES = {
    'XLF': 'Financials', 'XLC': 'Communication', 'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples', 'XLE': 'Energy', 'XLV': 'Healthcare',
    'XLI': 'Industrials', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLK': 'Technology', 'XLU': 'Utilities'
}

# ====================== STOCHASTIC PROCESSES ======================

class StochasticFeatureEngineering:
    """
    Advanced feature engineering with stochastic processes
    """
    
    def __init__(self):
        self.garch_models = {}
        self.ou_params = {}
        
    def fit_garch(self, returns: pd.Series, p=1, q=1) -> Dict:
        """
        Fit GARCH(p,q) model for volatility forecasting
        """
        # Scale returns to percentage
        returns_pct = returns * 100
        
        # Fit GARCH model
        model = arch_model(returns_pct.dropna(), vol='Garch', p=p, q=q)
        model_fit = model.fit(disp='off')
        
        # Extract parameters
        params = {
            'omega': model_fit.params['omega'],
            'alpha': model_fit.params[f'alpha[1]'],
            'beta': model_fit.params[f'beta[1]'],
            'conditional_volatility': model_fit.conditional_volatility / 100
        }
        
        # Forecast next period volatility
        forecast = model_fit.forecast(horizon=5)
        params['volatility_forecast'] = np.sqrt(forecast.variance.values[-1, :]) / 100
        
        return params
    
    def fit_ornstein_uhlenbeck(self, prices: pd.Series) -> Dict:
        """
        Fit Ornstein-Uhlenbeck process for mean reversion
        dX_t = θ(μ - X_t)dt + σdW_t
        """
        log_prices = np.log(prices)
        
        # Estimate parameters using MLE
        n = len(log_prices)
        dt = 1/252  # Daily data
        
        Sx = np.sum(log_prices[:-1])
        Sy = np.sum(log_prices[1:])
        Sxx = np.sum(log_prices[:-1]**2)
        Sxy = np.sum(log_prices[:-1] * log_prices[1:])
        Syy = np.sum(log_prices[1:]**2)
        
        mu = (Sy * Sxx - Sx * Sxy) / (n * Sxx - Sx**2)
        theta = -np.log((Sxy - mu * Sx - mu * Sy + n * mu**2) / 
                       (Sxx - 2 * mu * Sx + n * mu**2)) / dt
        
        a = np.exp(-theta * dt)
        sigmah2 = (Syy - 2 * a * Sxy + a**2 * Sxx - 2 * mu * (1 - a) * 
                  (Sy - a * Sx) + n * mu**2 * (1 - a)**2) / n
        sigma = np.sqrt(sigmah2 * 2 * theta / (1 - a**2))
        
        # Half-life of mean reversion
        half_life = np.log(2) / theta
        
        return {
            'mu': mu,
            'theta': theta,
            'sigma': sigma,
            'half_life': half_life,
            'mean_reversion_speed': theta,
            'long_term_mean': mu
        }
    
    def calculate_fft_features(self, prices: pd.Series, n_components: int = 10) -> pd.DataFrame:
        """
        Extract frequency domain features using FFT
        """
        # Remove trend
        detrended = signal.detrend(prices.dropna().values)
        
        # Apply FFT
        fft_vals = fft(detrended)
        fft_freq = fftfreq(len(detrended))
        
        # Get power spectrum
        power = np.abs(fft_vals)**2
        
        # Find dominant frequencies
        idx = np.argsort(power)[::-1][:n_components]
        
        features = pd.DataFrame(index=prices.index)
        
        # Extract top frequency components
        for i, j in enumerate(idx):
            features[f'fft_freq_{i}'] = fft_freq[j]
            features[f'fft_power_{i}'] = power[j]
            
            # Reconstruct signal from this component
            component = np.zeros_like(fft_vals)
            component[j] = fft_vals[j]
            component[-j] = fft_vals[-j]  # Symmetric component
            reconstructed = np.real(ifft(component))
            
            # Pad to match original length
            padded = np.full(len(prices), np.nan)
            padded[len(padded)-len(reconstructed):] = reconstructed
            features[f'fft_component_{i}'] = padded
        
        return features
    
    def calculate_wavelet_features(self, prices: pd.Series, wavelet: str = 'db4', levels: int = 3) -> pd.DataFrame:
        """
        Wavelet decomposition for multi-scale analysis
        """
        features = pd.DataFrame(index=prices.index)
        
        # Handle missing values
        clean_prices = prices.dropna()
        if len(clean_prices) < 20:
            return features
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(clean_prices.values, wavelet, level=levels)
            
            # Calculate statistical features from coefficients
            for i, coeff in enumerate(coeffs):
                if i == 0:
                    feature_prefix = 'wavelet_approx'
                else:
                    feature_prefix = f'wavelet_detail_{i}'
                
                # Energy (as a single value, repeated for all rows)
                energy = np.sum(coeff**2) / len(coeff)
                features[f'{feature_prefix}_energy'] = energy
                
                # Standard deviation
                features[f'{feature_prefix}_std'] = np.std(coeff)
                
                # Max absolute value
                features[f'{feature_prefix}_max'] = np.max(np.abs(coeff))
            
            # Calculate relative energy
            total_energy = sum(np.sum(c**2) for c in coeffs)
            for i, coeff in enumerate(coeffs):
                level_energy = np.sum(coeff**2)
                features[f'wavelet_rel_energy_{i}'] = level_energy / total_energy if total_energy > 0 else 0
                
        except Exception as e:
            print(f"Warning: Wavelet transform failed: {e}")
        
        return features
    
    def detect_regime_switches(self, returns: pd.Series, n_regimes: int = 3) -> pd.Series:
        """
        Detect market regimes using Hidden Markov Model approach
        """
        # Simple regime detection using rolling statistics
        vol = returns.rolling(20).std()
        
        # Define regimes based on volatility percentiles
        low_vol = vol.quantile(0.33)
        high_vol = vol.quantile(0.67)
        
        regimes = pd.Series(index=returns.index, dtype=int)
        regimes[vol <= low_vol] = 0  # Low volatility regime
        regimes[(vol > low_vol) & (vol <= high_vol)] = 1  # Normal regime
        regimes[vol > high_vol] = 2  # High volatility regime
        
        return regimes.fillna(1)  # Default to normal regime
    
    def calculate_jump_diffusion_params(self, returns: pd.Series, threshold: float = 3) -> Dict:
        """
        Estimate jump diffusion parameters
        """
        # Identify jumps (returns beyond threshold standard deviations)
        std = returns.std()
        mean = returns.mean()
        
        jumps = np.abs(returns - mean) > threshold * std
        
        # Jump parameters
        jump_intensity = jumps.sum() / len(returns)  # Lambda (jump frequency)
        jump_returns = returns[jumps]
        
        if len(jump_returns) > 0:
            jump_mean = jump_returns.mean()
            jump_std = jump_returns.std()
        else:
            jump_mean = 0
            jump_std = 0
        
        # Diffusion parameters (excluding jumps)
        normal_returns = returns[~jumps]
        diffusion_mean = normal_returns.mean()
        diffusion_std = normal_returns.std()
        
        return {
            'jump_intensity': jump_intensity,
            'jump_mean': jump_mean,
            'jump_std': jump_std,
            'diffusion_mean': diffusion_mean,
            'diffusion_std': diffusion_std,
            'jump_ratio': jumps.sum() / len(returns)
        }


# ====================== ADVANCED NEURAL NETWORKS ======================

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for interpretable multi-horizon forecasting
    Simplified version focusing on key components
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Variable selection network
        self.variable_selection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, dropout=dropout, batch_first=True)
        
        # Gated Residual Network
        self.grn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Dropout(dropout),
            nn.GLU(dim=-1)
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Variable selection
        weights = self.variable_selection(x)
        x = x * weights
        
        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Gated residual connection
        grn_out = self.grn(attn_out)
        
        # Use last timestep for prediction
        final_hidden = grn_out[:, -1, :]
        
        return self.output_layer(final_hidden)


class NBeatsBlock(nn.Module):
    """
    N-BEATS block for neural basis expansion
    """
    
    def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden_dim, theta_dim)  # Backcast
        self.theta_f = nn.Linear(hidden_dim, theta_dim)  # Forecast
        
    def forward(self, x):
        x = self.layers(x)
        return self.theta_b(x), self.theta_f(x)


class NBeats(nn.Module):
    """
    N-BEATS: Neural Basis Expansion Analysis for Time Series
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, stack_types: List[str] = ['trend', 'seasonality', 'generic']):
        super().__init__()
        
        self.stacks = nn.ModuleList()
        
        for stack_type in stack_types:
            if stack_type == 'trend':
                theta_dim = 3  # Polynomial basis
            elif stack_type == 'seasonality':
                theta_dim = 8  # Fourier basis
            else:  # generic
                theta_dim = input_dim
            
            self.stacks.append(NBeatsBlock(input_dim, theta_dim))
        
        self.final_layer = nn.Linear(len(stack_types) * input_dim, output_dim)
        
    def forward(self, x):
        # Flatten time series input
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        stack_outputs = []
        residual = x
        
        for stack in self.stacks:
            backcast, forecast = stack(residual)
            residual = residual - backcast
            stack_outputs.append(forecast)
        
        # Combine forecasts
        combined = torch.cat(stack_outputs, dim=1)
        return self.final_layer(combined)


class LSTMGARCHHybrid(nn.Module):
    """
    LSTM with GARCH volatility modeling
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # LSTM for return prediction
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Separate paths for mean and volatility
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Predict mean and volatility
        mean = self.mean_head(last_hidden)
        vol = self.vol_head(last_hidden)
        
        return mean, vol


class WaveletLSTM(nn.Module):
    """
    LSTM with wavelet decomposition
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_scales: int = 4):
        super().__init__()
        
        # Separate LSTM for each wavelet scale
        self.scale_lstms = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim // n_scales, batch_first=True)
            for _ in range(n_scales)
        ])
        
        # Combine all scales
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_multi_scale):
        # x_multi_scale: list of tensors, one for each wavelet scale
        scale_outputs = []
        
        for scale_data, lstm in zip(x_multi_scale, self.scale_lstms):
            out, _ = lstm(scale_data)
            scale_outputs.append(out[:, -1, :])
        
        # Concatenate all scales
        combined = torch.cat(scale_outputs, dim=1)
        return self.combine(combined)


# ====================== DATA PIPELINE ======================

class MultiSectorDataPipeline:
    """
    Complete data pipeline for all 11 sectors
    """
    
    def __init__(self):
        self.stochastic = StochasticFeatureEngineering()
        self.sector_data = {}
        self.fred_data = None
        
    def fetch_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all ETF and SPY data
        """
        print("Fetching market data...")
        symbols = SECTOR_ETFS + ['SPY']
        
        # Download all at once for efficiency
        data = yf.download(symbols, start=start_date, end=end_date, 
                          auto_adjust=False, group_by='ticker', progress=True)
        
        # Process each symbol
        processed_data = {}
        for symbol in symbols:
            if symbol in data.columns.levels[0]:
                df = data[symbol].copy()
                df.columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']
                processed_data[symbol] = df
        
        return processed_data
    
    def fetch_fred_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch all FRED economic indicators
        """
        print("Fetching FRED data...")
        
        # Beta factors to fetch
        indicators = [
            # Interest rates
            'DFF', 'DGS2', 'DGS10', 'DGS30', 'T10Y2Y',
            # Economic indicators  
            'UNRATE', 'CPIAUCSL', 'GDP', 'INDPRO',
            # Market indicators
            'VIXCLS', 'DEXUSEU',
            # Commodities
            'DCOILWTICO', 'GOLDAMGBD228NLBM',
            # Financial conditions
            'NFCI', 'ANFCI',
            # Money supply
            'M2SL', 'WALCL'
        ]
        
        fred_data = {}
        for indicator in tqdm(indicators, desc="Fetching FRED indicators"):
            params = {
                'series_id': indicator,
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }
            
            try:
                response = requests.get(FRED_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                observations = data.get('observations', [])
                if observations:
                    df = pd.DataFrame(observations)
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df.set_index('date', inplace=True)
                    fred_data[indicator] = df['value']
            except:
                pass
        
        # Combine all indicators
        if fred_data:
            combined = pd.DataFrame(fred_data)
            # Forward fill missing values
            combined = combined.fillna(method='ffill').fillna(method='bfill')
            return combined
        
        return pd.DataFrame()
    
    def create_alpha_factors(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Create all alpha factors for a sector
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        close = df[f'{symbol}_AdjClose']
        high = df[f'{symbol}_High']
        low = df[f'{symbol}_Low']
        volume = df[f'{symbol}_Volume']
        
        # 1. Momentum features
        for period in [5, 10, 21, 63]:
            features[f'{symbol}_momentum_{period}d'] = close.pct_change(period)
            features[f'{symbol}_momentum_{period}d_rank'] = features[f'{symbol}_momentum_{period}d'].rolling(252).rank(pct=True)
        
        # 2. Volatility features
        returns = close.pct_change()
        for period in [5, 21, 63]:
            features[f'{symbol}_volatility_{period}d'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Volatility ratios (after all volatility features are computed)
        for period in [5, 63]:
            if f'{symbol}_volatility_{period}d' in features.columns and f'{symbol}_volatility_21d' in features.columns:
                features[f'{symbol}_volatility_ratio_{period}d'] = features[f'{symbol}_volatility_{period}d'] / features[f'{symbol}_volatility_21d']
        
        # 3. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features[f'{symbol}_rsi_14d'] = 100 - (100 / (1 + rs))
        
        # 4. Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        features[f'{symbol}_bb_upper'] = sma + 2 * std
        features[f'{symbol}_bb_lower'] = sma - 2 * std
        features[f'{symbol}_bb_position'] = (close - features[f'{symbol}_bb_lower']) / (features[f'{symbol}_bb_upper'] - features[f'{symbol}_bb_lower'])
        
        # 5. MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features[f'{symbol}_macd'] = ema12 - ema26
        features[f'{symbol}_macd_signal'] = features[f'{symbol}_macd'].ewm(span=9, adjust=False).mean()
        features[f'{symbol}_macd_hist'] = features[f'{symbol}_macd'] - features[f'{symbol}_macd_signal']
        
        # 6. Volume features
        features[f'{symbol}_volume_ratio'] = volume / volume.rolling(20).mean()
        features[f'{symbol}_volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        # Money Flow Index
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(14).sum()
        negative_flow_sum = negative_flow.rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_flow_sum / (negative_flow_sum + 1e-10)))
        features[f'{symbol}_mfi'] = mfi
        
        # 7. Price patterns
        features[f'{symbol}_high_low_ratio'] = high / low
        features[f'{symbol}_close_open_ratio'] = close / df[f'{symbol}_Open']
        
        # 8. Support/Resistance
        features[f'{symbol}_distance_from_high'] = close / close.rolling(252).max() - 1
        features[f'{symbol}_distance_from_low'] = close / close.rolling(252).min() - 1
        
        # 9. Relative to SPY
        spy_returns = df['SPY_AdjClose'].pct_change()
        etf_returns = close.pct_change()
        
        features[f'{symbol}_relative_strength'] = etf_returns - spy_returns
        features[f'{symbol}_relative_strength_ma'] = features[f'{symbol}_relative_strength'].rolling(20).mean()
        features[f'{symbol}_beta'] = etf_returns.rolling(60).cov(spy_returns) / spy_returns.rolling(60).var()
        
        # 10. Sharpe ratio
        features[f'{symbol}_sharpe'] = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        
        return features
    
    def add_stochastic_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add stochastic process features
        """
        close = df[f'{symbol}_AdjClose']
        returns = close.pct_change()
        
        # 1. GARCH features
        try:
            garch_params = self.stochastic.fit_garch(returns.dropna())
            df[f'{symbol}_garch_volatility'] = garch_params['conditional_volatility']
            df[f'{symbol}_garch_forecast_1d'] = garch_params['volatility_forecast'][0] if len(garch_params['volatility_forecast']) > 0 else np.nan
        except:
            df[f'{symbol}_garch_volatility'] = returns.rolling(20).std()
            df[f'{symbol}_garch_forecast_1d'] = df[f'{symbol}_garch_volatility']
        
        # 2. Ornstein-Uhlenbeck features
        try:
            ou_params = self.stochastic.fit_ornstein_uhlenbeck(close.dropna())
            df[f'{symbol}_ou_half_life'] = ou_params['half_life']
            df[f'{symbol}_ou_mean_reversion'] = ou_params['mean_reversion_speed']
            df[f'{symbol}_ou_distance_from_mean'] = np.log(close) - ou_params['mu']
        except:
            df[f'{symbol}_ou_half_life'] = np.nan
            df[f'{symbol}_ou_mean_reversion'] = np.nan
            df[f'{symbol}_ou_distance_from_mean'] = np.nan
        
        # 3. FFT features
        fft_features = self.stochastic.calculate_fft_features(close, n_components=5)
        for col in fft_features.columns:
            df[f'{symbol}_{col}'] = fft_features[col]
        
        # 4. Wavelet features
        wavelet_features = self.stochastic.calculate_wavelet_features(close, levels=3)
        for col in wavelet_features.columns:
            df[f'{symbol}_{col}'] = wavelet_features[col]
        
        # 5. Regime detection
        df[f'{symbol}_regime'] = self.stochastic.detect_regime_switches(returns)
        
        # 6. Jump diffusion parameters
        jump_params = self.stochastic.calculate_jump_diffusion_params(returns.dropna())
        df[f'{symbol}_jump_intensity'] = jump_params['jump_intensity']
        df[f'{symbol}_jump_ratio'] = jump_params['jump_ratio']
        
        return df
    
    def create_sector_dataframes(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Create complete dataframes for all sectors
        """
        print("\n" + "="*60)
        print("CREATING MULTI-SECTOR DATAFRAMES")
        print("="*60)
        
        # Fetch market data
        market_data = self.fetch_market_data(start_date, end_date)
        
        # Fetch FRED data
        self.fred_data = self.fetch_fred_data(start_date, end_date)
        
        # Process each sector
        sector_dataframes = {}
        
        for etf in tqdm(SECTOR_ETFS, desc="Processing sectors"):
            print(f"\nProcessing {etf} ({SECTOR_NAMES[etf]})...")
            
            # Create base dataframe
            df = pd.DataFrame(index=market_data['SPY'].index)
            
            # Add SPY data
            for col in ['AdjClose', 'Volume', 'High', 'Low', 'Open']:
                df[f'SPY_{col}'] = market_data['SPY'][col] if col in market_data['SPY'].columns else market_data['SPY']['Adj Close']
            
            # Add sector ETF data
            for col in ['AdjClose', 'Volume', 'High', 'Low', 'Open']:
                df[f'{etf}_{col}'] = market_data[etf][col] if col in market_data[etf].columns else market_data[etf]['Adj Close']
            
            # Add alpha factors
            alpha_features = self.create_alpha_factors(df, etf)
            df = pd.concat([df, alpha_features], axis=1)
            
            # Add stochastic features
            df = self.add_stochastic_features(df, etf)
            
            # Add FRED beta factors
            if not self.fred_data.empty:
                fred_aligned = self.fred_data.reindex(df.index, method='ffill')
                df = pd.concat([df, fred_aligned], axis=1)
            
            # Create target variable (5-day forward relative return)
            spy_returns = df['SPY_AdjClose'].pct_change(5).shift(-5)
            etf_returns = df[f'{etf}_AdjClose'].pct_change(5).shift(-5)
            df[f'{etf}_target'] = etf_returns - spy_returns
            
            # Store
            sector_dataframes[f'SPY_{etf}'] = df
            
            # Report
            alpha_cols = [c for c in df.columns if etf in c and c not in [f'{etf}_AdjClose', f'{etf}_Volume', f'{etf}_High', f'{etf}_Low', f'{etf}_Open', f'{etf}_target']]
            beta_cols = [c for c in self.fred_data.columns if c in df.columns] if not self.fred_data.empty else []
            
            print(f"  ✓ Created {len(alpha_cols)} alpha factors")
            print(f"  ✓ Added {len(beta_cols)} beta factors")
            print(f"  ✓ Total features: {len(df.columns)}")
        
        return sector_dataframes


# ====================== MODEL TRAINING ======================

class MultiSectorModelTrainer:
    """
    Train models for all sectors
    """
    
    def __init__(self, model_type: str = 'tft'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, sector: str, sequence_length: int = 30):
        """
        Prepare data for training
        """
        # Get feature columns
        feature_cols = [c for c in df.columns if c != f'{sector}_target' and 'target' not in c]
        
        # Get features and target
        X = df[feature_cols].fillna(method='ffill').fillna(0)
        y = df[f'{sector}_target'].fillna(0)
        
        # Remove inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        X_test = X.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[sector] = scaler
        
        return (X_train_scaled, y_train.values, 
                X_val_scaled, y_val.values,
                X_test_scaled, y_test.values)
    
    def create_model(self, input_dim: int):
        """
        Create model based on specified type
        """
        if self.model_type == 'tft':
            return TemporalFusionTransformer(input_dim)
        elif self.model_type == 'nbeats':
            return NBeats(input_dim)
        elif self.model_type == 'lstm_garch':
            return LSTMGARCHHybrid(input_dim)
        elif self.model_type == 'wavelet_lstm':
            return WaveletLSTM(input_dim)
        else:
            # Default to TFT
            return TemporalFusionTransformer(input_dim)
    
    def train_sector_model(self, df: pd.DataFrame, sector: str, epochs: int = 50):
        """
        Train model for a single sector
        """
        print(f"\nTraining model for {sector}...")
        
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df, sector)
        
        # Create sequences for LSTM-based models
        sequence_length = min(30, len(X_train) // 10)
        
        # Create model
        input_dim = X_train.shape[1]
        model = self.create_model(input_dim)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Convert to tensors
        X_train_seq = self._create_sequences(X_train, sequence_length)
        y_train_seq = y_train[sequence_length:]
        X_val_seq = self._create_sequences(X_val, sequence_length)
        y_val_seq = y_val[sequence_length:]
        
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.FloatTensor(y_val_seq)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            if self.model_type == 'lstm_garch':
                mean_pred, vol_pred = model(X_train_tensor)
                # Custom loss for GARCH model
                mse_loss = criterion(mean_pred.squeeze(), y_train_tensor)
                # Add negative log-likelihood for volatility
                nll_loss = torch.mean(torch.log(vol_pred) + (y_train_tensor - mean_pred.squeeze())**2 / (2 * vol_pred.squeeze()**2))
                loss = mse_loss + 0.1 * nll_loss
            else:
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if self.model_type == 'lstm_garch':
                    val_mean, val_vol = model(X_val_tensor)
                    val_loss = criterion(val_mean.squeeze(), y_val_tensor)
                else:
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                break
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Test evaluation
        X_test_seq = self._create_sequences(X_test, sequence_length)
        y_test_seq = y_test[sequence_length:]
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.FloatTensor(y_test_seq)
        
        model.eval()
        with torch.no_grad():
            if self.model_type == 'lstm_garch':
                test_mean, test_vol = model(X_test_tensor)
                test_predictions = test_mean.squeeze().numpy()
                test_volatility = test_vol.squeeze().numpy()
            else:
                test_outputs = model(X_test_tensor)
                test_predictions = test_outputs.squeeze().numpy()
                test_volatility = None
        
        # Calculate metrics
        mse = np.mean((test_predictions - y_test_seq)**2)
        mae = np.mean(np.abs(test_predictions - y_test_seq))
        direction_accuracy = np.mean(np.sign(test_predictions) == np.sign(y_test_seq))
        
        # Store results
        self.models[sector] = model
        self.results[sector] = {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'predictions': test_predictions,
            'actuals': y_test_seq,
            'volatility': test_volatility
        }
        
        print(f"  ✓ {sector} - MSE: {mse:.6f}, MAE: {mae:.6f}, Direction: {direction_accuracy:.2%}")
        
        return model
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Create sequences for time series models
        """
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
        return np.array(sequences)
    
    def train_all_sectors(self, sector_dataframes: Dict[str, pd.DataFrame]):
        """
        Train models for all sectors
        """
        print("\n" + "="*60)
        print(f"TRAINING {self.model_type.upper()} MODELS FOR ALL SECTORS")
        print("="*60)
        
        for sector_name, df in sector_dataframes.items():
            sector = sector_name.replace('SPY_', '')
            self.train_sector_model(df, sector)
        
        return self.models, self.results


# ====================== PORTFOLIO OPTIMIZATION ======================

class PortfolioOptimizer:
    """
    Multi-sector portfolio optimization
    """
    
    def __init__(self):
        self.weights = {}
        self.allocations = {}
        
    def optimize_mean_variance(self, predictions: Dict[str, float], 
                              covariance: pd.DataFrame,
                              risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Mean-variance optimization
        """
        sectors = list(predictions.keys())
        n = len(sectors)
        
        # Expected returns
        mu = np.array([predictions[s] for s in sectors])
        
        # Covariance matrix
        sigma = covariance.loc[sectors, sectors].values
        
        # Optimization
        def objective(w):
            portfolio_return = np.dot(w, mu)
            portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
            return -(portfolio_return - risk_aversion * portfolio_risk)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds (0 to 1 for each weight)
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess (equal weights)
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(sectors, result.x))
        else:
            # Fall back to equal weights
            weights = dict(zip(sectors, [1/n]*n))
        
        return weights
    
    def risk_parity(self, covariance: pd.DataFrame) -> Dict[str, float]:
        """
        Risk parity allocation
        """
        sectors = covariance.columns.tolist()
        n = len(sectors)
        
        # Inverse volatility weighting (simplified risk parity)
        volatilities = np.sqrt(np.diag(covariance.values))
        inv_vols = 1 / volatilities
        weights = inv_vols / inv_vols.sum()
        
        return dict(zip(sectors, weights))
    
    def kelly_criterion(self, predictions: Dict[str, float],
                       volatilities: Dict[str, float],
                       confidence: float = 0.25) -> Dict[str, float]:
        """
        Kelly criterion for position sizing
        """
        weights = {}
        
        for sector in predictions:
            if sector in volatilities and volatilities[sector] > 0:
                # Kelly fraction = expected return / variance
                kelly = predictions[sector] / (volatilities[sector]**2)
                # Apply confidence scaling and cap
                weights[sector] = min(max(kelly * confidence, 0), 0.25)
            else:
                weights[sector] = 0
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            # Equal weights if all zero
            weights = {k: 1/len(predictions) for k in predictions}
        
        return weights


# ====================== BACKTESTING ======================

class MultiSectorBacktester:
    """
    Backtest multi-sector portfolio strategy
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def backtest_portfolio(self, 
                          predictions: Dict[str, np.ndarray],
                          actual_prices: Dict[str, pd.Series],
                          weights_history: List[Dict[str, float]],
                          transaction_cost: float = 0.001) -> Dict:
        """
        Backtest portfolio strategy
        """
        n_periods = min(len(pred) for pred in predictions.values())
        
        # Initialize tracking
        portfolio_value = [self.initial_capital]
        positions = {sector: 0 for sector in predictions.keys()}
        cash = self.initial_capital
        
        for t in range(n_periods):
            # Get current weights
            if t < len(weights_history):
                target_weights = weights_history[t]
            else:
                target_weights = weights_history[-1]
            
            # Current portfolio value
            current_value = cash
            for sector in positions:
                if positions[sector] > 0 and t < len(actual_prices[sector]):
                    current_value += positions[sector] * actual_prices[sector].iloc[t]
            
            # Rebalance portfolio
            for sector in predictions.keys():
                if t < len(actual_prices[sector]):
                    target_value = current_value * target_weights.get(sector, 0)
                    current_position_value = positions[sector] * actual_prices[sector].iloc[t] if positions[sector] > 0 else 0
                    
                    # Trade amount
                    trade_value = target_value - current_position_value
                    
                    if abs(trade_value) > current_value * 0.01:  # Only trade if > 1% of portfolio
                        # Update positions
                        shares_to_trade = trade_value / actual_prices[sector].iloc[t]
                        positions[sector] += shares_to_trade
                        
                        # Update cash (including transaction costs)
                        cash -= trade_value * (1 + transaction_cost * np.sign(trade_value))
            
            # Track portfolio value
            portfolio_value.append(current_value)
        
        # Calculate metrics
        returns = pd.Series(portfolio_value).pct_change().dropna()
        
        metrics = {
            'total_return': (portfolio_value[-1] / self.initial_capital - 1) * 100,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_value),
            'volatility': returns.std() * np.sqrt(252),
            'portfolio_values': portfolio_value
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """
        Calculate maximum drawdown
        """
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        return max_dd * 100


# ====================== MAIN EXECUTION ======================

def main():
    """
    Complete multi-sector ETF trading system execution
    """
    print("\n" + "="*80)
    print("COMPLETE MULTI-SECTOR ETF TRADING SYSTEM")
    print("With Advanced Models and Stochastic Processes")
    print("="*80)
    
    # Configuration
    START_DATE = '2019-01-01'
    END_DATE = '2024-12-01'
    MODEL_TYPE = 'tft'  # Options: 'tft', 'nbeats', 'lstm_garch', 'wavelet_lstm'
    
    # Step 1: Data Pipeline
    print("\n📊 STEP 1: DATA PIPELINE")
    pipeline = MultiSectorDataPipeline()
    sector_dataframes = pipeline.create_sector_dataframes(START_DATE, END_DATE)
    
    # Step 2: Model Training
    print("\n🤖 STEP 2: MODEL TRAINING")
    trainer = MultiSectorModelTrainer(model_type=MODEL_TYPE)
    models, results = trainer.train_all_sectors(sector_dataframes)
    
    # Step 3: Portfolio Optimization
    print("\n💼 STEP 3: PORTFOLIO OPTIMIZATION")
    optimizer = PortfolioOptimizer()
    
    # Get predictions for all sectors
    all_predictions = {}
    all_volatilities = {}
    
    for sector in SECTOR_ETFS:
        if sector in results:
            all_predictions[sector] = results[sector]['predictions'].mean()
            all_volatilities[sector] = results[sector]['predictions'].std()
    
    # Calculate correlation matrix from historical returns
    returns_df = pd.DataFrame()
    for sector in SECTOR_ETFS:
        df = sector_dataframes[f'SPY_{sector}']
        returns_df[sector] = df[f'{sector}_AdjClose'].pct_change()
    
    covariance = returns_df.cov()
    
    # Optimize portfolio
    print("\nOptimizing portfolio weights...")
    mv_weights = optimizer.optimize_mean_variance(all_predictions, covariance)
    rp_weights = optimizer.risk_parity(covariance)
    kelly_weights = optimizer.kelly_criterion(all_predictions, all_volatilities)
    
    print("\nOptimal Portfolio Weights:")
    print("-" * 40)
    print("Sector       | Mean-Var | Risk-Par | Kelly")
    print("-" * 40)
    for sector in SECTOR_ETFS:
        print(f"{sector:12} | {mv_weights.get(sector, 0):.3f}    | "
              f"{rp_weights.get(sector, 0):.3f}    | {kelly_weights.get(sector, 0):.3f}")
    
    # Step 4: Backtesting
    print("\n📈 STEP 4: BACKTESTING")
    backtester = MultiSectorBacktester()
    
    # Prepare data for backtesting
    actual_prices = {}
    predictions_series = {}
    
    for sector in SECTOR_ETFS:
        df = sector_dataframes[f'SPY_{sector}']
        actual_prices[sector] = df[f'{sector}_AdjClose']
        if sector in results:
            predictions_series[sector] = results[sector]['predictions']
    
    # Backtest with mean-variance weights
    backtest_results = backtester.backtest_portfolio(
        predictions_series,
        actual_prices,
        [mv_weights] * 100,  # Constant weights for simplicity
        transaction_cost=0.001
    )
    
    # Step 5: Results Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 Model Performance by Sector:")
    print("-" * 50)
    print("Sector       | MSE      | MAE      | Direction")
    print("-" * 50)
    for sector in SECTOR_ETFS:
        if sector in results:
            r = results[sector]
            print(f"{SECTOR_NAMES[sector]:12} | {r['mse']:.6f} | {r['mae']:.6f} | {r['direction_accuracy']:.2%}")
    
    print("\n💼 Portfolio Performance:")
    print("-" * 50)
    print(f"Total Return:    {backtest_results['total_return']:.2f}%")
    print(f"Sharpe Ratio:    {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {backtest_results['max_drawdown']:.2f}%")
    print(f"Volatility:      {backtest_results['volatility']:.2%}")
    
    print("\n✅ System Complete!")
    print("="*80)
    
    return sector_dataframes, models, results, backtest_results


if __name__ == "__main__":
    sector_dataframes, models, results, backtest_results = main()