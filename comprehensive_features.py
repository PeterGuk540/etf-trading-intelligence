"""
Comprehensive Feature Engineering Module
Implements all 20 alpha factors and 62 beta factors from Data_Generation.ipynb
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFeatureEngine:
    """Complete feature extraction matching Data_Generation.ipynb"""
    
    def __init__(self, fred_api_key="ccf75f3e8501e936dafd9f3e77729525"):
        self.fred_api_key = fred_api_key
        
        # Complete list of 62 FRED beta factors from Data_Generation.ipynb
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
            'CAPACITY': 'capacity_utilization',
            'RETAILSL': 'retail_sales',
            'HOUST': 'housing_starts',
            'PERMIT': 'building_permits',
            'AUTOSOLD': 'auto_sales',
            'DEXPORT': 'exports',
            'DIMPORT': 'imports',
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
            'PPIFGS': 'ppi_finished_goods',
            'GASREGW': 'gas_price',
            'DCOILWTICO': 'oil_wti',
            'DCOILBRENTEU': 'oil_brent',
            'GOLDAMGBD228NLBM': 'gold',
            
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
            'DXY': 'dollar_index',
            
            # Consumer & Business Sentiment (5 indicators)
            'UMCSENT': 'consumer_sentiment',
            'CBCCI': 'consumer_confidence',
            'USSLIND': 'leading_index',
            'USCCCI': 'coincident_index',
            'OEMV': 'business_optimism'
        }
        
        # ETF sectors
        self.sector_etfs = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
    
    def compute_momentum(self, close_series, window, skip=0):
        """Compute momentum as defined in Data_Generation.ipynb"""
        return close_series.shift(skip) / close_series.shift(skip + window) - 1
    
    def compute_rsi(self, close_series, period=14):
        """Compute RSI"""
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def compute_macd(self, close_series, fast=12, slow=26, signal=9):
        """Compute MACD, MACD signal, and MACD histogram"""
        exp1 = close_series.ewm(span=fast, adjust=False).mean()
        exp2 = close_series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def compute_bollinger_bands(self, close_series, period=20, std_dev=2):
        """Compute Bollinger Bands and %B"""
        sma = close_series.rolling(period).mean()
        std = close_series.rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        pct_b = (close_series - lower) / (upper - lower)
        return upper, lower, pct_b
    
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
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def fetch_market_data(self, start_date='2019-01-01', end_date=None):
        """Fetch all ETF and SPY data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("📊 Fetching market data...")
        all_tickers = self.sector_etfs + ['SPY']
        data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)
        
        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            # Create proper column names
            new_columns = []
            for col in data.columns:
                if col[0] in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col[1])
            data.columns = new_columns
        
        print(f"✓ Market data shape: {data.shape}")
        return data
    
    def fetch_fred_data(self, start_date='2019-01-01', end_date=None):
        """Fetch all 62 FRED indicators"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("📊 Fetching FRED economic indicators (62 total)...")
        fred_data = pd.DataFrame()
        
        successful = 0
        failed = []
        
        for fred_code, name in self.fred_indicators.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': fred_code,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'observation_start': start_date,
                    'observation_end': end_date
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    fred_json = response.json()
                    if 'observations' in fred_json and fred_json['observations']:
                        df = pd.DataFrame(fred_json['observations'])
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        df[name] = pd.to_numeric(df['value'], errors='coerce')
                        
                        if fred_data.empty:
                            fred_data = df[[name]]
                        else:
                            fred_data = fred_data.join(df[[name]], how='outer')
                        
                        successful += 1
                        if successful % 10 == 0:
                            print(f"  Progress: {successful}/{len(self.fred_indicators)} indicators fetched")
                else:
                    failed.append(fred_code)
            except Exception as e:
                failed.append(fred_code)
        
        print(f"✓ Successfully fetched {successful}/{len(self.fred_indicators)} indicators")
        if failed:
            print(f"⚠️  Failed indicators: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}")
        
        # Forward fill and backward fill missing values
        fred_data = fred_data.ffill().bfill()
        
        return fred_data
    
    def create_alpha_factors(self, market_data, etf):
        """Create all 20 alpha factors for a single ETF"""
        df = pd.DataFrame(index=market_data.index)
        
        # Get price columns for this ETF
        close_col = f'Close_{etf}'
        open_col = f'Open_{etf}'
        high_col = f'High_{etf}'
        low_col = f'Low_{etf}'
        volume_col = f'Volume_{etf}'
        
        # Extract data
        close = market_data[close_col] if close_col in market_data.columns else market_data[etf]
        open_price = market_data[open_col] if open_col in market_data.columns else close
        high = market_data[high_col] if high_col in market_data.columns else close
        low = market_data[low_col] if low_col in market_data.columns else close
        volume = market_data[volume_col] if volume_col in market_data.columns else pd.Series(1000000, index=market_data.index)
        
        # SPY data for relative calculations
        spy_close = market_data['Close_SPY'] if 'Close_SPY' in market_data.columns else market_data['SPY']
        
        # 1-2. Momentum (1 week, 1 month)
        df[f'{etf}_momentum_1w'] = self.compute_momentum(close, 5)
        df[f'{etf}_momentum_1m'] = self.compute_momentum(close, 21)
        
        # 3. RSI (14-day)
        df[f'{etf}_rsi_14d'] = self.compute_rsi(close, 14)
        
        # 4. Volatility (21-day)
        df[f'{etf}_volatility_21d'] = close.pct_change().rolling(21).std() * np.sqrt(252)
        
        # 5. Sharpe Ratio (10-day)
        returns = close.pct_change()
        df[f'{etf}_sharpe_10d'] = (returns.rolling(10).mean() / returns.rolling(10).std()) * np.sqrt(252)
        
        # 6. Ratio Momentum (ETF/SPY momentum)
        etf_spy_ratio = close / spy_close
        df[f'{etf}_ratio_momentum'] = self.compute_momentum(etf_spy_ratio, 21)
        
        # 7. Volume Ratio (5d/20d)
        df[f'{etf}_volume_ratio'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        # 8-10. MACD, Signal, Histogram
        macd, macd_signal, macd_hist = self.compute_macd(close)
        df[f'{etf}_macd'] = macd
        df[f'{etf}_macd_signal'] = macd_signal
        df[f'{etf}_macd_hist'] = macd_hist
        
        # 11. Bollinger Band %B
        _, _, pct_b = self.compute_bollinger_bands(close)
        df[f'{etf}_bb_pctb'] = pct_b
        
        # 12-14. KDJ Indicators
        k, d, j = self.compute_kdj(high, low, close)
        df[f'{etf}_kdj_k'] = k
        df[f'{etf}_kdj_d'] = d
        df[f'{etf}_kdj_j'] = j
        
        # 15. ATR (Average True Range)
        df[f'{etf}_atr_14d'] = self.compute_atr(high, low, close, 14)
        
        # 16-17. Price Breakouts
        df[f'{etf}_high_20d'] = close / close.rolling(20).max()
        df[f'{etf}_low_20d'] = close / close.rolling(20).min()
        
        # 18. MFI (Money Flow Index)
        df[f'{etf}_mfi_14d'] = self.compute_mfi(high, low, close, volume, 14)
        
        # 19. Volume-Weighted Price
        df[f'{etf}_vwap'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        
        # 20. Price Position (0-1 normalized)
        rolling_min = close.rolling(63).min()
        rolling_max = close.rolling(63).max()
        df[f'{etf}_price_position'] = (close - rolling_min) / (rolling_max - rolling_min)
        
        return df
    
    def create_beta_factors(self, fred_data, etf_prefix=""):
        """Create all 62 beta factors with proper naming"""
        df = pd.DataFrame(index=fred_data.index)
        
        for col in fred_data.columns:
            # Add raw value
            df[f'{etf_prefix}fred_{col}'] = fred_data[col]
            
            # Add 1-month change
            df[f'{etf_prefix}fred_{col}_chg_1m'] = fred_data[col].pct_change(21)
            
            # Add 3-month change
            df[f'{etf_prefix}fred_{col}_chg_3m'] = fred_data[col].pct_change(63)
        
        return df
    
    def create_complete_features(self, market_data, fred_data):
        """Create complete feature set for all ETFs"""
        all_features = {}
        
        for etf in self.sector_etfs:
            print(f"\nProcessing {etf}...")
            
            # Create alpha factors (20 per ETF)
            alpha_df = self.create_alpha_factors(market_data, etf)
            
            # Create beta factors (62 * 3 = 186 total)
            beta_df = self.create_beta_factors(fred_data)
            
            # Combine all features
            features = pd.concat([alpha_df, beta_df], axis=1)
            
            # Add target variable (21-day forward relative return)
            close_col = f'Close_{etf}'
            spy_col = 'Close_SPY'
            
            etf_close = market_data[close_col] if close_col in market_data.columns else market_data[etf]
            spy_close = market_data[spy_col] if spy_col in market_data.columns else market_data['SPY']
            
            etf_return = etf_close.pct_change(21).shift(-21)
            spy_return = spy_close.pct_change(21).shift(-21)
            features['target'] = etf_return - spy_return
            
            # Remove NaN values and infinite values
            features = features.dropna()
            
            # Replace infinite values with NaN and then forward fill
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.ffill().bfill()
            features = features.dropna()
            
            all_features[etf] = features
            print(f"  ✓ Created {len(features.columns)-1} features (20 alpha + {len(beta_df.columns)} beta)")
        
        return all_features
    
    def get_feature_stats(self, features_dict):
        """Print statistics about the features"""
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        
        for etf, features in features_dict.items():
            alpha_cols = [col for col in features.columns if etf in col and 'fred' not in col]
            beta_cols = [col for col in features.columns if 'fred' in col]
            
            print(f"\n{etf}:")
            print(f"  Total features: {len(features.columns)-1}")
            print(f"  Alpha factors: {len(alpha_cols)}")
            print(f"  Beta factors: {len(beta_cols)}")
            print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}")
            print(f"  Total samples: {len(features)}")
        
        # Show sample of features
        sample_etf = list(features_dict.keys())[0]
        sample_features = features_dict[sample_etf]
        
        print(f"\nSample features from {sample_etf}:")
        print("Alpha factors (first 5):")
        alpha_sample = [col for col in sample_features.columns if sample_etf in col and 'fred' not in col][:5]
        for col in alpha_sample:
            print(f"  - {col}")
        
        print("\nBeta factors (first 5):")
        beta_sample = [col for col in sample_features.columns if 'fred' in col][:5]
        for col in beta_sample:
            print(f"  - {col}")


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE FEATURE ENGINEERING TEST")
    print("="*80)
    
    # Initialize feature engine
    engine = ComprehensiveFeatureEngine()
    
    # Fetch data
    market_data = engine.fetch_market_data()
    fred_data = engine.fetch_fred_data()
    
    # Create complete features
    all_features = engine.create_complete_features(market_data, fred_data)
    
    # Show statistics
    engine.get_feature_stats(all_features)
    
    print("\n✅ Feature engineering complete!")
    print(f"   Created features for {len(all_features)} ETFs")
    print(f"   Each ETF has 20 alpha factors + {len(fred_data.columns)*3} beta factors")