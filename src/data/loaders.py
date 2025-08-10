"""Data loading and fetching utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from fredapi import Fred
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
import json
import hashlib


logger = logging.getLogger(__name__)


class DataCache:
    """Redis-based data caching"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, ttl: int = 3600):
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache"""
        try:
            data = self.redis_client.get(key)
            if data:
                return pd.read_json(data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
        
    def set(self, key: str, data: pd.DataFrame) -> None:
        """Set data in cache"""
        try:
            json_data = data.to_json()
            self.redis_client.setex(key, self.ttl, json_data)
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            
    def generate_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()


class MarketDataLoader:
    """Load market data from various sources"""
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def fetch_etf_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch ETF data for multiple symbols"""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Check cache first
        if self.cache:
            cache_key = self.cache.generate_key('etf', symbols, start_date, end_date, interval)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        data = {}
        
        def fetch_single(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                return symbol, df
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        # Parallel fetching
        futures = [self.executor.submit(fetch_single, symbol) for symbol in symbols]
        
        for future in futures:
            symbol, df = future.result()
            if not df.empty:
                data[symbol] = self._process_market_data(df)
                
        # Cache the result
        if self.cache and data:
            self.cache.set(cache_key, data)
            
        return data
    
    def _process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw market data"""
        # Remove timezone info for consistency
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Calculate additional fields
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Dollar_Volume'] = df['Close'] * df['Volume']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(0)
        
        return df


class EconomicDataLoader:
    """Load economic data from FRED API"""
    
    def __init__(self, api_key: str, cache: Optional[DataCache] = None):
        self.fred = Fred(api_key=api_key)
        self.cache = cache
        
    def fetch_indicators(
        self,
        indicators: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch economic indicators"""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key('econ', indicators, start_date, end_date)
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        data = {}
        
        for indicator in indicators:
            try:
                series = self.fred.get_series(
                    indicator,
                    start_date,
                    end_date
                )
                data[indicator] = series
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
                
        # Combine into DataFrame
        df = pd.DataFrame(data)
        df = self._process_economic_data(df)
        
        # Cache result
        if self.cache and not df.empty:
            self.cache.set(cache_key, df)
            
        return df
    
    def _process_economic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process economic data"""
        # Forward fill missing values (economic data often has gaps)
        df = df.fillna(method='ffill')
        
        # Calculate changes and growth rates
        for col in df.columns:
            df[f'{col}_Change'] = df[col].diff()
            df[f'{col}_PctChange'] = df[col].pct_change()
            df[f'{col}_MA30'] = df[col].rolling(window=30).mean()
            
        return df


class AlternativeDataLoader:
    """Load alternative data sources"""
    
    def __init__(self, cache: Optional[DataCache] = None):
        self.cache = cache
        
    async def fetch_news_sentiment(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        api_key: str
    ) -> pd.DataFrame:
        """Fetch news sentiment data"""
        
        # Implementation for news API
        # This is a placeholder - implement actual API calls
        sentiment_data = pd.DataFrame()
        
        return sentiment_data
    
    async def fetch_social_sentiment(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        api_key: str
    ) -> pd.DataFrame:
        """Fetch social media sentiment"""
        
        # Implementation for social media APIs
        # This is a placeholder - implement actual API calls
        social_data = pd.DataFrame()
        
        return social_data


class DataPipeline:
    """Unified data pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache = DataCache() if config.get('cache', {}).get('enabled') else None
        
        self.market_loader = MarketDataLoader(cache=self.cache)
        
        fred_key = config.get('economic_api_key')
        if fred_key:
            self.economic_loader = EconomicDataLoader(fred_key, cache=self.cache)
        else:
            self.economic_loader = None
            
        self.alternative_loader = AlternativeDataLoader(cache=self.cache)
        
    def load_all_data(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load all data sources"""
        
        data = {}
        
        # Load market data
        etf_symbols = self.config['etfs']['sectors'] + [self.config['etfs']['benchmark']]
        market_data = self.market_loader.fetch_etf_data(
            etf_symbols,
            start_date,
            end_date
        )
        data['market'] = market_data
        
        # Load economic data
        if self.economic_loader and self.config.get('economic_indicators'):
            economic_data = self.economic_loader.fetch_indicators(
                self.config['economic_indicators'],
                start_date,
                end_date
            )
            data['economic'] = economic_data
            
        # Load alternative data (async)
        # asyncio.run(self._load_alternative_data(start_date, end_date))
        
        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate loaded data"""
        
        for source, df_dict in data.items():
            if source == 'market':
                for symbol, df in df_dict.items():
                    if df.empty:
                        logger.error(f"Empty data for {symbol}")
                        return False
                    
                    # Check for required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_cols):
                        logger.error(f"Missing columns in {symbol}")
                        return False
                        
                    # Check for excessive missing values
                    if df.isnull().sum().sum() > len(df) * 0.1:
                        logger.warning(f"Excessive missing values in {symbol}")
                        
        return True
    
    def align_data(
        self,
        data: Dict[str, pd.DataFrame],
        frequency: str = 'D'
    ) -> pd.DataFrame:
        """Align all data sources to common timeline"""
        
        # Get all dataframes
        all_dfs = []
        
        # Process market data
        if 'market' in data:
            for symbol, df in data['market'].items():
                df_copy = df.copy()
                # Prefix columns with symbol
                df_copy.columns = [f"{symbol}_{col}" for col in df_copy.columns]
                all_dfs.append(df_copy)
                
        # Process economic data
        if 'economic' in data:
            all_dfs.append(data['economic'])
            
        # Merge all dataframes
        if all_dfs:
            merged = pd.concat(all_dfs, axis=1)
            
            # Resample to desired frequency
            merged = merged.resample(frequency).last()
            
            # Forward fill missing values
            merged = merged.fillna(method='ffill')
            
            return merged
            
        return pd.DataFrame()