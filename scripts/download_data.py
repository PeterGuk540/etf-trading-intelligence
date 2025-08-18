#!/usr/bin/env python
"""Download initial data for training"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import MarketDataLoader, EconomicDataLoader
import pandas as pd
import argparse
from datetime import datetime
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration"""
    config_path = Path("config/data_configs.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Download market data')
    parser.add_argument('--start', type=str, default='2018-01-01', 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, 
                       help='End date (YYYY-MM-DD), default is today')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize data loaders
    market_loader = MarketDataLoader()
    
    # Get ETF symbols
    symbols = config['data']['etfs']['sectors'] + [config['data']['etfs']['benchmark']]
    
    logger.info(f"Downloading data for {len(symbols)} symbols from {args.start} to {args.end or 'today'}")
    
    # Download market data
    market_data = market_loader.fetch_etf_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end
    )
    
    # Save to files
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol, df in market_data.items():
        output_file = output_dir / f"{symbol}.csv"
        df.to_csv(output_file)
        logger.info(f"Saved {symbol} data to {output_file} ({len(df)} rows)")
    
    # Create combined dataset
    combined_data = pd.DataFrame()
    for symbol, df in market_data.items():
        df_copy = df[['Close', 'Volume', 'Returns']].copy()
        df_copy.columns = [f"{symbol}_{col}" for col in df_copy.columns]
        if combined_data.empty:
            combined_data = df_copy
        else:
            combined_data = combined_data.join(df_copy, how='outer')
    
    # Save combined data
    combined_file = output_dir / "combined_etf_data.csv"
    combined_data.to_csv(combined_file)
    logger.info(f"Saved combined data to {combined_file}")
    
    # Download economic data if API key is available
    from dotenv import load_dotenv
    load_dotenv()
    
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key and fred_key != 'your_fred_api_key_here':
        logger.info("Downloading economic indicators...")
        econ_loader = EconomicDataLoader(api_key=fred_key)
        
        indicators = ['DGS10', 'DGS2', 'DFF', 'UNRATE', 'CPIAUCSL', 'GDP']
        econ_data = econ_loader.fetch_indicators(
            indicators=indicators,
            start_date=args.start,
            end_date=args.end
        )
        
        econ_file = output_dir / "economic_indicators.csv"
        econ_data.to_csv(econ_file)
        logger.info(f"Saved economic data to {econ_file}")
    else:
        logger.warning("FRED API key not configured. Skipping economic data.")
    
    logger.info("Data download completed successfully!")

if __name__ == "__main__":
    main()