"""
Multi-Sector ETF Trading System
Processes all 11 sectors with proper data structure
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from tqdm import tqdm

# Configuration
FRED_API_KEY = "ccf75f3e8501e936dafd9f3e77729525"
SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
START_DATE = '2020-01-01'
END_DATE = '2024-12-01'

print("="*60)
print("MULTI-SECTOR ETF ANALYSIS")
print("Processing all 11 sectors as in Data_Generation.ipynb")
print("="*60)

# Step 1: Fetch all ETF data
print("\n1. Fetching ETF Data...")
all_data = yf.download(SECTOR_ETFS + ['SPY'], start=START_DATE, end=END_DATE, auto_adjust=False)

# Step 2: Create sector-specific DataFrames
print("\n2. Creating Sector DataFrames (SPY + Sector)...")
sector_dataframes = {}

for etf in SECTOR_ETFS:
    print(f"   Processing SPY_{etf}...")
    
    # Create combined DataFrame
    df = pd.DataFrame()
    
    # Add SPY data
    df['SPY_AdjClose'] = all_data['Adj Close']['SPY']
    df['SPY_Volume'] = all_data['Volume']['SPY']
    df['SPY_High'] = all_data['High']['SPY']
    df['SPY_Low'] = all_data['Low']['SPY']
    
    # Add sector ETF data
    df[f'{etf}_AdjClose'] = all_data['Adj Close'][etf]
    df[f'{etf}_Volume'] = all_data['Volume'][etf]
    df[f'{etf}_High'] = all_data['High'][etf]
    df[f'{etf}_Low'] = all_data['Low'][etf]
    
    # Store DataFrame
    sector_dataframes[f'SPY_{etf}'] = df
    
print(f"\n✅ Created {len(sector_dataframes)} sector DataFrames")

# Step 3: Show structure
print("\n3. Data Structure Summary:")
for name, df in sector_dataframes.items():
    print(f"   {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    if name == 'SPY_XLF':  # Show example
        print(f"      Columns: {df.columns.tolist()}")

# Step 4: Calculate alpha factors for each sector
print("\n4. Alpha Factors Calculation...")

def calculate_alpha_factors(df, etf):
    """Calculate alpha factors for a sector"""
    # Momentum
    df[f'{etf}_momentum_1w'] = df[f'{etf}_AdjClose'].pct_change(5)
    df[f'{etf}_momentum_1m'] = df[f'{etf}_AdjClose'].pct_change(21)
    
    # RSI
    delta = df[f'{etf}_AdjClose'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df[f'{etf}_rsi_14d'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df[f'{etf}_volatility_21d'] = df[f'{etf}_AdjClose'].pct_change().rolling(21).std() * np.sqrt(252)
    
    # Relative strength vs SPY
    df[f'{etf}_relative_strength'] = (
        df[f'{etf}_AdjClose'].pct_change() - df['SPY_AdjClose'].pct_change()
    ).rolling(20).mean()
    
    return df

# Apply to all sectors
for etf in SECTOR_ETFS:
    df_name = f'SPY_{etf}'
    sector_dataframes[df_name] = calculate_alpha_factors(sector_dataframes[df_name], etf)
    alpha_cols = [col for col in sector_dataframes[df_name].columns if etf in col and 'AdjClose' not in col and 'Volume' not in col and 'High' not in col and 'Low' not in col]
    print(f"   {etf}: Added {len(alpha_cols)} alpha factors")

# Step 5: Create targets
print("\n5. Creating Target Variables (5-day relative returns)...")

for etf in SECTOR_ETFS:
    df_name = f'SPY_{etf}'
    df = sector_dataframes[df_name]
    
    # 5-day forward returns
    spy_fwd_return = df['SPY_AdjClose'].pct_change(5).shift(-5)
    etf_fwd_return = df[f'{etf}_AdjClose'].pct_change(5).shift(-5)
    
    # Relative return (alpha)
    df[f'{etf}_target_relative_return'] = etf_fwd_return - spy_fwd_return
    
    sector_dataframes[df_name] = df

print("✅ Targets created for all sectors")

# Step 6: Summary statistics
print("\n6. Summary Statistics:")
print("-"*50)

for etf in SECTOR_ETFS[:3]:  # Show first 3 as example
    df_name = f'SPY_{etf}'
    df = sector_dataframes[df_name]
    
    # Get valid target data
    target_col = f'{etf}_target_relative_return'
    valid_targets = df[target_col].dropna()
    
    print(f"\n{etf} Sector:")
    print(f"  Data points: {len(df)}")
    print(f"  Features: {len(df.columns) - 1}")  # Exclude target
    print(f"  Target mean: {valid_targets.mean():.4f}")
    print(f"  Target std: {valid_targets.std():.4f}")
    print(f"  Target range: [{valid_targets.min():.4f}, {valid_targets.max():.4f}]")

print("\n" + "="*60)
print("CORRECT DATA STRUCTURE ACHIEVED\!")
print("="*60)
print("\n✅ Each sector has its own DataFrame with:")
print("   • SPY benchmark data (4 columns)")
print("   • Sector ETF data (4 columns)")
print("   • Alpha factors (5+ technical indicators)")
print("   • Target variable (5-day relative return)")
print("\n💡 Ready for modeling:")
print("   • Can train 11 separate models (one per sector)")
print("   • Can train multi-output model (predict all sectors)")
print("   • Can analyze sector rotation patterns")
