"""
Download Actual October 2025 Market Data for Validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json

# ETF universe
SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

print("="*80)
print("DOWNLOADING ACTUAL OCTOBER 2025 MARKET DATA")
print("="*80)
print(f"Download Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Define October 2025 date range
# October 2025: October 1 - October 31 (21 trading days)
october_start = '2025-10-01'
october_end = '2025-10-31'
september_end = '2025-09-30'  # Last day before October for baseline

print(f"📅 Date Range:")
print(f"   October Start: {october_start}")
print(f"   October End: {october_end}")
print(f"   Baseline Date: {september_end}")
print()

# Download data
print("📊 Downloading ETF and SPY data...")
all_tickers = SECTOR_ETFS + ['SPY']

try:
    # Get data from Sept 1 to Nov 3 to ensure we have complete October
    data = yf.download(all_tickers, start='2025-09-01', end='2025-11-04', auto_adjust=True, progress=False)

    if data.empty:
        print("❌ ERROR: No data downloaded!")
        exit(1)

    print(f"✅ Downloaded data: {len(data)} days")
    print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
    print()

    # Calculate September 30 baseline prices
    sept_30_data = data[data.index <= september_end]
    if len(sept_30_data) == 0:
        print("❌ ERROR: No September data available!")
        exit(1)

    baseline_date = sept_30_data.index[-1]
    print(f"📍 Baseline Date: {baseline_date.date()}")
    print()

    # Calculate October end prices (last trading day in October)
    october_data = data[(data.index >= october_start) & (data.index <= october_end)]

    if len(october_data) == 0:
        print("❌ ERROR: No October data available!")
        exit(1)

    october_last_date = october_data.index[-1]
    print(f"📍 October Last Trading Day: {october_last_date.date()}")
    print(f"📊 October Trading Days: {len(october_data)}")
    print()

    # Calculate returns for each ETF and SPY
    print("="*80)
    print("ACTUAL OCTOBER 2025 RETURNS")
    print("="*80)
    print()

    results = {}

    # Get baseline and end prices
    baseline_prices = {}
    october_end_prices = {}

    for ticker in all_tickers:
        try:
            # Check if multi-index columns (multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                baseline_prices[ticker] = data['Close'][ticker].loc[baseline_date]
                october_end_prices[ticker] = data['Close'][ticker].loc[october_last_date]
            else:
                # Single ticker - direct access
                baseline_prices[ticker] = data.loc[baseline_date, 'Close']
                october_end_prices[ticker] = data.loc[october_last_date, 'Close']
        except Exception as e:
            print(f"⚠️  Warning: Could not get prices for {ticker}: {e}")
            baseline_prices[ticker] = None
            october_end_prices[ticker] = None

    # Calculate SPY return
    if baseline_prices['SPY'] and october_end_prices['SPY']:
        spy_return = (october_end_prices['SPY'] / baseline_prices['SPY'] - 1)
        print(f"SPY (Benchmark) Return:")
        print(f"   Price: ${baseline_prices['SPY']:.2f} → ${october_end_prices['SPY']:.2f}")
        print(f"   Return: {spy_return:+.4f} ({spy_return*100:+.2f}%)")
        print()
    else:
        print("❌ ERROR: Could not calculate SPY return!")
        exit(1)

    # Calculate relative returns for each ETF
    print("Sector ETF Returns vs SPY:")
    print("-"*80)

    for ticker in SECTOR_ETFS:
        if baseline_prices[ticker] and october_end_prices[ticker]:
            etf_return = (october_end_prices[ticker] / baseline_prices[ticker] - 1)
            relative_return = etf_return - spy_return

            results[ticker] = {
                'baseline_price': baseline_prices[ticker],
                'october_end_price': october_end_prices[ticker],
                'absolute_return': float(etf_return),
                'spy_return': float(spy_return),
                'relative_return': float(relative_return)
            }

            direction = "↑" if relative_return > 0 else "↓"
            print(f"{ticker:<6}: ${baseline_prices[ticker]:>7.2f} → ${october_end_prices[ticker]:>7.2f} | "
                  f"Abs: {etf_return:+.4f} ({etf_return*100:+.2f}%) | "
                  f"Rel: {relative_return:+.4f} ({relative_return*100:+.2f}%) {direction}")
        else:
            print(f"{ticker:<6}: ❌ ERROR - Missing data")
            results[ticker] = None

    print()
    print("="*80)

    # Save results to JSON
    output_file = '/home/aojie_ju/etf-trading-intelligence/october_2025_actual_returns.json'

    save_data = {
        'download_date': datetime.now().isoformat(),
        'baseline_date': baseline_date.isoformat(),
        'october_last_date': october_last_date.isoformat(),
        'october_trading_days': len(october_data),
        'spy_return': float(spy_return),
        'sector_returns': results
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"✅ Actual returns saved to: {output_file}")
    print()

    # Display summary statistics
    relative_returns = [r['relative_return'] for r in results.values() if r is not None]

    if relative_returns:
        print("📊 SUMMARY STATISTICS:")
        print(f"   Mean Relative Return: {np.mean(relative_returns):+.4f} ({np.mean(relative_returns)*100:+.2f}%)")
        print(f"   Std Dev: {np.std(relative_returns):.4f} ({np.std(relative_returns)*100:.2f}%)")
        print(f"   Max: {np.max(relative_returns):+.4f} ({np.max(relative_returns)*100:+.2f}%)")
        print(f"   Min: {np.min(relative_returns):+.4f} ({np.min(relative_returns)*100:+.2f}%)")

        # Count outperformers vs underperformers
        outperform = sum(1 for r in relative_returns if r > 0)
        underperform = sum(1 for r in relative_returns if r < 0)
        print(f"   Outperformers: {outperform}/11")
        print(f"   Underperformers: {underperform}/11")

    print()
    print("="*80)
    print("✅ DOWNLOAD COMPLETE")
    print("="*80)

except Exception as e:
    print(f"❌ ERROR downloading data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
