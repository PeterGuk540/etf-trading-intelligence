#!/usr/bin/env python
"""Run backtesting on trained model"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBacktester:
    """Simple backtesting engine"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
    def backtest(self, predictions, actual_returns, transaction_cost=0.001):
        """Run simple backtest"""
        capital = self.initial_capital
        positions = []
        returns = []
        
        for i in range(len(predictions)):
            # Generate signal based on prediction
            if predictions[i] > 0.001:  # Buy signal
                position = 1
            elif predictions[i] < -0.001:  # Sell signal
                position = -1
            else:  # No position
                position = 0
            
            # Calculate return
            if i > 0:
                ret = position * actual_returns[i] - abs(position - positions[-1]) * transaction_cost
                capital *= (1 + ret)
                returns.append(ret)
            
            positions.append(position)
        
        # Calculate metrics
        returns = np.array(returns)
        
        total_return = (capital / self.initial_capital - 1) * 100
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_trades = sum(1 for r in returns if r > 0)
        total_trades = len([r for r in returns if r != 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_capital': capital,
            'num_trades': total_trades
        }

def main():
    parser = argparse.ArgumentParser(description='Run backtest on trained model')
    parser.add_argument('--model-path', type=str, 
                       help='Path to saved model')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Backtest start date')
    args = parser.parse_args()
    
    logger.info("Starting backtest...")
    
    # Load test data
    data_path = Path("data/raw/combined_etf_data.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Filter to test period
    df = df[df.index >= args.start_date]
    
    # Generate dummy predictions for demonstration
    # In production, load your trained model and generate real predictions
    np.random.seed(42)
    predictions = np.random.randn(len(df)) * 0.01  # Random predictions
    actual_returns = df['SPY_Returns'].fillna(0).values
    
    # Run backtest
    backtester = SimpleBacktester(initial_capital=100000)
    results = backtester.backtest(predictions, actual_returns)
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Initial Capital: $100,000")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print("="*50)
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df['backtest_date'] = datetime.now()
    results_df['period_start'] = df.index[0]
    results_df['period_end'] = df.index[-1]
    
    output_path = Path("data/processed/backtest_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        existing = pd.read_csv(output_path)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()