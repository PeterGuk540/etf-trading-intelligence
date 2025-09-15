"""
August 2025 Prediction Evaluation Module
Evaluates model predictions made for August 2025 against actual market data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# ETF symbols to evaluate
ETF_SYMBOLS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

def fetch_august_data(start_date: str = "2025-08-01", end_date: str = "2025-08-31") -> pd.DataFrame:
    """
    Fetch actual August 2025 market data for evaluation
    """
    print("Fetching August 2025 actual market data...")
    
    # Include SPY for relative return calculation
    symbols = ETF_SYMBOLS + ['SPY']
    
    # Fetch data with buffer for return calculations
    buffer_start = pd.to_datetime(start_date) - timedelta(days=30)
    buffer_end = pd.to_datetime(end_date) + timedelta(days=30)
    
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=buffer_start, end=buffer_end)
            if not hist.empty:
                data[symbol] = hist['Close']
                print(f"  ✓ Fetched {symbol}: {len(hist)} days")
            else:
                print(f"  ✗ No data for {symbol}")
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
    
    if not data:
        raise ValueError("Failed to fetch any market data")
    
    df = pd.DataFrame(data)
    df = df.loc[start_date:end_date]
    
    return df

def calculate_actual_returns(df: pd.DataFrame, horizon: int = 21) -> Dict[str, float]:
    """
    Calculate actual returns for August 2025
    
    Args:
        df: DataFrame with daily close prices
        horizon: Trading days for return calculation (21 = ~1 month)
    
    Returns:
        Dictionary of actual returns for each ETF
    """
    print(f"\nCalculating actual {horizon}-day returns for August 2025...")
    
    returns = {}
    
    # Get first and last valid trading days in August
    first_day = df.index[0]
    
    # Find the day that is 'horizon' trading days after the first day
    if len(df) >= horizon:
        target_day_idx = min(horizon - 1, len(df) - 1)
        target_day = df.index[target_day_idx]
    else:
        target_day = df.index[-1]
        print(f"  Warning: Only {len(df)} trading days available, less than {horizon}")
    
    print(f"  Period: {first_day.date()} to {target_day.date()} ({target_day_idx + 1} trading days)")
    
    # Calculate returns for each ETF
    for symbol in ETF_SYMBOLS:
        if symbol in df.columns:
            start_price = df[symbol].iloc[0]
            end_price = df[symbol].iloc[target_day_idx]
            
            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                etf_return = (end_price - start_price) / start_price
                
                # Calculate SPY return for the same period
                spy_start = df['SPY'].iloc[0]
                spy_end = df['SPY'].iloc[target_day_idx]
                spy_return = (spy_end - spy_start) / spy_start if spy_start > 0 else 0
                
                # Relative return (ETF - SPY)
                relative_return = etf_return - spy_return
                returns[symbol] = {
                    'absolute_return': etf_return,
                    'spy_return': spy_return,
                    'relative_return': relative_return,
                    'start_price': start_price,
                    'end_price': end_price
                }
                
                print(f"  {symbol}: {relative_return:+.4f} (ETF: {etf_return:+.4f}, SPY: {spy_return:+.4f})")
    
    return returns

def load_model_predictions(prediction_file: str = '/home/aojie_ju/etf-trading-intelligence/august_2025_predictions.json') -> Dict[str, float]:
    """
    Load model predictions made for August 2025
    This loads from saved model outputs or uses placeholder if not available
    """
    print("\nLoading model predictions for August 2025...")
    
    # Try to load actual predictions if file exists
    try:
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        print(f"  ✓ Loaded actual model predictions from {prediction_file}")
        
        # Display loaded predictions
        for symbol, pred in predictions.items():
            print(f"    {symbol}: {pred:+.4f}")
            
    except FileNotFoundError:
        print(f"  ⚠ Prediction file not found: {prediction_file}")
        print("  ⚠ Using placeholder predictions for demonstration")
        
        # Placeholder predictions for demonstration
        predictions = {
            'XLF': 0.015,   # Predicted relative return
            'XLC': -0.008,
            'XLY': 0.022,
            'XLP': -0.012,
            'XLE': 0.035,
            'XLV': -0.005,
            'XLI': 0.018,
            'XLB': 0.012,
            'XLRE': -0.015,
            'XLK': 0.028,
            'XLU': -0.010
        }
    except Exception as e:
        print(f"  ✗ Error loading predictions: {e}")
        print("  ⚠ Using placeholder predictions")
        
        # Fallback placeholder predictions
        predictions = {
            'XLF': 0.015,
            'XLC': -0.008,
            'XLY': 0.022,
            'XLP': -0.012,
            'XLE': 0.035,
            'XLV': -0.005,
            'XLI': 0.018,
            'XLB': 0.012,
            'XLRE': -0.015,
            'XLK': 0.028,
            'XLU': -0.010
        }
    
    return predictions

def evaluate_predictions(actual_returns: Dict, predictions: Dict) -> Dict:
    """
    Evaluate prediction accuracy against actual returns
    """
    print("\nEvaluating prediction accuracy...")
    
    results = {
        'etf_metrics': {},
        'overall_metrics': {},
        'rankings': {}
    }
    
    # Collect data for overall metrics
    actual_values = []
    predicted_values = []
    direction_correct = []
    
    for symbol in ETF_SYMBOLS:
        if symbol in actual_returns and symbol in predictions:
            actual = actual_returns[symbol]['relative_return']
            predicted = predictions[symbol]
            
            actual_values.append(actual)
            predicted_values.append(predicted)
            
            # Calculate metrics for each ETF
            error = predicted - actual
            abs_error = abs(error)
            squared_error = error ** 2
            
            # Direction accuracy (both positive or both negative)
            direction_match = (actual > 0) == (predicted > 0)
            direction_correct.append(direction_match)
            
            # Magnitude accuracy (within 50% of actual)
            if actual != 0:
                pct_error = abs(error / actual)
            else:
                pct_error = abs(error) if error != 0 else 0
            
            results['etf_metrics'][symbol] = {
                'actual_return': actual,
                'predicted_return': predicted,
                'error': error,
                'abs_error': abs_error,
                'squared_error': squared_error,
                'direction_correct': direction_match,
                'pct_error': pct_error
            }
    
    # Calculate overall metrics
    if actual_values:
        actual_array = np.array(actual_values)
        predicted_array = np.array(predicted_values)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predicted_array - actual_array))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((predicted_array - actual_array) ** 2))
        
        # Correlation
        if len(actual_values) > 1:
            correlation = np.corrcoef(actual_array, predicted_array)[0, 1]
        else:
            correlation = 0
        
        # Direction Accuracy
        direction_accuracy = np.mean(direction_correct)
        
        # Ranking correlation (Spearman)
        from scipy.stats import spearmanr
        if len(actual_values) > 1:
            rank_correlation, _ = spearmanr(actual_array, predicted_array)
        else:
            rank_correlation = 0
        
        # Top/Bottom accuracy
        actual_top3 = set(sorted(results['etf_metrics'].keys(), 
                                key=lambda x: results['etf_metrics'][x]['actual_return'], 
                                reverse=True)[:3])
        predicted_top3 = set(sorted(results['etf_metrics'].keys(), 
                                   key=lambda x: results['etf_metrics'][x]['predicted_return'], 
                                   reverse=True)[:3])
        top3_overlap = len(actual_top3.intersection(predicted_top3)) / 3
        
        results['overall_metrics'] = {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'direction_accuracy': direction_accuracy,
            'rank_correlation': rank_correlation,
            'top3_accuracy': top3_overlap,
            'n_etfs_evaluated': len(actual_values)
        }
        
        # Store rankings
        results['rankings']['actual_ranking'] = sorted(
            results['etf_metrics'].keys(),
            key=lambda x: results['etf_metrics'][x]['actual_return'],
            reverse=True
        )
        results['rankings']['predicted_ranking'] = sorted(
            results['etf_metrics'].keys(),
            key=lambda x: results['etf_metrics'][x]['predicted_return'],
            reverse=True
        )
    
    return results

def generate_evaluation_report(results: Dict, actual_returns: Dict) -> str:
    """
    Generate a formatted evaluation report
    """
    report = []
    report.append("=" * 80)
    report.append("AUGUST 2025 PREDICTION EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall Performance Metrics
    report.append("OVERALL PERFORMANCE METRICS")
    report.append("-" * 40)
    metrics = results['overall_metrics']
    report.append(f"Direction Accuracy:     {metrics['direction_accuracy']:.1%}")
    report.append(f"Correlation:           {metrics['correlation']:.3f}")
    report.append(f"Rank Correlation:      {metrics['rank_correlation']:.3f}")
    report.append(f"Mean Absolute Error:   {metrics['mae']:.4f}")
    report.append(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    report.append(f"Top 3 Overlap:         {metrics['top3_accuracy']:.1%}")
    report.append("")
    
    # Individual ETF Performance
    report.append("INDIVIDUAL ETF PERFORMANCE")
    report.append("-" * 40)
    report.append(f"{'ETF':<6} {'Actual':>10} {'Predicted':>10} {'Error':>10} {'Direction':>10}")
    report.append("-" * 56)
    
    for symbol in ETF_SYMBOLS:
        if symbol in results['etf_metrics']:
            m = results['etf_metrics'][symbol]
            direction = "✓" if m['direction_correct'] else "✗"
            report.append(
                f"{symbol:<6} {m['actual_return']:>10.4f} {m['predicted_return']:>10.4f} "
                f"{m['error']:>10.4f} {direction:>10}"
            )
    
    report.append("")
    
    # Ranking Comparison
    report.append("RANKING COMPARISON")
    report.append("-" * 40)
    report.append(f"{'Rank':<6} {'Actual Best':>15} {'Predicted Best':>15}")
    report.append("-" * 36)
    
    for i in range(min(5, len(results['rankings']['actual_ranking']))):
        actual_etf = results['rankings']['actual_ranking'][i]
        predicted_etf = results['rankings']['predicted_ranking'][i]
        actual_ret = results['etf_metrics'][actual_etf]['actual_return']
        predicted_ret = results['etf_metrics'][predicted_etf]['predicted_return']
        
        match = "✓" if actual_etf == predicted_etf else ""
        report.append(
            f"{i+1:<6} {actual_etf:>8} ({actual_ret:+.3f}) "
            f"{predicted_etf:>8} ({predicted_ret:+.3f}) {match}"
        )
    
    report.append("")
    
    # Trading Strategy Performance
    report.append("TRADING STRATEGY PERFORMANCE")
    report.append("-" * 40)
    
    # Long top 3, short bottom 3 strategy
    top3_predicted = results['rankings']['predicted_ranking'][:3]
    bottom3_predicted = results['rankings']['predicted_ranking'][-3:]
    
    long_return = np.mean([results['etf_metrics'][etf]['actual_return'] for etf in top3_predicted])
    short_return = np.mean([results['etf_metrics'][etf]['actual_return'] for etf in bottom3_predicted])
    strategy_return = long_return - short_return
    
    report.append(f"Long Top 3 ETFs:    {', '.join(top3_predicted)}")
    report.append(f"Short Bottom 3 ETFs: {', '.join(bottom3_predicted)}")
    report.append(f"Long Return:        {long_return:+.4f}")
    report.append(f"Short Return:       {short_return:+.4f}")
    report.append(f"Strategy Return:    {strategy_return:+.4f}")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def run_august_evaluation():
    """
    Main function to run the complete August 2025 evaluation
    """
    print("Starting August 2025 Prediction Evaluation")
    print("=" * 60)
    
    try:
        # Fetch actual August data
        august_data = fetch_august_data()
        
        # Calculate actual returns
        actual_returns = calculate_actual_returns(august_data)
        
        # Load model predictions (you'll need to update this with actual prediction file)
        predictions = load_model_predictions()
        
        # Evaluate predictions
        results = evaluate_predictions(actual_returns, predictions)
        
        # Generate report
        report = generate_evaluation_report(results, actual_returns)
        
        # Save report
        report_file = '/home/aojie_ju/etf-trading-intelligence/AUGUST_2025_EVALUATION.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Evaluation complete! Report saved to {report_file}")
        print("\n" + report)
        
        return results, report
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, report = run_august_evaluation()