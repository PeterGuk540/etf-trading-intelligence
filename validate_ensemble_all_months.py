"""
Validate ensemble predictions for ALL three months: August, September, October 2025
Compares TRUE 4-model ensemble predictions against actual market data
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

def load_data(month):
    """Load predictions and actual data for a specific month"""
    pred_file = f"{month.lower()}_2025_predictions.json"
    actual_file = f"{month.lower()}_2025_actual_returns.json"

    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    with open(actual_file, 'r') as f:
        actuals_data = json.load(f)

    # Extract actual relative returns
    actuals = {etf: data['relative_return']
              for etf, data in actuals_data['sector_returns'].items()}

    return predictions, actuals, actuals_data

def calculate_metrics(predictions, actuals):
    """Calculate validation metrics"""
    # Align predictions and actuals
    common_etfs = sorted(set(predictions.keys()) & set(actuals.keys()))
    pred_values = np.array([predictions[etf] for etf in common_etfs])
    actual_values = np.array([actuals[etf] for etf in common_etfs])

    # Direction accuracy
    pred_signs = np.sign(pred_values)
    actual_signs = np.sign(actual_values)
    direction_matches = (pred_signs == actual_signs)
    direction_accuracy = np.mean(direction_matches) * 100
    correct_count = np.sum(direction_matches)

    # Correlation
    correlation = np.corrcoef(pred_values, actual_values)[0, 1]

    # Error metrics
    mae = np.mean(np.abs(pred_values - actual_values))
    rmse = np.sqrt(np.mean((pred_values - actual_values) ** 2))

    # R-squared
    ss_res = np.sum((actual_values - pred_values) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf

    # Top/Bottom 3 analysis
    top3_pred = set(sorted(common_etfs, key=lambda x: predictions[x], reverse=True)[:3])
    top3_actual = set(sorted(common_etfs, key=lambda x: actuals[x], reverse=True)[:3])
    top3_overlap = len(top3_pred & top3_actual)

    bottom3_pred = set(sorted(common_etfs, key=lambda x: predictions[x])[:3])
    bottom3_actual = set(sorted(common_etfs, key=lambda x: actuals[x])[:3])
    bottom3_overlap = len(bottom3_pred & bottom3_actual)

    # Trading strategy (long top 3, short bottom 3)
    predicted_return = sum(predictions[etf] for etf in top3_pred) / 3 - \
                       sum(predictions[etf] for etf in bottom3_pred) / 3
    actual_return = sum(actuals[etf] for etf in top3_pred) / 3 - \
                    sum(actuals[etf] for etf in bottom3_pred) / 3

    return {
        'common_etfs': common_etfs,
        'pred_values': pred_values,
        'actual_values': actual_values,
        'direction_accuracy': direction_accuracy,
        'correct_count': correct_count,
        'total_count': len(common_etfs),
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'top3_overlap': top3_overlap,
        'bottom3_overlap': bottom3_overlap,
        'predicted_return': predicted_return,
        'actual_return': actual_return,
        'top3_pred': top3_pred,
        'top3_actual': top3_actual,
        'bottom3_pred': bottom3_pred,
        'bottom3_actual': bottom3_actual
    }

def print_month_results(month, predictions, actuals, actuals_data):
    """Print validation results for a specific month"""
    print("=" * 80)
    print(f"{month.upper()} 2025 ENSEMBLE VALIDATION")
    print("=" * 80)

    metrics = calculate_metrics(predictions, actuals)

    # Handle different date field names
    baseline_date = actuals_data['baseline_date']
    if isinstance(baseline_date, str) and 'T' in baseline_date:
        baseline_date = baseline_date.split('T')[0]

    target_date = actuals_data.get('target_date') or actuals_data.get('october_last_date', 'N/A')
    if isinstance(target_date, str) and 'T' in target_date:
        target_date = target_date.split('T')[0]

    print(f"\n📊 Dataset:")
    print(f"  Baseline Date: {baseline_date}")
    print(f"  Target Date: {target_date}")
    print(f"  SPY Return: {actuals_data['spy_return']:+.4f} ({actuals_data['spy_return']*100:+.2f}%)")
    print(f"  Sectors Evaluated: {metrics['total_count']}")

    print(f"\n🎯 Performance Metrics:")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.1f}% ({metrics['correct_count']}/{metrics['total_count']} correct)")

    # Threshold analysis
    if metrics['direction_accuracy'] >= 80:
        status = "🏆 EXCELLENT"
    elif metrics['direction_accuracy'] >= 70:
        status = "✅ VERY GOOD"
    elif metrics['direction_accuracy'] >= 60:
        status = "👍 GOOD"
    elif metrics['direction_accuracy'] >= 55:
        status = "⚡ PROFITABLE"
    else:
        status = "⚠️ BELOW THRESHOLD"
    print(f"  Status: {status}")

    print(f"  Correlation: {metrics['correlation']:.3f}")
    print(f"  Mean Absolute Error: {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
    print(f"  Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r_squared']:.3f}")

    print(f"\n🎲 Top/Bottom 3 Identification:")
    print(f"  Top 3 Predicted: {', '.join(sorted(metrics['top3_pred']))}")
    print(f"  Top 3 Actual: {', '.join(sorted(metrics['top3_actual']))}")
    print(f"  Top 3 Overlap: {metrics['top3_overlap']}/3 ({metrics['top3_overlap']/3*100:.1f}%)")

    print(f"  Bottom 3 Predicted: {', '.join(sorted(metrics['bottom3_pred']))}")
    print(f"  Bottom 3 Actual: {', '.join(sorted(metrics['bottom3_actual']))}")
    print(f"  Bottom 3 Overlap: {metrics['bottom3_overlap']}/3 ({metrics['bottom3_overlap']/3*100:.1f}%)")

    print(f"\n💰 Trading Strategy (Long Top 3, Short Bottom 3):")
    print(f"  Predicted Return: {metrics['predicted_return']:+.4f} ({metrics['predicted_return']*100:+.2f}%)")
    print(f"  Actual Return: {metrics['actual_return']:+.4f} ({metrics['actual_return']*100:+.2f}%)")
    print(f"  Error: {metrics['actual_return'] - metrics['predicted_return']:+.4f}")

    print(f"\n📋 Sector-by-Sector Results:")
    print(f"{'ETF':<8} {'Predicted':<12} {'Actual':<12} {'Error':<12} {'Direction':<10}")
    print("-" * 60)

    for etf in metrics['common_etfs']:
        pred = predictions[etf]
        actual = actuals[etf]
        error = actual - pred
        direction = "✅" if np.sign(pred) == np.sign(actual) else "❌"
        print(f"{etf:<8} {pred:+.4f} ({pred*100:+6.2f}%)  {actual:+.4f} ({actual*100:+6.2f}%)  "
              f"{error:+.4f} ({error*100:+6.2f}%)  {direction}")

    print()

    return metrics

def main():
    """Main validation function"""
    print("\n" + "=" * 80)
    print("ENSEMBLE PREDICTIONS VALIDATION - ALL MONTHS")
    print("=" * 80)
    print("\n🎯 TRUE 4-Model Ensemble:")
    print("  • LSTM (baseline)")
    print("  • TFT (Temporal Fusion Transformer with attention)")
    print("  • N-BEATS (Neural Basis Expansion)")
    print("  • LSTM-GARCH (volatility modeling)")
    print("\n📊 Adaptive Weighting:")
    print("  • Sector-specific base weights")
    print("  • VIX regime adjustments (21-day lagged)")
    print("\n" + "=" * 80)
    print()

    months = ["August", "September", "October"]
    all_metrics = {}

    for month in months:
        try:
            predictions, actuals, actuals_data = load_data(month)
            metrics = print_month_results(month, predictions, actuals, actuals_data)
            all_metrics[month] = metrics
        except FileNotFoundError as e:
            print(f"⚠️ {month} data not found: {e}\n")
            continue

    # Summary across all months
    if len(all_metrics) > 0:
        print("=" * 80)
        print("SUMMARY ACROSS ALL MONTHS")
        print("=" * 80)

        avg_direction = np.mean([m['direction_accuracy'] for m in all_metrics.values()])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics.values()])
        avg_mae = np.mean([m['mae'] for m in all_metrics.values()])
        total_top3 = sum(m['top3_overlap'] for m in all_metrics.values())
        total_possible = len(all_metrics) * 3

        print(f"\n📊 Average Metrics:")
        print(f"  Average Direction Accuracy: {avg_direction:.1f}%")
        print(f"  Average Correlation: {avg_correlation:.3f}")
        print(f"  Average MAE: {avg_mae:.4f} ({avg_mae*100:.2f}%)")
        print(f"  Top 3 Identification: {total_top3}/{total_possible} ({total_top3/total_possible*100:.1f}%)")

        print(f"\n🏆 Month-by-Month Direction Accuracy:")
        for month, metrics in all_metrics.items():
            status = "🏆" if metrics['direction_accuracy'] >= 80 else \
                     "✅" if metrics['direction_accuracy'] >= 70 else \
                     "👍" if metrics['direction_accuracy'] >= 60 else \
                     "⚡" if metrics['direction_accuracy'] >= 55 else "⚠️"
            print(f"  {month:12s}: {metrics['direction_accuracy']:5.1f}% {status}")

        print(f"\n💰 Trading Strategy Performance:")
        for month, metrics in all_metrics.items():
            status = "✅" if metrics['actual_return'] > 0 else "❌"
            print(f"  {month:12s}: {metrics['actual_return']:+.4f} ({metrics['actual_return']*100:+.2f}%) {status}")

        total_strategy_return = sum(m['actual_return'] for m in all_metrics.values())
        print(f"  {'TOTAL':12s}: {total_strategy_return:+.4f} ({total_strategy_return*100:+.2f}%)")

        print(f"\n✅ Validation complete for {len(all_metrics)} month(s)")
        print("=" * 80)

        # Save summary
        summary = {
            'validation_date': datetime.now().isoformat(),
            'ensemble_type': 'TRUE_4_MODEL_ENSEMBLE',
            'months_validated': list(all_metrics.keys()),
            'summary_metrics': {
                'avg_direction_accuracy': float(avg_direction),
                'avg_correlation': float(avg_correlation),
                'avg_mae': float(avg_mae),
                'total_strategy_return': float(total_strategy_return)
            },
            'monthly_results': {
                month: {
                    'direction_accuracy': m['direction_accuracy'],
                    'correlation': m['correlation'],
                    'mae': m['mae'],
                    'strategy_return': m['actual_return']
                }
                for month, m in all_metrics.items()
            }
        }

        with open('ensemble_validation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Summary saved to: ensemble_validation_summary.json")

if __name__ == "__main__":
    main()
