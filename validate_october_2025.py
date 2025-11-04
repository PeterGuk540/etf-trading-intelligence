"""
Validate October 2025 Predictions Against Actual Market Data
Compares model predictions with real outcomes
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("OCTOBER 2025 PREDICTION VALIDATION")
print("="*80)
print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load actual returns
print("📊 Loading actual October 2025 returns...")
with open('/home/aojie_ju/etf-trading-intelligence/october_2025_actual_returns.json', 'r') as f:
    actual_data = json.load(f)

actual_returns = {ticker: data['relative_return']
                  for ticker, data in actual_data['sector_returns'].items()}

print(f"✅ Loaded actual returns for {len(actual_returns)} sectors")
print(f"   Period: {actual_data['baseline_date'][:10]} to {actual_data['october_last_date'][:10]}")
print(f"   SPY Return: {actual_data['spy_return']:+.4f} ({actual_data['spy_return']*100:+.2f}%)")
print()

# Try to load model predictions
prediction_sources = [
    '/home/aojie_ju/etf-trading-intelligence/october_2025_predictions.json',
    '/home/aojie_ju/etf-trading-intelligence/predictions_october_2025.json'
]

predictions = None
prediction_file = None

for file_path in prediction_sources:
    try:
        with open(file_path, 'r') as f:
            predictions = json.load(f)
        prediction_file = file_path
        print(f"✅ Loaded model predictions from: {prediction_file}")
        break
    except FileNotFoundError:
        continue

if predictions is None:
    print("❌ ERROR: No model prediction file found!")
    print("   Expected files:")
    for file_path in prediction_sources:
        print(f"     - {file_path}")
    print()
    print("   Please run etf_monthly_prediction_system.py first to generate predictions.")
    exit(1)

print(f"   Loaded predictions for {len(predictions)} sectors")
print()

# Align predictions and actuals
common_sectors = sorted(set(predictions.keys()) & set(actual_returns.keys()))

if len(common_sectors) != 11:
    print(f"⚠️  Warning: Only {len(common_sectors)} common sectors found (expected 11)")
    print(f"   Common sectors: {common_sectors}")
    print()

pred_values = np.array([predictions[s] for s in common_sectors])
actual_values = np.array([actual_returns[s] for s in common_sectors])

# Calculate metrics
print("="*80)
print("VALIDATION METRICS")
print("="*80)
print()

# 1. Direction Accuracy
pred_directions = np.sign(pred_values)
actual_directions = np.sign(actual_values)
direction_accuracy = np.mean(pred_directions == actual_directions) * 100

print(f"📈 DIRECTION ACCURACY: {direction_accuracy:.1f}% ({int(np.sum(pred_directions == actual_directions))}/11 correct)")
print()

# 2. Correlation
correlation, p_value = pearsonr(pred_values, actual_values)
print(f"📊 CORRELATION:")
print(f"   Pearson r: {correlation:+.4f}")
print(f"   p-value: {p_value:.4f}")
print(f"   Interpretation: {'Strong' if abs(correlation) > 0.5 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'} {'positive' if correlation > 0 else 'negative'} correlation")
print()

# 3. Mean Absolute Error
mae = np.mean(np.abs(pred_values - actual_values))
print(f"📏 MEAN ABSOLUTE ERROR (MAE): {mae:.4f} ({mae*100:.2f}%)")
print()

# 4. Root Mean Squared Error
rmse = np.sqrt(np.mean((pred_values - actual_values)**2))
print(f"📐 ROOT MEAN SQUARED ERROR (RMSE): {rmse:.4f} ({rmse*100:.2f}%)")
print()

# 5. R-squared
ss_res = np.sum((actual_values - pred_values)**2)
ss_tot = np.sum((actual_values - np.mean(actual_values))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"📊 R² SCORE: {r2:.4f}")
print()

# Detailed comparison by sector
print("="*80)
print("SECTOR-BY-SECTOR COMPARISON")
print("="*80)
print()
print(f"{'Sector':<6} | {'Predicted':<12} | {'Actual':<12} | {'Error':<12} | {'Direction':<10}")
print("-"*80)

correct_count = 0
for sector in common_sectors:
    pred = predictions[sector]
    actual = actual_returns[sector]
    error = pred - actual

    pred_dir = "↑" if pred > 0 else "↓"
    actual_dir = "↑" if actual > 0 else "↓"
    correct = "✓" if np.sign(pred) == np.sign(actual) else "✗"

    if correct == "✓":
        correct_count += 1

    print(f"{sector:<6} | {pred:+.4f} ({pred*100:+5.2f}%) | {actual:+.4f} ({actual*100:+5.2f}%) | "
          f"{error:+.4f} ({error*100:+5.2f}%) | {pred_dir} vs {actual_dir} {correct}")

print()

# Top 3 / Bottom 3 Analysis
print("="*80)
print("TOP 3 / BOTTOM 3 ANALYSIS")
print("="*80)
print()

pred_sorted = sorted(common_sectors, key=lambda s: predictions[s], reverse=True)
actual_sorted = sorted(common_sectors, key=lambda s: actual_returns[s], reverse=True)

top3_pred = set(pred_sorted[:3])
top3_actual = set(actual_sorted[:3])
top3_overlap = len(top3_pred & top3_actual)

bottom3_pred = set(pred_sorted[-3:])
bottom3_actual = set(actual_sorted[-3:])
bottom3_overlap = len(bottom3_pred & bottom3_actual)

print(f"Top 3 Predicted: {', '.join(pred_sorted[:3])}")
print(f"Top 3 Actual:    {', '.join(actual_sorted[:3])}")
print(f"Overlap: {top3_overlap}/3 ({top3_overlap/3*100:.0f}%)")
print()

print(f"Bottom 3 Predicted: {', '.join(pred_sorted[-3:])}")
print(f"Bottom 3 Actual:    {', '.join(actual_sorted[-3:])}")
print(f"Overlap: {bottom3_overlap}/3 ({bottom3_overlap/3*100:.0f}%)")
print()

# Trading Strategy Performance
print("="*80)
print("TRADING STRATEGY PERFORMANCE")
print("="*80)
print()

# Long top 3, short bottom 3
long_pred = np.mean([predictions[s] for s in pred_sorted[:3]])
short_pred = np.mean([predictions[s] for s in pred_sorted[-3:]])
strategy_pred = long_pred - short_pred

long_actual = np.mean([actual_returns[s] for s in pred_sorted[:3]])
short_actual = np.mean([actual_returns[s] for s in pred_sorted[-3:]])
strategy_actual = long_actual - short_actual

print(f"Strategy Based on Predictions:")
print(f"   Long  (top 3):    {long_pred:+.4f} ({long_pred*100:+.2f}%)")
print(f"   Short (bottom 3): {short_pred:+.4f} ({short_pred*100:+.2f}%)")
print(f"   Expected Return:  {strategy_pred:+.4f} ({strategy_pred*100:+.2f}%)")
print()
print(f"Actual Strategy Performance:")
print(f"   Long  (top 3):    {long_actual:+.4f} ({long_actual*100:+.2f}%)")
print(f"   Short (bottom 3): {short_actual:+.4f} ({short_actual*100:+.2f}%)")
print(f"   Actual Return:    {strategy_actual:+.4f} ({strategy_actual*100:+.2f}%)")
print()
print(f"Strategy Accuracy: {strategy_actual:+.4f} vs {strategy_pred:+.4f} (error: {(strategy_actual-strategy_pred)*100:+.2f}%)")
print()

# Performance Assessment
print("="*80)
print("PERFORMANCE ASSESSMENT")
print("="*80)
print()

# Determine performance level
if direction_accuracy >= 70:
    assessment = "EXCELLENT"
elif direction_accuracy >= 60:
    assessment = "GOOD"
elif direction_accuracy >= 55:
    assessment = "PROFITABLE"
elif direction_accuracy >= 50:
    assessment = "MARGINAL"
else:
    assessment = "POOR"

print(f"Overall Performance: {assessment}")
print()

if direction_accuracy > 55:
    print("✅ Direction accuracy ABOVE profitable threshold (>55%)")
else:
    print("❌ Direction accuracy BELOW profitable threshold (>55%)")

if abs(correlation) > 0.3:
    print(f"✅ Correlation shows {'strong' if abs(correlation) > 0.5 else 'moderate'} relationship")
else:
    print("❌ Correlation shows weak relationship")

if strategy_actual > 0:
    print(f"✅ Trading strategy was profitable: {strategy_actual*100:+.2f}%")
else:
    print(f"❌ Trading strategy lost money: {strategy_actual*100:+.2f}%")

print()

# Save validation results
validation_results = {
    'validation_date': datetime.now().isoformat(),
    'period': f"{actual_data['baseline_date'][:10]} to {actual_data['october_last_date'][:10]}",
    'prediction_source': prediction_file,
    'metrics': {
        'direction_accuracy': float(direction_accuracy),
        'correct_predictions': int(np.sum(pred_directions == actual_directions)),
        'total_predictions': len(common_sectors),
        'correlation': float(correlation),
        'correlation_pvalue': float(p_value),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    },
    'top3_analysis': {
        'predicted': pred_sorted[:3],
        'actual': actual_sorted[:3],
        'overlap': int(top3_overlap)
    },
    'bottom3_analysis': {
        'predicted': pred_sorted[-3:],
        'actual': actual_sorted[-3:],
        'overlap': int(bottom3_overlap)
    },
    'trading_strategy': {
        'predicted_return': float(strategy_pred),
        'actual_return': float(strategy_actual),
        'error': float(strategy_actual - strategy_pred)
    },
    'sector_details': {
        sector: {
            'predicted': float(predictions[sector]),
            'actual': float(actual_returns[sector]),
            'error': float(predictions[sector] - actual_returns[sector]),
            'direction_correct': bool(np.sign(predictions[sector]) == np.sign(actual_returns[sector]))
        }
        for sector in common_sectors
    },
    'assessment': assessment
}

output_file = '/home/aojie_ju/etf-trading-intelligence/OCTOBER_2025_VALIDATION.json'
with open(output_file, 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f"✅ Validation results saved to: {output_file}")
print()

print("="*80)
print("VALIDATION COMPLETE")
print("="*80)
