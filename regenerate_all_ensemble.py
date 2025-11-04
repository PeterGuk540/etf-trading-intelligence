"""
Regenerate ALL predictions (August, September, October) with TRUE ensemble
"""

from datetime import datetime
from generate_ensemble_predictions import generate_ensemble_predictions

print("="*80)
print("REGENERATING ALL PREDICTIONS WITH TRUE 4-MODEL ENSEMBLE")
print("="*80)
print("\nThis will regenerate:")
print("  • August 2025 predictions")
print("  • September 2025 predictions")
print("  • October 2025 predictions")
print("\nUsing proper sliding-window methodology with ensemble models.")
print("="*80)
print()

# August 2025: Train through July, Validate on July, Predict August
print("\n" + "="*80)
print("1/3: GENERATING AUGUST 2025 ENSEMBLE PREDICTIONS")
print("="*80)
august_results = generate_ensemble_predictions(
    month="August",
    year=2025,
    train_end_date=datetime(2025, 7, 31),
    val_start_date=datetime(2025, 7, 1),
    val_end_date=datetime(2025, 7, 31)
)

# September 2025: Train through August, Validate on August, Predict September
print("\n" + "="*80)
print("2/3: GENERATING SEPTEMBER 2025 ENSEMBLE PREDICTIONS")
print("="*80)
september_results = generate_ensemble_predictions(
    month="September",
    year=2025,
    train_end_date=datetime(2025, 8, 31),
    val_start_date=datetime(2025, 8, 1),
    val_end_date=datetime(2025, 8, 31)
)

# October 2025: Train through August, Validate on September, Predict October
print("\n" + "="*80)
print("3/3: GENERATING OCTOBER 2025 ENSEMBLE PREDICTIONS")
print("="*80)
october_results = generate_ensemble_predictions(
    month="October",
    year=2025,
    train_end_date=datetime(2025, 8, 31),
    val_start_date=datetime(2025, 9, 1),
    val_end_date=datetime(2025, 9, 30)
)

print("\n" + "="*80)
print("✅ ALL ENSEMBLE PREDICTIONS GENERATED")
print("="*80)
print(f"\nFiles created:")
print(f"  • august_2025_predictions.json ({len(august_results)} predictions)")
print(f"  • september_2025_predictions.json ({len(september_results)} predictions)")
print(f"  • october_2025_predictions.json ({len(october_results)} predictions)")
print("\nAll predictions generated using 4-model ensemble:")
print("  • LSTM (sector-specific weighting)")
print("  • TFT (attention mechanism)")
print("  • N-BEATS (neural basis expansion)")
print("  • LSTM-GARCH (volatility modeling)")
print("\nWith adaptive weighting:")
print("  • Sector-specific base weights")
print("  • VIX regime adjustments (21-day lagged)")
print("\n" + "="*80)
