"""
Generate November 2025 ensemble predictions
Following sliding-window methodology: Train through Sept, Validate on Oct, Predict Nov
"""

from datetime import datetime
from generate_ensemble_predictions import generate_ensemble_predictions

print("=" * 80)
print("GENERATING NOVEMBER 2025 ENSEMBLE PREDICTIONS")
print("=" * 80)
print("\nSliding-Window Configuration:")
print("  Training: Jan 2020 - Sep 30, 2025")
print("  Validation: Oct 1-31, 2025 (using actual October data)")
print("  Prediction: November 2025")
print("=" * 80)
print()

# November 2025: Train through Sept, Validate on Oct, Predict November
november_results = generate_ensemble_predictions(
    month="November",
    year=2025,
    train_end_date=datetime(2025, 9, 30),
    val_start_date=datetime(2025, 10, 1),
    val_end_date=datetime(2025, 10, 31)
)

print("\n" + "=" * 80)
print("✅ NOVEMBER 2025 ENSEMBLE PREDICTIONS GENERATED")
print("=" * 80)
print(f"\nFile created:")
print(f"  • november_2025_predictions.json ({len(november_results)} predictions)")
print("\nPredictions generated using 4-model ensemble:")
print("  • LSTM (sector-specific weighting)")
print("  • TFT (attention mechanism)")
print("  • N-BEATS (neural basis expansion)")
print("  • LSTM-GARCH (volatility modeling)")
print("\nWith adaptive weighting:")
print("  • Sector-specific base weights")
print("  • VIX regime adjustments (21-day lagged)")
print("\n" + "=" * 80)
