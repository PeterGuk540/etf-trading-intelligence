"""
Regenerate ALL predictions (August 2025 - March 2026) with TRUE 7-model ensemble
"""

from datetime import datetime
from generate_ensemble_predictions import generate_ensemble_predictions

MONTHS = [
    # (month, year, train_end, val_start, val_end)
    ("August",    2025, datetime(2025, 7, 31), datetime(2025, 7, 1),  datetime(2025, 7, 31)),
    ("September", 2025, datetime(2025, 8, 31), datetime(2025, 8, 1),  datetime(2025, 8, 31)),
    ("October",   2025, datetime(2025, 8, 31), datetime(2025, 9, 1),  datetime(2025, 9, 30)),
    ("November",  2025, datetime(2025, 9, 30), datetime(2025, 10, 1), datetime(2025, 10, 31)),
    ("December",  2025, datetime(2025, 10, 31), datetime(2025, 11, 1), datetime(2025, 11, 30)),
    ("January",   2026, datetime(2025, 11, 30), datetime(2025, 12, 1), datetime(2025, 12, 31)),
    ("February",  2026, datetime(2025, 12, 31), datetime(2026, 1, 1),  datetime(2026, 1, 31)),
    ("March",     2026, datetime(2026, 2, 27),  datetime(2026, 2, 1),  datetime(2026, 2, 27)),
]

print("="*80)
print("REGENERATING ALL PREDICTIONS WITH TRUE 7-MODEL ENSEMBLE")
print("="*80)
print(f"\nThis will regenerate {len(MONTHS)} months:")
for m, y, *_ in MONTHS:
    print(f"  • {m} {y} predictions")
print("\nUsing proper sliding-window methodology with 7-model ensemble.")
print("="*80)
print()

all_results = {}
for idx, (month, year, train_end, val_start, val_end) in enumerate(MONTHS, 1):
    print("\n" + "="*80)
    print(f"{idx}/{len(MONTHS)}: GENERATING {month.upper()} {year} ENSEMBLE PREDICTIONS")
    print("="*80)
    results = generate_ensemble_predictions(
        month=month,
        year=year,
        train_end_date=train_end,
        val_start_date=val_start,
        val_end_date=val_end,
    )
    all_results[(month, year)] = results

print("\n" + "="*80)
print("✅ ALL ENSEMBLE PREDICTIONS GENERATED")
print("="*80)
print(f"\nFiles created:")
for (month, year), results in all_results.items():
    print(f"  • {month.lower()}_{year}_predictions.json ({len(results)} predictions)")
print("\nAll predictions generated using 7-model ensemble:")
print("  • LSTM (sector-specific weighting)")
print("  • TFT (attention mechanism)")
print("  • N-BEATS (neural basis expansion)")
print("  • LSTM-GARCH (volatility modeling)")
print("  • LightGBM (gradient boosting regression)")
print("  • CatBoost (gradient boosting classification)")
print("  • SARIMAX (time-series with feature selection)")
print("\nWith adaptive weighting:")
print("  • Sector-specific base weights")
print("  • VIX regime adjustments (21-day lagged)")
print("\n" + "="*80)
