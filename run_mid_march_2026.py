"""
Mid-March 2026 Refresh: Ad-hoc prediction update due to recent market volatility.
Train through 2026-03-23, predict 21 trading days forward (~mid-April 2026).
"""
import sys
import json
from datetime import datetime

sys.path.insert(0, '/home/aojie_ju/etf-trading-intelligence')

# Override module-level dates BEFORE importing the pipeline
import etf_monthly_prediction_system as eps

train_end = datetime(2026, 3, 23)
val_start = datetime(2026, 3, 1)
val_end = datetime(2026, 3, 23)

eps.TRAIN_END = train_end
eps.VALIDATION_START = val_start
eps.VALIDATION_END = val_end
eps.PREDICTION_START = datetime(2026, 3, 24)

from generate_ensemble_predictions import generate_ensemble_predictions

print("\n" + "="*80)
print("MID-MARCH 2026 REFRESH — Ad-hoc update due to recent volatility")
print("="*80)
print(f"Training through: {train_end.date()}")
print(f"Prediction target: ~21 trading days from {train_end.date()}")
print()

predictions = generate_ensemble_predictions(
    month="Mid_march",
    year=2026,
    train_end_date=train_end,
    val_start_date=val_start,
    val_end_date=val_end,
)

if predictions:
    output_file = "mid_march_2026_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✅ Mid-month predictions saved to: {output_file}")
    print(f"\n📊 Sector Rankings (predicted relative return vs SPY):")
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    for i, (etf, pred) in enumerate(sorted_preds, 1):
        tag = "LONG" if i <= 3 else ("SHORT" if i >= 9 else "NEUTRAL")
        print(f"  {i:2d}. {etf}: {pred*100:+.2f}%  [{tag}]")

    # Compare with original March predictions
    try:
        with open("march_2026_predictions.json") as f:
            orig = json.load(f)
        print(f"\n📈 Change from original March predictions (trained through Feb 27):")
        for etf, new_pred in sorted_preds:
            old_pred = orig.get(etf, 0)
            delta = (new_pred - old_pred) * 100
            print(f"  {etf}: {old_pred*100:+.2f}% → {new_pred*100:+.2f}%  (Δ {delta:+.2f}pp)")
    except FileNotFoundError:
        pass
else:
    print("❌ Prediction generation failed")
    sys.exit(1)
