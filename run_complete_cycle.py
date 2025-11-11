"""
Complete Monthly Cycle Runner
Runs the full pipeline: prediction → feature importance → validation → report update

Usage:
    # Generate new prediction with feature importance
    python run_complete_cycle.py --month november --year 2025 --train-end 2025-09-30

    # Add validation for existing prediction
    python run_complete_cycle.py --validate --month october --year 2025
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"🔄 {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ {description} - COMPLETED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def generate_new_prediction(month, year, train_end):
    """Generate new prediction with feature importance"""
    print("\n" + "="*80)
    print(f"GENERATING NEW PREDICTION CYCLE: {month.upper()} {year}")
    print("="*80)
    print(f"Training cutoff: {train_end}")
    print()

    # Determine validation dates based on training end
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
    import calendar

    train_end_date = datetime.strptime(train_end, '%Y-%m-%d')

    # Validation is the month before prediction
    val_end_date = train_end_date
    val_start_date = datetime(val_end_date.year, val_end_date.month, 1)

    # Prediction month
    pred_date = train_end_date + relativedelta(months=1)
    pred_month = pred_date.month

    print(f"Validation period: {val_start_date.date()} to {val_end_date.date()}")
    print(f"Prediction month: {pred_month}")
    print()

    # Step 1: Generate predictions
    print("\n" + "="*80)
    print("STEP 1: GENERATING ENSEMBLE PREDICTIONS")
    print("="*80)

    # Import and run directly
    from generate_ensemble_predictions import generate_ensemble_predictions

    predictions = generate_ensemble_predictions(
        month=month.capitalize(),
        year=year,
        train_end_date=train_end_date,
        val_start_date=val_start_date,
        val_end_date=val_end_date
    )

    if not predictions:
        print("❌ Prediction generation failed")
        return False

    print(f"✅ Predictions generated: {len(predictions)} sectors")

    # Step 2: Calculate feature importance
    print("\n" + "="*80)
    print("STEP 2: CALCULATING FEATURE IMPORTANCE")
    print("="*80)

    success = run_command([
        'python', 'calculate_feature_importance_real.py',
        '--month', month.lower(),
        '--year', str(year),
        '--train-end', train_end,
        '--repeats', '10',
        '--top-n', '50'
    ], "Feature Importance Calculation")

    if not success:
        print("⚠️ Warning: Feature importance calculation failed, continuing anyway...")

    # Step 3: Update tracking report
    print("\n" + "="*80)
    print("STEP 3: UPDATING MONTHLY TRACKING REPORT")
    print("="*80)

    success = run_command([
        'python', 'update_monthly_tracking.py'
    ], "Monthly Tracking Report Update")

    if not success:
        print("❌ Report update failed")
        return False

    print("\n" + "="*80)
    print("✅ COMPLETE CYCLE FINISHED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  • {month.lower()}_{year}_predictions.json")
    print(f"  • feature_importance_{month.lower()}_{year}.json")
    print(f"  • MONTHLY_TRACKING_REPORT.md (updated)")
    print(f"  • plots/ (interactive visualizations)")
    print()
    print(f"Next steps:")
    print(f"  1. Wait for {month} {year} to complete")
    print(f"  2. Collect actual returns data")
    print(f"  3. Run: python run_complete_cycle.py --validate --month {month.lower()} --year {year}")

    return True


def add_validation(month, year):
    """Add validation for existing prediction"""
    print("\n" + "="*80)
    print(f"ADDING VALIDATION FOR: {month.upper()} {year}")
    print("="*80)
    print()

    # Check if prediction file exists
    pred_file = Path(f"{month.lower()}_{year}_predictions.json")
    if not pred_file.exists():
        print(f"❌ Prediction file not found: {pred_file}")
        return False

    # Check if actual returns file exists
    actual_file = Path(f"{month.lower()}_{year}_actual_returns.json")
    if not actual_file.exists():
        print(f"❌ Actual returns file not found: {actual_file}")
        print(f"\nPlease create this file with actual market data first.")
        print(f"Expected format:")
        print("""
{
  "spy_return": 0.0XXX,
  "baseline_date": "YYYY-MM-DD",
  "target_date": "YYYY-MM-DD",
  "sector_returns": {
    "XLF": {"absolute_return": 0.XXX, "relative_return": 0.XXX},
    ...
  }
}
        """)
        return False

    print(f"✅ Found prediction file: {pred_file}")
    print(f"✅ Found actual returns file: {actual_file}")

    # Update tracking report (validation will be automatically included)
    print("\n" + "="*80)
    print("UPDATING MONTHLY TRACKING REPORT WITH VALIDATION")
    print("="*80)

    success = run_command([
        'python', 'update_monthly_tracking.py'
    ], "Monthly Tracking Report Update")

    if not success:
        print("❌ Report update failed")
        return False

    print("\n" + "="*80)
    print("✅ VALIDATION ADDED SUCCESSFULLY!")
    print("="*80)
    print(f"\nUpdated files:")
    print(f"  • MONTHLY_TRACKING_REPORT.md (updated with validation)")
    print(f"  • plots/ (new validation visualizations)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete monthly cycle: prediction + feature importance + report'
    )

    # Mode selection
    parser.add_argument('--validate', action='store_true',
                       help='Validation mode: add validation for existing prediction')

    # Required for both modes
    parser.add_argument('--month', type=str, required=True,
                       help='Month name (e.g., november, october)')
    parser.add_argument('--year', type=int, required=True,
                       help='Year (e.g., 2025)')

    # Only for prediction mode
    parser.add_argument('--train-end', type=str,
                       help='Training end date YYYY-MM-DD (required for prediction mode)')

    args = parser.parse_args()

    # Validate arguments
    if not args.validate and not args.train_end:
        parser.error("--train-end is required when not in --validate mode")

    if args.validate:
        # Validation mode
        success = add_validation(args.month, args.year)
    else:
        # Prediction mode
        success = generate_new_prediction(args.month, args.year, args.train_end)

    if success:
        print("\n🎉 All operations completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some operations failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
