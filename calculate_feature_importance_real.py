"""
Real Feature Importance Calculation using Permutation Importance
Calculates actual feature importance from trained ensemble models

Usage:
    python calculate_feature_importance_real.py --month august --year 2025 --train-end 2025-07-31
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import torch

from generate_ensemble_predictions import EnsemblePredictor
from etf_monthly_prediction_system import MonthlyPredictionPipeline

SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']


class RealFeatureImportanceCalculator:
    """Calculate real feature importance using permutation on trained ensemble"""

    def __init__(self, month, year, train_end_date):
        self.month = month
        self.year = year
        self.train_end_date = pd.to_datetime(train_end_date)
        self.output_file = Path(f"feature_importance_{month.lower()}_{year}.json")

    def calculate_for_all_sectors(self, n_repeats=10, top_n=50):
        """Calculate feature importance for all sectors"""
        print("="*80)
        print(f"REAL FEATURE IMPORTANCE CALCULATION - {self.month.upper()} {self.year}")
        print("="*80)
        print(f"\nMethod: Permutation Importance")
        print(f"Training Cutoff: {self.train_end_date.date()}")
        print(f"Repeats per feature: {n_repeats}")
        print(f"Top features to save: {top_n}")
        print("="*80)
        print()

        # Fetch data
        print("📊 Fetching data...")
        pipeline = MonthlyPredictionPipeline()
        market_data, fred_data = pipeline.fetch_all_data()

        print("🔧 Creating features...")
        features = pipeline.create_features(market_data, fred_data)

        results = {
            'month': self.month,
            'year': self.year,
            'train_end_date': self.train_end_date.strftime('%Y-%m-%d'),
            'calculation_timestamp': datetime.now().isoformat(),
            'method': 'permutation_importance',
            'n_repeats': n_repeats,
            'sector_importance': {}
        }

        for etf in SECTOR_ETFS:
            print(f"\n{'='*60}")
            print(f"Processing {etf}...")
            print('-'*60)

            importance = self._calculate_sector_importance(
                etf, features[etf], n_repeats, top_n
            )

            if importance:
                results['sector_importance'][etf] = importance
                print(f"✅ {etf} complete - Top feature: {importance['top_features'][0]['feature']}")
            else:
                print(f"⚠️ {etf} skipped - insufficient data")

        # Calculate aggregate importance across all sectors
        print(f"\n{'='*60}")
        print("Calculating aggregate importance...")
        print('-'*60)
        results['aggregate_importance'] = self._calculate_aggregate_importance(results)

        # Save results
        print(f"\n💾 Saving results to {self.output_file}...")
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*80)
        print("✅ FEATURE IMPORTANCE CALCULATION COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_file}")
        print(f"Sectors processed: {len(results['sector_importance'])}")

        # Print summary
        self._print_summary(results)

        return results

    def _calculate_sector_importance(self, etf, df, n_repeats, top_n):
        """Calculate permutation importance for a single sector"""

        # Clean data
        feature_cols = [col for col in df.columns if col != 'target']
        df = df.copy()
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        df = df.dropna(subset=['target'])

        # Get training data
        train_data = df[df.index <= self.train_end_date]

        if len(train_data) < 100:
            print(f"  ⚠️ Insufficient data: {len(train_data)} samples")
            return None

        print(f"  Training samples: {len(train_data)}")

        # Prepare data
        X = train_data[feature_cols].values
        y = train_data['target'].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create sequences for LSTM
        seq_length = 20
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X_scaled)):
            X_seq.append(X_scaled[i-seq_length:i])
            y_seq.append(y[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        if len(X_seq) < 50:
            print(f"  ⚠️ Insufficient sequences: {len(X_seq)}")
            return None

        print(f"  Training sequences: {len(X_seq)}")

        # Train ensemble model
        print("  Training 4-model ensemble...")
        ensemble = EnsemblePredictor(X_scaled.shape[1])
        ensemble.train_models(X_seq, y_seq, etf, epochs=30, lr=0.001)

        # Prepare validation set for permutation importance
        val_size = min(200, len(X_seq) // 5)
        val_start = len(X_seq) - val_size
        X_val_seq = X_seq[val_start:]
        y_val = y_seq[val_start:]

        print(f"  Validation samples: {len(y_val)}")

        # Create wrapper for permutation importance
        class EnsembleWrapper:
            """Wrapper to make ensemble compatible with sklearn's permutation_importance"""
            def __init__(self, ensemble_model, etf_name, seq_length):
                self.ensemble = ensemble_model
                self.etf = etf_name
                self.seq_length = seq_length
                self.feature_history = []

            def fit(self, X, y):
                """Dummy fit method (already trained)"""
                return self

            def predict(self, X):
                """Predict method for permutation importance"""
                # X shape: (n_samples, n_features)
                # Need to create sequences from single timesteps
                predictions = []

                for i in range(len(X)):
                    # Store this sample for building sequences
                    self.feature_history.append(X[i])

                    # Build sequence (use repeated sample if not enough history)
                    if len(self.feature_history) >= self.seq_length:
                        seq = np.array(self.feature_history[-self.seq_length:])
                    else:
                        # Pad with repeated samples
                        seq = np.tile(X[i], (self.seq_length, 1))

                    # Reshape for ensemble
                    seq_input = seq.reshape(1, self.seq_length, -1)

                    # Get ensemble prediction
                    pred, _, _, _ = self.ensemble.predict_ensemble(
                        seq_input, self.etf, vix_level=None
                    )
                    predictions.append(pred if isinstance(pred, float) else pred[0])

                return np.array(predictions)

        wrapper = EnsembleWrapper(ensemble, etf, seq_length)

        # Extract features from last timestep of each sequence
        X_val_flat = X_val_seq[:, -1, :]

        print(f"  Calculating permutation importance (this may take a few minutes)...")

        # Calculate permutation importance
        perm_result = permutation_importance(
            wrapper, X_val_flat, y_val,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_absolute_error'
        )

        # Extract importance scores
        feature_importance = {}
        for idx, feature_name in enumerate(feature_cols):
            importance = perm_result.importances_mean[idx]
            std = perm_result.importances_std[idx]
            feature_importance[feature_name] = {
                'importance': float(importance),
                'std': float(std),
                'abs_importance': float(abs(importance))
            }

        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1]['abs_importance'],
            reverse=True
        )

        # Normalize top features to percentages
        top_features = sorted_features[:top_n]
        total_importance = sum(f[1]['abs_importance'] for f in top_features)

        top_features_list = []
        for feature, scores in top_features:
            if total_importance > 0:
                normalized_pct = (scores['abs_importance'] / total_importance) * 100
            else:
                normalized_pct = 0

            # Categorize feature
            category = self._categorize_feature(feature)

            top_features_list.append({
                'feature': feature,
                'importance_score': scores['importance'],
                'importance_std': scores['std'],
                'importance_pct': normalized_pct,
                'category': category
            })

        # Calculate category breakdown
        category_breakdown = self._calculate_category_breakdown(top_features_list)

        return {
            'etf': etf,
            'total_features': len(feature_cols),
            'top_n': top_n,
            'top_features': top_features_list,
            'category_breakdown': category_breakdown
        }

    def _categorize_feature(self, feature_name):
        """Categorize a feature into Alpha/Beta/VIX/Derived"""
        if 'vix' in feature_name.lower() and 'lag21' in feature_name:
            return 'VIX Regime'
        elif feature_name.startswith('fred_'):
            # Further subcategorize beta
            if any(kw in feature_name for kw in ['treasury', 'yield', 'fed_funds', 'mortgage']):
                return 'Beta - Interest Rates'
            elif any(kw in feature_name for kw in ['gdp', 'industrial', 'retail', 'housing', 'employment']):
                return 'Beta - Economic'
            elif any(kw in feature_name for kw in ['cpi', 'ppi', 'inflation', 'oil', 'gas']):
                return 'Beta - Inflation'
            elif any(kw in feature_name for kw in ['m1', 'm2', 'monetary', 'credit', 'loans']):
                return 'Beta - Money Supply'
            elif any(kw in feature_name for kw in ['vix', 'usd', 'dollar']):
                return 'Beta - Market'
            elif any(kw in feature_name for kw in ['sentiment', 'confidence', 'optimism']):
                return 'Beta - Sentiment'
            else:
                return 'Beta - Other'
        elif feature_name in ['yield_curve_10y2y', 'yield_curve_10y3m', 'real_rate_10y']:
            return 'Derived'
        else:
            return 'Alpha - Technical'

    def _calculate_category_breakdown(self, top_features_list):
        """Calculate importance breakdown by category"""
        category_totals = {}
        for feature in top_features_list:
            cat = feature['category']
            if cat not in category_totals:
                category_totals[cat] = 0
            category_totals[cat] += feature['importance_pct']

        return category_totals

    def _calculate_aggregate_importance(self, results):
        """Calculate aggregate feature importance across all sectors"""
        # Collect all features across sectors
        all_feature_scores = {}

        for etf, data in results['sector_importance'].items():
            for feature_info in data['top_features']:
                feature_name = feature_info['feature']
                if feature_name not in all_feature_scores:
                    all_feature_scores[feature_name] = {
                        'scores': [],
                        'category': feature_info['category']
                    }
                all_feature_scores[feature_name]['scores'].append(
                    feature_info['importance_pct']
                )

        # Calculate average importance across sectors
        aggregate = []
        for feature_name, data in all_feature_scores.items():
            avg_importance = np.mean(data['scores'])
            std_importance = np.std(data['scores'])
            n_sectors = len(data['scores'])

            aggregate.append({
                'feature': feature_name,
                'avg_importance_pct': float(avg_importance),
                'std_importance_pct': float(std_importance),
                'n_sectors': n_sectors,
                'category': data['category']
            })

        # Sort by average importance
        aggregate.sort(key=lambda x: x['avg_importance_pct'], reverse=True)

        # Calculate category breakdown
        category_totals = {}
        for feature in aggregate[:50]:  # Top 50 for aggregate
            cat = feature['category']
            if cat not in category_totals:
                category_totals[cat] = 0
            category_totals[cat] += feature['avg_importance_pct']

        return {
            'top_features': aggregate[:50],
            'category_breakdown': category_totals
        }

    def _print_summary(self, results):
        """Print summary of feature importance results"""
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE SUMMARY")
        print("="*80)

        # Aggregate top features
        print("\n📊 TOP 20 FEATURES (Aggregate Across All Sectors):")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Category':<25}")
        print("-"*80)

        for i, feature in enumerate(results['aggregate_importance']['top_features'][:20], 1):
            print(f"{i:<6} {feature['feature']:<35} {feature['avg_importance_pct']:>6.2f}% "
                  f"(±{feature['std_importance_pct']:.2f})  {feature['category']:<25}")

        # Category breakdown
        print("\n📈 CATEGORY IMPORTANCE BREAKDOWN:")
        print("-"*80)

        category_breakdown = results['aggregate_importance']['category_breakdown']
        for category, importance in sorted(category_breakdown.items(),
                                          key=lambda x: x[1], reverse=True):
            print(f"{category:<30} {importance:>6.2f}%")

        # Sector-specific highlights
        print("\n🏢 SECTOR-SPECIFIC TOP FEATURES:")
        print("-"*80)

        for etf in SECTOR_ETFS:
            if etf in results['sector_importance']:
                top_feature = results['sector_importance'][etf]['top_features'][0]
                print(f"{etf:<6} {top_feature['feature']:<35} {top_feature['importance_pct']:>6.2f}% "
                      f"({top_feature['category']})")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Calculate real feature importance using permutation'
    )
    parser.add_argument('--month', type=str, required=True,
                       help='Month name (e.g., august, september)')
    parser.add_argument('--year', type=int, required=True,
                       help='Year (e.g., 2025)')
    parser.add_argument('--train-end', type=str, required=True,
                       help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--repeats', type=int, default=10,
                       help='Number of permutation repeats (default: 10)')
    parser.add_argument('--top-n', type=int, default=50,
                       help='Number of top features to save (default: 50)')

    args = parser.parse_args()

    calculator = RealFeatureImportanceCalculator(
        month=args.month,
        year=args.year,
        train_end_date=args.train_end
    )

    results = calculator.calculate_for_all_sectors(
        n_repeats=args.repeats,
        top_n=args.top_n
    )

    print(f"\n✅ Feature importance calculation complete!")
    print(f"📄 Results: {calculator.output_file}")


if __name__ == "__main__":
    main()
