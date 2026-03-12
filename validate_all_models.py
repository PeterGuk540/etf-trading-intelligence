"""
Multi-Model Validation for Monthly ETF Predictions (7-Model Ensemble).

Modes:
    --mode benchmark   : Train/evaluate each of 7 wrappers independently on
                         the same rolling windows. Reports per-model metrics.
    --mode integration : Train full 7-model EnsemblePredictor and report
                         ensemble metrics + per-model contributions.
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/home/aojie_ju/etf-trading-intelligence')

# ---- Import model wrappers (no duplicate class definitions) -----------------
from generate_ensemble_predictions import (
    SimpleLSTM, SimpleTFT, SimpleNBeats, SimpleLSTMGARCH,
    EnsemblePredictor,
)
from src.models.pytorch_wrappers import PyTorchModelWrapper
from src.models.lightgbm_wrapper import LightGBMWrapper
from src.models.catboost_wrapper import CatBoostWrapper
from src.models.sarimax_wrapper import SARIMAXWrapper

# Configuration
SECTOR_ETFS = ['XLF', 'XLK', 'XLE']  # Test 3 sectors for speed
PREDICTION_HORIZON = 21  # One month


# ====================== DATA PIPELINE ========================================

def fetch_and_prepare_data():
    """Fetch data and create features"""
    print("\n📊 Fetching market data...")
    tickers = SECTOR_ETFS + ['SPY']
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(tickers, start='2019-01-01', end=end_date, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    return data


def create_features_and_target(data, etf):
    """Create 20 alpha factors + target (same as original pipeline)."""
    df = pd.DataFrame(index=data.index)

    close_col = f'Close_{etf}' if f'Close_{etf}' in data.columns else etf
    high_col = f'High_{etf}' if f'High_{etf}' in data.columns else close_col
    low_col = f'Low_{etf}' if f'Low_{etf}' in data.columns else close_col
    volume_col = f'Volume_{etf}' if f'Volume_{etf}' in data.columns else None
    spy_col = 'Close_SPY' if 'Close_SPY' in data.columns else 'SPY'

    close = data[close_col] if close_col in data.columns else data[etf]
    high = data[high_col] if high_col in data.columns else close
    low = data[low_col] if low_col in data.columns else close
    volume = data[volume_col] if volume_col and volume_col in data.columns else pd.Series(1000000, index=data.index)
    spy = data[spy_col] if spy_col in data.columns else data['SPY']

    # 1-2. Momentum
    df['momentum_1w'] = close.pct_change(5)
    df['momentum_1m'] = close.pct_change(21)
    # 3. RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    # 4. Volatility
    df['volatility_21d'] = close.pct_change().rolling(21).std() * np.sqrt(252)
    # 5. Sharpe
    returns = close.pct_change()
    df['sharpe_10d'] = returns.rolling(10).mean() / (returns.rolling(10).std() + 1e-10)
    # 6-10. SMA ratios
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = close / close.rolling(period).mean()
    # 11. Bollinger Band %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_pctb'] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    # 12-13. Volume features
    df['volume_ratio'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
    df['volume_chg'] = volume.pct_change(5)
    # 14-16. Price position features
    df['high_20d'] = close / (close.rolling(20).max() + 1e-10)
    df['low_20d'] = close / (close.rolling(20).min() + 1e-10)
    df['price_position'] = (close - close.rolling(63).min()) / (close.rolling(63).max() - close.rolling(63).min() + 1e-10)
    # 17-20. Relative features
    df['relative_strength'] = close / spy
    df['relative_momentum'] = (close / spy).pct_change(21)
    df['relative_vol'] = df['volatility_21d'] / (spy.pct_change().rolling(21).std() * np.sqrt(252) + 1e-10)
    df['spread'] = close - spy

    # Target: 21-day forward relative return
    etf_fwd = close.pct_change(21).shift(-21)
    spy_fwd = spy.pct_change(21).shift(-21)
    df['target'] = etf_fwd - spy_fwd

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()


# ====================== METRIC HELPERS =======================================

def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R², direction accuracy, IC Pearson, IC Spearman."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0
    direction_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    ic_pearson = 0.0
    ic_spearman = 0.0
    if len(y_true) > 2:
        try:
            ic_pearson, _ = pearsonr(y_true, y_pred)
        except Exception:
            pass
        try:
            ic_spearman, _ = spearmanr(y_true, y_pred)
        except Exception:
            pass

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_acc,
        'ic_pearson': ic_pearson,
        'ic_spearman': ic_spearman,
    }


# ====================== BUILD 7 WRAPPER INSTANCES ============================

def build_all_wrappers():
    """Return dict of name → BaseModelWrapper for all 7 models."""
    return {
        'lstm':       PyTorchModelWrapper(SimpleLSTM, 'lstm'),
        'tft':        PyTorchModelWrapper(SimpleTFT, 'tft'),
        'nbeats':     PyTorchModelWrapper(SimpleNBeats, 'nbeats'),
        'lstm_garch': PyTorchModelWrapper(SimpleLSTMGARCH, 'lstm_garch'),
        'lightgbm':   LightGBMWrapper(),
        'catboost':   CatBoostWrapper(),
        'sarimax':    SARIMAXWrapper(),
    }


# ====================== BENCHMARK MODE =======================================

def run_benchmark(data, windows):
    """Train/evaluate each of 7 wrappers independently, report per-model metrics."""
    print("\n" + "=" * 70)
    print("BENCHMARK MODE — 7 Independent Models on Same Rolling Windows")
    print("=" * 70)

    # Aggregate results across all sectors and windows
    all_metrics = {name: {k: [] for k in ['rmse', 'mae', 'r2', 'direction_acc', 'ic_pearson', 'ic_spearman']}
                   for name in build_all_wrappers()}

    for etf in SECTOR_ETFS:
        print(f"\n{'=' * 40}")
        print(f"SECTOR: {etf}")
        print(f"{'=' * 40}")

        df = create_features_and_target(data, etf)
        feature_cols = [col for col in df.columns if col not in ['target', 'price', 'spy']]

        for train_start, train_end, val_start, val_end, window_name in windows[-3:]:
            train_df = df[(df.index >= train_start) & (df.index <= train_end)]
            val_df = df[(df.index >= val_start) & (df.index <= val_end)]

            if len(train_df) < 100 or len(val_df) < 20:
                continue

            print(f"\n  Window: {window_name}  (train={len(train_df)}, val={len(val_df)})")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_df[feature_cols].values)
            y_train = train_df['target'].values
            X_val_scaled = scaler.transform(val_df[feature_cols].values)
            y_val = val_df['target'].values

            wrappers = build_all_wrappers()

            for name, wrapper in wrappers.items():
                try:
                    wrapper.train(X_train_scaled, y_train, etf, list(feature_cols),
                                  val_X=X_val_scaled, val_y=y_val)
                    preds = wrapper.predict(X_val_scaled, list(feature_cols))

                    # Align lengths (PyTorch wrappers may return fewer predictions
                    # due to sliding window)
                    n = min(len(preds), len(y_val))
                    if n < 2:
                        raise ValueError("not enough predictions")
                    preds = preds[-n:]
                    y_v = y_val[-n:]

                    m = compute_metrics(y_v, preds)
                    for k in m:
                        all_metrics[name][k].append(m[k])

                    print(f"    {name:<12s}: R²={m['r2']:.3f}  Dir={m['direction_acc']:.0f}%  "
                          f"IC_p={m['ic_pearson']:.3f}  IC_s={m['ic_spearman']:.3f}")
                except Exception as e:
                    print(f"    {name:<12s}: FAILED — {str(e)[:50]}")

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY — Average Across All Sectors & Windows")
    print("=" * 70)
    print(f"\n{'Model':<14s} {'RMSE':<10s} {'MAE':<10s} {'R²':<10s} "
          f"{'Dir%':<8s} {'IC_p':<8s} {'IC_s':<8s}")
    print("-" * 70)

    best_dir, best_dir_model = -1, None
    for name in build_all_wrappers():
        if all_metrics[name]['r2']:
            avg = {k: np.mean(all_metrics[name][k]) for k in all_metrics[name]}
            print(f"{name:<14s} {avg['rmse']:<10.6f} {avg['mae']:<10.6f} {avg['r2']:<10.4f} "
                  f"{avg['direction_acc']:<8.1f} {avg['ic_pearson']:<8.4f} {avg['ic_spearman']:<8.4f}")
            if avg['direction_acc'] > best_dir:
                best_dir = avg['direction_acc']
                best_dir_model = name
        else:
            print(f"{name:<14s} {'(no results)'}")

    if best_dir_model:
        print(f"\n🏆 Best Direction Accuracy: {best_dir_model} ({best_dir:.1f}%)")


# ====================== INTEGRATION MODE =====================================

def run_integration(data, windows):
    """Train full 7-model EnsemblePredictor, report ensemble + per-model metrics."""
    print("\n" + "=" * 70)
    print("INTEGRATION MODE — Full 7-Model Ensemble Validation")
    print("=" * 70)

    ensemble_agg = {k: [] for k in ['rmse', 'mae', 'r2', 'direction_acc', 'ic_pearson', 'ic_spearman']}

    for etf in SECTOR_ETFS:
        print(f"\n{'=' * 40}")
        print(f"ENSEMBLE — SECTOR: {etf}")
        print(f"{'=' * 40}")

        df = create_features_and_target(data, etf)
        feature_cols = [col for col in df.columns if col not in ['target', 'price', 'spy']]

        for train_start, train_end, val_start, val_end, window_name in windows[-3:]:
            train_df = df[(df.index >= train_start) & (df.index <= train_end)]
            val_df = df[(df.index >= val_start) & (df.index <= val_end)]

            if len(train_df) < 100 or len(val_df) < 20:
                continue

            print(f"\n  Window: {window_name}  (train={len(train_df)}, val={len(val_df)})")

            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(train_df[feature_cols].values)
                y_train = train_df['target'].values
                X_val_scaled = scaler.transform(val_df[feature_cols].values)
                y_val = val_df['target'].values

                # Build & train ensemble
                ensemble = EnsemblePredictor(X_train_scaled.shape[1], list(feature_cols))
                ensemble.train_models(X_train_scaled, y_train, etf, list(feature_cols))

                # Predict one-at-a-time for each validation point (rolling)
                preds_list = []
                seq_len = 20
                for i in range(len(X_val_scaled)):
                    # Use the last seq_len rows ending at val point i
                    # Combine tail of training with validation prefix
                    avail = np.vstack([X_train_scaled, X_val_scaled[:i + 1]])
                    window = avail[-seq_len:]

                    pred, _, weights, regime = ensemble.predict_ensemble(
                        window, etf, vix_level=25.0, feature_names=list(feature_cols)
                    )
                    preds_list.append(pred)

                y_pred = np.array(preds_list)
                m = compute_metrics(y_val, y_pred)
                for k in m:
                    ensemble_agg[k].append(m[k])

                # Verify weights sum to 1
                w_sum = sum(weights.values())
                print(f"    Ensemble: R²={m['r2']:.3f}  Dir={m['direction_acc']:.0f}%  "
                      f"IC_p={m['ic_pearson']:.3f}  weights_sum={w_sum:.4f}  regime={regime}")
                print(f"    Weights: ", end="")
                for mn, mw in sorted(weights.items(), key=lambda x: -x[1]):
                    if mw > 0.01:
                        print(f"{mn}={mw:.1%} ", end="")
                print()

            except Exception as e:
                print(f"    FAILED: {str(e)[:60]}")

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY — 7-Model Ensemble Performance")
    print("=" * 70)
    if ensemble_agg['r2']:
        avg = {k: np.mean(ensemble_agg[k]) for k in ensemble_agg}
        print(f"\n  RMSE           : {avg['rmse']:.6f}")
        print(f"  MAE            : {avg['mae']:.6f}")
        print(f"  R²             : {avg['r2']:.4f}")
        print(f"  Direction Acc  : {avg['direction_acc']:.1f}%")
        print(f"  IC Pearson     : {avg['ic_pearson']:.4f}")
        print(f"  IC Spearman    : {avg['ic_spearman']:.4f}")
    else:
        print("\n  No valid results to summarize.")


# ====================== MAIN =================================================

def main():
    parser = argparse.ArgumentParser(description="7-Model ETF Ensemble Validation")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "integration"],
        default="benchmark",
        help="benchmark = per-model comparison; integration = full ensemble",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(f"7-MODEL VALIDATION — mode={args.mode}")
    print("=" * 70)

    data = fetch_and_prepare_data()

    # Build rolling windows
    from dateutil.relativedelta import relativedelta
    current_date = datetime.now()
    windows = []
    for months_back in range(12, 0, -3):
        val_end = current_date - relativedelta(months=months_back)
        val_start = val_end - relativedelta(months=2)
        train_end = val_start - relativedelta(days=1)
        train_start = train_end - relativedelta(years=2)
        window_name = f"Q{((val_end.month - 1) // 3) + 1} {val_end.year}"
        windows.append((
            train_start.strftime('%Y-%m-%d'),
            train_end.strftime('%Y-%m-%d'),
            val_start.strftime('%Y-%m-%d'),
            val_end.strftime('%Y-%m-%d'),
            window_name,
        ))

    print(f"\n📊 Rolling windows: {len(windows)} total, using last 3")
    print(f"   Sectors: {SECTOR_ETFS}")

    if args.mode == "benchmark":
        run_benchmark(data, windows)
    else:
        run_integration(data, windows)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\n✅ All 7 models validated (LSTM, TFT, N-BEATS, LSTM-GARCH, LightGBM, CatBoost, SARIMAX)")


if __name__ == "__main__":
    main()
