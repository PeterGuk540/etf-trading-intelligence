# ETF Trading Intelligence System

A Python-based ETF sector rotation prediction system using an ensemble of 4 neural network architectures to forecast monthly relative returns across 11 major sector ETFs.

## 📊 System Overview

This system predicts 21-day forward relative returns (ETF return - SPY return) for sector rotation strategies. It uses 219 features per ETF combining technical indicators, economic data, and VIX regime detection to train an ensemble of 4 neural network models with sector-specific weighting.

## 🎯 Actual Performance

### Model Validation Results (Historical Data)
- **Best Model**: TFT with Attention
- **Average Direction Accuracy**: 48.1% across rolling windows
- **Average R²**: -4.25 (negative due to noisy relative returns)
- **MAE**: 3.77%

### Real-World Performance (TRUE 4-Model Ensemble with Adaptive Weighting)

| Month | Direction Accuracy | Correlation | MAE | Strategy Return | Status |
|-------|-------------------|-------------|-----|-----------------|--------|
| Aug 2025 | **72.7%** (8/11) ✅ | 0.776 | 1.67% | **+4.63%** | Profitable |
| Sep 2025 | **72.7%** (8/11) ✅ | 0.739 | 1.87% | **+5.59%** | Profitable |
| Oct 2025 | **63.6%** (7/11) 👍 | 0.268 | 3.29% | **+3.05%** | Profitable |
| Nov 2025 | 45.5% (5/11) ⚠️ | 0.333 | 4.15% | **+0.94%** | Profitable |
| Dec 2025 | 36.4% (4/11) ⚠️ | -0.183 | 3.19% | **+0.02%** | Profitable |
| Jan 2026 | 54.5% (6/11) ⚠️ | -0.217 | 4.55% | **-2.68%** | Loss |
| Feb 2026 | **72.7%** (8/11) ✅ | 0.614 | 4.48% | **+9.60%** | Profitable |

**Overall Summary (7-Month Period: Aug 2025 - Feb 2026):**
- **Average Direction Accuracy**: 59.7%
- **Win Rate**: 6/7 months profitable (86%)
- **Cumulative Strategy Return**: +21.14%
- **Best Month**: February 2026 (+9.60% return, 72.7% accuracy)
- **Worst Month**: January 2026 (-2.68% loss, regime shift from Tech to Physical Economy)

**March 2026 Predictions (trained through Feb 27, 2026):**
- **Top 3 (LONG)**: XLE (+9.67%), XLU (+9.34%), XLB (+6.98%)
- **Bottom 3 (SHORT)**: XLY (-3.31%), XLF (-1.66%), XLC (+0.75%)
- **Trading Strategy**: Long top 3, Short bottom 3
- **Status**: Awaiting validation at end of March 2026

## 🛠️ Actual Implementation

### Features (219 per ETF) ✅ WITH VIX REGIME & 100% FRED DATA QUALITY
- **20 Alpha Factors**: Technical indicators (RSI, MACD, Bollinger Bands, momentum, volatility)
- **186 Beta Factors**: 62 FRED economic indicators with 3 variations each
- **10 VIX Regime Features**: **21-day lagged** volatility regime detection (LOW/MEDIUM/HIGH)
- **3 Derived Features**: Yield curves and real rates
- **Total**: 2,409 features across all 11 ETFs (219 × 11)
- **Data Quality**: ✅ 62/62 FRED indicators working (100% success rate)
- **VIX Regime**: ✅ Properly lagged 21 days to prevent data leakage

### Model Architectures
The system implements an ensemble of 4 neural network architectures with sector-specific weighting:

1. **LSTM (Baseline)**: 2-layer LSTM with 64 hidden units
   - Direction Accuracy: 48.1%
   - Best for: XLK (Technology) - 57.1% accuracy

2. **TFT (Temporal Fusion Transformer)**: LSTM with multi-head attention
   - Direction Accuracy: 44.4%
   - Best for: XLF (Financials) - 50.8% accuracy

3. **N-BEATS**: Neural basis expansion analysis
   - Direction Accuracy: 48.1%
   - General purpose forecasting

4. **LSTM-GARCH**: LSTM with volatility modeling
   - Direction Accuracy: 47.6%
   - Best for: XLE (Energy) - 77.8% accuracy

### 🔗 Ensemble Methodology: Adaptive Weighted Averaging

The system uses a **sophisticated multi-level weighted averaging approach** (not simple averaging or bagging):

#### **Level 1: Sector-Specific Base Weights**
Each sector has optimized weights based on validation performance:
```
XLE (Energy):     LSTM-GARCH: 70%, LSTM: 20%, TFT: 10%, N-BEATS: 0%
XLK (Technology): LSTM: 60%, N-BEATS: 30%, TFT: 10%, LSTM-GARCH: 0%
XLF (Financials): TFT: 50%, LSTM: 30%, N-BEATS: 20%, LSTM-GARCH: 0%
Other Sectors:    LSTM: 30%, TFT: 30%, N-BEATS: 20%, LSTM-GARCH: 20%
```

#### **Level 2: VIX Regime Adjustments (21-day lagged)**
Base weights are multiplied by regime-specific factors:
- **LOW_VOL (VIX < 20)**: LSTM ×1.2, TFT ×1.1, N-BEATS ×1.0, LSTM-GARCH ×0.8
- **MEDIUM_VOL (20-30)**: All models ×1.0 (no adjustment)
- **HIGH_VOL (VIX > 30)**: LSTM ×0.8, TFT ×0.9, N-BEATS ×1.0, LSTM-GARCH ×1.3

#### **Level 3: Final Ensemble Calculation**
```python
adjusted_weight = base_weight × vix_adjustment
normalized_weight = adjusted_weight / sum(all_adjusted_weights)
ensemble_prediction = Σ(normalized_weight[i] × model_prediction[i])
uncertainty = std_deviation(all_model_predictions)
```

#### **Key Advantages**
- **Not simple averaging**: Dynamic weights based on performance and market conditions
- **Not bagging**: All models see full data, no bootstrap sampling
- **Sector-adaptive**: Optimized for each sector's characteristics
- **Regime-adaptive**: Adjusts to market volatility environment
- **Uncertainty quantification**: Model disagreement provides confidence intervals

### ETF Coverage
11 Major Sector ETFs:
- XLK (Technology)
- XLF (Financials)
- XLE (Energy)
- XLV (Healthcare)
- XLI (Industrials)
- XLY (Consumer Discretionary)
- XLP (Consumer Staples)
- XLB (Materials)
- XLU (Utilities)
- XLRE (Real Estate)
- XLC (Communications)

## 📁 Project Structure

### Core Production Scripts

| Script | Purpose |
|--------|---------|
| **`run_complete_cycle.py`** | 🎯 **Master workflow** - Orchestrates complete monthly cycle |
| `generate_ensemble_predictions.py` | 4-model ensemble prediction engine |
| `calculate_feature_importance_real.py` | Real permutation-based feature importance |
| `update_monthly_tracking.py` | Auto-updates monthly tracking report |
| `etf_monthly_prediction_system.py` | Data fetching & feature engineering pipeline |
| `validate_ensemble_all_months.py` | Multi-month validation utility |
| `regenerate_all_ensemble.py` | Batch regenerate multiple historical months |
| `validate_all_models.py` | Model comparison & selection (research) |

### Living Documentation

| File | Description |
|------|-------------|
| **`MONTHLY_TRACKING_REPORT.md`** | 📊 **Auto-generated living report** with performance, feature importance, and predictions |
| `README.md` | Project overview and quick start guide |

### Data Files

```
├── Predictions
│   ├── august_2025_predictions.json
│   ├── september_2025_predictions.json
│   ├── october_2025_predictions.json
│   ├── november_2025_predictions.json
│   ├── december_2025_predictions.json
│   ├── january_2026_predictions.json
│   ├── february_2026_predictions.json
│   └── march_2026_predictions.json       # Latest prediction
│
├── Actual Returns (Real Market Data)
│   ├── august_2025_actual_returns.json
│   ├── september_2025_actual_returns.json
│   ├── october_2025_actual_returns.json
│   ├── november_2025_actual_returns.json
│   ├── december_2025_actual_returns.json
│   ├── january_2026_actual_returns.json
│   └── february_2026_actual_returns.json  # Latest validation
│
├── Feature Importance (Real Permutation Data)
│   └── feature_importance_{month}_{year}.json
│
└── Interactive Visualizations
    └── plots/
        ├── performance_timeline.html
        ├── pred_vs_actual_{month}_{year}.html
        ├── error_distribution_{month}_{year}.html
        └── feature_importance_{sector}_{month}_{year}.html
```

### Archive

```
archive/
├── deprecated_feature_importance/    # Old hardcoded feature importance (removed)
├── single_month_scripts/             # One-off monthly generation scripts
├── testing_utilities/                # Development & testing tools
├── static_reports/                   # Superseded by MONTHLY_TRACKING_REPORT.md
└── interim_summaries/                # Historical LSTM-only results
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Virtual environment with packages from requirements.txt

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd etf-trading-intelligence

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Running the System

#### 🎯 Complete Monthly Cycle (Recommended)

**Generate new monthly prediction:**
```bash
python run_complete_cycle.py --month march --year 2026 --train-end 2026-02-27
```

This automatically:
1. ✅ Generates 4-model ensemble predictions
2. ✅ Calculates real feature importance (permutation-based)
3. ✅ Updates `MONTHLY_TRACKING_REPORT.md`
4. ✅ Creates interactive HTML plots

**Add validation after month completes:**
```bash
# First: Collect actual returns and save to march_2026_actual_returns.json
# Then run:
python run_complete_cycle.py --validate --month march --year 2026
```

This automatically:
1. ✅ Loads predictions and actual market data
2. ✅ Calculates all validation metrics
3. ✅ Updates `MONTHLY_TRACKING_REPORT.md` with validation results
4. ✅ Creates validation visualizations

##**Monthly workflow (backtest + predict next month):**
```bash
# 1. Collect actual returns for the completed month (e.g., via yfinance)
# 2. Backtest the completed month:
python run_complete_cycle.py --validate --month february --year 2026
# 3. Generate next month's predictions (retrained through end of completed month):
python run_complete_cycle.py --month march --year 2026 --train-end 2026-02-27
```

#### 📊 View Results

```bash
# View the living tracking report (updated automatically)
cat MONTHLY_TRACKING_REPORT.md

# Or open in browser to see interactive plots
# plots/performance_timeline.html
# plots/pred_vs_actual_february_2026.html
```

#### 🔧 Advanced Usage

**Regenerate all historical predictions:**
```bash
python regenerate_all_ensemble.py
```

**Manually update tracking report:**
```bash
python update_monthly_tracking.py
```

**Calculate feature importance separately:**
```bash
python calculate_feature_importance_real.py --month november --year 2025 --train-end 2025-09-30
```

## 📈 Feature Importance

**Method:** Real permutation-based importance on trained ensemble models (not hardcoded!)

Feature importance is calculated for each monthly cycle and saved to:
- `feature_importance_{month}_{year}.json`
- Interactive visualizations in `plots/`

**Key characteristics:**
- ✅ Calculated from actual model performance
- ✅ Uses permutation importance (10 repeats)
- ✅ Sector-specific and aggregate importance
- ✅ Updated automatically with each new prediction cycle

View the latest feature importance in **`MONTHLY_TRACKING_REPORT.md`**

## ⚠️ Known Issues & Limitations

1. **Data Availability**: System may encounter "insufficient data" errors when validation period is too recent
2. **Negative R² Values**: This is expected for relative return predictions due to high noise
3. **Regime Shifts**: Model struggles with unprecedented macroeconomic catalysts (e.g., Jan 2026 "Great Rotation" from Tech to Physical Economy)
4. **Feature Importance**: scikit-learn >= 1.8 requires `BaseEstimator` inheritance for custom wrappers (fixed in `calculate_feature_importance_real.py`)
5. **✅ SOLVED**: Direction accuracy averages 59.7% over 7 months with 86% win rate
6. **✅ SOLVED**: VIX regime detection with corrected 21-day lag methodology
7. **✅ SOLVED**: Data leakage issue fixed with lagged VIX regime features
8. **✅ SOLVED**: All 62 FRED indicators working with robust retry/alternative mechanisms

## 📊 Validation Methodology

The system employs two distinct approaches:

### Model Validation (Rolling Windows)
Used in `validate_all_models.py` for robust model selection:
- **Training Window**: 2 years (500+ samples)
- **Validation Window**: 3 months (60+ samples)
- **Step Size**: 3 months
- **Coverage**: Multiple market conditions from 2024-2025

### Production Predictions (Single Split)
Used in `etf_monthly_prediction_system.py` for actual predictions:
- **Training Period**: Jan 2020 to end of last completed month
- **Validation Period**: Previous month (for model selection)
- **Prediction Period**: Next month forward
- **Latest**: March 2026 predictions trained through Feb 27, 2026 (1526 training samples per ETF)

## 🔍 Key Insights

1. **Technical Indicators Dominate**: Momentum and volatility account for 31.2% of predictive power
2. **Economic Matters**: Economic indicators (beta) contribute significant predictive power across multiple categories
3. **Sector Specificity**: Each sector responds differently to economic drivers (e.g., XLE best predicted by LSTM-GARCH, XLK by LSTM)
4. **Direction > Magnitude**: Direction accuracy matters more than R² for trading profitability
5. **Regime Vulnerability**: Model performs best in trending markets (72.7% in Aug/Sep/Feb) but struggles during regime shifts (36.4% in Dec, -2.68% loss in Jan)

## 📝 Documentation & Reports

### 📊 Primary Documentation (Living Document)

**`MONTHLY_TRACKING_REPORT.md`** - 🎯 **Auto-generated tracking report**
- ✅ Performance timeline across all months
- ✅ Real feature importance per cycle (permutation-based)
- ✅ Validation results with interactive plots
- ✅ Latest predictions with model contributions
- ✅ Updated automatically after each cycle

This is the **single source of truth** for all model performance and predictions.

### 📁 Archive Documentation

All static reports have been archived to `archive/static_reports/`:
- Historical ensemble evaluation reports
- Feature selection reports (based on old hardcoded data)
- Validation snapshots
- Technical documentation

**View archived reports:** `archive/static_reports/` and `archive/interim_summaries/`

## ⚖️ License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This system is for educational and research purposes. Always perform your own due diligence before making investment decisions.