================================================================================
MID-MONTH SEPTEMBER 2025 PREDICTION EVALUATION REPORT
================================================================================
Generated: 2025-10-20 19:26:44

EXECUTIVE SUMMARY
----------------------------------------
This report evaluates the ETF Trading Intelligence System's ensemble
predictions made on September 15, 2025 for the next 21 trading days.
This provides additional validation using a mid-month timeframe.

Prediction Date: September 15, 2025
Evaluation Period: September 15 - October 13, 2025 (21 trading days)
Prediction Horizon: 21-day forward relative returns (ETF return - SPY return)
Number of ETFs: 11 major sector ETFs
Model Used: 4-Model Ensemble with Sector-Specific & VIX Regime Weighting

ENSEMBLE MODEL INFORMATION
----------------------------------------
🔗 ENSEMBLE Model Architecture:
  • 4 Neural Network Models with Sector-Specific Weighting:
    - LSTM (Baseline): Best for XLK (Technology) - 57.1% validation accuracy
    - TFT (Temporal Fusion Transformer): Best for XLF (Financials) - 50.8% validation accuracy
    - N-BEATS (Neural Basis Expansion): General purpose forecasting
    - LSTM-GARCH (Hybrid): Best for XLE (Energy) - 77.8% validation accuracy
  • Ensemble Strategy: Weighted combination based on sector validation performance
  • VIX Regime Detection: 21-day lagged regime classification for adaptive weighting
  • Uncertainty Quantification: Model disagreement analysis

Features Used (217 per ETF):
  • 20 Alpha Factors: RSI, MACD, Bollinger Bands, momentum, volatility
  • 186 Beta Factors: 62 FRED economic indicators with 3 variations
  • 11 VIX Regime Features: 21-day lagged regime classification

Training Data:
  • Period: January 2020 - August 2025
  • Prediction Made: September 15, 2025 for next 21 days
  • VIX Regime (21-day lag): Late August VIX = 18.2 → LOW_VOL regime

ENSEMBLE PREDICTIONS (Made September 15, 2025)
----------------------------------------
Predicted 21-day Relative Returns (ETF - SPY):
  XLC    +0.0306 (+3.06%) - BUY
  XLF    +0.0124 (+1.24%) - BUY
  XLY    +0.0019 (+0.19%) - HOLD
  XLK    -0.0001 (-0.01%) - HOLD
  XLI    -0.0031 (-0.31%) - HOLD
  XLU    -0.0159 (-1.59%) - SELL
  XLV    -0.0241 (-2.41%) - SELL
  XLRE   -0.0321 (-3.21%) - SELL
  XLE    -0.0366 (-3.66%) - SELL
  XLP    -0.0380 (-3.80%) - SELL

ACTUAL MARKET RESULTS (September 15 - October 13, 2025)
----------------------------------------
Actual 21-day Relative Returns:
  XLU    +0.0617 (+6.17%) | Absolute: +6.77% | SPY: +0.60%
  XLK    +0.0387 (+3.87%) | Absolute: +4.47% | SPY: +0.60%
  XLV    +0.0354 (+3.54%) | Absolute: +4.14% | SPY: +0.60%
  XLI    -0.0073 (-0.73%) | Absolute: -0.13% | SPY: +0.60%
  XLP    -0.0189 (-1.89%) | Absolute: -1.29% | SPY: +0.60%
  XLE    -0.0194 (-1.94%) | Absolute: -1.34% | SPY: +0.60%
  XLF    -0.0218 (-2.18%) | Absolute: -1.58% | SPY: +0.60%
  XLRE   -0.0315 (-3.15%) | Absolute: -2.55% | SPY: +0.60%
  XLY    -0.0327 (-3.27%) | Absolute: -2.67% | SPY: +0.60%
  XLC    -0.0460 (-4.60%) | Absolute: -4.00% | SPY: +0.60%

ENSEMBLE PERFORMANCE METRICS
----------------------------------------
Direction Accuracy:      40.0% (4/11 correct)
Correlation:            -0.210 (Pearson correlation)
Rank Correlation:       -0.345 (Spearman correlation)
Mean Absolute Error:    0.0362 (3.62%)
Root Mean Squared Error: 0.0447 (4.47%)
Top 3 Identification:   0.0% (0/3 correct)

PERFORMANCE COMPARISON
----------------------------------------
September Month-End (Sept 1-30) Ensemble Results:
  • Direction Accuracy: 72.7% (8/11 correct)
  • Correlation: 0.628
  • MAE: 2.92%
  • Trading Return: +4.58%

August 2025 Ensemble Results:
  • Direction Accuracy: 45.5% (5/11 correct)
  • Correlation: -0.022
  • MAE: 2.39%

Mid-Month September vs Month-End September:
  • Direction Accuracy: -32.7% difference (40.0% vs 72.7%)
  • MAE: +0.0070 difference (0.0362 vs 0.0292)

PERFORMANCE INTERPRETATION
----------------------------------------
✗ Weak direction accuracy (<50%) - Ensemble struggles with direction prediction
✗ Weak/negative correlation - Ensemble predictions poorly aligned with actual returns
✗ High prediction error (>3%) - Significant magnitude prediction errors

DETAILED ETF-BY-ETF ANALYSIS
----------------------------------------
ETF    Sector                      Actual  Predicted    Error  Direction
-----------------------------------------------------------------------------
XLF    Financials                  -0.022      0.012    0.034    ✗ Wrong
XLC    Communication Services      -0.046      0.031    0.077    ✗ Wrong
XLY    Consumer Discretionary      -0.033      0.002    0.035    ✗ Wrong
XLP    Consumer Staples            -0.019     -0.038   -0.019  ✓ Correct
XLE    Energy                      -0.019     -0.037   -0.017  ✓ Correct
XLV    Health Care                  0.035     -0.024   -0.059    ✗ Wrong
XLI    Industrials                 -0.007     -0.003    0.004  ✓ Correct
XLRE   Real Estate                 -0.031     -0.032   -0.001  ✓ Correct
XLK    Technology                   0.039     -0.000   -0.039    ✗ Wrong
XLU    Utilities                    0.062     -0.016   -0.078    ✗ Wrong

BEST AND WORST PREDICTIONS
----------------------------------------
Most Accurate Predictions (smallest error):
  XLRE: Error = -0.0006 (Predicted: -0.0321, Actual: -0.0315)
  XLI: Error = +0.0042 (Predicted: -0.0031, Actual: -0.0073)
  XLE: Error = -0.0172 (Predicted: -0.0366, Actual: -0.0194)

Least Accurate Predictions (largest error):
  XLV: Error = -0.0595 (Predicted: -0.0241, Actual: +0.0354)
  XLC: Error = +0.0766 (Predicted: +0.0306, Actual: -0.0460)
  XLU: Error = -0.0776 (Predicted: -0.0159, Actual: +0.0617)

RANKING COMPARISON
----------------------------------------
Rank       Actual Best  Predicted Best
------------------------------------
1           XLU (+0.062)      XLC (+0.031) 
2           XLK (+0.039)      XLF (+0.012) 
3           XLV (+0.035)      XLY (+0.002) 
4           XLI (-0.007)      XLK (-0.000) 
5           XLP (-0.019)      XLI (-0.003) 

TRADING STRATEGY PERFORMANCE
----------------------------------------
Long Top 3 ETFs:    XLC, XLF, XLY
Short Bottom 3 ETFs: XLRE, XLE, XLP
Long Return:        -0.0335 (-3.35%)
Short Return:       -0.0233 (-2.33%)
Strategy Return:    -0.0102 (-1.02%)

✗ Trading strategy unprofitable - Strategy would have lost money

================================================================================