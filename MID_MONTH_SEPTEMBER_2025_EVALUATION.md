================================================================================
MID-MONTH SEPTEMBER 2025 PREDICTION EVALUATION REPORT
================================================================================
Generated: 2025-10-20 17:05:52

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
  XLK    +0.0220 (+2.20%) - BUY
  XLY    +0.0150 (+1.50%) - BUY
  XLC    +0.0100 (+1.00%) - BUY
  XLF    +0.0080 (+0.80%) - BUY
  XLV    +0.0040 (+0.40%) - HOLD
  XLI    +0.0020 (+0.20%) - HOLD
  XLB    +0.0000 (+0.00%) - HOLD
  XLP    -0.0040 (-0.40%) - HOLD
  XLRE   -0.0060 (-0.60%) - SELL
  XLE    -0.0080 (-0.80%) - SELL
  XLU    -0.0100 (-1.00%) - SELL

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
  XLB    -0.0312 (-3.12%) | Absolute: -2.52% | SPY: +0.60%
  XLRE   -0.0315 (-3.15%) | Absolute: -2.55% | SPY: +0.60%
  XLY    -0.0327 (-3.27%) | Absolute: -2.67% | SPY: +0.60%
  XLC    -0.0460 (-4.60%) | Absolute: -4.00% | SPY: +0.60%

ENSEMBLE PERFORMANCE METRICS
----------------------------------------
Direction Accuracy:      54.5% (6/11 correct)
Correlation:            -0.035 (Pearson correlation)
Rank Correlation:       -0.236 (Spearman correlation)
Mean Absolute Error:    0.0314 (3.14%)
Root Mean Squared Error: 0.0366 (3.66%)
Top 3 Identification:   33.3% (1/3 correct)

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
  • Direction Accuracy: -18.2% difference (54.5% vs 72.7%)
  • MAE: +0.0022 difference (0.0314 vs 0.0292)

PERFORMANCE INTERPRETATION
----------------------------------------
⚠ Moderate direction accuracy (50-55%) - Ensemble shows predictive ability above random chance
✗ Weak/negative correlation - Ensemble predictions poorly aligned with actual returns
✗ High prediction error (>3%) - Significant magnitude prediction errors

DETAILED ETF-BY-ETF ANALYSIS
----------------------------------------
ETF    Sector                      Actual  Predicted    Error  Direction
-----------------------------------------------------------------------------
XLF    Financials                  -0.022      0.008    0.030    ✗ Wrong
XLC    Communication Services      -0.046      0.010    0.056    ✗ Wrong
XLY    Consumer Discretionary      -0.033      0.015    0.048    ✗ Wrong
XLP    Consumer Staples            -0.019     -0.004    0.015  ✓ Correct
XLE    Energy                      -0.019     -0.008    0.011  ✓ Correct
XLV    Health Care                  0.035      0.004   -0.031  ✓ Correct
XLI    Industrials                 -0.007      0.002    0.009    ✗ Wrong
XLB    Materials                   -0.031      0.000    0.031  ✓ Correct
XLRE   Real Estate                 -0.031     -0.006    0.025  ✓ Correct
XLK    Technology                   0.039      0.022   -0.017  ✓ Correct
XLU    Utilities                    0.062     -0.010   -0.072    ✗ Wrong

BEST AND WORST PREDICTIONS
----------------------------------------
Most Accurate Predictions (smallest error):
  XLI: Error = +0.0093 (Predicted: +0.0020, Actual: -0.0073)
  XLE: Error = +0.0114 (Predicted: -0.0080, Actual: -0.0194)
  XLP: Error = +0.0149 (Predicted: -0.0040, Actual: -0.0189)

Least Accurate Predictions (largest error):
  XLY: Error = +0.0477 (Predicted: +0.0150, Actual: -0.0327)
  XLC: Error = +0.0560 (Predicted: +0.0100, Actual: -0.0460)
  XLU: Error = -0.0717 (Predicted: -0.0100, Actual: +0.0617)

RANKING COMPARISON
----------------------------------------
Rank       Actual Best  Predicted Best
------------------------------------
1           XLU (+0.062)      XLK (+0.022) 
2           XLK (+0.039)      XLY (+0.015) 
3           XLV (+0.035)      XLC (+0.010) 
4           XLI (-0.007)      XLF (+0.008) 
5           XLP (-0.019)      XLV (+0.004) 

TRADING STRATEGY PERFORMANCE
----------------------------------------
Long Top 3 ETFs:    XLK, XLY, XLC
Short Bottom 3 ETFs: XLRE, XLE, XLU
Long Return:        -0.0133 (-1.33%)
Short Return:       +0.0036 (+0.36%)
Strategy Return:    -0.0170 (-1.70%)

✗ Trading strategy unprofitable - Strategy would have lost money

================================================================================