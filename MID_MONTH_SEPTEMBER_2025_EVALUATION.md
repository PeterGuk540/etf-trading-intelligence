================================================================================
MID-MONTH SEPTEMBER 2025 PREDICTION EVALUATION REPORT
================================================================================
Generated: 2025-10-21 16:41:56

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
  XLV    +0.0156 (+1.56%) - BUY
  XLE    -0.0004 (-0.04%) - HOLD
  XLY    -0.0010 (-0.10%) - HOLD
  XLK    -0.0026 (-0.26%) - HOLD
  XLRE   -0.0048 (-0.48%) - HOLD
  XLF    -0.0130 (-1.30%) - SELL
  XLI    -0.0151 (-1.51%) - SELL
  XLP    -0.0167 (-1.67%) - SELL
  XLU    -0.0170 (-1.70%) - SELL
  XLB    -0.0467 (-4.67%) - SELL

ACTUAL MARKET RESULTS (September 15 - October 13, 2025)
----------------------------------------
Actual 21-day Relative Returns:
  XLU    +0.0617 (+6.17%) | Absolute: +6.77% | SPY: +0.60%
  XLK    +0.0387 (+3.87%) | Absolute: +4.47% | SPY: +0.60%
  XLV    +0.0354 (+3.54%) | Absolute: +4.14% | SPY: +0.60%
  XLI    -0.0073 (-0.73%) | Absolute: -0.13% | SPY: +0.60%
  XLP    -0.0189 (-1.89%) | Absolute: -1.29% | SPY: +0.60%
  XLE    -0.0194 (-1.94%) | Absolute: -1.34% | SPY: +0.60%
  XLB    -0.0312 (-3.12%) | Absolute: -2.52% | SPY: +0.60%
  XLRE   -0.0315 (-3.15%) | Absolute: -2.55% | SPY: +0.60%
  XLY    -0.0327 (-3.27%) | Absolute: -2.67% | SPY: +0.60%

ENSEMBLE PERFORMANCE METRICS
----------------------------------------
Direction Accuracy:      77.8% (8/11 correct)
Correlation:            0.287 (Pearson correlation)
Rank Correlation:       -0.083 (Spearman correlation)
Mean Absolute Error:    0.0270 (2.70%)
Root Mean Squared Error: 0.0344 (3.44%)
Top 3 Identification:   33.3% (1/3 correct)

PERFORMANCE COMPARISON
----------------------------------------
September Month-End (Sept 1-30) Ensemble Results:
  • Direction Accuracy: 54.5% (6/11 correct)
  • Correlation: 0.180
  • MAE: 2.84%

August 2025 Ensemble Results:
  • Direction Accuracy: 63.6% (7/11 correct)
  • Correlation: 0.413
  • MAE: 2.08%

Mid-Month September vs Month-End September:
  • Direction Accuracy: +23.3% difference (77.8% vs 54.5%)
  • MAE: -0.0014 difference (0.0270 vs 0.0284)

PERFORMANCE INTERPRETATION
----------------------------------------
✓ Strong direction accuracy (>60%) - Ensemble successfully predicts outperformance/underperformance
✗ Weak/negative correlation - Ensemble predictions poorly aligned with actual returns
⚠ Moderate prediction error (2-3%) - Reasonable magnitude accuracy

DETAILED ETF-BY-ETF ANALYSIS
----------------------------------------
ETF    Sector                      Actual  Predicted    Error  Direction
-----------------------------------------------------------------------------
XLY    Consumer Discretionary      -0.033     -0.001    0.032  ✓ Correct
XLP    Consumer Staples            -0.019     -0.017    0.002  ✓ Correct
XLE    Energy                      -0.019     -0.000    0.019  ✓ Correct
XLV    Health Care                  0.035      0.016   -0.020  ✓ Correct
XLI    Industrials                 -0.007     -0.015   -0.008  ✓ Correct
XLB    Materials                   -0.031     -0.047   -0.015  ✓ Correct
XLRE   Real Estate                 -0.031     -0.005    0.027  ✓ Correct
XLK    Technology                   0.039     -0.003   -0.041    ✗ Wrong
XLU    Utilities                    0.062     -0.017   -0.079    ✗ Wrong

BEST AND WORST PREDICTIONS
----------------------------------------
Most Accurate Predictions (smallest error):
  XLP: Error = +0.0023 (Predicted: -0.0167, Actual: -0.0189)
  XLI: Error = -0.0078 (Predicted: -0.0151, Actual: -0.0073)
  XLB: Error = -0.0155 (Predicted: -0.0467, Actual: -0.0312)

Least Accurate Predictions (largest error):
  XLY: Error = +0.0317 (Predicted: -0.0010, Actual: -0.0327)
  XLK: Error = -0.0413 (Predicted: -0.0026, Actual: +0.0387)
  XLU: Error = -0.0787 (Predicted: -0.0170, Actual: +0.0617)

RANKING COMPARISON
----------------------------------------
Rank       Actual Best  Predicted Best
------------------------------------
1           XLU (+0.062)      XLV (+0.016) 
2           XLK (+0.039)      XLE (-0.000) 
3           XLV (+0.035)      XLY (-0.001) 
4           XLI (-0.007)      XLK (-0.003) 
5           XLP (-0.019)     XLRE (-0.005) 

TRADING STRATEGY PERFORMANCE
----------------------------------------
Long Top 3 ETFs:    XLV, XLE, XLY
Short Bottom 3 ETFs: XLP, XLU, XLB
Long Return:        -0.0056 (-0.56%)
Short Return:       +0.0039 (+0.39%)
Strategy Return:    -0.0094 (-0.94%)

✗ Trading strategy unprofitable - Strategy would have lost money

================================================================================