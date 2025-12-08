# November 2025 Model Underperformance Analysis

**Date:** December 8, 2025
**Author:** ETF Trading Intelligence System
**Period Analyzed:** November 1-28, 2025

---

## Executive Summary

The ensemble model's November 2025 predictions showed significantly lower direction accuracy (45.5%) compared to the previous three months (average 69.7%). Despite this, the strategy remained profitable (+0.94%). This analysis confirms all data is real (verified against live yfinance feeds) and examines the market conditions that contributed to the underperformance.

---

## 1. Data Verification

### Prediction Data
All November predictions were generated using the 4-model ensemble trained through September 30, 2025, with October 2025 as validation. The predictions contain unique decimal values from actual model inference:

| ETF | Predicted Return |
|-----|-----------------|
| XLV | +2.54% |
| XLU | +1.04% |
| XLY | -0.47% |
| XLK | -1.97% |
| XLB | -3.91% |

### Actual Returns Verification
**All 11 sector returns verified against fresh yfinance data on December 8, 2025.**

| ETF | Stored | Fresh yfinance | Status |
|-----|--------|----------------|--------|
| XLF | +1.64% | +1.64% | ✅ Match |
| XLC | +0.31% | +0.31% | ✅ Match |
| XLY | -1.64% | -1.64% | ✅ Match |
| XLP | +3.86% | +3.86% | ✅ Match |
| XLE | +2.44% | +2.44% | ✅ Match |
| XLV | +9.09% | +9.09% | ✅ Match |
| XLI | -1.08% | -1.08% | ✅ Match |
| XLB | +4.16% | +4.16% | ✅ Match |
| XLRE | +1.69% | +1.69% | ✅ Match |
| XLK | -5.00% | -5.00% | ✅ Match |
| XLU | +1.52% | +1.52% | ✅ Match |

**Conclusion: All data is authentic and sourced from real market data.**

---

## 2. Performance Comparison

| Month | Direction Accuracy | Strategy Return | Market Context |
|-------|-------------------|-----------------|----------------|
| August 2025 | 72.7% | +4.63% | Normal market |
| September 2025 | 72.7% | +5.59% | Normal market |
| October 2025 | 63.6% | +3.05% | Pre-election volatility |
| **November 2025** | **45.5%** | **+0.94%** | **Extreme rotation** |

---

## 3. November 2025 Market Context: An Extraordinary Month

### Key Market Events

November 2025 witnessed one of the most dramatic sector rotations in recent years:

1. **Healthcare (XLV) Surge: +9.09%** (vs SPY)
   - The Health Care Select Sector SPDR Fund (XLV) outperformed Technology (XLK) by **14 percentage points** - the widest performance gap since February 2002
   - Healthcare posted its best monthly gain since October 2022 (+9.14%)
   - This was unexpected because Healthcare had underperformed for three consecutive years (2022-2024)

2. **Technology (XLK) Collapse: -5.00%** (vs SPY)
   - Information Technology declined 4.36% in November after massive gains in prior years
   - Nvidia earnings and AI bubble fears triggered the sell-off
   - Tech valuations remained stretched at ~30x forward P/E vs Healthcare at ~20x

3. **Defensive Rotation**
   - Consumer Staples (XLP): +3.86% vs SPY
   - Utilities (XLU): +1.52% vs SPY
   - Materials (XLB): +4.16% vs SPY (unexpected given poor October)

### Sources
- [etf.com: XLV vs XLK Time to Rotate](https://www.etf.com/sections/advisor-center/xlv-vs-xlk-time-rotate-healthcare)
- [Healthcare Stocks Surge as Tech Faces AI Bubble Fears](https://www.ainvest.com/news/healthcare-stocks-surge-tech-faces-ai-bubble-fears-november-2025-sector-rotation-reveals-defensive-winners-2511/)
- [YCharts Monthly Market Wrap: November 2025](https://get.ycharts.com/resources/blog/monthly-market-wrap/)
- [S&P Global US Equities Market Attributes November 2025](https://www.spglobal.com/spdji/en/commentary/article/us-equities-market-attributes/)
- [Benzinga: Tech Stocks Drop, Pharma Gains](https://www.benzinga.com/markets/equities/25/11/48932450/markets-today-wall-street-tuesday-sector-rotation-tech-pharma-healthcare-nvidia-eli-lilly-fed-ou)

---

## 4. Analysis: Why the Model Underperformed

### 4.1 Prediction vs Reality

| Sector | Predicted | Actual | Error | Direction |
|--------|-----------|--------|-------|-----------|
| **XLB** | -3.91% | **+4.16%** | +8.06% | ❌ |
| **XLE** | -3.56% | **+2.44%** | +6.00% | ❌ |
| **XLP** | -2.26% | **+3.86%** | +6.11% | ❌ |
| **XLF** | -1.43% | **+1.64%** | +3.06% | ❌ |
| **XLRE** | -3.83% | **+1.69%** | +5.52% | ❌ |
| **XLC** | -3.86% | **+0.31%** | +4.17% | ❌ |
| XLV | +2.54% | +9.09% | +6.55% | ✅ |
| XLU | +1.04% | +1.52% | +0.48% | ✅ |
| XLK | -1.97% | -5.00% | -3.04% | ✅ |
| XLI | -2.60% | -1.08% | +1.52% | ✅ |
| XLY | -0.47% | -1.64% | -1.17% | ✅ |

### 4.2 Root Causes

**1. Unprecedented Defensive Rotation**
The model was trained on data through September 2025, during which defensive sectors (Healthcare, Staples, Utilities) had underperformed for years. November saw a sudden, massive rotation into defensives that historical patterns couldn't anticipate.

**2. Mean Reversion Surprise**
- Materials (XLB) fell -6.79% in October, leading the model to predict continued weakness (-3.91%)
- Instead, XLB experienced aggressive mean reversion (+4.16%)
- Same pattern for Energy (XLE) and Consumer Staples (XLP)

**3. Tech Valuation Correction**
- The model correctly predicted XLK underperformance (-1.97%)
- But the magnitude was much larger than expected (-5.00%)
- This indicates the model captured the direction but underestimated the severity of the tech rotation

**4. Healthcare Surge Magnitude**
- The model correctly predicted XLV as top performer (+2.54%)
- Actual performance was nearly 4x higher (+9.09%)
- This was driven by sector-specific catalysts (pharma/biotech strength, Eli Lilly momentum) not captured in the feature set

### 4.3 What the Model Got Right

Despite low direction accuracy, the model:
1. **Correctly identified XLV as #1 performer** (predicted: +2.54%, actual: +9.09%)
2. **Correctly predicted XLK weakness** (predicted: -1.97%, actual: -5.00%)
3. **Correctly predicted XLY and XLI underperformance**
4. **Remained profitable** (+0.94% strategy return)

---

## 5. Strategy Performance Despite Low Accuracy

**Trading Strategy:** Long top 3 predicted, Short bottom 3 predicted

| Position | ETF | Predicted | Actual | Contribution |
|----------|-----|-----------|--------|--------------|
| LONG | XLV | +2.54% | +9.09% | +9.09% |
| LONG | XLU | +1.04% | +1.52% | +1.52% |
| LONG | XLY | -0.47% | -1.64% | -1.64% |
| SHORT | XLB | -3.91% | +4.16% | -4.16% |
| SHORT | XLC | -3.86% | +0.31% | -0.31% |
| SHORT | XLRE | -3.83% | +1.69% | -1.69% |

**Gross Return:** (9.09 + 1.52 - 1.64)/3 - (-4.16 - 0.31 - 1.69)/3 = **+0.94%**

The strategy remained profitable because:
1. XLV (long position) massively outperformed, offsetting losses
2. XLK underperformance (not in top 3) didn't hurt longs
3. Short positions underperformed but didn't cause catastrophic losses

---

## 6. Lessons and Recommendations

### 6.1 Model Limitations Identified

1. **Regime Change Detection**: The model struggled to detect the shift from risk-on to risk-off (defensive rotation)
2. **Mean Reversion Timing**: After extreme moves (October's XLB -6.79%), the model didn't anticipate bounce-back
3. **Magnitude Estimation**: Even when direction was correct, magnitude was often underestimated

### 6.2 Potential Improvements

1. **Add regime detection features**: Market breadth, put/call ratios, credit spreads
2. **Incorporate momentum reversal signals**: Extreme oversold conditions triggering mean reversion
3. **Sector-specific sentiment**: Analyst upgrades/downgrades, fund flows
4. **Valuation spreads**: PE ratio differentials between sectors

### 6.3 Positive Takeaways

1. **Strategy remained profitable** despite lowest accuracy month
2. **Correct identification of top performer** (XLV) saved the month
3. **Risk management worked**: No catastrophic drawdown
4. **4/4 months profitable** with cumulative return of +14.20%

---

## 7. Conclusion

November 2025's underperformance was primarily driven by an extraordinary market regime shift - the largest Healthcare vs Technology rotation in over 20 years. While the model's direction accuracy fell to 45.5%, it:

1. Correctly identified the #1 performer (XLV)
2. Correctly predicted Technology weakness
3. Maintained profitability (+0.94%)
4. Kept the 100% win rate intact (4/4 months profitable)

The underperformance highlights the model's limitation in detecting sudden regime changes and mean reversion after extreme moves. However, the overall system design (ensemble approach, sector-specific weighting) provided enough resilience to remain profitable even in challenging conditions.

---

*Report generated: December 8, 2025*
*Data sources: yfinance, FRED API, market news*
