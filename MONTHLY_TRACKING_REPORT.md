# ETF Trading Intelligence - Monthly Tracking Report
*Living document tracking model performance, feature importance, and predictions*

**Last Updated:** 2025-11-10 19:54 UTC

---

## 📊 Latest Status Dashboard

| Metric | Value |
|--------|-------|
| **Latest Prediction** | November 2025 |
| **Last Validated Month** | October 2025 |
| **Overall Direction Accuracy** | 69.7% |
| **Win Rate (Profitable Months)** | 3/3 (100%) if validated else 'N/A' |
| **Cumulative Strategy Return** | +13.26% |
| **Total Cycles Tracked** | 5 |

---

## 📈 Performance Timeline

<iframe src="plots/performance_timeline.html" width="100%" height="850" frameborder="0"></iframe>


| Month | Direction Accuracy | Correlation | MAE | Strategy Return | Status | Training Through |
|-------|-------------------|-------------|-----|-----------------|--------|------------------|
| November 2025 | *Pending* | *Pending* | *Pending* | *Pending* | 🔮 Predicted | *Unknown* |
| October 2025 | 63.6% | 0.268 | 3.29% | +3.05% | ✅ Validated | 2025-09-30T00:00:00 |
| September 2025 | 72.7% | 0.739 | 1.87% | +5.59% | ✅ Validated | 2025-08-29 |
| August 2025 | 72.7% | 0.776 | 1.67% | +4.63% | ✅ Validated | 2025-07-31 |
| Mid_september 2025 | *Pending* | *Pending* | *Pending* | *Pending* | 🔮 Predicted | *Unknown* |

---

## 🔮 November 2025 Cycle

**Latest Prediction - Awaiting Validation**

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLV 🟢 | +3.74% | LONG |
| 2 | XLU 🟢 | +2.78% | LONG |
| 3 | XLK 🟢 | +1.48% | LONG |
| 4 | XLP ⚪ | +0.46% | NEUTRAL |
| 5 | XLC ⚪ | +0.33% | NEUTRAL |
| 6 | XLY ⚪ | -0.49% | NEUTRAL |
| 7 | XLRE ⚪ | -0.68% | NEUTRAL |
| 8 | XLE ⚪ | -1.67% | NEUTRAL |
| 9 | XLB 🔴 | -2.41% | SHORT |
| 10 | XLF 🔴 | -3.47% | SHORT |
| 11 | XLI 🔴 | -3.53% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

---

## ✅ October 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLC 🟢 | +5.00% | LONG |
| 2 | XLY 🟢 | +1.73% | LONG |
| 3 | XLU 🟢 | +1.19% | LONG |
| 4 | XLK ⚪ | +0.73% | NEUTRAL |
| 5 | XLE ⚪ | -0.10% | NEUTRAL |
| 6 | XLRE ⚪ | -0.79% | NEUTRAL |
| 7 | XLV ⚪ | -1.21% | NEUTRAL |
| 8 | XLI ⚪ | -1.37% | NEUTRAL |
| 9 | XLF 🔴 | -2.78% | SHORT |
| 10 | XLB 🔴 | -3.70% | SHORT |
| 11 | XLP 🔴 | -5.26% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-09-30T00:00:00 to 2025-10-31T00:00:00
**SPY Return:** +2.38%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **63.6%** (7/11) | 👍 GOOD |
| Correlation | 0.268 | Weak |
| Mean Absolute Error | 3.29% | - |
| R² Score | -0.683 | - |
| **Strategy Return** | **+3.05%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_october_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -3.70% | -6.79% | -3.10% | ✅ |
| XLC | +5.00% | -5.39% | -10.39% | ❌ |
| XLE | -0.10% | -3.74% | -3.63% | ✅ |
| XLF | -2.78% | -5.17% | -2.38% | ✅ |
| XLI | -1.37% | -1.85% | -0.48% | ✅ |
| XLK | +0.73% | +4.29% | +3.57% | ✅ |
| XLP | -5.26% | -5.05% | +0.21% | ✅ |
| XLRE | -0.79% | -5.30% | -4.51% | ✅ |
| XLU | +1.19% | -0.22% | -1.40% | ❌ |
| XLV | -1.21% | +1.27% | +2.48% | ❌ |
| XLY | +1.73% | -2.26% | -4.00% | ❌ |

<iframe src="plots/error_distribution_october_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLC, XLU, XLY
**Top 3 Actual:** XLK, XLU, XLV
**Overlap:** 1/3 (33%)

**Bottom 3 Predicted:** XLB, XLF, XLP
**Bottom 3 Actual:** XLB, XLC, XLRE
**Overlap:** 1/3 (33%)

---

## ✅ September 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLC 🟢 | +3.74% | LONG |
| 2 | XLK 🟢 | +1.22% | LONG |
| 3 | XLV 🟢 | +0.13% | LONG |
| 4 | XLY ⚪ | -1.04% | NEUTRAL |
| 5 | XLP ⚪ | -1.06% | NEUTRAL |
| 6 | XLU ⚪ | -1.27% | NEUTRAL |
| 7 | XLRE ⚪ | -1.43% | NEUTRAL |
| 8 | XLF ⚪ | -1.81% | NEUTRAL |
| 9 | XLI 🔴 | -2.71% | SHORT |
| 10 | XLE 🔴 | -3.98% | SHORT |
| 11 | XLB 🔴 | -8.82% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-08-29 to 2025-09-30
**SPY Return:** +3.56%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **72.7%** (8/11) | ✅ VERY GOOD |
| Correlation | 0.739 | Very Strong |
| Mean Absolute Error | 1.87% | - |
| R² Score | 0.499 | - |
| **Strategy Return** | **+5.59%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_september_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -8.82% | -5.98% | +2.83% | ✅ |
| XLC | +3.74% | +3.07% | -0.67% | ✅ |
| XLE | -3.98% | -3.88% | +0.10% | ✅ |
| XLF | -1.81% | -3.46% | -1.64% | ✅ |
| XLI | -2.71% | -1.68% | +1.03% | ✅ |
| XLK | +1.22% | +3.97% | +2.75% | ✅ |
| XLP | -1.06% | -5.88% | -4.82% | ✅ |
| XLRE | -1.43% | -3.23% | -1.80% | ✅ |
| XLU | -1.27% | +0.56% | +1.83% | ❌ |
| XLV | +0.13% | -1.83% | -1.96% | ❌ |
| XLY | -1.04% | +0.03% | +1.07% | ❌ |

<iframe src="plots/error_distribution_september_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLC, XLK, XLV
**Top 3 Actual:** XLC, XLK, XLU
**Overlap:** 2/3 (67%)

**Bottom 3 Predicted:** XLB, XLE, XLI
**Bottom 3 Actual:** XLB, XLE, XLP
**Overlap:** 2/3 (67%)

---

## ✅ August 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLY 🟢 | +2.00% | LONG |
| 2 | XLV 🟢 | +1.31% | LONG |
| 3 | XLRE 🟢 | +0.70% | LONG |
| 4 | XLB ⚪ | +0.56% | NEUTRAL |
| 5 | XLP ⚪ | -0.13% | NEUTRAL |
| 6 | XLC ⚪ | -0.32% | NEUTRAL |
| 7 | XLE ⚪ | -2.04% | NEUTRAL |
| 8 | XLF ⚪ | -2.35% | NEUTRAL |
| 9 | XLK 🔴 | -3.44% | SHORT |
| 10 | XLU 🔴 | -3.47% | SHORT |
| 11 | XLI 🔴 | -3.55% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-07-31 to 2025-08-29
**SPY Return:** +2.05%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **72.7%** (8/11) | ✅ VERY GOOD |
| Correlation | 0.776 | Very Strong |
| Mean Absolute Error | 1.67% | - |
| R² Score | 0.189 | - |
| **Strategy Return** | **+4.63%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_august_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | +0.56% | +3.13% | +2.57% | ✅ |
| XLC | -0.32% | +1.65% | +1.97% | ❌ |
| XLE | -2.04% | +1.59% | +3.63% | ❌ |
| XLF | -2.35% | +1.04% | +3.39% | ❌ |
| XLI | -3.55% | -2.05% | +1.49% | ✅ |
| XLK | -3.44% | -2.16% | +1.28% | ✅ |
| XLP | -0.13% | -0.80% | -0.67% | ✅ |
| XLRE | +0.70% | +0.12% | -0.57% | ✅ |
| XLU | -3.47% | -3.63% | -0.15% | ✅ |
| XLV | +1.31% | +3.31% | +2.01% | ✅ |
| XLY | +2.00% | +2.60% | +0.60% | ✅ |

<iframe src="plots/error_distribution_august_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLRE, XLV, XLY
**Top 3 Actual:** XLB, XLV, XLY
**Overlap:** 2/3 (67%)

**Bottom 3 Predicted:** XLI, XLK, XLU
**Bottom 3 Actual:** XLI, XLK, XLU
**Overlap:** 3/3 (100%)

---

## 🔮 Mid_september 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLV 🟢 | +1.56% | LONG |
| 2 | XLE 🟢 | -0.04% | LONG |
| 3 | XLY 🟢 | -0.10% | LONG |
| 4 | XLK ⚪ | -0.26% | NEUTRAL |
| 5 | XLRE ⚪ | -0.48% | NEUTRAL |
| 6 | XLF ⚪ | -1.30% | NEUTRAL |
| 7 | XLI ⚪ | -1.51% | NEUTRAL |
| 8 | XLP ⚪ | -1.67% | NEUTRAL |
| 9 | XLU 🔴 | -1.70% | SHORT |
| 10 | XLB 🔴 | -4.67% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

---


---

*Report auto-generated by `update_monthly_tracking.py`*
*Framework: 4-Model Ensemble (LSTM, TFT, N-BEATS, LSTM-GARCH) with Adaptive Weighting*
*Feature Set: 219 features (20 Alpha + 186 Beta + 10 VIX + 3 Derived)*
*Feature Importance: Permutation-based on trained ensemble models*
