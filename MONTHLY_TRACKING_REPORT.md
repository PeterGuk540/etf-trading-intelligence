# ETF Trading Intelligence - Monthly Tracking Report
*Living document tracking model performance, feature importance, and predictions*

**Last Updated:** 2026-02-02 15:04 UTC

---

## 📊 Latest Status Dashboard

| Metric | Value |
|--------|-------|
| **Latest Prediction** | February 2026 |
| **Last Validated Month** | January 2026 |
| **Overall Direction Accuracy** | 57.6% |
| **Win Rate (Profitable Months)** | 5/6 (83%) if validated else 'N/A' |
| **Cumulative Strategy Return** | +11.54% |
| **Total Cycles Tracked** | 7 |

---

## 📈 Performance Timeline

<iframe src="plots/performance_timeline.html" width="100%" height="850" frameborder="0"></iframe>


| Month | Direction Accuracy | Correlation | MAE | Strategy Return | Status | Training Through |
|-------|-------------------|-------------|-----|-----------------|--------|------------------|
| February 2026 | *Pending* | *Pending* | *Pending* | *Pending* | 🔮 Predicted | 2026-01-30 |
| January 2026 | 54.5% | -0.217 | 4.55% | -2.68% | ✅ Validated | 2025-12-31 |
| December 2025 | 36.4% | -0.183 | 3.19% | +0.02% | ✅ Validated | 2025-11-28 |
| November 2025 | 45.5% | 0.333 | 4.15% | +0.94% | ✅ Validated | 2025-10-31 |
| October 2025 | 63.6% | 0.268 | 3.29% | +3.05% | ✅ Validated | 2025-09-30T00:00:00 |
| September 2025 | 72.7% | 0.739 | 1.87% | +5.59% | ✅ Validated | 2025-08-29 |
| August 2025 | 72.7% | 0.776 | 1.67% | +4.63% | ✅ Validated | 2025-07-31 |

---

## 🔮 February 2026 Cycle

**Latest Prediction - Awaiting Validation**

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLB 🟢 | +6.73% | LONG |
| 2 | XLE 🟢 | +6.33% | LONG |
| 3 | XLI 🟢 | +4.61% | LONG |
| 4 | XLK ⚪ | +4.17% | NEUTRAL |
| 5 | XLU ⚪ | +1.71% | NEUTRAL |
| 6 | XLP ⚪ | +1.68% | NEUTRAL |
| 7 | XLRE ⚪ | -0.04% | NEUTRAL |
| 8 | XLC ⚪ | -0.10% | NEUTRAL |
| 9 | XLV 🔴 | -1.46% | SHORT |
| 10 | XLY 🔴 | -2.59% | SHORT |
| 11 | XLF 🔴 | -5.49% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

---

## ✅ January 2026 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLI 🟢 | +3.77% | LONG |
| 2 | XLY 🟢 | +2.38% | LONG |
| 3 | XLC 🟢 | +2.10% | LONG |
| 4 | XLF ⚪ | +1.71% | NEUTRAL |
| 5 | XLB ⚪ | -0.64% | NEUTRAL |
| 6 | XLK ⚪ | -0.71% | NEUTRAL |
| 7 | XLP ⚪ | -1.23% | NEUTRAL |
| 8 | XLV ⚪ | -1.56% | NEUTRAL |
| 9 | XLE 🔴 | -2.88% | SHORT |
| 10 | XLRE 🔴 | -3.03% | SHORT |
| 11 | XLU 🔴 | -3.46% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-12-31 to 2026-01-30
**SPY Return:** +1.47%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **54.5%** (6/11) | ⚠️ BELOW THRESHOLD |
| Correlation | -0.217 | Weak |
| Mean Absolute Error | 4.55% | - |
| R² Score | -0.800 | - |
| **Strategy Return** | **-2.68%** | ❌ LOSS |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_january_2026.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -0.64% | +7.17% | +7.81% | ❌ |
| XLC | +2.10% | +0.53% | -1.57% | ✅ |
| XLE | -2.88% | +12.71% | +15.58% | ❌ |
| XLF | +1.71% | -3.90% | -5.61% | ❌ |
| XLI | +3.77% | +5.18% | +1.41% | ✅ |
| XLK | -0.71% | -1.54% | -0.83% | ✅ |
| XLP | -1.23% | +6.03% | +7.26% | ❌ |
| XLRE | -3.03% | +1.20% | +4.23% | ❌ |
| XLU | -3.46% | -0.16% | +3.30% | ✅ |
| XLV | -1.56% | -1.51% | +0.05% | ✅ |
| XLY | +2.38% | +0.00% | -2.38% | ✅ |

<iframe src="plots/error_distribution_january_2026.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLC, XLI, XLY
**Top 3 Actual:** XLB, XLE, XLP
**Overlap:** 0/3 (0%)

**Bottom 3 Predicted:** XLE, XLRE, XLU
**Bottom 3 Actual:** XLF, XLK, XLV
**Overlap:** 0/3 (0%)

---

## ✅ December 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLV 🟢 | +8.21% | LONG |
| 2 | XLY 🟢 | +2.44% | LONG |
| 3 | XLB 🟢 | +1.98% | LONG |
| 4 | XLRE ⚪ | +1.72% | NEUTRAL |
| 5 | XLP ⚪ | +1.54% | NEUTRAL |
| 6 | XLF ⚪ | +0.86% | NEUTRAL |
| 7 | XLU ⚪ | +0.04% | NEUTRAL |
| 8 | XLC ⚪ | -0.17% | NEUTRAL |
| 9 | XLE 🔴 | -1.17% | SHORT |
| 10 | XLI 🔴 | -1.84% | SHORT |
| 11 | XLK 🔴 | -2.88% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-11-28 to 2025-12-31
**SPY Return:** +0.08%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **36.4%** (4/11) | ⚠️ BELOW THRESHOLD |
| Correlation | -0.183 | Weak |
| Mean Absolute Error | 3.19% | - |
| R² Score | -2.172 | - |
| **Strategy Return** | **+0.02%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_december_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | +1.98% | +1.90% | -0.08% | ✅ |
| XLC | -0.17% | +2.27% | +2.44% | ❌ |
| XLE | -1.17% | -0.38% | +0.79% | ✅ |
| XLF | +0.86% | +2.98% | +2.12% | ✅ |
| XLI | -1.84% | +1.20% | +3.04% | ❌ |
| XLK | -2.88% | +0.67% | +3.55% | ❌ |
| XLP | +1.54% | -1.42% | -2.96% | ❌ |
| XLRE | +1.72% | -2.19% | -3.91% | ❌ |
| XLU | +0.04% | -5.17% | -5.21% | ❌ |
| XLV | +8.21% | -1.47% | -9.68% | ❌ |
| XLY | +2.44% | +1.12% | -1.32% | ✅ |

<iframe src="plots/error_distribution_december_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLB, XLV, XLY
**Top 3 Actual:** XLB, XLC, XLF
**Overlap:** 1/3 (33%)

**Bottom 3 Predicted:** XLE, XLI, XLK
**Bottom 3 Actual:** XLRE, XLU, XLV
**Overlap:** 0/3 (0%)

---

## ✅ November 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLV 🟢 | +2.54% | LONG |
| 2 | XLU 🟢 | +1.04% | LONG |
| 3 | XLY 🟢 | -0.47% | LONG |
| 4 | XLF ⚪ | -1.43% | NEUTRAL |
| 5 | XLK ⚪ | -1.97% | NEUTRAL |
| 6 | XLP ⚪ | -2.26% | NEUTRAL |
| 7 | XLI ⚪ | -2.60% | NEUTRAL |
| 8 | XLE ⚪ | -3.56% | NEUTRAL |
| 9 | XLRE 🔴 | -3.83% | SHORT |
| 10 | XLC 🔴 | -3.86% | SHORT |
| 11 | XLB 🔴 | -3.91% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### 🔬 Feature Importance
**Calculated:** 2025-11-10
**Method:** Permutation Importance (10 repeats)

<iframe src="plots/aggregate_feature_importance_november_2025.html" width="100%" height="700" frameborder="0"></iframe>

#### Top 20 Features (Aggregate)

| Rank | Feature | Avg Importance | Std | Category |
|------|---------|----------------|-----|----------|
| 1 | momentum_1w | 5.18% | ±2.39% | Alpha - Technical |
| 2 | fred_consumer_credit_chg_1m | 5.03% | ±2.60% | Beta - Money Supply |
| 3 | fred_consumer_credit_chg_3m | 3.92% | ±0.72% | Beta - Money Supply |
| 4 | fred_consumer_sentiment_chg_3m | 3.84% | ±1.61% | Beta - Sentiment |
| 5 | atr_14d | 3.56% | ±0.33% | Alpha - Technical |
| 6 | fred_consumer_sentiment_chg_1m | 3.52% | ±2.46% | Beta - Sentiment |
| 7 | fred_dollar_index_chg_3m | 3.44% | ±1.78% | Beta - Market |
| 8 | vix_volatility_lag21 | 3.23% | ±0.00% | VIX Regime |
| 9 | fred_usd_eur_chg_3m | 3.15% | ±1.51% | Beta - Market |
| 10 | fred_ppi_metals_chg_3m | 2.94% | ±1.47% | Beta - Inflation |
| 11 | macd_signal | 2.90% | ±1.80% | Alpha - Technical |
| 12 | fred_ppi_metals | 2.68% | ±1.83% | Beta - Inflation |
| 13 | macd_hist | 2.60% | ±0.73% | Alpha - Technical |
| 14 | fred_ppi_metals_chg_1m | 2.57% | ±1.26% | Beta - Inflation |
| 15 | fred_usd_jpy_chg_3m | 2.50% | ±0.88% | Beta - Market |
| 16 | fred_dollar_index | 2.42% | ±0.76% | Beta - Market |
| 17 | fred_usd_eur_chg_1m | 2.31% | ±1.00% | Beta - Market |
| 18 | rsi_14d | 2.31% | ±1.23% | Alpha - Technical |
| 19 | fred_usd_eur | 2.27% | ±0.73% | Beta - Market |
| 20 | momentum_1m | 2.15% | ±1.15% | Alpha - Technical |

#### Category Importance Breakdown

| Category | Importance |
|----------|------------|
| Alpha - Technical | 31.20% |
| Beta - Other | 18.16% |
| Beta - Inflation | 16.33% |
| Beta - Market | 16.09% |
| Beta - Money Supply | 12.99% |
| Beta - Interest Rates | 10.12% |
| Beta - Sentiment | 7.36% |
| Beta - Economic | 6.07% |
| VIX Regime | 3.23% |
| Derived | 2.02% |

<details>
<summary><b>📋 Sector-Specific Feature Importance (Click to expand)</b></summary>


**XLF** - Top 5:
1. fred_consumer_credit_chg_1m (9.52%) - Beta - Money Supply
2. fred_usd_eur_chg_3m (6.59%) - Beta - Market
3. fred_ppi_metals_chg_3m (5.17%) - Beta - Inflation
4. fred_ppi_metals_chg_1m (4.39%) - Beta - Inflation
5. rsi_14d (4.24%) - Alpha - Technical

**XLC** - Top 5:
1. fred_consumer_sentiment_chg_1m (8.28%) - Beta - Sentiment
2. fred_ppi_metals (4.09%) - Beta - Inflation
3. macd_hist (2.96%) - Alpha - Technical
4. macd (2.64%) - Alpha - Technical
5. fred_building_permits_chg_1m (2.54%) - Beta - Other

**XLY** - Top 5:
1. fred_consumer_credit_chg_1m (6.23%) - Beta - Money Supply
2. fred_usd_eur (3.45%) - Beta - Market
3. atr_14d (3.05%) - Alpha - Technical
4. high_20d (2.64%) - Alpha - Technical
5. price_position (2.51%) - Alpha - Technical

**XLP** - Top 5:
1. fred_investment_grade_spread_chg_1m (2.18%) - Beta - Other
2. fred_treasury_5y_chg_3m (2.13%) - Beta - Interest Rates
3. fred_capacity_utilization_chg_1m (2.10%) - Beta - Other
4. fred_gas_price_chg_3m (2.10%) - Beta - Inflation
5. vix_above_sma20_lag21 (2.09%) - VIX Regime

**XLE** - Top 5:
1. fred_ppi_metals (6.05%) - Beta - Inflation
2. momentum_1w (5.35%) - Alpha - Technical
3. fred_consumer_sentiment_chg_3m (5.34%) - Beta - Sentiment
4. fred_usd_eur_chg_1m (4.41%) - Beta - Market
5. fred_consumer_credit_chg_3m (4.35%) - Beta - Money Supply

**XLV** - Top 5:
1. momentum_1w (2.82%) - Alpha - Technical
2. fred_inflation_5y_chg_1m (2.25%) - Beta - Inflation
3. fred_building_permits (2.20%) - Beta - Other
4. fred_usd_eur_chg_3m (2.12%) - Beta - Market
5. fred_investment_grade_spread (2.08%) - Beta - Other

**XLI** - Top 5:
1. momentum_1w (9.00%) - Alpha - Technical
2. fred_consumer_sentiment_chg_3m (6.55%) - Beta - Sentiment
3. fred_dollar_index_chg_3m (5.83%) - Beta - Market
4. macd_signal (5.19%) - Alpha - Technical
5. fred_consumer_credit_chg_3m (4.43%) - Beta - Money Supply

**XLB** - Top 5:
1. fred_consumer_sentiment_chg_3m (3.96%) - Beta - Sentiment
2. macd_hist (3.91%) - Alpha - Technical
3. fred_business_loans_chg_1m (3.79%) - Beta - Money Supply
4. momentum_1w (3.54%) - Alpha - Technical
5. fred_cpi_chg_1m (3.27%) - Beta - Inflation

**XLRE** - Top 5:
1. fred_building_permits_chg_3m (2.16%) - Beta - Other
2. fred_prime_rate_chg_3m (2.14%) - Beta - Other
3. fred_capacity_utilization_chg_1m (2.12%) - Beta - Other
4. fred_inflation_10y_chg_3m (2.12%) - Beta - Inflation
5. fred_imports_chg_3m (2.09%) - Beta - Other

**XLK** - Top 5:
1. fred_consumer_sentiment_chg_3m (4.28%) - Beta - Sentiment
2. momentum_1m (4.04%) - Alpha - Technical
3. fred_consumer_credit_chg_1m (3.84%) - Beta - Money Supply
4. atr_14d (3.61%) - Alpha - Technical
5. rsi_14d (3.18%) - Alpha - Technical

**XLU** - Top 5:
1. fred_consumer_sentiment_chg_3m (2.37%) - Beta - Sentiment
2. fred_ppi_metals (2.29%) - Beta - Inflation
3. fred_yield_curve_10y3m_chg_3m (2.17%) - Beta - Interest Rates
4. fred_usd_eur (2.10%) - Beta - Market
5. high_20d (2.09%) - Alpha - Technical

</details>

### ✅ Validation Results
**Period:** 2025-10-31 to 2025-11-28
**SPY Return:** +0.20%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **45.5%** (5/11) | ⚠️ BELOW THRESHOLD |
| Correlation | 0.333 | Moderate |
| Mean Absolute Error | 4.15% | - |
| R² Score | -0.915 | - |
| **Strategy Return** | **+0.94%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_november_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -3.91% | +4.16% | +8.06% | ❌ |
| XLC | -3.86% | +0.31% | +4.17% | ❌ |
| XLE | -3.56% | +2.44% | +6.00% | ❌ |
| XLF | -1.43% | +1.64% | +3.06% | ❌ |
| XLI | -2.60% | -1.08% | +1.52% | ✅ |
| XLK | -1.97% | -5.00% | -3.04% | ✅ |
| XLP | -2.26% | +3.86% | +6.11% | ❌ |
| XLRE | -3.83% | +1.69% | +5.52% | ❌ |
| XLU | +1.04% | +1.52% | +0.48% | ✅ |
| XLV | +2.54% | +9.09% | +6.55% | ✅ |
| XLY | -0.47% | -1.64% | -1.17% | ✅ |

<iframe src="plots/error_distribution_november_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLU, XLV, XLY
**Top 3 Actual:** XLB, XLP, XLV
**Overlap:** 1/3 (33%)

**Bottom 3 Predicted:** XLB, XLC, XLRE
**Bottom 3 Actual:** XLI, XLK, XLY
**Overlap:** 0/3 (0%)

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


---

*Report auto-generated by `update_monthly_tracking.py`*
*Framework: 4-Model Ensemble (LSTM, TFT, N-BEATS, LSTM-GARCH) with Adaptive Weighting*
*Feature Set: 219 features (20 Alpha + 186 Beta + 10 VIX + 3 Derived)*
*Feature Importance: Permutation-based on trained ensemble models*
