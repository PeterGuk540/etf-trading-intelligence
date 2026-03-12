# ETF Trading Intelligence - Monthly Tracking Report
*Living document tracking model performance, feature importance, and predictions*

**Last Updated:** 2026-03-12 15:19 UTC

---

## 📊 Latest Status Dashboard

| Metric | Value |
|--------|-------|
| **Latest Prediction** | March 2026 |
| **Last Validated Month** | February 2026 |
| **Overall Direction Accuracy** | 59.7% |
| **Win Rate (Profitable Months)** | 5/7 (71%) if validated else 'N/A' |
| **Cumulative Strategy Return** | +12.65% |
| **Total Cycles Tracked** | 8 |

---

## 📈 Performance Timeline

<iframe src="plots/performance_timeline.html" width="100%" height="850" frameborder="0"></iframe>


| Month | Direction Accuracy | Correlation | MAE | Strategy Return | Status | Training Through |
|-------|-------------------|-------------|-----|-----------------|--------|------------------|
| March 2026 | *Pending* | *Pending* | *Pending* | *Pending* | 🔮 Predicted | *Unknown* |
| February 2026 | 63.6% | 0.476 | 5.19% | +4.76% | ✅ Validated | 2026-01-30 |
| January 2026 | 54.5% | -0.172 | 4.71% | -4.17% | ✅ Validated | 2025-12-31 |
| December 2025 | 36.4% | -0.120 | 2.71% | +1.15% | ✅ Validated | 2025-11-28 |
| November 2025 | 45.5% | 0.029 | 3.48% | -0.47% | ✅ Validated | 2025-10-31 |
| October 2025 | 90.9% | 0.551 | 2.66% | +5.85% | ✅ Validated | 2025-09-30T00:00:00 |
| September 2025 | 81.8% | -0.096 | 2.88% | +4.45% | ✅ Validated | 2025-08-29 |
| August 2025 | 45.5% | 0.406 | 1.84% | +1.07% | ✅ Validated | 2025-07-31 |

---

## 🔮 March 2026 Cycle

**Latest Prediction - Awaiting Validation**

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLU 🟢 | +5.71% | LONG |
| 2 | XLE 🟢 | +5.07% | LONG |
| 3 | XLRE 🟢 | +3.49% | LONG |
| 4 | XLI ⚪ | +2.05% | NEUTRAL |
| 5 | XLP ⚪ | +1.84% | NEUTRAL |
| 6 | XLB ⚪ | +0.90% | NEUTRAL |
| 7 | XLC ⚪ | +0.82% | NEUTRAL |
| 8 | XLV ⚪ | +0.20% | NEUTRAL |
| 9 | XLK 🔴 | -0.36% | SHORT |
| 10 | XLY 🔴 | -1.99% | SHORT |
| 11 | XLF 🔴 | -2.48% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### 🔬 Feature Importance
**Calculated:** 2026-02-28
**Method:** Permutation Importance (10 repeats)

<iframe src="plots/aggregate_feature_importance_march_2026.html" width="100%" height="700" frameborder="0"></iframe>

#### Top 20 Features (Aggregate)

| Rank | Feature | Avg Importance | Std | Category |
|------|---------|----------------|-----|----------|
| 1 | fred_consumer_sentiment_chg_3m | 4.31% | ±1.73% | Beta - Sentiment |
| 2 | fred_consumer_credit_chg_1m | 3.61% | ±2.84% | Beta - Money Supply |
| 3 | fred_consumer_credit_chg_3m | 3.42% | ±1.35% | Beta - Money Supply |
| 4 | fred_ppi_metals_chg_1m | 3.12% | ±1.16% | Beta - Inflation |
| 5 | atr_14d | 3.06% | ±1.54% | Alpha - Technical |
| 6 | high_20d | 3.05% | ±0.00% | Alpha - Technical |
| 7 | fred_business_loans_chg_1m | 2.99% | ±0.77% | Beta - Money Supply |
| 8 | mfi_14d | 2.98% | ±2.37% | Alpha - Technical |
| 9 | fred_exports_chg_1m | 2.94% | ±0.62% | Beta - Other |
| 10 | fred_usd_eur_chg_1m | 2.74% | ±1.34% | Beta - Market |
| 11 | low_20d | 2.71% | ±0.11% | Alpha - Technical |
| 12 | fred_bank_reserves_chg_1m | 2.70% | ±1.15% | Beta - Other |
| 13 | macd | 2.61% | ±1.17% | Alpha - Technical |
| 14 | fred_imports_chg_1m | 2.59% | ±0.60% | Beta - Other |
| 15 | fred_exports_chg_3m | 2.59% | ±0.72% | Beta - Other |
| 16 | price_position | 2.54% | ±1.51% | Alpha - Technical |
| 17 | fred_imports_chg_3m | 2.51% | ±0.81% | Beta - Other |
| 18 | macd_hist | 2.46% | ±0.71% | Alpha - Technical |
| 19 | fred_yield_curve_10y3m_chg_3m | 2.43% | ±0.85% | Beta - Interest Rates |
| 20 | vix_volatility_lag21 | 2.42% | ±1.16% | VIX Regime |

#### Category Importance Breakdown

| Category | Importance |
|----------|------------|
| Alpha - Technical | 32.39% |
| Beta - Other | 21.30% |
| Beta - Money Supply | 18.44% |
| Beta - Interest Rates | 12.63% |
| Beta - Market | 11.06% |
| Beta - Economic | 7.97% |
| Beta - Inflation | 7.57% |
| Beta - Sentiment | 6.52% |
| VIX Regime | 2.42% |

<details>
<summary><b>📋 Sector-Specific Feature Importance (Click to expand)</b></summary>


**XLF** - Top 5:
1. fred_consumer_credit_chg_3m (4.90%) - Beta - Money Supply
2. fred_business_loans_chg_1m (4.19%) - Beta - Money Supply
3. vix_volatility_lag21 (3.71%) - VIX Regime
4. fred_core_cpi_chg_3m (3.28%) - Beta - Inflation
5. high_20d (3.05%) - Alpha - Technical

**XLC** - Top 5:
1. fred_consumer_credit_chg_1m (2.71%) - Beta - Money Supply
2. fred_vix_chg_1m (2.64%) - Beta - Market
3. fred_exports_chg_3m (2.53%) - Beta - Other
4. fred_usd_jpy_chg_1m (2.50%) - Beta - Market
5. fred_m2_money_chg_1m (2.41%) - Beta - Money Supply

**XLY** - Top 5:
1. atr_14d (5.25%) - Alpha - Technical
2. fred_consumer_sentiment_chg_3m (4.61%) - Beta - Sentiment
3. fred_exports_chg_1m (3.78%) - Beta - Other
4. fred_exports_chg_3m (2.96%) - Beta - Other
5. fred_ppi_metals (2.92%) - Beta - Inflation

**XLP** - Top 5:
1. fred_consumer_credit_chg_3m (5.24%) - Beta - Money Supply
2. fred_yield_curve_10y3m_chg_3m (4.05%) - Beta - Interest Rates
3. momentum_1w (4.03%) - Alpha - Technical
4. vix_volatility_lag21 (3.93%) - VIX Regime
5. macd_hist (3.09%) - Alpha - Technical

**XLE** - Top 5:
1. momentum_1m (2.23%) - Alpha - Technical
2. fred_vix (2.19%) - Beta - Market
3. fred_high_yield_spread_chg_3m (2.18%) - Beta - Interest Rates
4. fred_cpi_chg_1m (2.16%) - Beta - Inflation
5. fred_capacity_utilization_chg_1m (2.16%) - Beta - Other

**XLV** - Top 5:
1. fred_exports_chg_1m (3.48%) - Beta - Other
2. fred_consumer_sentiment_chg_1m (3.02%) - Beta - Sentiment
3. fred_business_loans_chg_3m (2.75%) - Beta - Money Supply
4. ratio_momentum (2.61%) - Alpha - Technical
5. fred_consumer_credit_chg_3m (2.58%) - Beta - Money Supply

**XLI** - Top 5:
1. fred_consumer_credit_chg_1m (9.61%) - Beta - Money Supply
2. atr_14d (4.51%) - Alpha - Technical
3. fred_business_loans_chg_3m (3.44%) - Beta - Money Supply
4. fred_imports_chg_3m (3.23%) - Beta - Other
5. fred_ppi_metals_chg_1m (3.12%) - Beta - Inflation

**XLB** - Top 5:
1. fred_consumer_sentiment_chg_3m (7.79%) - Beta - Sentiment
2. price_position (5.82%) - Alpha - Technical
3. fred_usd_eur_chg_1m (4.97%) - Beta - Market
4. fred_imports_chg_1m (3.69%) - Beta - Other
5. fred_exports_chg_1m (3.18%) - Beta - Other

**XLRE** - Top 5:
1. fred_consumer_sentiment_chg_3m (5.52%) - Beta - Sentiment
2. fred_imports_chg_3m (3.67%) - Beta - Other
3. fred_consumer_sentiment_chg_1m (3.48%) - Beta - Sentiment
4. fred_imports_chg_1m (2.66%) - Beta - Other
5. vix_regime_low_vol_lag21 (2.53%) - VIX Regime

**XLK** - Top 5:
1. fred_consumer_credit_chg_1m (7.15%) - Beta - Money Supply
2. macd (5.14%) - Alpha - Technical
3. fred_ppi_metals_chg_1m (5.06%) - Beta - Inflation
4. fred_bank_reserves_chg_1m (4.64%) - Beta - Other
5. fred_consumer_sentiment_chg_3m (4.02%) - Beta - Sentiment

**XLU** - Top 5:
1. mfi_14d (7.07%) - Alpha - Technical
2. fred_consumer_credit_chg_3m (5.24%) - Beta - Money Supply
3. macd_signal (4.39%) - Alpha - Technical
4. fred_ppi_metals_chg_1m (4.10%) - Beta - Inflation
5. fred_exports_chg_3m (3.95%) - Beta - Other

</details>

---

## ✅ February 2026 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLP 🟢 | +3.64% | LONG |
| 2 | XLI 🟢 | +3.44% | LONG |
| 3 | XLE 🟢 | +3.10% | LONG |
| 4 | XLB ⚪ | +2.52% | NEUTRAL |
| 5 | XLK ⚪ | +1.38% | NEUTRAL |
| 6 | XLC ⚪ | +0.55% | NEUTRAL |
| 7 | XLRE ⚪ | +0.13% | NEUTRAL |
| 8 | XLY ⚪ | -0.87% | NEUTRAL |
| 9 | XLU 🔴 | -1.05% | SHORT |
| 10 | XLF 🔴 | -1.52% | SHORT |
| 11 | XLV 🔴 | -1.67% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2026-01-30 to 2026-02-27
**SPY Return:** -0.86%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **63.6%** (7/11) | 👍 GOOD |
| Correlation | 0.476 | Moderate |
| Mean Absolute Error | 5.19% | - |
| R² Score | -0.232 | - |
| **Strategy Return** | **+4.76%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_february_2026.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | +2.52% | +9.27% | +6.75% | ✅ |
| XLC | +0.55% | -0.83% | -1.37% | ❌ |
| XLE | +3.10% | +10.40% | +7.30% | ✅ |
| XLF | -1.52% | -2.90% | -1.38% | ✅ |
| XLI | +3.44% | +7.94% | +4.50% | ✅ |
| XLK | +1.38% | -2.69% | -4.07% | ❌ |
| XLP | +3.64% | +8.65% | +5.01% | ✅ |
| XLRE | +0.13% | +6.68% | +6.55% | ✅ |
| XLU | -1.05% | +11.22% | +12.28% | ❌ |
| XLV | -1.67% | +4.39% | +6.06% | ❌ |
| XLY | -0.87% | -2.69% | -1.82% | ✅ |

<iframe src="plots/error_distribution_february_2026.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLE, XLI, XLP
**Top 3 Actual:** XLB, XLE, XLU
**Overlap:** 1/3 (33%)

**Bottom 3 Predicted:** XLF, XLU, XLV
**Bottom 3 Actual:** XLF, XLK, XLY
**Overlap:** 1/3 (33%)

---

## ✅ January 2026 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLK 🟢 | +3.62% | LONG |
| 2 | XLY 🟢 | +2.03% | LONG |
| 3 | XLF 🟢 | +0.93% | LONG |
| 4 | XLC ⚪ | +0.62% | NEUTRAL |
| 5 | XLI ⚪ | +0.41% | NEUTRAL |
| 6 | XLB ⚪ | +0.21% | NEUTRAL |
| 7 | XLE ⚪ | -0.25% | NEUTRAL |
| 8 | XLV ⚪ | -1.14% | NEUTRAL |
| 9 | XLRE 🔴 | -1.76% | SHORT |
| 10 | XLP 🔴 | -1.91% | SHORT |
| 11 | XLU 🔴 | -3.87% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-12-31 to 2026-01-30
**SPY Return:** +1.47%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **54.5%** (6/11) | ⚠️ BELOW THRESHOLD |
| Correlation | -0.172 | Weak |
| Mean Absolute Error | 4.71% | - |
| R² Score | -0.595 | - |
| **Strategy Return** | **-4.17%** | ❌ LOSS |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_january_2026.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | +0.21% | +7.17% | +6.96% | ✅ |
| XLC | +0.62% | +0.53% | -0.09% | ✅ |
| XLE | -0.25% | +12.71% | +12.96% | ❌ |
| XLF | +0.93% | -3.90% | -4.84% | ❌ |
| XLI | +0.41% | +5.18% | +4.77% | ✅ |
| XLK | +3.62% | -1.54% | -5.16% | ❌ |
| XLP | -1.91% | +6.03% | +7.94% | ❌ |
| XLRE | -1.76% | +1.20% | +2.96% | ❌ |
| XLU | -3.87% | -0.16% | +3.71% | ✅ |
| XLV | -1.14% | -1.51% | -0.37% | ✅ |
| XLY | +2.03% | +0.00% | -2.03% | ✅ |

<iframe src="plots/error_distribution_january_2026.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLF, XLK, XLY
**Top 3 Actual:** XLB, XLE, XLP
**Overlap:** 0/3 (0%)

**Bottom 3 Predicted:** XLP, XLRE, XLU
**Bottom 3 Actual:** XLF, XLK, XLV
**Overlap:** 0/3 (0%)

---

## ✅ December 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLV 🟢 | +4.09% | LONG |
| 2 | XLP 🟢 | +1.69% | LONG |
| 3 | XLF 🟢 | +1.68% | LONG |
| 4 | XLRE ⚪ | +1.13% | NEUTRAL |
| 5 | XLE ⚪ | +0.76% | NEUTRAL |
| 6 | XLB ⚪ | +0.34% | NEUTRAL |
| 7 | XLC ⚪ | +0.14% | NEUTRAL |
| 8 | XLI ⚪ | -0.64% | NEUTRAL |
| 9 | XLU 🔴 | -0.73% | SHORT |
| 10 | XLY 🔴 | -1.20% | SHORT |
| 11 | XLK 🔴 | -2.41% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-11-28 to 2025-12-31
**SPY Return:** +0.08%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **36.4%** (4/11) | ⚠️ BELOW THRESHOLD |
| Correlation | -0.120 | Weak |
| Mean Absolute Error | 2.71% | - |
| R² Score | -0.758 | - |
| **Strategy Return** | **+1.15%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_december_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | +0.34% | +1.90% | +1.55% | ✅ |
| XLC | +0.14% | +2.27% | +2.13% | ✅ |
| XLE | +0.76% | -0.38% | -1.13% | ❌ |
| XLF | +1.68% | +2.98% | +1.30% | ✅ |
| XLI | -0.64% | +1.20% | +1.84% | ❌ |
| XLK | -2.41% | +0.67% | +3.08% | ❌ |
| XLP | +1.69% | -1.42% | -3.11% | ❌ |
| XLRE | +1.13% | -2.19% | -3.32% | ❌ |
| XLU | -0.73% | -5.17% | -4.44% | ✅ |
| XLV | +4.09% | -1.47% | -5.55% | ❌ |
| XLY | -1.20% | +1.12% | +2.32% | ❌ |

<iframe src="plots/error_distribution_december_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLF, XLP, XLV
**Top 3 Actual:** XLB, XLC, XLF
**Overlap:** 1/3 (33%)

**Bottom 3 Predicted:** XLK, XLU, XLY
**Bottom 3 Actual:** XLRE, XLU, XLV
**Overlap:** 1/3 (33%)

---

## ✅ November 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLU 🟢 | +1.79% | LONG |
| 2 | XLK 🟢 | +1.73% | LONG |
| 3 | XLE 🟢 | +1.15% | LONG |
| 4 | XLV ⚪ | +0.86% | NEUTRAL |
| 5 | XLP ⚪ | -0.89% | NEUTRAL |
| 6 | XLI ⚪ | -1.18% | NEUTRAL |
| 7 | XLF ⚪ | -1.53% | NEUTRAL |
| 8 | XLB ⚪ | -2.03% | NEUTRAL |
| 9 | XLY 🔴 | -2.32% | SHORT |
| 10 | XLRE 🔴 | -2.36% | SHORT |
| 11 | XLC 🔴 | -2.46% | SHORT |

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
| Correlation | 0.029 | Weak |
| Mean Absolute Error | 3.48% | - |
| R² Score | -0.601 | - |
| **Strategy Return** | **-0.47%** | ❌ LOSS |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_november_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -2.03% | +4.16% | +6.19% | ❌ |
| XLC | -2.46% | +0.31% | +2.77% | ❌ |
| XLE | +1.15% | +2.44% | +1.29% | ✅ |
| XLF | -1.53% | +1.64% | +3.17% | ❌ |
| XLI | -1.18% | -1.08% | +0.11% | ✅ |
| XLK | +1.73% | -5.00% | -6.74% | ❌ |
| XLP | -0.89% | +3.86% | +4.74% | ❌ |
| XLRE | -2.36% | +1.69% | +4.05% | ❌ |
| XLU | +1.79% | +1.52% | -0.26% | ✅ |
| XLV | +0.86% | +9.09% | +8.24% | ✅ |
| XLY | -2.32% | -1.64% | +0.68% | ✅ |

<iframe src="plots/error_distribution_november_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLE, XLK, XLU
**Top 3 Actual:** XLB, XLP, XLV
**Overlap:** 0/3 (0%)

**Bottom 3 Predicted:** XLC, XLRE, XLY
**Bottom 3 Actual:** XLI, XLK, XLY
**Overlap:** 1/3 (33%)

---

## ✅ October 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLK 🟢 | +2.83% | LONG |
| 2 | XLU 🟢 | -0.15% | LONG |
| 3 | XLY 🟢 | -0.79% | LONG |
| 4 | XLV ⚪ | -0.87% | NEUTRAL |
| 5 | XLI ⚪ | -1.12% | NEUTRAL |
| 6 | XLF ⚪ | -1.41% | NEUTRAL |
| 7 | XLB ⚪ | -1.81% | NEUTRAL |
| 8 | XLE ⚪ | -2.26% | NEUTRAL |
| 9 | XLRE 🔴 | -2.54% | SHORT |
| 10 | XLP 🔴 | -3.82% | SHORT |
| 11 | XLC 🔴 | -14.53% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-09-30T00:00:00 to 2025-10-31T00:00:00
**SPY Return:** +2.38%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **90.9%** (10/11) | 🏆 EXCELLENT |
| Correlation | 0.551 | Strong |
| Mean Absolute Error | 2.66% | - |
| R² Score | -0.244 | - |
| **Strategy Return** | **+5.85%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_october_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -1.81% | -6.79% | -4.98% | ✅ |
| XLC | -14.53% | -5.39% | +9.14% | ✅ |
| XLE | -2.26% | -3.74% | -1.48% | ✅ |
| XLF | -1.41% | -5.17% | -3.76% | ✅ |
| XLI | -1.12% | -1.85% | -0.73% | ✅ |
| XLK | +2.83% | +4.29% | +1.47% | ✅ |
| XLP | -3.82% | -5.05% | -1.23% | ✅ |
| XLRE | -2.54% | -5.30% | -2.76% | ✅ |
| XLU | -0.15% | -0.22% | -0.06% | ✅ |
| XLV | -0.87% | +1.27% | +2.14% | ❌ |
| XLY | -0.79% | -2.26% | -1.48% | ✅ |

<iframe src="plots/error_distribution_october_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLK, XLU, XLY
**Top 3 Actual:** XLK, XLU, XLV
**Overlap:** 2/3 (67%)

**Bottom 3 Predicted:** XLC, XLP, XLRE
**Bottom 3 Actual:** XLB, XLC, XLRE
**Overlap:** 2/3 (67%)

---

## ✅ September 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLK 🟢 | +1.47% | LONG |
| 2 | XLU 🟢 | +1.25% | LONG |
| 3 | XLY 🟢 | +0.61% | LONG |
| 4 | XLE ⚪ | +0.07% | NEUTRAL |
| 5 | XLI ⚪ | -0.93% | NEUTRAL |
| 6 | XLV ⚪ | -1.98% | NEUTRAL |
| 7 | XLRE ⚪ | -2.01% | NEUTRAL |
| 8 | XLF ⚪ | -3.04% | NEUTRAL |
| 9 | XLB 🔴 | -3.08% | SHORT |
| 10 | XLP 🔴 | -3.99% | SHORT |
| 11 | XLC 🔴 | -13.55% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-08-29 to 2025-09-30
**SPY Return:** +3.56%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **81.8%** (9/11) | 🏆 EXCELLENT |
| Correlation | -0.096 | Weak |
| Mean Absolute Error | 2.88% | - |
| R² Score | -1.871 | - |
| **Strategy Return** | **+4.45%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_september_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -3.08% | -5.98% | -2.90% | ✅ |
| XLC | -13.55% | +3.07% | +16.62% | ❌ |
| XLE | +0.07% | -3.88% | -3.95% | ❌ |
| XLF | -3.04% | -3.46% | -0.41% | ✅ |
| XLI | -0.93% | -1.68% | -0.75% | ✅ |
| XLK | +1.47% | +3.97% | +2.50% | ✅ |
| XLP | -3.99% | -5.88% | -1.89% | ✅ |
| XLRE | -2.01% | -3.23% | -1.21% | ✅ |
| XLU | +1.25% | +0.56% | -0.69% | ✅ |
| XLV | -1.98% | -1.83% | +0.15% | ✅ |
| XLY | +0.61% | +0.03% | -0.58% | ✅ |

<iframe src="plots/error_distribution_september_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLK, XLU, XLY
**Top 3 Actual:** XLC, XLK, XLU
**Overlap:** 2/3 (67%)

**Bottom 3 Predicted:** XLB, XLC, XLP
**Bottom 3 Actual:** XLB, XLE, XLP
**Overlap:** 2/3 (67%)

---

## ✅ August 2025 Cycle

### 📊 Predicted Returns (vs SPY)

| Rank | ETF | Predicted Return | Recommendation |
|------|-----|------------------|----------------|
| 1 | XLV 🟢 | +4.98% | LONG |
| 2 | XLY 🟢 | +1.94% | LONG |
| 3 | XLK 🟢 | +1.30% | LONG |
| 4 | XLC ⚪ | +1.10% | NEUTRAL |
| 5 | XLP ⚪ | +0.30% | NEUTRAL |
| 6 | XLRE ⚪ | -0.09% | NEUTRAL |
| 7 | XLI ⚪ | -0.14% | NEUTRAL |
| 8 | XLE ⚪ | -0.48% | NEUTRAL |
| 9 | XLF 🔴 | -0.69% | SHORT |
| 10 | XLU 🔴 | -0.96% | SHORT |
| 11 | XLB 🔴 | -1.04% | SHORT |

**Trading Strategy:** Long top 3, Short bottom 3

### ✅ Validation Results
**Period:** 2025-07-31 to 2025-08-29
**SPY Return:** +2.05%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **45.5%** (5/11) | ⚠️ BELOW THRESHOLD |
| Correlation | 0.406 | Moderate |
| Mean Absolute Error | 1.84% | - |
| R² Score | 0.043 | - |
| **Strategy Return** | **+1.07%** | ✅ PROFITABLE |

#### Prediction vs Actual

<iframe src="plots/pred_vs_actual_august_2025.html" width="100%" height="600" frameborder="0"></iframe>

| ETF | Predicted | Actual | Error | Direction |
|-----|-----------|--------|-------|-----------|
| XLB | -1.04% | +3.13% | +4.17% | ❌ |
| XLC | +1.10% | +1.65% | +0.55% | ✅ |
| XLE | -0.48% | +1.59% | +2.07% | ❌ |
| XLF | -0.69% | +1.04% | +1.73% | ❌ |
| XLI | -0.14% | -2.05% | -1.91% | ✅ |
| XLK | +1.30% | -2.16% | -3.46% | ❌ |
| XLP | +0.30% | -0.80% | -1.10% | ❌ |
| XLRE | -0.09% | +0.12% | +0.21% | ❌ |
| XLU | -0.96% | -3.63% | -2.67% | ✅ |
| XLV | +4.98% | +3.31% | -1.67% | ✅ |
| XLY | +1.94% | +2.60% | +0.66% | ✅ |

<iframe src="plots/error_distribution_august_2025.html" width="100%" height="500" frameborder="0"></iframe>

#### Top/Bottom 3 Analysis

**Top 3 Predicted:** XLK, XLV, XLY
**Top 3 Actual:** XLB, XLV, XLY
**Overlap:** 2/3 (67%)

**Bottom 3 Predicted:** XLB, XLF, XLU
**Bottom 3 Actual:** XLI, XLK, XLU
**Overlap:** 1/3 (33%)

---


---

*Report auto-generated by `update_monthly_tracking.py`*
*Framework: 7-Model Ensemble (LSTM, TFT, N-BEATS, LSTM-GARCH, LightGBM, CatBoost, SARIMAX) with Adaptive Weighting*
*Feature Set: 219 features (20 Alpha + 186 Beta + 10 VIX + 3 Derived)*
*Feature Importance: Permutation-based on trained ensemble models*
