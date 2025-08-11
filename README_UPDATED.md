# ETF Trading Intelligence System (Updated)

## Overview
Advanced deep learning system for ETF sector rotation prediction using neural networks (no boosting methods).

## Key Results
- **Direction Accuracy:** 52.6% (profitable threshold >50%)
- **Validation Method:** Rolling window across Q4 2023 - Q2 2024
- **Best Model:** LSTM-GARCH
- **Data:** 1400+ days from 2019-2024

## Files in This Repository

### Core System
- `validate_all_models.py` - Main validation with rolling windows
- `etf_multi_sector_complete.py` - Complete multi-sector pipeline
- `etf_monthly_prediction_system.py` - Monthly predictions

### Documentation
- `COMPREHENSIVE_REPORT.md` - Full technical report with results
- `VALIDATION_REPORT.md` - Detailed validation metrics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run validation
python validate_all_models.py

# Generate predictions
python etf_monthly_prediction_system.py
```

## Understanding Negative R²

The system shows negative R² scores, which is **normal and expected** for monthly relative return predictions:
- Relative returns (ETF - SPY) are mostly noise
- Target variance is extremely small (~0.02%)
- Direction accuracy (52.6%) is more important for trading

## Models Implemented

1. **LSTM-GARCH** - Best direction accuracy (52.6%)
2. **N-BEATS** - Neural basis expansion
3. **Temporal Fusion Transformer** - Attention mechanism
4. **Wavelet-LSTM** - Multi-scale analysis

All models are deep learning based - no XGBoost, LightGBM, or other boosting methods.

## August 2024 Predictions

- **Long:** XLK (Technology) +2.8%
- **Short:** XLE (Energy) -2.9%

## License
MIT

## Contact
For questions, see COMPREHENSIVE_REPORT.md for technical details.