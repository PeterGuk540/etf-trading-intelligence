"""
ETF Trading Intelligence System Status
Shows current configuration and feature set
"""

from etf_monthly_prediction_system import MonthlyPredictionPipeline
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

def show_system_status():
    """Display complete system status and configuration"""
    
    print("="*80)
    print("ETF TRADING INTELLIGENCE SYSTEM STATUS")
    print("="*80)
    
    # Date configuration
    TODAY = datetime.now()
    CURRENT_MONTH = datetime(TODAY.year, TODAY.month, 1)
    TWO_MONTHS_AGO = TODAY - relativedelta(months=2)
    VAL_END = datetime(TWO_MONTHS_AGO.year, TWO_MONTHS_AGO.month, 1) + relativedelta(months=1) - relativedelta(days=1)
    VAL_START = datetime(TWO_MONTHS_AGO.year, TWO_MONTHS_AGO.month, 1)
    TRAIN_END = VAL_START - relativedelta(days=1)
    
    print(f"\n📅 DYNAMIC DATE CONFIGURATION (Today: {TODAY.date()}):")
    print(f"  Training Period: 2019-01-01 to {TRAIN_END.date()}")
    print(f"  Validation Period: {VAL_START.date()} to {VAL_END.date()}")
    print(f"  Prediction Month: {CURRENT_MONTH.strftime('%B %Y')}")
    print(f"  Prediction Horizon: 21 trading days (1 month)")
    
    # ETF sectors
    SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
    
    print(f"\n🏢 ETF SECTORS ({len(SECTOR_ETFS)} total):")
    sectors = {
        'XLF': 'Financials',
        'XLC': 'Communication Services',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLK': 'Technology',
        'XLU': 'Utilities'
    }
    for etf in SECTOR_ETFS:
        print(f"  {etf}: {sectors[etf]}")
    
    # Feature configuration
    pipeline = MonthlyPredictionPipeline()
    
    print(f"\n📊 FEATURE CONFIGURATION:")
    print(f"\n  ALPHA FACTORS (20 Technical Indicators per ETF):")
    alpha_factors = [
        "1. Momentum (1 week)",
        "2. Momentum (1 month)",
        "3. RSI (14-day)",
        "4. Volatility (21-day)",
        "5. Sharpe Ratio (10-day)",
        "6. Ratio Momentum (ETF/SPY)",
        "7. Volume Ratio (5d/20d)",
        "8. MACD",
        "9. MACD Signal",
        "10. MACD Histogram",
        "11. Bollinger Band %B",
        "12. KDJ K",
        "13. KDJ D",
        "14. KDJ J",
        "15. ATR (14-day)",
        "16. Price High (20-day)",
        "17. Price Low (20-day)",
        "18. MFI (14-day)",
        "19. VWAP",
        "20. Price Position"
    ]
    for factor in alpha_factors:
        print(f"    {factor}")
    
    print(f"\n  BETA FACTORS ({len(pipeline.fred_indicators)} Economic Indicators x 3 variations):")
    
    # Categorize FRED indicators
    categories = {
        'Interest Rates & Yields': ['DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30', 
                                   'DFEDTARU', 'DFEDTARL', 'TB3MS', 'TB6MS', 'MORTGAGE30US'],
        'Yield Curves & Spreads': ['T10Y2Y', 'T10Y3M', 'T5YIE', 'T10YIE', 
                                   'TEDRATE', 'BAMLH0A0HYM2', 'BAMLC0A0CM', 'DPRIME'],
        'Economic Activity': ['GDP', 'GDPC1', 'INDPRO', 'CAPACITY', 'RETAILSL', 
                             'HOUST', 'PERMIT', 'AUTOSOLD', 'DEXPORT', 'DIMPORT', 
                             'NETEXP', 'BUSLOANS'],
        'Employment': ['UNRATE', 'EMRATIO', 'CIVPART', 'NFCI', 'ICSA', 
                      'PAYEMS', 'AWHAETP', 'AWHMAN'],
        'Inflation & Prices': ['CPIAUCSL', 'CPILFESL', 'PPIACO', 'PPIFGS', 
                              'GASREGW', 'DCOILWTICO', 'DCOILBRENTEU', 'GOLDAMGBD228NLBM'],
        'Money Supply & Credit': ['M1SL', 'M2SL', 'BOGMBASE', 'TOTRESNS', 
                                 'CONSUMER', 'TOTALSL'],
        'Market Indicators': ['VIXCLS', 'DEXUSEU', 'DEXJPUS', 'DEXUSUK', 'DXY'],
        'Sentiment': ['UMCSENT', 'CBCCI', 'USSLIND', 'USCCCI', 'OEMV']
    }
    
    for category, indicators in categories.items():
        print(f"    {category}: {len(indicators)} indicators")
    
    print(f"\n    Each indicator has 3 variations:")
    print(f"      • Raw value")
    print(f"      • 1-month change")
    print(f"      • 3-month change")
    print(f"    Total beta features: {len(pipeline.fred_indicators) * 3} per ETF")
    
    # Total features
    total_features_per_etf = 20 + (len(pipeline.fred_indicators) * 3)
    total_features_all = total_features_per_etf * len(SECTOR_ETFS)
    
    print(f"\n📈 TOTAL FEATURE COUNT:")
    print(f"  Per ETF: {total_features_per_etf} features")
    print(f"  All ETFs: {total_features_all} features")
    
    # Model information
    print(f"\n🤖 MODELS:")
    print("  Primary Model: LSTM-GARCH")
    print("    • 2 LSTM layers with 64 hidden units")
    print("    • GARCH(1,1) volatility modeling")
    print("    • Dropout: 0.2")
    print("    • Optimizer: Adam (lr=0.001)")
    print("    • Loss: MSE")
    
    print("\n  Alternative Models:")
    print("    • Temporal Fusion Transformer (TFT)")
    print("    • N-BEATS")
    print("    • Wavelet-LSTM")
    
    # System capabilities
    print(f"\n✅ SYSTEM CAPABILITIES:")
    print("  • Dynamic date configuration (auto-updates based on current date)")
    print("  • Comprehensive feature extraction matching Data_Generation.ipynb")
    print("  • Rolling window validation")
    print("  • Monthly predictions with 21-day horizon")
    print("  • Portfolio optimization and allocation")
    print("  • Direction accuracy focus (more important than R²)")
    
    # File structure
    print(f"\n📁 KEY FILES:")
    files = {
        "etf_monthly_prediction_system.py": "Main prediction system with comprehensive features",
        "comprehensive_features.py": "Complete feature extraction module",
        "validate_all_models.py": "Model validation with rolling windows",
        "COMPREHENSIVE_REPORT.md": "Full technical documentation",
        "VALIDATION_REPORT.md": "Validation results and metrics"
    }
    
    for file, desc in files.items():
        print(f"  {file}")
    print(f"    └─ {desc}")
    
    print("\n" + "="*80)
    print("SYSTEM READY FOR PREDICTIONS")
    print("="*80)

if __name__ == "__main__":
    show_system_status()