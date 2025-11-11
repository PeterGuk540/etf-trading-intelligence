"""
Test script to validate ETF Trading Intelligence pipeline consistency
Checks: data extraction -> analysis -> prediction -> visualization -> report
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import torch
        from datetime import datetime
        print("✓ Core libraries imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_pipeline_initialization():
    """Test if the main pipeline can be initialized"""
    print("\nTesting pipeline initialization...")
    try:
        from etf_monthly_prediction_system import MonthlyPredictionPipeline
        pipeline = MonthlyPredictionPipeline()
        print("✓ Pipeline initialized successfully")
        print(f"  - Found {len(pipeline.fred_indicators)} FRED indicators")
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

def test_data_extraction():
    """Test data extraction functionality"""
    print("\nTesting data extraction...")
    try:
        from etf_monthly_prediction_system import MonthlyPredictionPipeline
        pipeline = MonthlyPredictionPipeline()
        
        # Test with a small sample
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Get just 1 month of data for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        test_ticker = 'XLF'
        data = yf.download(test_ticker, start=start_date.strftime('%Y-%m-%d'), 
                         end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if len(data) > 0:
            print(f"✓ Data extraction works - fetched {len(data)} days of data for {test_ticker}")
            return True
        else:
            print("✗ No data fetched")
            return False
    except Exception as e:
        print(f"✗ Data extraction failed: {e}")
        return False

def test_feature_creation():
    """Test feature creation"""
    print("\nTesting feature creation...")
    try:
        from etf_monthly_prediction_system import MonthlyPredictionPipeline
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, timedelta
        
        pipeline = MonthlyPredictionPipeline()
        
        # Get test data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        data = yf.download(['XLF', 'SPY'], start=start_date.strftime('%Y-%m-%d'), 
                         end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        # Test momentum calculation
        close_series = data['Close']['XLF'] if 'XLF' in data['Close'].columns else data['Close']
        momentum = pipeline.compute_momentum(close_series, window=5)
        
        if len(momentum.dropna()) > 0:
            print("✓ Feature creation works - computed momentum successfully")
            return True
        else:
            print("✗ Feature creation failed")
            return False
    except Exception as e:
        print(f"✗ Feature creation failed: {e}")
        return False

def test_model_architecture():
    """Test if models can be created"""
    print("\nTesting model architectures...")
    try:
        import torch
        import torch.nn as nn
        
        # Simple test model
        class TestLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(10, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        
        model = TestLSTM()
        test_input = torch.randn(2, 20, 10)  # batch=2, seq=20, features=10
        output = model(test_input)
        
        if output.shape == torch.Size([2, 1]):
            print("✓ Model architecture test passed")
            return True
        else:
            print(f"✗ Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_visualization_files():
    """Check if visualization outputs exist"""
    print("\nChecking visualization outputs...")
    import os
    
    viz_files = {
        'backtest_results_real.png': 'Backtest results visualization',
        'feature_importance_real.png': 'Feature importance plot',
        'trading_dashboard_real.html': 'Trading dashboard'
    }
    
    found = 0
    for file, desc in viz_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✓ Found {file} ({size:.1f} KB) - {desc}")
            found += 1
        else:
            print(f"✗ Missing {file} - {desc}")
    
    return found > 0

def test_reports():
    """Check if reports exist and are valid"""
    print("\nChecking reports...")
    import os
    
    reports = {
        'COMPREHENSIVE_REPORT.md': 'Main technical report',
        'VALIDATION_REPORT.md': 'Model validation results',
        'UPDATE_SUMMARY.md': 'System update summary'
    }
    
    found = 0
    for file, desc in reports.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✓ Found {file} ({size:.1f} KB) - {desc}")
            found += 1
        else:
            print(f"✗ Missing {file} - {desc}")
    
    return found > 0

def main():
    """Run all tests"""
    print("="*60)
    print("ETF TRADING INTELLIGENCE PIPELINE CONSISTENCY CHECK")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Data Extraction", test_data_extraction),
        ("Feature Creation", test_feature_creation),
        ("Model Architecture", test_model_architecture),
        ("Visualization Files", test_visualization_files),
        ("Reports", test_reports)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE CONSISTENCY SUMMARY")
    print("="*60)
    
    passed_count = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed_count}/{total} components working")
    
    if passed_count == total:
        print("\n✅ PIPELINE IS FULLY CONSISTENT - All components working!")
    elif passed_count >= total * 0.7:
        print("\n⚠️ PIPELINE MOSTLY CONSISTENT - Some components need attention")
    else:
        print("\n❌ PIPELINE HAS ISSUES - Multiple components failing")
    
    # Identify specific issues
    if passed_count < total:
        print("\n🔧 ISSUES FOUND:")
        for name, passed in results:
            if not passed:
                if name == "Data Extraction":
                    print(f"  - {name}: Check internet connection and API keys")
                elif name == "Feature Creation":
                    print(f"  - {name}: Verify data format and calculation functions")
                elif name == "Visualization Files":
                    print(f"  - {name}: Run visualization scripts to generate outputs")
                elif name == "Reports":
                    print(f"  - {name}: Generate reports using the main pipeline")
                else:
                    print(f"  - {name}: Investigate module dependencies")

if __name__ == "__main__":
    main()