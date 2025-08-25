"""
Advanced Feature Selection Module for ETF Trading Intelligence
Implements sector-specific and universal feature selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_regression, 
    SelectKBest, 
    f_regression,
    RFE
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SectorFeatureSelector:
    """
    Advanced feature selection for sector-specific ETF predictions
    """
    
    def __init__(self, n_features_to_select=50):
        """
        Initialize feature selector
        
        Args:
            n_features_to_select: Target number of features to select
        """
        self.n_features = n_features_to_select
        self.feature_scores = {}
        self.selected_features = {}
        self.universal_features = []
        self.sector_specific_features = {}
        
    def mutual_information_selection(self, X, y, feature_names):
        """
        Select features using mutual information
        """
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
        return mi_scores
    
    def lasso_selection(self, X, y, feature_names, alpha=0.01):
        """
        Select features using LASSO regularization
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        importance = np.abs(lasso.coef_)
        lasso_scores = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        return lasso_scores
    
    def random_forest_selection(self, X, y, feature_names):
        """
        Select features using Random Forest importance
        """
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        rf_scores = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        return rf_scores
    
    def correlation_selection(self, X, y, feature_names):
        """
        Select features based on correlation with target
        """
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))
        
        corr_scores = pd.Series(correlations, index=feature_names).sort_values(ascending=False)
        return corr_scores
    
    def recursive_feature_elimination(self, X, y, feature_names, n_features=50):
        """
        Select features using Recursive Feature Elimination
        """
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=5)
        selector.fit(X, y)
        
        selected = feature_names[selector.support_]
        ranking = selector.ranking_
        
        return selected, ranking
    
    def ensemble_selection(self, X, y, feature_names, sector_name):
        """
        Combine multiple feature selection methods
        """
        print(f"\n🔍 Performing feature selection for {sector_name}...")
        
        # Get scores from different methods
        mi_scores = self.mutual_information_selection(X, y, feature_names)
        lasso_scores = self.lasso_selection(X, y, feature_names)
        rf_scores = self.random_forest_selection(X, y, feature_names)
        corr_scores = self.correlation_selection(X, y, feature_names)
        
        # Normalize scores to 0-1 range
        mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
        lasso_norm = (lasso_scores - lasso_scores.min()) / (lasso_scores.max() - lasso_scores.min() + 1e-10)
        rf_norm = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min() + 1e-10)
        corr_norm = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min() + 1e-10)
        
        # Ensemble score (weighted average)
        ensemble_scores = (
            0.3 * mi_norm +
            0.2 * lasso_norm +
            0.3 * rf_norm +
            0.2 * corr_norm
        ).sort_values(ascending=False)
        
        # Store scores
        self.feature_scores[sector_name] = {
            'mutual_info': mi_scores,
            'lasso': lasso_scores,
            'random_forest': rf_scores,
            'correlation': corr_scores,
            'ensemble': ensemble_scores
        }
        
        # Select top features
        selected = ensemble_scores.head(self.n_features).index.tolist()
        self.selected_features[sector_name] = selected
        
        return selected, ensemble_scores
    
    def identify_universal_features(self, min_sectors=8):
        """
        Identify features that are important across multiple sectors
        
        Args:
            min_sectors: Minimum number of sectors where feature must be important
        """
        if not self.selected_features:
            return []
        
        # Count feature occurrences across sectors
        feature_counts = {}
        for sector, features in self.selected_features.items():
            for feature in features:
                if feature not in feature_counts:
                    feature_counts[feature] = []
                feature_counts[feature].append(sector)
        
        # Identify universal features
        self.universal_features = []
        for feature, sectors in feature_counts.items():
            if len(sectors) >= min_sectors:
                self.universal_features.append({
                    'feature': feature,
                    'sectors': sectors,
                    'coverage': len(sectors)
                })
        
        # Sort by coverage
        self.universal_features = sorted(
            self.universal_features, 
            key=lambda x: x['coverage'], 
            reverse=True
        )
        
        return self.universal_features
    
    def identify_sector_specific_features(self, top_n=10):
        """
        Identify features unique to specific sectors
        """
        if not self.feature_scores:
            return {}
        
        self.sector_specific_features = {}
        
        for sector, scores in self.feature_scores.items():
            # Get top features for this sector
            top_features = scores['ensemble'].head(top_n).index.tolist()
            
            # Find features unique to this sector (not in universal)
            universal_feature_names = [f['feature'] for f in self.universal_features]
            unique_features = [f for f in top_features if f not in universal_feature_names]
            
            self.sector_specific_features[sector] = unique_features[:5]  # Top 5 unique
        
        return self.sector_specific_features
    
    def generate_feature_importance_report(self):
        """
        Generate comprehensive feature importance report
        """
        report = {
            'universal_features': self.universal_features,
            'sector_specific': self.sector_specific_features,
            'feature_scores': self.feature_scores,
            'selected_features': self.selected_features
        }
        
        return report


class SectorSpecificFeatureAnalyzer:
    """
    Analyze feature importance patterns across sectors
    """
    
    def __init__(self):
        self.sector_characteristics = {
            'XLF': {  # Financials
                'name': 'Financials',
                'key_indicators': ['interest_rates', 'yield_curve', 'credit_spreads', 'monetary_policy'],
                'expected_features': ['treasury_10y', 'yield_curve_10y2y', 'ted_spread', 'fed_funds']
            },
            'XLE': {  # Energy
                'name': 'Energy',
                'key_indicators': ['oil_prices', 'gas_prices', 'commodity_trends'],
                'expected_features': ['oil_wti', 'oil_brent', 'gas_price', 'dollar_index']
            },
            'XLK': {  # Technology
                'name': 'Technology',
                'key_indicators': ['growth_metrics', 'innovation', 'nasdaq_correlation'],
                'expected_features': ['nasdaq_momentum', 'volatility', 'consumer_sentiment']
            },
            'XLV': {  # Healthcare
                'name': 'Healthcare',
                'key_indicators': ['demographics', 'regulatory', 'innovation'],
                'expected_features': ['employment_ratio', 'gdp', 'consumer_sentiment']
            },
            'XLI': {  # Industrials
                'name': 'Industrials',
                'key_indicators': ['manufacturing', 'infrastructure', 'trade'],
                'expected_features': ['industrial_production', 'capacity_utilization', 'exports']
            },
            'XLY': {  # Consumer Discretionary
                'name': 'Consumer Discretionary',
                'key_indicators': ['consumer_spending', 'employment', 'confidence'],
                'expected_features': ['retail_sales', 'consumer_sentiment', 'unemployment_rate']
            },
            'XLP': {  # Consumer Staples
                'name': 'Consumer Staples',
                'key_indicators': ['inflation', 'defensive', 'stability'],
                'expected_features': ['cpi', 'core_cpi', 'food_prices']
            },
            'XLU': {  # Utilities
                'name': 'Utilities',
                'key_indicators': ['interest_rates', 'regulation', 'defensive'],
                'expected_features': ['treasury_10y', 'inflation_10y', 'dividend_yield']
            },
            'XLRE': {  # Real Estate
                'name': 'Real Estate',
                'key_indicators': ['interest_rates', 'housing', 'mortgage'],
                'expected_features': ['mortgage_30y', 'housing_starts', 'building_permits']
            },
            'XLB': {  # Materials
                'name': 'Materials',
                'key_indicators': ['commodities', 'global_growth', 'dollar'],
                'expected_features': ['gold', 'dollar_index', 'industrial_production']
            },
            'XLC': {  # Communication Services
                'name': 'Communication Services',
                'key_indicators': ['technology', 'consumer', 'growth'],
                'expected_features': ['consumer_sentiment', 'gdp', 'volatility']
            }
        }
    
    def analyze_feature_alignment(self, selected_features):
        """
        Analyze how well selected features align with sector characteristics
        """
        alignment_scores = {}
        
        for sector, features in selected_features.items():
            if sector in self.sector_characteristics:
                expected = self.sector_characteristics[sector]['expected_features']
                
                # Check how many expected features were selected
                matches = sum(1 for f in features 
                             if any(exp in f.lower() for exp in expected))
                
                alignment_scores[sector] = {
                    'matches': matches,
                    'expected': len(expected),
                    'score': matches / len(expected) if expected else 0
                }
        
        return alignment_scores
    
    def categorize_features(self, feature_list):
        """
        Categorize features into groups
        """
        categories = {
            'Technical': [],
            'Macroeconomic': [],
            'Market Structure': [],
            'Sentiment': [],
            'Monetary': []
        }
        
        for feature in feature_list:
            feature_lower = feature.lower()
            
            if any(x in feature_lower for x in ['momentum', 'rsi', 'macd', 'bollinger', 'kdj', 'atr']):
                categories['Technical'].append(feature)
            elif any(x in feature_lower for x in ['treasury', 'yield', 'rate', 'fed_funds']):
                categories['Monetary'].append(feature)
            elif any(x in feature_lower for x in ['gdp', 'unemployment', 'inflation', 'cpi', 'production']):
                categories['Macroeconomic'].append(feature)
            elif any(x in feature_lower for x in ['sentiment', 'confidence', 'optimism']):
                categories['Sentiment'].append(feature)
            elif any(x in feature_lower for x in ['volume', 'volatility', 'vix', 'spread']):
                categories['Market Structure'].append(feature)
            else:
                # Try to categorize based on other patterns
                if 'oil' in feature_lower or 'gold' in feature_lower:
                    categories['Macroeconomic'].append(feature)
                else:
                    categories['Technical'].append(feature)
        
        return categories


def perform_sector_feature_selection(data, targets, feature_names, sector_etfs):
    """
    Main function to perform feature selection for all sectors
    """
    selector = SectorFeatureSelector(n_features_to_select=50)
    analyzer = SectorSpecificFeatureAnalyzer()
    
    results = {}
    
    for etf in sector_etfs:
        # Assuming data is structured as dict with ETF keys
        X = data[etf]
        y = targets[etf]
        
        # Perform ensemble feature selection
        selected, scores = selector.ensemble_selection(X, y, feature_names, etf)
        
        results[etf] = {
            'selected_features': selected,
            'scores': scores,
            'top_10': selected[:10]
        }
    
    # Identify universal and sector-specific features
    universal = selector.identify_universal_features(min_sectors=7)
    sector_specific = selector.identify_sector_specific_features(top_n=10)
    
    # Analyze alignment with expected patterns
    alignment = analyzer.analyze_feature_alignment(selector.selected_features)
    
    # Categorize universal features
    universal_names = [f['feature'] for f in universal]
    categories = analyzer.categorize_features(universal_names)
    
    return {
        'sector_results': results,
        'universal_features': universal,
        'sector_specific_features': sector_specific,
        'alignment_scores': alignment,
        'feature_categories': categories,
        'selector': selector
    }


if __name__ == "__main__":
    print("="*80)
    print("SECTOR-SPECIFIC FEATURE SELECTION MODULE")
    print("="*80)
    print("\n✅ Module initialized successfully")
    print("\nCapabilities:")
    print("  • Mutual Information Selection")
    print("  • LASSO Regularization")
    print("  • Random Forest Importance")
    print("  • Correlation-based Selection")
    print("  • Ensemble Method Combination")
    print("  • Universal Feature Identification")
    print("  • Sector-Specific Feature Analysis")
    print("\nReady for integration with ETF prediction pipeline")