"""
Monthly Tracking Report Auto-Updater
Automatically updates the living performance report after each backtesting cycle

Usage:
    python update_monthly_tracking.py  # Full update (recommended)
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

SECTOR_ETFS = ['XLF', 'XLC', 'XLY', 'XLP', 'XLE', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']
PLOT_DIR = Path('plots')
PLOT_DIR.mkdir(exist_ok=True)


class MonthlyTrackingUpdater:
    """Updates the monthly tracking report with latest cycle data"""

    def __init__(self):
        self.report_path = Path('MONTHLY_TRACKING_REPORT.md')
        self.cycles = []
        self.load_existing_cycles()

    def load_existing_cycles(self):
        """Load all existing cycle data from JSON files"""
        prediction_files = sorted(Path('.').glob('*_2025_predictions.json'))

        for pred_file in prediction_files:
            month = pred_file.stem.replace('_2025_predictions', '').capitalize()

            cycle = {
                'month': month,
                'year': 2025,
                'prediction_file': str(pred_file),
                'has_validation': False,
                'has_feature_importance': False,
                'predictions': None,
                'validation': None,
                'feature_importance': None
            }

            # Load predictions
            with open(pred_file, 'r') as f:
                cycle['predictions'] = json.load(f)

            # Check for validation
            actual_file = Path(f"{month.lower()}_2025_actual_returns.json")
            if actual_file.exists():
                with open(actual_file, 'r') as f:
                    actuals_data = json.load(f)
                    cycle['has_validation'] = True
                    cycle['validation'] = self._calculate_validation_metrics(
                        cycle['predictions'],
                        actuals_data
                    )

            # Check for feature importance
            fi_file = Path(f"feature_importance_{month.lower()}_2025.json")
            if fi_file.exists():
                with open(fi_file, 'r') as f:
                    cycle['feature_importance'] = json.load(f)
                    cycle['has_feature_importance'] = True

            self.cycles.append(cycle)

        # Sort by month (newest first)
        month_order = {
            'December': 12, 'November': 11, 'October': 10, 'September': 9,
            'August': 8, 'July': 7, 'June': 6
        }
        self.cycles.sort(key=lambda x: month_order.get(x['month'], 0), reverse=True)

    def _calculate_validation_metrics(self, predictions, actuals_data):
        """Calculate validation metrics from predictions and actuals"""
        actuals = {etf: data['relative_return']
                  for etf, data in actuals_data['sector_returns'].items()}

        common_etfs = sorted(set(predictions.keys()) & set(actuals.keys()))
        pred_values = np.array([predictions[etf] for etf in common_etfs])
        actual_values = np.array([actuals[etf] for etf in common_etfs])

        # Calculate metrics
        pred_signs = np.sign(pred_values)
        actual_signs = np.sign(actual_values)
        direction_accuracy = np.mean(pred_signs == actual_signs) * 100
        correct_count = np.sum(pred_signs == actual_signs)

        correlation = np.corrcoef(pred_values, actual_values)[0, 1]
        mae = np.mean(np.abs(pred_values - actual_values))
        rmse = np.sqrt(np.mean((pred_values - actual_values) ** 2))

        # R-squared
        ss_res = np.sum((actual_values - pred_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf

        # Top/Bottom 3
        top3_pred = set(sorted(common_etfs, key=lambda x: predictions[x], reverse=True)[:3])
        top3_actual = set(sorted(common_etfs, key=lambda x: actuals[x], reverse=True)[:3])
        bottom3_pred = set(sorted(common_etfs, key=lambda x: predictions[x])[:3])
        bottom3_actual = set(sorted(common_etfs, key=lambda x: actuals[x])[:3])

        # Trading strategy return
        actual_return = (sum(actuals[etf] for etf in top3_pred) / 3 -
                        sum(actuals[etf] for etf in bottom3_pred) / 3)

        return {
            'direction_accuracy': direction_accuracy,
            'correct_count': int(correct_count),
            'total_count': len(common_etfs),
            'correlation': correlation,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'top3_overlap': len(top3_pred & top3_actual),
            'bottom3_overlap': len(bottom3_pred & bottom3_actual),
            'strategy_return': actual_return,
            'top3_pred': sorted(list(top3_pred)),
            'top3_actual': sorted(list(top3_actual)),
            'bottom3_pred': sorted(list(bottom3_pred)),
            'bottom3_actual': sorted(list(bottom3_actual)),
            'spy_return': actuals_data['spy_return'],
            'baseline_date': actuals_data['baseline_date'],
            'target_date': actuals_data.get('target_date') or actuals_data.get('october_last_date'),
            'sector_details': {
                etf: {
                    'predicted': predictions[etf],
                    'actual': actuals[etf],
                    'error': actuals[etf] - predictions[etf],
                    'direction_correct': np.sign(predictions[etf]) == np.sign(actuals[etf])
                }
                for etf in common_etfs
            }
        }

    def create_prediction_vs_actual_plot(self, cycle):
        """Create interactive scatter plot of predictions vs actuals"""
        if not cycle['has_validation']:
            return None

        val = cycle['validation']
        sectors = list(val['sector_details'].keys())
        predictions = [val['sector_details'][etf]['predicted'] * 100 for etf in sectors]
        actuals = [val['sector_details'][etf]['actual'] * 100 for etf in sectors]
        correct = [val['sector_details'][etf]['direction_correct'] for etf in sectors]

        colors = ['#2ecc71' if c else '#e74c3c' for c in correct]

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=predictions,
            y=actuals,
            mode='markers+text',
            text=sectors,
            textposition='top center',
            marker=dict(
                size=14,
                color=colors,
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Predicted: %{x:.2f}%<br>' +
                         'Actual: %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))

        # Add diagonal line (perfect prediction)
        min_val = min(min(predictions), min(actuals)) - 1
        max_val = max(max(predictions), max(actuals)) + 1
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray', width=2),
            name='Perfect Prediction',
            showlegend=True
        ))

        fig.update_layout(
            title=f"{cycle['month']} {cycle['year']} - Predictions vs Actuals",
            xaxis_title="Predicted Relative Return (%)",
            yaxis_title="Actual Relative Return (%)",
            template='plotly_white',
            height=550,
            font=dict(size=12)
        )

        filename = PLOT_DIR / f"pred_vs_actual_{cycle['month'].lower()}_{cycle['year']}.html"
        fig.write_html(filename)
        return f"plots/{filename.name}"

    def create_feature_importance_plot(self, cycle, sector):
        """Create interactive bar chart of top 20 features for a sector"""
        if not cycle['has_feature_importance']:
            return None

        fi_data = cycle['feature_importance']['sector_importance'].get(sector)
        if not fi_data:
            return None

        top_features = fi_data['top_features'][:20]

        features = [f['feature'] for f in top_features]
        importances = [f['importance_pct'] for f in top_features]
        categories = [f['category'] for f in top_features]

        # Color mapping
        color_map = {
            'Alpha - Technical': '#3498db',
            'Beta - Interest Rates': '#e67e22',
            'Beta - Economic': '#f39c12',
            'Beta - Inflation': '#d35400',
            'Beta - Money Supply': '#c0392b',
            'Beta - Market': '#8e44ad',
            'Beta - Sentiment': '#2c3e50',
            'Beta - Other': '#95a5a6',
            'VIX Regime': '#27ae60',
            'Derived': '#16a085'
        }
        colors = [color_map.get(c, '#7f8c8d') for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=1, color='white')
            ),
            text=[f'{imp:.1f}%' for imp in importances],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Importance: %{x:.2f}%<br>' +
                         'Category: ' + categories[0] + '<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title=f"{sector} - Top 20 Features ({cycle['month']} {cycle['year']})",
            xaxis_title="Importance (%)",
            yaxis_title="Feature",
            template='plotly_white',
            height=650,
            font=dict(size=11),
            showlegend=False
        )

        filename = PLOT_DIR / f"feature_importance_{sector}_{cycle['month'].lower()}_{cycle['year']}.html"
        fig.write_html(filename)
        return f"plots/{filename.name}"

    def create_aggregate_feature_importance_plot(self, cycle):
        """Create aggregate feature importance plot across all sectors"""
        if not cycle['has_feature_importance']:
            return None

        aggregate = cycle['feature_importance']['aggregate_importance']
        top_features = aggregate['top_features'][:20]

        features = [f['feature'] for f in top_features]
        importances = [f['avg_importance_pct'] for f in top_features]
        stds = [f['std_importance_pct'] for f in top_features]
        categories = [f['category'] for f in top_features]

        # Color mapping (same as before)
        color_map = {
            'Alpha - Technical': '#3498db',
            'Beta - Interest Rates': '#e67e22',
            'Beta - Economic': '#f39c12',
            'Beta - Inflation': '#d35400',
            'Beta - Money Supply': '#c0392b',
            'Beta - Market': '#8e44ad',
            'Beta - Sentiment': '#2c3e50',
            'Beta - Other': '#95a5a6',
            'VIX Regime': '#27ae60',
            'Derived': '#16a085'
        }
        colors = [color_map.get(c, '#7f8c8d') for c in categories]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(width=1, color='white')
            ),
            error_x=dict(type='data', array=stds),
            text=[f'{imp:.1f}%' for imp in importances],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Importance: %{x:.2f}% (±%{error_x.array:.2f}%)<br>' +
                         '<extra></extra>'
        ))

        fig.update_layout(
            title=f"Aggregate Feature Importance - {cycle['month']} {cycle['year']}",
            xaxis_title="Average Importance Across Sectors (%)",
            yaxis_title="Feature",
            template='plotly_white',
            height=650,
            font=dict(size=11)
        )

        filename = PLOT_DIR / f"aggregate_feature_importance_{cycle['month'].lower()}_{cycle['year']}.html"
        fig.write_html(filename)
        return f"plots/{filename.name}"

    def create_performance_timeline_plot(self):
        """Create timeline of performance metrics"""
        validated_cycles = [c for c in self.cycles if c['has_validation']]

        if not validated_cycles:
            return None

        months = [f"{c['month']} {c['year']}" for c in validated_cycles][::-1]
        direction_acc = [c['validation']['direction_accuracy'] for c in validated_cycles][::-1]
        strategy_return = [c['validation']['strategy_return'] * 100 for c in validated_cycles][::-1]
        correlations = [c['validation']['correlation'] for c in validated_cycles][::-1]

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Direction Accuracy (%)', 'Strategy Return (%)', 'Correlation'),
            vertical_spacing=0.12
        )

        # Direction accuracy
        fig.add_trace(
            go.Scatter(
                x=months, y=direction_acc,
                mode='lines+markers',
                name='Direction Accuracy',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10),
                hovertemplate='%{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        fig.add_hline(y=55, line_dash="dash", line_color="#27ae60",
                     annotation_text="Profitable (55%)", row=1, col=1)

        # Strategy return
        colors = ['#27ae60' if r > 0 else '#e74c3c' for r in strategy_return]
        fig.add_trace(
            go.Bar(
                x=months, y=strategy_return,
                name='Strategy Return',
                marker=dict(color=colors),
                hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Correlation
        fig.add_trace(
            go.Scatter(
                x=months, y=correlations,
                mode='lines+markers',
                name='Correlation',
                line=dict(color='#9b59b6', width=3),
                marker=dict(size=10),
                hovertemplate='%{x}<br>Correlation: %{y:.3f}<extra></extra>'
            ),
            row=3, col=1
        )

        fig.update_layout(
            title_text="Performance Timeline",
            template='plotly_white',
            height=800,
            showlegend=False,
            font=dict(size=12)
        )

        filename = PLOT_DIR / "performance_timeline.html"
        fig.write_html(filename)
        return f"plots/{filename.name}"

    def create_error_distribution_plot(self, cycle):
        """Create error distribution plot for a validation cycle"""
        if not cycle['has_validation']:
            return None

        val = cycle['validation']
        sectors = list(val['sector_details'].keys())
        errors = [val['sector_details'][etf]['error'] * 100 for etf in sectors]
        correct = [val['sector_details'][etf]['direction_correct'] for etf in sectors]

        colors = ['#2ecc71' if c else '#e74c3c' for c in correct]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sectors,
            y=errors,
            marker=dict(
                color=colors,
                line=dict(width=1, color='white')
            ),
            text=[f'{e:+.1f}%' for e in errors],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Error: %{y:+.2f}%<extra></extra>'
        ))

        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=2)

        fig.update_layout(
            title=f"{cycle['month']} {cycle['year']} - Prediction Errors",
            xaxis_title="Sector",
            yaxis_title="Error (Actual - Predicted, %)",
            template='plotly_white',
            height=450,
            font=dict(size=12)
        )

        filename = PLOT_DIR / f"error_distribution_{cycle['month'].lower()}_{cycle['year']}.html"
        fig.write_html(filename)
        return f"plots/{filename.name}"

    def generate_report(self):
        """Generate the complete markdown report"""
        print("\n📝 Generating MONTHLY_TRACKING_REPORT.md...")

        # Calculate summary statistics
        validated = [c for c in self.cycles if c['has_validation']]
        if validated:
            avg_direction = np.mean([c['validation']['direction_accuracy'] for c in validated])
            total_return = sum([c['validation']['strategy_return'] for c in validated]) * 100
            profitable_months = len([c for c in validated if c['validation']['strategy_return'] > 0])
            win_rate = f"{profitable_months}/{len(validated)}"
        else:
            avg_direction = 0
            total_return = 0
            profitable_months = 0
            win_rate = "0/0"

        # Get latest cycle info
        latest = self.cycles[0] if self.cycles else None
        latest_validated = validated[0] if validated else None

        # Create timeline plot
        timeline_plot = self.create_performance_timeline_plot()

        # Start building markdown
        md = f"""# ETF Trading Intelligence - Monthly Tracking Report
*Living document tracking model performance, feature importance, and predictions*

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

---

## 📊 Latest Status Dashboard

| Metric | Value |
|--------|-------|
| **Latest Prediction** | {latest['month']} {latest['year']} |
| **Last Validated Month** | {latest_validated['month']} {latest_validated['year'] if latest_validated else 'N/A'} |
| **Overall Direction Accuracy** | {avg_direction:.1f}% |
| **Win Rate (Profitable Months)** | {win_rate} ({profitable_months/len(validated)*100:.0f}%) if validated else 'N/A' |
| **Cumulative Strategy Return** | {total_return:+.2f}% |
| **Total Cycles Tracked** | {len(self.cycles)} |

---

## 📈 Performance Timeline

"""

        if timeline_plot:
            md += f'<iframe src="{timeline_plot}" width="100%" height="850" frameborder="0"></iframe>\n\n'

        md += """
| Month | Direction Accuracy | Correlation | MAE | Strategy Return | Status | Training Through |
|-------|-------------------|-------------|-----|-----------------|--------|------------------|
"""

        for cycle in self.cycles:
            if cycle['has_validation']:
                val = cycle['validation']
                status = "✅ Validated"
                dir_acc = f"{val['direction_accuracy']:.1f}%"
                corr = f"{val['correlation']:.3f}"
                mae = f"{val['mae']*100:.2f}%"
                ret = f"{val['strategy_return']*100:+.2f}%"
                train_through = val['baseline_date']
            else:
                status = "🔮 Predicted"
                dir_acc = "*Pending*"
                corr = "*Pending*"
                mae = "*Pending*"
                ret = "*Pending*"
                train_through = "*Unknown*"

            md += f"| {cycle['month']} {cycle['year']} | {dir_acc} | {corr} | {mae} | {ret} | {status} | {train_through} |\n"

        md += "\n---\n\n"

        # Add detailed cycle information (newest first)
        for idx, cycle in enumerate(self.cycles):
            md += self._generate_cycle_section(cycle, idx == 0)

        md += f"""
---

*Report auto-generated by `update_monthly_tracking.py`*
*Framework: 4-Model Ensemble (LSTM, TFT, N-BEATS, LSTM-GARCH) with Adaptive Weighting*
*Feature Set: 219 features (20 Alpha + 186 Beta + 10 VIX + 3 Derived)*
*Feature Importance: Permutation-based on trained ensemble models*
"""

        # Write to file
        with open(self.report_path, 'w') as f:
            f.write(md)

        print(f"✅ Report generated: {self.report_path}")

    def _generate_cycle_section(self, cycle, is_latest=False):
        """Generate markdown section for a single cycle"""
        emoji = '🔮' if not cycle['has_validation'] else '✅'
        md = f"## {emoji} {cycle['month']} {cycle['year']} Cycle\n\n"

        if is_latest and not cycle['has_validation']:
            md += "**Latest Prediction - Awaiting Validation**\n\n"

        # Prediction section
        md += "### 📊 Predicted Returns (vs SPY)\n\n"
        md += "| Rank | ETF | Predicted Return | Recommendation |\n"
        md += "|------|-----|------------------|----------------|\n"

        sorted_preds = sorted(cycle['predictions'].items(),
                            key=lambda x: x[1], reverse=True)

        for rank, (etf, pred) in enumerate(sorted_preds, 1):
            emoji = "🟢" if rank <= 3 else ("🔴" if rank >= 9 else "⚪")
            rec = "LONG" if rank <= 3 else ("SHORT" if rank >= 9 else "NEUTRAL")
            md += f"| {rank} | {etf} {emoji} | {pred*100:+.2f}% | {rec} |\n"

        md += "\n**Trading Strategy:** Long top 3, Short bottom 3\n\n"

        # Feature Importance section
        if cycle['has_feature_importance']:
            md += self._generate_feature_importance_section(cycle)

        # Validation section (if available)
        if cycle['has_validation']:
            md += self._generate_validation_section(cycle)

        md += "---\n\n"

        return md

    def _generate_feature_importance_section(self, cycle):
        """Generate feature importance section"""
        fi = cycle['feature_importance']

        md = f"### 🔬 Feature Importance\n"
        md += f"**Calculated:** {fi['calculation_timestamp'][:10]}\n"
        md += f"**Method:** Permutation Importance ({fi['n_repeats']} repeats)\n\n"

        # Aggregate importance plot
        agg_plot = self.create_aggregate_feature_importance_plot(cycle)
        if agg_plot:
            md += f'<iframe src="{agg_plot}" width="100%" height="700" frameborder="0"></iframe>\n\n'

        # Top features table
        md += "#### Top 20 Features (Aggregate)\n\n"
        md += "| Rank | Feature | Avg Importance | Std | Category |\n"
        md += "|------|---------|----------------|-----|----------|\n"

        for i, feature in enumerate(fi['aggregate_importance']['top_features'][:20], 1):
            md += f"| {i} | {feature['feature']} | {feature['avg_importance_pct']:.2f}% | "
            md += f"±{feature['std_importance_pct']:.2f}% | {feature['category']} |\n"

        # Category breakdown
        md += "\n#### Category Importance Breakdown\n\n"
        md += "| Category | Importance |\n"
        md += "|----------|------------|\n"

        cat_breakdown = fi['aggregate_importance']['category_breakdown']
        for category, importance in sorted(cat_breakdown.items(),
                                          key=lambda x: x[1], reverse=True):
            md += f"| {category} | {importance:.2f}% |\n"

        md += "\n"

        # Sector-specific top features
        md += "<details>\n"
        md += "<summary><b>📋 Sector-Specific Feature Importance (Click to expand)</b></summary>\n\n"

        for etf in SECTOR_ETFS:
            if etf in fi['sector_importance']:
                sector_data = fi['sector_importance'][etf]
                top5 = sector_data['top_features'][:5]

                md += f"\n**{etf}** - Top 5:\n"
                for i, feature in enumerate(top5, 1):
                    md += f"{i}. {feature['feature']} ({feature['importance_pct']:.2f}%) - {feature['category']}\n"

        md += "\n</details>\n\n"

        return md

    def _generate_validation_section(self, cycle):
        """Generate validation section for a cycle"""
        val = cycle['validation']

        md = f"""### ✅ Validation Results
**Period:** {val['baseline_date']} to {val['target_date']}
**SPY Return:** {val['spy_return']*100:+.2f}%

#### Overall Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Direction Accuracy** | **{val['direction_accuracy']:.1f}%** ({val['correct_count']}/{val['total_count']}) | {'🏆 EXCELLENT' if val['direction_accuracy'] >= 80 else '✅ VERY GOOD' if val['direction_accuracy'] >= 70 else '👍 GOOD' if val['direction_accuracy'] >= 60 else '⚡ PROFITABLE' if val['direction_accuracy'] >= 55 else '⚠️ BELOW THRESHOLD'} |
| Correlation | {val['correlation']:.3f} | {'Very Strong' if abs(val['correlation']) > 0.7 else 'Strong' if abs(val['correlation']) > 0.5 else 'Moderate' if abs(val['correlation']) > 0.3 else 'Weak'} |
| Mean Absolute Error | {val['mae']*100:.2f}% | - |
| R² Score | {val['r_squared']:.3f} | - |
| **Strategy Return** | **{val['strategy_return']*100:+.2f}%** | {'✅ PROFITABLE' if val['strategy_return'] > 0 else '❌ LOSS'} |

#### Prediction vs Actual

"""

        # Add interactive plot
        plot_file = self.create_prediction_vs_actual_plot(cycle)
        if plot_file:
            md += f'<iframe src="{plot_file}" width="100%" height="600" frameborder="0"></iframe>\n\n'

        # Detailed sector results
        md += "| ETF | Predicted | Actual | Error | Direction |\n"
        md += "|-----|-----------|--------|-------|-----------|\n"

        for etf in sorted(val['sector_details'].keys()):
            details = val['sector_details'][etf]
            check = "✅" if details['direction_correct'] else "❌"
            md += f"| {etf} | {details['predicted']*100:+.2f}% | {details['actual']*100:+.2f}% | "
            md += f"{details['error']*100:+.2f}% | {check} |\n"

        md += "\n"

        # Error distribution plot
        error_plot = self.create_error_distribution_plot(cycle)
        if error_plot:
            md += f'<iframe src="{error_plot}" width="100%" height="500" frameborder="0"></iframe>\n\n'

        # Top/Bottom analysis
        md += f"""#### Top/Bottom 3 Analysis

**Top 3 Predicted:** {', '.join(val['top3_pred'])}
**Top 3 Actual:** {', '.join(val['top3_actual'])}
**Overlap:** {val['top3_overlap']}/3 ({val['top3_overlap']/3*100:.0f}%)

**Bottom 3 Predicted:** {', '.join(val['bottom3_pred'])}
**Bottom 3 Actual:** {', '.join(val['bottom3_actual'])}
**Overlap:** {val['bottom3_overlap']}/3 ({val['bottom3_overlap']/3*100:.0f}%)

"""

        return md


def main():
    """Main entry point"""
    print("="*80)
    print("MONTHLY TRACKING REPORT UPDATER")
    print("="*80)
    print()

    updater = MonthlyTrackingUpdater()
    updater.generate_report()

    print("\n" + "="*80)
    print("✅ Monthly tracking report updated successfully!")
    print("="*80)
    print(f"\n📊 View report: MONTHLY_TRACKING_REPORT.md")
    print(f"📈 Interactive plots: {PLOT_DIR}/")
    print(f"\nCycles tracked: {len(updater.cycles)}")
    validated = len([c for c in updater.cycles if c['has_validation']])
    print(f"  - Validated: {validated}")
    print(f"  - Pending: {len(updater.cycles) - validated}")


if __name__ == "__main__":
    main()
