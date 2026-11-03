import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def visualize_metric_ff_results(metric_name: str, save_path: Optional[str] = None):
    """
    Visualize Fama-French regression results for a specific metric.

    Parameters:
    -----------
    metric_name : str
        Name of the social metric to visualize (e.g., 'AnalyticEmploymentCreationScore')
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    # Load Fama-French regression results
    results_file = Path("data/results/ff_regression/fama_french_regression_results.csv")

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)

    # Filter for the specific metric
    metric_data = df[df['metric'] == metric_name].copy()

    if metric_data.empty:
        raise ValueError(f"No data found for metric: {metric_name}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Fama-French Regression Results: {metric_name}', fontsize=16, fontweight='bold')

    # 1. Alpha visualization with confidence intervals
    ax1 = axes[0, 0]

    portfolios = ['Portfolio_1', 'Portfolio_2', 'Portfolio_3']
    alphas = metric_data['alpha'].values
    alpha_errors = np.asarray(metric_data['alpha_t'].values, dtype=float) * 0.1  # Approximate confidence interval

    bars = ax1.bar(portfolios, alphas, yerr=alpha_errors, capsize=5,
                   color=['#2E86AB', '#A23B72', '#F18F01'],
                   alpha=0.8, edgecolor='black', linewidth=1)

    ax1.set_ylabel('Alpha (Monthly %)', fontsize=12)
    ax1.set_title('Portfolio Alphas with 90% Confidence Intervals', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, alpha, t_stat in zip(bars, alphas, metric_data['alpha_t']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                '.3f', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')

    # 2. Factor betas comparison
    ax2 = axes[0, 1]

    factors = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'WML']
    factor_names = ['Market', 'Size', 'Value', 'Profitability', 'Investment', 'Momentum']

    x = np.arange(len(portfolios))
    width = 0.13

    for i, (factor, name) in enumerate(zip(factors, factor_names)):
        betas = metric_data[f'{factor.lower()}_beta'].values
        ax2.bar(x + i*width - width*2.5, betas, width, label=name, alpha=0.8)

    ax2.set_xlabel('Portfolio')
    ax2.set_ylabel('Beta Coefficient', fontsize=12)
    ax2.set_title('Factor Betas by Portfolio', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(portfolios)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. R-squared comparison
    ax3 = axes[1, 0]

    r_squared = metric_data['r_squared'].values
    adj_r_squared = metric_data['adj_r_squared'].values

    x = np.arange(len(portfolios))
    width = 0.35

    ax3.bar(x - width/2, r_squared, width, label='R²', alpha=0.8, color='#2E86AB')
    ax3.bar(x + width/2, adj_r_squared, width, label='Adjusted R²', alpha=0.8, color='#A23B72')

    ax3.set_ylabel('R² Value', fontsize=12)
    ax3.set_title('Model Fit (R²)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(portfolios)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for i, (r2, adj_r2) in enumerate(zip(r_squared, adj_r_squared)):
        ax3.text(i - width/2, r2 + 0.01, '.3f', ha='center', va='bottom', fontsize=9)
        ax3.text(i + width/2, adj_r2 + 0.01, '.3f', ha='center', va='bottom', fontsize=9)

    # 4. Statistical significance summary
    ax4 = axes[1, 1]

    # Count significant results (p < 0.05)
    significant_alpha = (metric_data['alpha_p'] < 0.05).sum()
    significant_market = (metric_data['mkt_rf_p'] < 0.05).sum()

    categories = ['Significant Alphas\n(p<0.05)', 'Significant Market Betas\n(p<0.05)']
    values = [significant_alpha, significant_market]
    total_possible = [len(metric_data), len(metric_data)]

    bars = ax4.bar(categories, values, color=['#F18F01', '#2E86AB'], alpha=0.8, edgecolor='black', linewidth=1)

    # Add percentage labels
    for bar, value, total in zip(bars, values, total_possible):
        height = bar.get_height()
        percentage = (value / total) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(percentage)}%\n({value}/{total})', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax4.set_ylabel('Number of Portfolios', fontsize=12)
    ax4.set_title('Statistical Significance Summary', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, len(metric_data) + 0.5)
    ax4.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        # Check if we're in an interactive environment
        try:
            plt.show()
        except:
            # If display fails, save to default location
            default_path = f"data/results/ff_regression/{metric_name}_visualization.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"Display not available. Visualization saved to: {default_path}")
            plt.close()

    # Print summary statistics
    print(f"\n=== Fama-French Results Summary: {metric_name} ===")
    print(f"Number of portfolios analyzed: {len(metric_data)}")
    print(f"Average R²: {metric_data['r_squared'].mean():.4f}")
    print(f"Average Adjusted R²: {metric_data['adj_r_squared'].mean():.4f}")
    print(f"Portfolios with significant alpha (p<0.05): {significant_alpha}/{len(metric_data)}")
    print(f"Portfolios with significant market beta (p<0.05): {significant_market}/{len(metric_data)}")

    # Portfolio-specific details
    print("\nPortfolio Details:")
    for _, row in metric_data.iterrows():
        portfolio = row['portfolio']
        alpha = row['alpha']
        alpha_p = row['alpha_p']
        r2 = row['r_squared']
        print(f"  {portfolio}: α={alpha:.4f} (p={alpha_p:.4f}), R²={r2:.3f}")

if __name__ == "__main__":
    # Example usage - force save to file
    metric = "WomenEmployees"
    save_path = f"data/results/ff_regression/{metric}_visualization.png"
    visualize_metric_ff_results(metric, save_path=save_path)