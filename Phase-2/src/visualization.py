"""
Visualization Module - Creates all plots for the project
Author: Pushkarjay Ajay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PLOTS_DIR
from src.utils import print_header, print_section, print_success, print_info

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def create_all_visualizations(df, output_dir=PLOTS_DIR):
    """
    Create all visualization plots for the dataset
    
    Args:
        df: pandas DataFrame with crop data
        output_dir: Directory to save plots
    """
    print_header("VISUALIZATION GENERATOR")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print_section("Creating Plots")
    
    # Plot 1: Yield Distribution
    create_yield_distribution(df, output_dir)
    
    # Plot 2: Correlation Matrix
    create_correlation_matrix(df, output_dir)
    
    # Plot 3: Crop Yield Comparison
    create_crop_comparison(df, output_dir)
    
    # Plot 4: State Yield Comparison
    create_state_comparison(df, output_dir)
    
    # Plot 5: Weather & Soil vs Yield
    create_weather_soil_plots(df, output_dir)
    
    print_success(f"All visualizations saved to: {output_dir}")


def create_yield_distribution(df, output_dir):
    """Create yield distribution plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Normal distribution
    axes[0].hist(df['Yield_kg_per_hectare'], bins=50, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Yield (kg/hectare)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Crop Yield')
    axes[0].axvline(df['Yield_kg_per_hectare'].mean(), color='red', linestyle='--', label=f'Mean: {df["Yield_kg_per_hectare"].mean():,.0f}')
    axes[0].legend()
    
    # Log-transformed
    axes[1].hist(np.log10(df['Yield_kg_per_hectare'] + 1), bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Log10(Yield)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Log-Transformed Yield Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_yield_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 01_yield_distribution.png")


def create_correlation_matrix(df, output_dir):
    """Create correlation matrix heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(16, 12))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=0.5, annot_kws={'size': 8})
    plt.title('Correlation Matrix - All Numeric Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 02_correlation_matrix.png")


def create_crop_comparison(df, output_dir):
    """Create crop-wise yield comparison"""
    plt.figure(figsize=(14, 6))
    
    crop_yield = df.groupby('Crop')['Yield_kg_per_hectare'].mean().sort_values(ascending=False)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(crop_yield)))
    
    bars = plt.bar(range(len(crop_yield)), crop_yield.values, color=colors, edgecolor='black')
    plt.xticks(range(len(crop_yield)), crop_yield.index, rotation=45, ha='right')
    plt.xlabel('Crop')
    plt.ylabel('Average Yield (kg/hectare)')
    plt.title('Average Yield by Crop Type (75K Synthetic Dataset)')
    
    # Add value labels
    for bar, val in zip(bars, crop_yield.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'{val:,.0f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_crop_yield_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 03_crop_yield_comparison.png")


def create_state_comparison(df, output_dir):
    """Create state-wise yield comparison"""
    plt.figure(figsize=(14, 6))
    
    state_yield = df.groupby('State')['Yield_kg_per_hectare'].mean().sort_values(ascending=False)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(state_yield)))
    
    bars = plt.bar(range(len(state_yield)), state_yield.values, color=colors, edgecolor='black')
    plt.xticks(range(len(state_yield)), state_yield.index, rotation=45, ha='right')
    plt.xlabel('State')
    plt.ylabel('Average Yield (kg/hectare)')
    plt.title('Average Yield by State (20 Indian States)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_state_yield_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 04_state_yield_comparison.png")


def create_weather_soil_plots(df, output_dir):
    """Create weather and soil vs yield scatter plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    plots = [
        ('Temperature_C', 'Temperature (°C)', axes[0, 0]),
        ('Rainfall_mm', 'Rainfall (mm)', axes[0, 1]),
        ('Humidity', 'Humidity (%)', axes[0, 2]),
        ('Nitrogen', 'Nitrogen (kg/ha)', axes[1, 0]),
        ('Fertilizer_Amount_kg_per_hectare', 'Fertilizer (kg/ha)', axes[1, 1]),
        ('Soil_Quality', 'Soil Quality', axes[1, 2]),
    ]
    
    for col, label, ax in plots:
        ax.scatter(df[col], df['Yield_kg_per_hectare'], alpha=0.1, s=1, c='green')
        ax.set_xlabel(label)
        ax.set_ylabel('Yield (kg/ha)')
        ax.set_title(f'{label} vs Yield')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_weather_soil_yield.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 05_weather_soil_yield.png")


def create_feature_importance_plot(feature_importances, feature_names, output_dir=PLOTS_DIR):
    """
    Create feature importance bar plot
    
    Args:
        feature_importances: Array of importance values
        feature_names: List of feature names
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by importance
    indices = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    # Use top 15 features
    n_features = min(15, len(sorted_names))
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, n_features))[::-1]
    
    bars = plt.barh(range(n_features), sorted_importances[:n_features], color=colors, edgecolor='black')
    plt.yticks(range(n_features), sorted_names[:n_features])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance - Gradient Boosting Model')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, sorted_importances[:n_features]):
        plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 06_feature_importance.png")


def create_prediction_analysis_plot(y_true, y_pred, model_name, metrics, output_dir=PLOTS_DIR):
    """
    Create actual vs predicted scatter plot
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        metrics: Dict with r2, mae, rmse
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.3, s=5, c='green')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Yield (kg/ha)')
    ax1.set_ylabel('Predicted Yield (kg/ha)')
    ax1.set_title(f'{model_name} - Actual vs Predicted')
    ax1.legend()
    
    # Add metrics text
    metrics_text = f"R² = {metrics['r2']:.4f}\nMAE = {metrics['mae']:,.0f}\nRMSE = {metrics['rmse']:,.0f}"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residuals plot
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=5, c='steelblue')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Yield (kg/ha)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_prediction_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 07_prediction_analysis.png")


def create_outlier_analysis_plot(df, output_dir=PLOTS_DIR):
    """
    Create outlier analysis visualization
    
    Args:
        df: DataFrame with data
        output_dir: Directory to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Select numeric columns for outlier analysis
    numeric_cols = ['Temperature_C', 'Rainfall_mm', 'Yield_kg_per_hectare', 
                    'Fertilizer_Amount', 'Nitrogen', 'Soil_Quality']
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 3, idx % 3]
        
        if col in df.columns:
            data = df[col].dropna()
            
            # Box plot
            bp = ax.boxplot(data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][0].set_edgecolor('darkgreen')
            
            # Calculate outlier stats
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((data < lower) | (data > upper)).sum()
            
            ax.set_title(f'{col}\n(Outliers: {outliers})')
            ax.set_ylabel('Value')
    
    plt.suptitle('Outlier Analysis - Box Plots (75K Synthetic Dataset)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_outlier_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Created: 07_outlier_analysis.png")


if __name__ == "__main__":
    from src.utils import log_output, get_log_filename
    from src.config import BASE_DIR
    
    log_file = get_log_filename("visualization")
    
    with log_output(log_file):
        # Load dataset
        df = pd.read_csv(os.path.join(BASE_DIR, 'unified_dataset.csv'))
        print_info(f"Loaded dataset: {len(df):,} rows")
        
        # Create all visualizations
        create_all_visualizations(df)
        create_outlier_analysis_plot(df)
    
    print(f"\n📄 Log saved to: {log_file}")
