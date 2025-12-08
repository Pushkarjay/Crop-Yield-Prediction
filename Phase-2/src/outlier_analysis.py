"""
Outlier Analysis Module - Detect and analyze outliers in crop yield data
Author: Pushkarjay Ajay
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BASE_DIR, PLOTS_DIR, NUMERIC_FEATURES
from src.utils import (
    print_header, print_section, print_success, print_warning, 
    print_info, print_error
)
from src.visualization import create_outlier_analysis_plot


class OutlierAnalyzer:
    """
    Analyze and handle outliers in the dataset
    """
    
    def __init__(self, df):
        """
        Initialize with a DataFrame
        
        Args:
            df: pandas DataFrame with numeric columns
        """
        self.df = df.copy()
        self.outlier_stats = {}
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    def detect_iqr_outliers(self, column, multiplier=1.5):
        """
        Detect outliers using IQR method
        
        Args:
            column: Column name to analyze
            multiplier: IQR multiplier (default 1.5 for mild outliers)
            
        Returns:
            dict: Outlier statistics
        """
        data = self.df[column].dropna()
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers_low = data[data < lower_bound]
        outliers_high = data[data > upper_bound]
        
        return {
            'column': column,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers_low': len(outliers_low),
            'n_outliers_high': len(outliers_high),
            'n_outliers_total': len(outliers_low) + len(outliers_high),
            'pct_outliers': 100 * (len(outliers_low) + len(outliers_high)) / len(data)
        }
    
    def detect_zscore_outliers(self, column, threshold=3):
        """
        Detect outliers using Z-score method
        
        Args:
            column: Column name to analyze
            threshold: Z-score threshold (default 3)
            
        Returns:
            dict: Outlier statistics
        """
        data = self.df[column].dropna()
        
        mean = data.mean()
        std = data.std()
        z_scores = np.abs((data - mean) / std)
        
        outliers = data[z_scores > threshold]
        
        return {
            'column': column,
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'n_outliers': len(outliers),
            'pct_outliers': 100 * len(outliers) / len(data)
        }
    
    def analyze_all_columns(self, method='iqr'):
        """
        Analyze outliers in all numeric columns
        
        Args:
            method: 'iqr' or 'zscore'
            
        Returns:
            DataFrame: Summary of outlier analysis
        """
        print_section(f"Outlier Analysis ({method.upper()} Method)")
        
        results = []
        
        for col in self.numeric_columns:
            if method == 'iqr':
                stats = self.detect_iqr_outliers(col)
                results.append({
                    'Column': col,
                    'Q1': f"{stats['Q1']:.2f}",
                    'Q3': f"{stats['Q3']:.2f}",
                    'Lower': f"{stats['lower_bound']:.2f}",
                    'Upper': f"{stats['upper_bound']:.2f}",
                    'Outliers': stats['n_outliers_total'],
                    'Pct': f"{stats['pct_outliers']:.2f}%"
                })
            else:
                stats = self.detect_zscore_outliers(col)
                results.append({
                    'Column': col,
                    'Mean': f"{stats['mean']:.2f}",
                    'Std': f"{stats['std']:.2f}",
                    'Threshold': stats['threshold'],
                    'Outliers': stats['n_outliers'],
                    'Pct': f"{stats['pct_outliers']:.2f}%"
                })
            
            self.outlier_stats[col] = stats
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def handle_outliers(self, column, method='cap'):
        """
        Handle outliers in a column
        
        Args:
            column: Column name
            method: 'cap' (winsorize), 'remove', or 'median'
            
        Returns:
            Modified DataFrame
        """
        data = self.df[column]
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        if method == 'cap':
            # Winsorize: Cap values at bounds
            self.df[column] = data.clip(lower=lower, upper=upper)
            print_info(f"Capped {column} to [{lower:.2f}, {upper:.2f}]")
            
        elif method == 'remove':
            # Remove outlier rows
            mask = (data >= lower) & (data <= upper)
            self.df = self.df[mask]
            print_info(f"Removed {(~mask).sum()} outlier rows from {column}")
            
        elif method == 'median':
            # Replace with median
            median = data.median()
            mask = (data < lower) | (data > upper)
            self.df.loc[mask, column] = median
            print_info(f"Replaced {mask.sum()} outliers in {column} with median {median:.2f}")
        
        return self.df
    
    def get_summary(self):
        """
        Get a summary of the dataset and outlier analysis
        
        Returns:
            str: Summary string
        """
        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"  DATASET SUMMARY")
        summary.append(f"{'='*60}")
        summary.append(f"  Rows: {len(self.df):,}")
        summary.append(f"  Columns: {len(self.df.columns)}")
        summary.append(f"  Numeric Columns: {len(self.numeric_columns)}")
        
        if self.outlier_stats:
            total_outliers = sum(
                s.get('n_outliers_total', s.get('n_outliers', 0)) 
                for s in self.outlier_stats.values()
            )
            summary.append(f"\n  Total Outliers Detected: {total_outliers:,}")
        
        return '\n'.join(summary)


def run_outlier_analysis(data_path=None):
    """
    Run complete outlier analysis
    
    Args:
        data_path: Path to dataset CSV
        
    Returns:
        OutlierAnalyzer: Analyzer object with results
    """
    print_header("OUTLIER ANALYSIS")
    
    # Load data
    if data_path is None:
        data_path = os.path.join(BASE_DIR, 'unified_dataset.csv')
    
    print_info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print_success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Run IQR analysis
    iqr_results = analyzer.analyze_all_columns(method='iqr')
    print(f"\n{iqr_results.to_string(index=False)}")
    
    # Run Z-score analysis
    zscore_results = analyzer.analyze_all_columns(method='zscore')
    print(f"\n{zscore_results.to_string(index=False)}")
    
    # Print summary
    print(analyzer.get_summary())
    
    # Create visualization
    print_section("Creating Outlier Visualization")
    create_outlier_analysis_plot(df)
    
    # Key insights
    print_section("Key Insights")
    
    key_cols = ['Yield_kg_per_hectare', 'Temperature_C', 'Rainfall_mm', 'Fertilizer_Amount']
    for col in key_cols:
        if col in analyzer.outlier_stats:
            stats = analyzer.outlier_stats[col]
            outlier_count = stats.get('n_outliers_total', stats.get('n_outliers', 0))
            pct = stats.get('pct_outliers', 0)
            
            if outlier_count > 0:
                if pct > 5:
                    print_warning(f"{col}: {outlier_count:,} outliers ({pct:.2f}%) - High")
                elif pct > 1:
                    print_info(f"{col}: {outlier_count:,} outliers ({pct:.2f}%) - Moderate")
                else:
                    print_success(f"{col}: {outlier_count:,} outliers ({pct:.2f}%) - Low")
    
    print_header("ANALYSIS COMPLETE")
    
    return analyzer


if __name__ == "__main__":
    from src.utils import log_output, get_log_filename
    
    log_file = get_log_filename("outlier_analysis")
    
    with log_output(log_file):
        analyzer = run_outlier_analysis()
    
    print(f"\n📄 Log saved to: {log_file}")
