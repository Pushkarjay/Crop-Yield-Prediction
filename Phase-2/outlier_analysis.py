"""
Outlier Analysis and Data Cleaning Script
Author: Pushkarjay Ajay
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('unified_dataset.csv')

print("=" * 70)
print("🚨 DATA QUALITY ANALYSIS - OUTLIER DETECTION")
print("=" * 70)

# =============================================================================
# 1. TEMPERATURE ANALYSIS (CRITICAL ISSUE)
# =============================================================================
print("\n" + "=" * 70)
print("🌡️  TEMPERATURE ANALYSIS (CRITICAL)")
print("=" * 70)

temp_issues = df[df['Temperature_C'] > 50]
print(f"\n❌ Rows with Temperature > 50°C: {len(temp_issues)} rows")
print(f"   Row range: {temp_issues.index.min()} to {temp_issues.index.max()}")
print(f"   Values: {temp_issues['Temperature_C'].min():.0f}°C to {temp_issues['Temperature_C'].max():.0f}°C")

print("\n🔍 Sample of problematic rows:")
print(temp_issues[['Crop', 'State', 'Temperature_C', 'Yield_kg_per_hectare']].head(10))

# If these were Fahrenheit, converting to Celsius:
print(f"\n📐 If these were Fahrenheit values:")
print(f"   52°F → {(52-32)*5/9:.1f}°C (reasonable)")
print(f"   102°F → {(102-32)*5/9:.1f}°C (reasonable)")

# =============================================================================
# 2. OTHER STATISTICAL OUTLIERS (IQR Method)
# =============================================================================
print("\n" + "=" * 70)
print("📊 OTHER STATISTICAL OUTLIERS (IQR Method)")
print("=" * 70)

numeric_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall_mm', 
                'Temperature_C', 'Humidity', 'Soil_Moisture']

for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        if len(outliers) > 0:
            print(f"\n📌 {col}:")
            print(f"   Valid range: {lower:.2f} to {upper:.2f}")
            print(f"   Outliers: {len(outliers)} rows ({100*len(outliers)/len(df):.1f}%)")
            print(f"   Min outlier: {outliers[col].min():.2f}")
            print(f"   Max outlier: {outliers[col].max():.2f}")

# =============================================================================
# 3. CREATE VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("📈 Creating Visualization...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Temperature distribution
ax1 = axes[0, 0]
ax1.hist(df['Temperature_C'].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(x=50, color='red', linestyle='--', linewidth=2, label='Max realistic temp (50°C)')
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Frequency')
ax1.set_title('Temperature Distribution\n(Red line shows unrealistic values > 50°C)')
ax1.legend()

# 2. Temperature by row index
ax2 = axes[0, 1]
ax2.scatter(range(len(df)), df['Temperature_C'], alpha=0.5, s=1, c='steelblue')
ax2.axhline(y=50, color='red', linestyle='--', linewidth=2)
ax2.fill_between([4000, 4800], 0, 110, color='red', alpha=0.2, label='Problem rows (4000-4800)')
ax2.set_xlabel('Row Index')
ax2.set_ylabel('Temperature (°C)')
ax2.set_title('Temperature by Row Index\n(Highlighting problematic region)')
ax2.legend()

# 3. Phosphorus and Potassium
ax3 = axes[1, 0]
ax3.scatter(df['Phosphorus'], df['Potassium'], alpha=0.5, s=5, c='green')
ax3.axhline(y=90, color='orange', linestyle='--', label='K outlier threshold (90)')
ax3.axvline(x=130, color='red', linestyle='--', label='P outlier threshold (130)')
ax3.set_xlabel('Phosphorus')
ax3.set_ylabel('Potassium')
ax3.set_title('Phosphorus vs Potassium\n(Grapes from Punjab have high values)')
ax3.legend()

# 4. Summary text
ax4 = axes[1, 1]
ax4.axis('off')
summary = """
OUTLIER SUMMARY

CRITICAL ISSUE:
- Temperature > 50C: 800 rows (rows 4000-4799)
  Appears to be Fahrenheit values not converted to Celsius
  Max value: 102F = 39C (reasonable)

STATISTICAL OUTLIERS (may be valid):
- Rainfall > 1162mm: 48 rows
- Phosphorus > 130: 126 rows (Grapes in Punjab)
- Potassium > 90: 200 rows (Grapes in Punjab)

RECOMMENDATION:
1. FIX Temperature: Convert F to C for rows 4000-4799
2. KEEP P/K outliers: Grapes need more nutrients
3. KEEP Rainfall: High rainfall regions are valid
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plots/07_outlier_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: plots/07_outlier_analysis.png")
plt.close()

# =============================================================================
# 4. SHOW SPECIFIC ROWS
# =============================================================================
print("\n" + "=" * 70)
print("📋 SPECIFIC OUTLIER ROWS")
print("=" * 70)

print("\n🌡️ TEMPERATURE OUTLIERS (rows 4000-4010 sample):")
print(df.loc[4000:4010, ['Crop', 'State', 'Temperature_C', 'Rainfall_mm', 'Yield_kg_per_hectare']])

print("\n🧪 PHOSPHORUS/POTASSIUM OUTLIERS (Grapes-Punjab):")
grapes_punjab = df[(df['Crop'] == 'Grapes') & (df['State'] == 'Punjab')]
print(grapes_punjab[['Crop', 'State', 'Phosphorus', 'Potassium', 'Yield_kg_per_hectare']].head(10))

print("\n🌧️ RAINFALL OUTLIERS (>1162mm):")
rain_outliers = df[df['Rainfall_mm'] > 1162]
print(rain_outliers[['Crop', 'State', 'Rainfall_mm', 'Yield_kg_per_hectare']].head(10))

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)
