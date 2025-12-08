"""
Fix Temperature Outliers - Convert Fahrenheit to Celsius
Author: Pushkarjay Ajay
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('unified_dataset.csv')

print("=" * 70)
print("🌡️ FIX TEMPERATURE OUTLIERS")
print("=" * 70)

# Find rows with temperature > 50°C (these are likely Fahrenheit)
temp_outliers = df['Temperature_C'] > 50
count_outliers = temp_outliers.sum()

print(f"\n📊 Rows with Temperature > 50°C: {count_outliers}")

if count_outliers > 0:
    # Show before
    print(f"\n📋 Before Fix (sample):")
    print(df.loc[temp_outliers, ['Temperature_C', 'Yield_kg_per_hectare']].head(5))
    
    # Convert Fahrenheit to Celsius: C = (F - 32) * 5/9
    original_temps = df.loc[temp_outliers, 'Temperature_C'].copy()
    df.loc[temp_outliers, 'Temperature_C'] = (df.loc[temp_outliers, 'Temperature_C'] - 32) * 5/9
    
    # Show after
    print(f"\n📋 After Fix (sample):")
    print(df.loc[temp_outliers, ['Temperature_C', 'Yield_kg_per_hectare']].head(5))
    
    print(f"\n✅ Converted {count_outliers} temperature values from Fahrenheit to Celsius")
    print(f"   Range before: {original_temps.min():.1f}°F to {original_temps.max():.1f}°F")
    print(f"   Range after: {df.loc[temp_outliers, 'Temperature_C'].min():.1f}°C to {df.loc[temp_outliers, 'Temperature_C'].max():.1f}°C")

# Verify final temperature range
print(f"\n📊 Final Temperature Statistics:")
print(f"   Min: {df['Temperature_C'].min():.1f}°C")
print(f"   Max: {df['Temperature_C'].max():.1f}°C")
print(f"   Mean: {df['Temperature_C'].mean():.1f}°C")
print(f"   Rows > 50°C remaining: {(df['Temperature_C'] > 50).sum()}")

# Save fixed dataset
df.to_csv('unified_dataset.csv', index=False)
print(f"\n✅ Saved fixed dataset to: unified_dataset.csv")

print("\n" + "=" * 70)
print("✅ TEMPERATURE FIX COMPLETE")
print("=" * 70)
print("\n⚠️ RECOMMENDATION: Re-run full_workflow.py to retrain the model with fixed data!")
