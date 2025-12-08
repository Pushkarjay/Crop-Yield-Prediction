"""
Analyze Data Quality by Source
"""
import pandas as pd

df = pd.read_csv('unified_dataset_BACKUP.csv')

print("Source dataset distribution:")
print(df['Source_Dataset'].value_counts())
print("\n" + "="*60)

for src in df['Source_Dataset'].unique():
    subset = df[df['Source_Dataset'] == src]
    has_crop = subset['Crop'].notna().sum()
    has_yield = subset['Yield_kg_per_hectare'].notna().sum()
    has_both = (subset['Crop'].notna() & subset['Yield_kg_per_hectare'].notna()).sum()
    
    print(f"\n{src} ({len(subset)} rows):")
    print(f"  Has Crop: {has_crop}")
    print(f"  Has Yield: {has_yield}")
    print(f"  Has BOTH: {has_both}")

# Find rows that have both Crop and Yield
usable = df[df['Crop'].notna() & df['Yield_kg_per_hectare'].notna()]
print(f"\n" + "="*60)
print(f"TOTAL USABLE ROWS (has Crop AND Yield): {len(usable)}")

if len(usable) > 0:
    print("\nSample usable rows:")
    print(usable[['Crop', 'State', 'Temperature_C', 'Yield_kg_per_hectare']].head(10))
