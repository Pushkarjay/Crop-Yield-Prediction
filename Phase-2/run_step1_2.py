# ============================================
# STEP 1-2: Load and Display All Datasets
# ============================================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

print("=" * 70)
print("✅ STEP 1: All libraries imported successfully!")
print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🐍 Pandas Version: {pd.__version__}")
print(f"🔢 NumPy Version: {np.__version__}")
print("=" * 70)

# STEP 2: Load ALL Datasets
DATASET_FOLDER = '../Dataset'
csv_files = glob.glob(os.path.join(DATASET_FOLDER, '*.csv'))
excel_files = glob.glob(os.path.join(DATASET_FOLDER, '*.xlsx'))
all_files = csv_files + excel_files

print(f"\n📂 Dataset Folder: {os.path.abspath(DATASET_FOLDER)}")
print(f"📊 Found {len(csv_files)} CSV files and {len(excel_files)} Excel files")
print(f"📁 Total files to load: {len(all_files)}")

datasets = {}
for file_path in all_files:
    filename = os.path.basename(file_path)
    print(f"\n{'='*70}")
    print(f"📄 Loading: {filename}")
    print(f"{'='*70}")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        
        key_name = filename.replace('.csv', '').replace('.xlsx', '').replace(' ', '_').lower()
        datasets[key_name] = df
        
        print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n📋 Columns ({len(df.columns)}):")
        print(f"   {list(df.columns)}")
        print(f"\n🔍 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        print(f"\n📊 First 5 Rows:")
        print(df.head().to_string())
        print(f"\n✅ Successfully loaded: {filename}")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'='*70}")
print(f"📦 SUMMARY: Loaded {len(datasets)} datasets successfully")
print(f"{'='*70}")

# Summary table
print("\n📊 DATASET OVERVIEW SUMMARY:")
print("-" * 70)
print(f"{'Dataset':<30} {'Rows':>8} {'Cols':>6} {'Missing':>10} {'Missing%':>10}")
print("-" * 70)
for name, df in datasets.items():
    missing = df.isnull().sum().sum()
    total = df.shape[0] * df.shape[1]
    pct = (missing/total*100) if total > 0 else 0
    print(f"{name:<30} {df.shape[0]:>8} {df.shape[1]:>6} {missing:>10} {pct:>9.2f}%")
print("-" * 70)

total_rows = sum([df.shape[0] for df in datasets.values()])
print(f"\n📈 Total rows across all datasets: {total_rows:,}")

# All unique columns
print("\n📋 ALL UNIQUE COLUMNS ACROSS ALL DATASETS:")
print("-" * 70)
all_columns = set()
for name, df in datasets.items():
    all_columns.update(df.columns.tolist())
    print(f"\n🔹 {name}:")
    print(f"   {list(df.columns)}")

print(f"\n{'='*70}")
print(f"📊 Total Unique Columns Found: {len(all_columns)}")
print(f"{'='*70}")
print(sorted(all_columns))

print("\n" + "="*70)
print("🛑 CHECKPOINT: Steps 1-2 Complete!")
print("="*70)
print("""
✅ All required libraries imported
✅ All datasets (CSV and Excel) loaded from Dataset folder
✅ Each dataset's shape, columns, and preview displayed

DATASETS LOADED:
1. agricultural_yield_test.csv - Yield, soil quality, weather data
2. Data.csv - Soil humidity, air temperature, pressure, wind data
3. indiancrop_dataset.csv - N/P/K soil data, state, crop, price
4. Crop Data.xlsx - Excel file with crop data
5. crop yield data sheet.xlsx - Excel file with yield data
6. data.xlsx - Excel file with additional data

⏸️ Waiting for your confirmation to proceed with Steps 3-13
   (Merging, EDA, Modeling, API, Dashboard)
""")
