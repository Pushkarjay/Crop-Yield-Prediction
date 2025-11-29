# ============================================
# CROP YIELD PREDICTION - FULL ML WORKFLOW
# Phase-2: Steps 3-13 Complete Implementation
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import joblib
import warnings
from datetime import datetime
from scipy import stats

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (12, 6)
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 70)
print("🌾 CROP YIELD PREDICTION - FULL ML WORKFLOW")
print("=" * 70)

# ============================================
# STEP 3: CREATE MASTER COLUMN SCHEMA
# ============================================
print("\n" + "=" * 70)
print("📋 STEP 3: CREATE MASTER COLUMN SCHEMA")
print("=" * 70)

MASTER_COLUMNS = {
    # A. Identification
    'State': None,
    'District': None,
    'Year': None,
    'Season': None,
    'Crop': None,
    
    # B. Weather
    'Rainfall_mm': None,
    'Temperature_C': None,
    'Humidity': None,
    'Sunshine_hours': None,
    'GDD': None,  # Growing Degree Days
    'Rainfall_Anomaly': None,
    'Pressure_KPa': None,
    'Wind_Speed_Kmh': None,
    
    # C. Soil
    'Soil_Type': None,
    'Soil_pH': None,
    'Soil_Quality': None,
    'OrganicCarbon': None,
    'Nitrogen': None,
    'Phosphorus': None,
    'Potassium': None,
    'Soil_Humidity': None,
    'Soil_Health_Score': None,
    
    # D. Management
    'Irrigation_Type': None,
    'Irrigation_Schedule': None,
    'Sowing_Date': None,
    'Fertilizer_Amount_kg_per_hectare': None,
    'Seed_Variety': None,
    'Pesticide_Used': None,
    
    # E. Economic
    'Crop_Price': None,
    
    # F. Target
    'Yield_kg_per_hectare': None
}

print(f"✅ Master schema created with {len(MASTER_COLUMNS)} columns")
print("\n📋 Master Column Categories:")
print("   A. Identification: State, District, Year, Season, Crop")
print("   B. Weather: Rainfall_mm, Temperature_C, Humidity, Sunshine_hours, GDD, etc.")
print("   C. Soil: Soil_Type, Soil_pH, Nitrogen, Phosphorus, Potassium, etc.")
print("   D. Management: Irrigation_Type, Fertilizer_Amount, Seed_Variety, etc.")
print("   E. Economic: Crop_Price")
print("   F. Target: Yield_kg_per_hectare")

# ============================================
# STEP 4: LOAD AND MERGE ALL DATASETS
# ============================================
print("\n" + "=" * 70)
print("🔄 STEP 4: MERGE ALL DATASETS INTO UNIFIED DATASET")
print("=" * 70)

# Column mapping from source to master schema
COLUMN_MAPPING = {
    # Soil nutrients
    'N_SOIL': 'Nitrogen',
    'P_SOIL': 'Phosphorus',
    'K_SOIL': 'Potassium',
    'Nitrogen (N)': 'Nitrogen',
    'Phosphorus (P)': 'Phosphorus',
    'Potassium (K)': 'Potassium',
    
    # Temperature
    'TEMPERATURE': 'Temperature_C',
    'Air temperature (C)': 'Temperature_C',
    'Temperatue': 'Temperature_C',
    'Mean Temp': 'Temperature_C',
    
    # Humidity
    'HUMIDITY': 'Humidity',
    'Air humidity (%)': 'Humidity',
    'Average Humidity': 'Humidity',
    
    # Soil
    'ph': 'Soil_pH',
    'Soil_Quality': 'Soil_Quality',
    'Soil humidity 1': 'Soil_Humidity',
    'Moisture': 'Soil_Humidity',
    
    # Rainfall
    'RAINFALL': 'Rainfall_mm',
    'Rainfall_mm': 'Rainfall_mm',
    'Rain Fall (mm)': 'Rainfall_mm',
    'rainfall': 'Rainfall_mm',
    
    # Location & Crop
    'STATE': 'State',
    'CROP': 'Crop',
    'CROP_PRICE': 'Crop_Price',
    
    # Management
    'Fertilizer_Amount_kg_per_hectare': 'Fertilizer_Amount_kg_per_hectare',
    'Fertilizer': 'Fertilizer_Amount_kg_per_hectare',
    'Seed_Variety': 'Seed_Variety',
    'Irrigation_Schedule': 'Irrigation_Schedule',
    
    # Weather
    'Sunny_Days': 'Sunshine_hours',
    'Pressure (KPa)': 'Pressure_KPa',
    'Wind speed (Km/h)': 'Wind_Speed_Kmh',
    
    # Yield (Target)
    'Yield_kg_per_hectare': 'Yield_kg_per_hectare',
    'Crop Yield': 'Yield_kg_per_hectare',
    'Yeild (Q/acre)': 'Yield_kg_per_hectare',
    'millet yield': 'Yield_kg_per_hectare',
}

# Load datasets
DATASET_FOLDER = '../Dataset'
datasets = {}

csv_files = glob.glob(os.path.join(DATASET_FOLDER, '*.csv'))
excel_files = glob.glob(os.path.join(DATASET_FOLDER, '*.xlsx'))

for file_path in csv_files + excel_files:
    filename = os.path.basename(file_path)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')
        
        key_name = filename.replace('.csv', '').replace('.xlsx', '').replace(' ', '_').lower()
        
        # Skip metadata-only files
        if df.shape[1] <= 2 and df.shape[0] < 50:
            print(f"⚠️ Skipping {filename} (metadata only)")
            continue
            
        datasets[key_name] = df
        print(f"✅ Loaded: {filename} ({df.shape[0]} rows, {df.shape[1]} cols)")
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")

# Process and map each dataset
processed_dfs = []

for name, df in datasets.items():
    print(f"\n📊 Processing: {name}")
    
    # Create empty dataframe with master columns
    processed = pd.DataFrame(columns=list(MASTER_COLUMNS.keys()))
    
    # Map columns
    for src_col in df.columns:
        if src_col in COLUMN_MAPPING:
            target_col = COLUMN_MAPPING[src_col]
            processed[target_col] = df[src_col].values
            print(f"   ✓ Mapped: {src_col} → {target_col}")
        elif src_col in MASTER_COLUMNS:
            processed[src_col] = df[src_col].values
            print(f"   ✓ Direct: {src_col}")
    
    # Set source identifier
    processed['Source_Dataset'] = name
    
    if len(processed.dropna(axis=1, how='all').columns) > 1:
        processed_dfs.append(processed)
        print(f"   📈 Added {len(processed)} rows")

# Concatenate all processed datasets
print("\n🔄 Concatenating all datasets...")
unified_df = pd.concat(processed_dfs, ignore_index=True)

# Ensure all master columns exist
for col in MASTER_COLUMNS.keys():
    if col not in unified_df.columns:
        unified_df[col] = np.nan

print(f"\n✅ Unified dataset created!")
print(f"   Shape: {unified_df.shape[0]} rows × {unified_df.shape[1]} columns")

# ============================================
# STEP 5: DATA CLEANING & PREPROCESSING
# ============================================
print("\n" + "=" * 70)
print("🧹 STEP 5: DATA CLEANING & PREPROCESSING")
print("=" * 70)

# Standardize text columns
text_columns = ['State', 'District', 'Season', 'Crop', 'Soil_Type', 'Irrigation_Type']
for col in text_columns:
    if col in unified_df.columns and unified_df[col].notna().any():
        unified_df[col] = unified_df[col].astype(str).str.strip().str.title()
        unified_df[col] = unified_df[col].replace('Nan', np.nan)

# Convert numeric columns
numeric_columns = ['Rainfall_mm', 'Temperature_C', 'Humidity', 'Soil_pH', 'Soil_Quality',
                   'Nitrogen', 'Phosphorus', 'Potassium', 'Fertilizer_Amount_kg_per_hectare',
                   'Yield_kg_per_hectare', 'Sunshine_hours', 'Pressure_KPa', 'Wind_Speed_Kmh',
                   'Soil_Humidity', 'Crop_Price']

for col in numeric_columns:
    if col in unified_df.columns:
        unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce')

# Missing value summary
print("\n📊 Missing Value Summary:")
print("-" * 50)
missing_summary = unified_df.isnull().sum()
missing_pct = (unified_df.isnull().sum() / len(unified_df) * 100).round(2)
for col in unified_df.columns:
    if missing_summary[col] > 0:
        print(f"   {col}: {missing_summary[col]} ({missing_pct[col]}%)")

# Save unified dataset
unified_df.to_csv('unified_dataset.csv', index=False)
print(f"\n✅ Saved: unified_dataset.csv ({unified_df.shape[0]} rows)")

# ============================================
# STEP 6: DESCRIPTIVE SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 6: DESCRIPTIVE SUMMARY")
print("=" * 70)

print("\n📈 Summary Statistics for Numeric Columns:")
print(unified_df.describe().to_string())

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Plot 1: Yield Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
yield_data = unified_df['Yield_kg_per_hectare'].dropna()
if len(yield_data) > 0:
    sns.histplot(yield_data, kde=True, bins=50, color='green')
    plt.title('Distribution of Crop Yield')
    plt.xlabel('Yield (kg/hectare)')
    plt.ylabel('Frequency')

# Plot 2: Rainfall vs Yield
plt.subplot(1, 2, 2)
rainfall = unified_df['Rainfall_mm'].dropna()
yield_vals = unified_df.loc[unified_df['Rainfall_mm'].notna(), 'Yield_kg_per_hectare']
if len(rainfall) > 0 and len(yield_vals) > 0:
    plt.scatter(rainfall[:len(yield_vals)], yield_vals[:len(rainfall)], alpha=0.5, c='blue')
    plt.title('Rainfall vs Yield')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Yield (kg/hectare)')

plt.tight_layout()
plt.savefig('plots/01_yield_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/01_yield_distribution.png")

# ============================================
# STEP 7: DETAILED EDA
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 7: DETAILED EDA")
print("=" * 70)

# Correlation Matrix
numeric_df = unified_df.select_dtypes(include=[np.number])
numeric_df = numeric_df.dropna(axis=1, how='all')

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(14, 10))
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix - Numeric Features')
    plt.tight_layout()
    plt.savefig('plots/02_correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/02_correlation_matrix.png")

# Crop-wise Yield (if Crop column has data)
if unified_df['Crop'].notna().sum() > 0:
    plt.figure(figsize=(14, 6))
    crop_yield = unified_df.groupby('Crop')['Yield_kg_per_hectare'].mean().sort_values(ascending=False)
    if len(crop_yield) > 0:
        crop_yield.head(20).plot(kind='bar', color='forestgreen', edgecolor='black')
        plt.title('Average Yield by Crop Type (Top 20)')
        plt.xlabel('Crop')
        plt.ylabel('Average Yield (kg/hectare)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/03_crop_yield_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved: plots/03_crop_yield_comparison.png")

# State-wise Yield (if State column has data)
if unified_df['State'].notna().sum() > 0:
    plt.figure(figsize=(14, 6))
    state_yield = unified_df.groupby('State')['Yield_kg_per_hectare'].mean().sort_values(ascending=False)
    if len(state_yield) > 0:
        state_yield.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Average Yield by State (Top 15)')
        plt.xlabel('State')
        plt.ylabel('Average Yield (kg/hectare)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/04_state_yield_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Saved: plots/04_state_yield_comparison.png")

# Weather vs Yield plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temperature vs Yield
ax1 = axes[0, 0]
temp = unified_df['Temperature_C'].dropna()
if len(temp) > 0:
    ax1.scatter(temp, unified_df.loc[temp.index, 'Yield_kg_per_hectare'], alpha=0.5, c='orange')
    ax1.set_title('Temperature vs Yield')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Yield (kg/hectare)')

# Humidity vs Yield
ax2 = axes[0, 1]
humidity = unified_df['Humidity'].dropna()
if len(humidity) > 0:
    ax2.scatter(humidity, unified_df.loc[humidity.index, 'Yield_kg_per_hectare'], alpha=0.5, c='blue')
    ax2.set_title('Humidity vs Yield')
    ax2.set_xlabel('Humidity (%)')
    ax2.set_ylabel('Yield (kg/hectare)')

# Nitrogen vs Yield
ax3 = axes[1, 0]
nitrogen = unified_df['Nitrogen'].dropna()
if len(nitrogen) > 0:
    ax3.scatter(nitrogen, unified_df.loc[nitrogen.index, 'Yield_kg_per_hectare'], alpha=0.5, c='green')
    ax3.set_title('Nitrogen vs Yield')
    ax3.set_xlabel('Nitrogen')
    ax3.set_ylabel('Yield (kg/hectare)')

# Fertilizer vs Yield
ax4 = axes[1, 1]
fert = unified_df['Fertilizer_Amount_kg_per_hectare'].dropna()
if len(fert) > 0:
    ax4.scatter(fert, unified_df.loc[fert.index, 'Yield_kg_per_hectare'], alpha=0.5, c='purple')
    ax4.set_title('Fertilizer Amount vs Yield')
    ax4.set_xlabel('Fertilizer (kg/hectare)')
    ax4.set_ylabel('Yield (kg/hectare)')

plt.tight_layout()
plt.savefig('plots/05_weather_soil_yield.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/05_weather_soil_yield.png")

# ============================================
# STEP 8: MODEL PREPARATION
# ============================================
print("\n" + "=" * 70)
print("🔧 STEP 8: MODEL PREPARATION")
print("=" * 70)

# Select features for modeling
feature_columns = ['Rainfall_mm', 'Temperature_C', 'Humidity', 'Soil_pH', 'Soil_Quality',
                   'Nitrogen', 'Phosphorus', 'Potassium', 'Fertilizer_Amount_kg_per_hectare',
                   'Sunshine_hours', 'Pressure_KPa', 'Wind_Speed_Kmh', 'Soil_Humidity',
                   'Irrigation_Schedule', 'Seed_Variety', 'Crop_Price']

target_column = 'Yield_kg_per_hectare'

# Filter columns that exist and have data
available_features = []
for col in feature_columns:
    if col in unified_df.columns and unified_df[col].notna().sum() > 100:
        available_features.append(col)

print(f"📊 Available features for modeling: {len(available_features)}")
print(f"   {available_features}")

# Prepare data - only use columns that exist in the dataframe
existing_features = [col for col in feature_columns if col in unified_df.columns]
model_df = unified_df[existing_features + [target_column]].copy()
model_df = model_df.dropna(subset=[target_column])

print(f"📊 Samples with target value: {len(model_df)}")

# Handle categorical columns with LabelEncoder
label_encoders = {}
categorical_cols = ['Irrigation_Schedule', 'Seed_Variety']

for col in categorical_cols:
    if col in model_df.columns:
        le = LabelEncoder()
        model_df[col] = model_df[col].fillna('Unknown')
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        label_encoders[col] = le
        print(f"   ✓ Encoded: {col}")

# Drop columns that are all NaN
model_df = model_df.dropna(axis=1, how='all')

# Get actual numeric features (exclude target)
actual_features = [col for col in model_df.columns if col != target_column]
print(f"📊 Actual features used: {len(actual_features)}")
print(f"   {actual_features}")

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X = model_df[actual_features]
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=actual_features)
y = model_df[target_column].values

# Update available_features for later use
available_features = actual_features

print(f"\n📊 Final dataset shape: X={X_imputed.shape}, y={y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"📊 Train set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# ============================================
# STEP 9: TRAIN MULTIPLE MODELS
# ============================================
print("\n" + "=" * 70)
print("🤖 STEP 9: TRAIN MULTIPLE MODELS")
print("=" * 70)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n🔄 Training: {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Model': name,
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse
    })
    
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")

# Results comparison
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON RESULTS")
print("=" * 70)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)
print(results_df.to_string(index=False))

# Select best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} (R² = {results_df.iloc[0]['R²']:.4f})")

# ============================================
# STEP 10: FEATURE IMPORTANCE
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 10: FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance from Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance (Random Forest):")
print(feature_importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('plots/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/06_feature_importance.png")

# ============================================
# STEP 11: SAVE FINAL MODEL
# ============================================
print("\n" + "=" * 70)
print("💾 STEP 11: SAVE FINAL MODEL")
print("=" * 70)

os.makedirs('model', exist_ok=True)

# Save best model
joblib.dump(best_model, 'model/model.pkl')
print("✅ Saved: model/model.pkl")

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')
print("✅ Saved: model/scaler.pkl")

# Save label encoders
joblib.dump(label_encoders, 'model/label_encoders.pkl')
print("✅ Saved: model/label_encoders.pkl")

# Save feature list
joblib.dump(available_features, 'model/feature_list.pkl')
print("✅ Saved: model/feature_list.pkl")

# Save imputer
joblib.dump(imputer, 'model/imputer.pkl')
print("✅ Saved: model/imputer.pkl")

# Save model info
model_info = {
    'model_name': best_model_name,
    'features': available_features,
    'r2_score': results_df.iloc[0]['R²'],
    'mae': results_df.iloc[0]['MAE'],
    'rmse': results_df.iloc[0]['RMSE'],
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(y_train),
    'test_samples': len(y_test)
}
joblib.dump(model_info, 'model/model_info.pkl')
print("✅ Saved: model/model_info.pkl")

print("\n" + "=" * 70)
print("🎉 WORKFLOW COMPLETE!")
print("=" * 70)
print(f"""
✅ Steps 1-11 completed successfully!

📁 Files Created:
   - unified_dataset.csv (merged dataset)
   - plots/01_yield_distribution.png
   - plots/02_correlation_matrix.png
   - plots/03_crop_yield_comparison.png
   - plots/04_state_yield_comparison.png
   - plots/05_weather_soil_yield.png
   - plots/06_feature_importance.png
   - model/model.pkl (best model: {best_model_name})
   - model/scaler.pkl
   - model/label_encoders.pkl
   - model/feature_list.pkl
   - model/imputer.pkl
   - model/model_info.pkl

🏆 Best Model Performance:
   Model: {best_model_name}
   R² Score: {results_df.iloc[0]['R²']:.4f}
   MAE: {results_df.iloc[0]['MAE']:.4f}
   RMSE: {results_df.iloc[0]['RMSE']:.4f}

📊 Now creating API and Dashboard (Steps 12-13)...
""")
