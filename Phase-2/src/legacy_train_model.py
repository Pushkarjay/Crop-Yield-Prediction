# ============================================
# CROP YIELD PREDICTION - ML TRAINING PIPELINE
# Uses Synthetic Dataset (75,000 records)
# Author: Pushkarjay Ajay
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 70)
print("🌾 CROP YIELD PREDICTION - ML TRAINING PIPELINE")
print("   Using Synthetic Dataset (75,000 records)")
print("=" * 70)

# ============================================
# STEP 1: LOAD DATASET
# ============================================
print("\n" + "=" * 70)
print("📂 STEP 1: LOAD DATASET")
print("=" * 70)

df = pd.read_csv('unified_dataset.csv')
print(f"\n✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📋 Columns: {df.columns.tolist()}")

print(f"\n📊 Data Types:")
print(df.dtypes)

print(f"\n📊 Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values!")
else:
    print(missing[missing > 0])

# ============================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

os.makedirs('plots', exist_ok=True)

print("\n📈 Summary Statistics:")
print(df.describe().round(2).to_string())

# Plot 1: Yield Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(df['Yield_kg_per_hectare'], bins=50, color='green', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Yield (kg/hectare)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Crop Yield')

ax2 = axes[1]
ax2.hist(np.log10(df['Yield_kg_per_hectare'] + 1), bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Log10(Yield)')
ax2.set_ylabel('Frequency')
ax2.set_title('Log-Transformed Yield Distribution')

plt.tight_layout()
plt.savefig('plots/01_yield_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/01_yield_distribution.png")

# Plot 2: Correlation Matrix
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(16, 12))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, annot_kws={'size': 8})
plt.title('Correlation Matrix - All Numeric Features')
plt.tight_layout()
plt.savefig('plots/02_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/02_correlation_matrix.png")

# Plot 3: Crop-wise Average Yield
plt.figure(figsize=(14, 6))
crop_yield = df.groupby('Crop')['Yield_kg_per_hectare'].mean().sort_values(ascending=False)
crop_yield.plot(kind='bar', color='forestgreen', edgecolor='black')
plt.title('Average Yield by Crop Type')
plt.xlabel('Crop')
plt.ylabel('Average Yield (kg/hectare)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/03_crop_yield_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/03_crop_yield_comparison.png")

# Plot 4: State-wise Average Yield
plt.figure(figsize=(14, 6))
state_yield = df.groupby('State')['Yield_kg_per_hectare'].mean().sort_values(ascending=False)
state_yield.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Average Yield by State')
plt.xlabel('State')
plt.ylabel('Average Yield (kg/hectare)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/04_state_yield_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/04_state_yield_comparison.png")

# Plot 5: Weather & Soil vs Yield
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Temperature vs Yield
ax = axes[0, 0]
ax.scatter(df['Temperature_C'], df['Yield_kg_per_hectare'], alpha=0.1, s=1)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Yield (kg/ha)')
ax.set_title('Temperature vs Yield')

# Rainfall vs Yield
ax = axes[0, 1]
ax.scatter(df['Rainfall_mm'], df['Yield_kg_per_hectare'], alpha=0.1, s=1)
ax.set_xlabel('Rainfall (mm)')
ax.set_ylabel('Yield (kg/ha)')
ax.set_title('Rainfall vs Yield')

# Humidity vs Yield
ax = axes[0, 2]
ax.scatter(df['Humidity'], df['Yield_kg_per_hectare'], alpha=0.1, s=1)
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('Yield (kg/ha)')
ax.set_title('Humidity vs Yield')

# Nitrogen vs Yield
ax = axes[1, 0]
ax.scatter(df['Nitrogen'], df['Yield_kg_per_hectare'], alpha=0.1, s=1)
ax.set_xlabel('Nitrogen')
ax.set_ylabel('Yield (kg/ha)')
ax.set_title('Nitrogen vs Yield')

# Fertilizer vs Yield
ax = axes[1, 1]
ax.scatter(df['Fertilizer_Amount_kg_per_hectare'], df['Yield_kg_per_hectare'], alpha=0.1, s=1)
ax.set_xlabel('Fertilizer (kg/ha)')
ax.set_ylabel('Yield (kg/ha)')
ax.set_title('Fertilizer vs Yield')

# Soil Quality vs Yield
ax = axes[1, 2]
ax.scatter(df['Soil_Quality'], df['Yield_kg_per_hectare'], alpha=0.1, s=1)
ax.set_xlabel('Soil Quality')
ax.set_ylabel('Yield (kg/ha)')
ax.set_title('Soil Quality vs Yield')

plt.tight_layout()
plt.savefig('plots/05_weather_soil_yield.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/05_weather_soil_yield.png")

# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================
print("\n" + "=" * 70)
print("🔧 STEP 3: FEATURE ENGINEERING")
print("=" * 70)

# Define feature columns and target
feature_columns = [
    # Weather features
    'Rainfall_mm', 'Temperature_C', 'Humidity', 'Sunshine_hours', 'GDD',
    'Pressure_KPa', 'Wind_Speed_Kmh',
    
    # Soil features
    'Soil_pH', 'Soil_Quality', 'OrganicCarbon', 'Nitrogen', 'Phosphorus', 
    'Potassium', 'Soil_Moisture',
    
    # Management features
    'Fertilizer_Amount_kg_per_hectare',
    
    # Economic
    'Crop_Price'
]

categorical_columns = ['Crop', 'State', 'Season', 'Soil_Type', 'Irrigation_Type', 'Seed_Variety']
target_column = 'Yield_kg_per_hectare'

print(f"\n📊 Numeric Features: {len(feature_columns)}")
print(f"📊 Categorical Features: {len(categorical_columns)}")

# Encode categorical variables
label_encoders = {}
df_encoded = df.copy()

for col in categorical_columns:
    if col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"   ✓ Encoded: {col} ({len(le.classes_)} categories)")

# Add encoded columns to features
encoded_features = [col + '_encoded' for col in categorical_columns if col in df_encoded.columns]
all_features = feature_columns + encoded_features

print(f"\n📊 Total Features: {len(all_features)}")

# ============================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================
print("\n" + "=" * 70)
print("📦 STEP 4: PREPARE DATA FOR MODELING")
print("=" * 70)

X = df_encoded[all_features]
y = df_encoded[target_column]

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target vector shape: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n📊 Training set: {X_train.shape[0]:,} samples")
print(f"📊 Test set: {X_test.shape[0]:,} samples")

# ============================================
# STEP 5: TRAIN MULTIPLE MODELS
# ============================================
print("\n" + "=" * 70)
print("🤖 STEP 5: TRAIN MULTIPLE MODELS")
print("=" * 70)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=8)
}

results = []

for name, model in models.items():
    print(f"\n🔄 Training: {name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'CV_R²_Mean': cv_scores.mean(),
        'CV_R²_Std': cv_scores.std()
    })
    
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================
# STEP 6: MODEL COMPARISON
# ============================================
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R²', ascending=False)
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['R²']
print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")

# ============================================
# STEP 7: FEATURE IMPORTANCE
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 7: FEATURE IMPORTANCE")
print("=" * 70)

# Use Random Forest for feature importance
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 15 Most Important Features (Random Forest):")
print(feature_importance.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='forestgreen')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/06_feature_importance.png")

# ============================================
# STEP 8: SAVE BEST MODEL
# ============================================
print("\n" + "=" * 70)
print("💾 STEP 8: SAVE BEST MODEL")
print("=" * 70)

os.makedirs('model', exist_ok=True)

best_model = models[best_model_name]

# Save model and artifacts
joblib.dump(best_model, 'model/model.pkl')
print("✅ Saved: model/model.pkl")

joblib.dump(scaler, 'model/scaler.pkl')
print("✅ Saved: model/scaler.pkl")

joblib.dump(label_encoders, 'model/label_encoders.pkl')
print("✅ Saved: model/label_encoders.pkl")

joblib.dump(all_features, 'model/feature_list.pkl')
print("✅ Saved: model/feature_list.pkl")

# Save model info
model_info = {
    'model_name': best_model_name,
    'r2_score': float(best_r2),
    'mae': float(results_df.iloc[0]['MAE']),
    'rmse': float(results_df.iloc[0]['RMSE']),
    'cv_r2_mean': float(results_df.iloc[0]['CV_R²_Mean']),
    'features': all_features,
    'numeric_features': feature_columns,
    'categorical_features': categorical_columns,
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'total_samples': int(len(df)),
    'crops': df['Crop'].unique().tolist(),
    'states': df['State'].unique().tolist(),
}
joblib.dump(model_info, 'model/model_info.pkl')
print("✅ Saved: model/model_info.pkl")

# ============================================
# STEP 9: PREDICTION VISUALIZATION
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 9: PREDICTION VISUALIZATION")
print("=" * 70)

y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
ax1 = axes[0]
ax1.scatter(y_test, y_pred_best, alpha=0.3, s=5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Yield (kg/ha)')
ax1.set_ylabel('Predicted Yield (kg/ha)')
ax1.set_title(f'Actual vs Predicted Yield\n{best_model_name} (R² = {best_r2:.4f})')

# Residuals
ax2 = axes[1]
residuals = y_test - y_pred_best
ax2.hist(residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Residual (Actual - Predicted)')
ax2.set_ylabel('Frequency')
ax2.set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('plots/07_prediction_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: plots/07_prediction_analysis.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("🎉 TRAINING PIPELINE COMPLETE!")
print("=" * 70)

print(f"""
📊 DATASET SUMMARY:
   - Total Samples: {len(df):,}
   - Features: {len(all_features)}
   - Crops: {df['Crop'].nunique()}
   - States: {df['State'].nunique()}
   - Years: {df['Year'].min()} - {df['Year'].max()}

🏆 BEST MODEL PERFORMANCE:
   - Model: {best_model_name}
   - R² Score: {best_r2:.4f}
   - MAE: {results_df.iloc[0]['MAE']:.2f} kg/ha
   - RMSE: {results_df.iloc[0]['RMSE']:.2f} kg/ha
   - CV R² (5-fold): {results_df.iloc[0]['CV_R²_Mean']:.4f} ± {results_df.iloc[0]['CV_R²_Std']:.4f}

📁 FILES CREATED:
   - plots/01_yield_distribution.png
   - plots/02_correlation_matrix.png
   - plots/03_crop_yield_comparison.png
   - plots/04_state_yield_comparison.png
   - plots/05_weather_soil_yield.png
   - plots/06_feature_importance.png
   - plots/07_prediction_analysis.png
   - model/model.pkl
   - model/scaler.pkl
   - model/label_encoders.pkl
   - model/feature_list.pkl
   - model/model_info.pkl

🚀 NEXT STEPS:
   1. Start the API: cd api && python app.py
   2. Open the Dashboard: dashboard/index.html
""")
