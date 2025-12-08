"""
Synthetic Dataset Generator for Crop Yield Prediction
Author: Pushkarjay Ajay
Generates ~75,000 realistic crop yield records with all features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("=" * 70)
print("🌾 SYNTHETIC DATASET GENERATOR")
print("=" * 70)

np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_ROWS = 75000

# Indian States with agricultural data
STATES = {
    'Punjab': {'temp_range': (15, 35), 'rainfall': (500, 900), 'soil_ph': (7.0, 8.5)},
    'Haryana': {'temp_range': (15, 38), 'rainfall': (400, 800), 'soil_ph': (7.0, 8.5)},
    'Uttar Pradesh': {'temp_range': (18, 40), 'rainfall': (600, 1200), 'soil_ph': (6.5, 8.0)},
    'Maharashtra': {'temp_range': (20, 38), 'rainfall': (600, 2500), 'soil_ph': (6.0, 7.5)},
    'Madhya Pradesh': {'temp_range': (18, 42), 'rainfall': (800, 1400), 'soil_ph': (6.5, 8.0)},
    'Rajasthan': {'temp_range': (20, 45), 'rainfall': (200, 600), 'soil_ph': (7.5, 9.0)},
    'Gujarat': {'temp_range': (22, 40), 'rainfall': (400, 1500), 'soil_ph': (7.0, 8.5)},
    'Karnataka': {'temp_range': (20, 35), 'rainfall': (600, 3000), 'soil_ph': (5.5, 7.5)},
    'Tamil Nadu': {'temp_range': (24, 38), 'rainfall': (800, 1500), 'soil_ph': (6.0, 7.5)},
    'Andhra Pradesh': {'temp_range': (22, 40), 'rainfall': (700, 1200), 'soil_ph': (6.0, 8.0)},
    'Telangana': {'temp_range': (22, 42), 'rainfall': (700, 1100), 'soil_ph': (6.5, 8.0)},
    'West Bengal': {'temp_range': (18, 35), 'rainfall': (1200, 2500), 'soil_ph': (5.5, 7.0)},
    'Bihar': {'temp_range': (18, 40), 'rainfall': (1000, 1500), 'soil_ph': (6.5, 8.0)},
    'Odisha': {'temp_range': (20, 38), 'rainfall': (1200, 1800), 'soil_ph': (5.5, 7.0)},
    'Assam': {'temp_range': (15, 32), 'rainfall': (1500, 3000), 'soil_ph': (4.5, 6.5)},
    'Kerala': {'temp_range': (23, 33), 'rainfall': (2000, 3500), 'soil_ph': (5.0, 6.5)},
    'Chhattisgarh': {'temp_range': (20, 42), 'rainfall': (1200, 1600), 'soil_ph': (5.5, 7.0)},
    'Jharkhand': {'temp_range': (18, 38), 'rainfall': (1100, 1500), 'soil_ph': (5.5, 7.0)},
    'Himachal Pradesh': {'temp_range': (5, 25), 'rainfall': (800, 2000), 'soil_ph': (5.5, 7.0)},
    'Uttarakhand': {'temp_range': (8, 30), 'rainfall': (1000, 2500), 'soil_ph': (5.5, 7.5)},
}

# Crops with their optimal conditions and yield characteristics
CROPS = {
    'Rice': {
        'optimal_temp': (22, 32), 'optimal_rainfall': (1000, 2000), 'optimal_ph': (5.5, 7.0),
        'npk': {'N': (80, 120), 'P': (30, 60), 'K': (30, 60)},
        'base_yield': (3000, 5000), 'seasons': ['Kharif', 'Rabi'],
        'price_range': (1800, 2500)
    },
    'Wheat': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (400, 800), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (100, 150), 'P': (40, 70), 'K': (30, 50)},
        'base_yield': (2500, 4500), 'seasons': ['Rabi'],
        'price_range': (1900, 2400)
    },
    'Maize': {
        'optimal_temp': (21, 30), 'optimal_rainfall': (500, 1000), 'optimal_ph': (5.5, 7.5),
        'npk': {'N': (120, 180), 'P': (50, 80), 'K': (40, 70)},
        'base_yield': (4000, 8000), 'seasons': ['Kharif', 'Rabi'],
        'price_range': (1700, 2200)
    },
    'Cotton': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (600, 1200), 'optimal_ph': (6.0, 8.0),
        'npk': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60)},
        'base_yield': (1500, 2500), 'seasons': ['Kharif'],
        'price_range': (5000, 7000)
    },
    'Sugarcane': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (1500, 2500), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (150, 250), 'P': (60, 100), 'K': (80, 120)},
        'base_yield': (60000, 100000), 'seasons': ['Annual'],
        'price_range': (280, 350)
    },
    'Soybean': {
        'optimal_temp': (20, 30), 'optimal_rainfall': (600, 1000), 'optimal_ph': (6.0, 7.0),
        'npk': {'N': (20, 40), 'P': (60, 90), 'K': (40, 60)},
        'base_yield': (1500, 2500), 'seasons': ['Kharif'],
        'price_range': (3500, 4500)
    },
    'Groundnut': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (500, 1000), 'optimal_ph': (6.0, 7.0),
        'npk': {'N': (20, 40), 'P': (40, 80), 'K': (40, 60)},
        'base_yield': (1200, 2000), 'seasons': ['Kharif', 'Summer'],
        'price_range': (4500, 6000)
    },
    'Mustard': {
        'optimal_temp': (10, 25), 'optimal_rainfall': (300, 600), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (60, 90), 'P': (30, 50), 'K': (20, 40)},
        'base_yield': (1000, 1800), 'seasons': ['Rabi'],
        'price_range': (4000, 5500)
    },
    'Chickpea': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (400, 700), 'optimal_ph': (6.0, 8.0),
        'npk': {'N': (15, 30), 'P': (40, 70), 'K': (20, 40)},
        'base_yield': (800, 1500), 'seasons': ['Rabi'],
        'price_range': (4500, 5500)
    },
    'Pigeon Pea': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (600, 1000), 'optimal_ph': (5.5, 7.0),
        'npk': {'N': (15, 30), 'P': (50, 80), 'K': (20, 40)},
        'base_yield': (700, 1200), 'seasons': ['Kharif'],
        'price_range': (5500, 7000)
    },
    'Lentil': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (300, 500), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (15, 25), 'P': (40, 60), 'K': (20, 35)},
        'base_yield': (600, 1200), 'seasons': ['Rabi'],
        'price_range': (4000, 5000)
    },
    'Sunflower': {
        'optimal_temp': (20, 30), 'optimal_rainfall': (500, 800), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (60, 100), 'P': (40, 70), 'K': (40, 60)},
        'base_yield': (1000, 1800), 'seasons': ['Kharif', 'Rabi'],
        'price_range': (4000, 5500)
    },
    'Potato': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (500, 800), 'optimal_ph': (5.5, 6.5),
        'npk': {'N': (120, 180), 'P': (80, 120), 'K': (100, 150)},
        'base_yield': (20000, 35000), 'seasons': ['Rabi'],
        'price_range': (800, 1500)
    },
    'Onion': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (400, 700), 'optimal_ph': (6.0, 7.0),
        'npk': {'N': (80, 120), 'P': (50, 80), 'K': (60, 90)},
        'base_yield': (15000, 25000), 'seasons': ['Rabi', 'Kharif'],
        'price_range': (1000, 3000)
    },
    'Tomato': {
        'optimal_temp': (20, 30), 'optimal_rainfall': (500, 800), 'optimal_ph': (6.0, 7.0),
        'npk': {'N': (100, 150), 'P': (60, 100), 'K': (80, 120)},
        'base_yield': (25000, 40000), 'seasons': ['Rabi', 'Kharif'],
        'price_range': (1200, 3500)
    },
    'Banana': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (1500, 2500), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (150, 250), 'P': (50, 100), 'K': (200, 350)},
        'base_yield': (30000, 50000), 'seasons': ['Annual'],
        'price_range': (1500, 2500)
    },
    'Mango': {
        'optimal_temp': (24, 35), 'optimal_rainfall': (800, 1500), 'optimal_ph': (5.5, 7.5),
        'npk': {'N': (80, 150), 'P': (40, 80), 'K': (80, 150)},
        'base_yield': (8000, 15000), 'seasons': ['Annual'],
        'price_range': (3000, 8000)
    },
    'Grapes': {
        'optimal_temp': (20, 35), 'optimal_rainfall': (500, 800), 'optimal_ph': (6.5, 7.5),
        'npk': {'N': (100, 150), 'P': (80, 130), 'K': (150, 220)},
        'base_yield': (15000, 25000), 'seasons': ['Annual'],
        'price_range': (4000, 10000)
    },
    'Tea': {
        'optimal_temp': (15, 28), 'optimal_rainfall': (1500, 3000), 'optimal_ph': (4.5, 5.5),
        'npk': {'N': (100, 180), 'P': (30, 60), 'K': (50, 100)},
        'base_yield': (1500, 2500), 'seasons': ['Annual'],
        'price_range': (15000, 35000)
    },
    'Coffee': {
        'optimal_temp': (18, 28), 'optimal_rainfall': (1500, 2500), 'optimal_ph': (5.0, 6.5),
        'npk': {'N': (80, 150), 'P': (30, 60), 'K': (80, 150)},
        'base_yield': (800, 1500), 'seasons': ['Annual'],
        'price_range': (20000, 40000)
    },
    'Jute': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (1500, 2500), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (40, 80), 'P': (20, 40), 'K': (40, 80)},
        'base_yield': (2000, 3000), 'seasons': ['Kharif'],
        'price_range': (4000, 5500)
    },
    'Tobacco': {
        'optimal_temp': (20, 30), 'optimal_rainfall': (500, 1000), 'optimal_ph': (5.5, 7.0),
        'npk': {'N': (40, 80), 'P': (60, 100), 'K': (80, 120)},
        'base_yield': (1500, 2500), 'seasons': ['Rabi'],
        'price_range': (12000, 18000)
    },
}

SOIL_TYPES = ['Alluvial', 'Black', 'Red', 'Laterite', 'Desert', 'Mountain', 'Saline', 'Peaty']
IRRIGATION_TYPES = ['Canal', 'Tubewell', 'Tank', 'Drip', 'Sprinkler', 'Rainfed', 'Flood']
SEED_VARIETIES = ['Hybrid', 'HYV', 'Traditional', 'Organic', 'GM']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_yield_factor(actual, optimal_min, optimal_max):
    """Calculate yield multiplier based on how close to optimal conditions"""
    if optimal_min <= actual <= optimal_max:
        return 1.0  # Perfect conditions
    elif actual < optimal_min:
        deviation = (optimal_min - actual) / optimal_min
        return max(0.3, 1 - deviation * 0.5)
    else:
        deviation = (actual - optimal_max) / optimal_max
        return max(0.3, 1 - deviation * 0.5)

def generate_realistic_yield(crop_data, temp, rainfall, ph, nitrogen, phosphorus, 
                             potassium, fertilizer, irrigation_type, soil_quality):
    """Generate realistic yield based on all factors"""
    base_yield = np.random.uniform(crop_data['base_yield'][0], crop_data['base_yield'][1])
    
    # Temperature factor
    temp_factor = calculate_yield_factor(temp, crop_data['optimal_temp'][0], crop_data['optimal_temp'][1])
    
    # Rainfall factor
    rain_factor = calculate_yield_factor(rainfall, crop_data['optimal_rainfall'][0], crop_data['optimal_rainfall'][1])
    
    # pH factor
    ph_factor = calculate_yield_factor(ph, crop_data['optimal_ph'][0], crop_data['optimal_ph'][1])
    
    # NPK factors
    n_factor = calculate_yield_factor(nitrogen, crop_data['npk']['N'][0], crop_data['npk']['N'][1])
    p_factor = calculate_yield_factor(phosphorus, crop_data['npk']['P'][0], crop_data['npk']['P'][1])
    k_factor = calculate_yield_factor(potassium, crop_data['npk']['K'][0], crop_data['npk']['K'][1])
    npk_factor = (n_factor + p_factor + k_factor) / 3
    
    # Fertilizer factor (more fertilizer = generally higher yield, up to a point)
    fert_factor = min(1.2, 0.7 + fertilizer / 300)
    
    # Irrigation factor
    irrigation_factors = {
        'Drip': 1.15, 'Sprinkler': 1.10, 'Canal': 1.05, 
        'Tubewell': 1.05, 'Tank': 1.0, 'Flood': 0.95, 'Rainfed': 0.80
    }
    irr_factor = irrigation_factors.get(irrigation_type, 1.0)
    
    # Soil quality factor
    soil_factor = 0.7 + (soil_quality / 100) * 0.5
    
    # Calculate final yield
    final_yield = base_yield * temp_factor * rain_factor * ph_factor * npk_factor * fert_factor * irr_factor * soil_factor
    
    # Add some random variation (±10%)
    final_yield *= np.random.uniform(0.9, 1.1)
    
    return max(100, final_yield)  # Minimum yield of 100

# =============================================================================
# GENERATE DATASET
# =============================================================================

print(f"\n📊 Generating {TARGET_ROWS:,} synthetic records...")
print("   This may take a minute...")

data = []
years = list(range(2015, 2025))

for i in range(TARGET_ROWS):
    # Select state and crop
    state = random.choice(list(STATES.keys()))
    state_data = STATES[state]
    crop = random.choice(list(CROPS.keys()))
    crop_data = CROPS[crop]
    
    # Year and Season
    year = random.choice(years)
    season = random.choice(crop_data['seasons'])
    
    # Weather conditions (based on state characteristics)
    temperature = np.random.uniform(state_data['temp_range'][0], state_data['temp_range'][1])
    rainfall = np.random.uniform(state_data['rainfall'][0], state_data['rainfall'][1])
    humidity = np.random.uniform(40, 95)
    
    # Add seasonal variation
    if season == 'Kharif':  # Monsoon
        rainfall *= np.random.uniform(1.1, 1.5)
        humidity = min(95, humidity * 1.2)
    elif season == 'Rabi':  # Winter
        temperature *= np.random.uniform(0.7, 0.9)
        rainfall *= np.random.uniform(0.3, 0.6)
    
    # Soil conditions
    soil_type = random.choice(SOIL_TYPES)
    soil_ph = np.random.uniform(state_data['soil_ph'][0], state_data['soil_ph'][1])
    soil_quality = np.random.uniform(40, 100)
    soil_moisture = np.random.uniform(20, 80)
    organic_carbon = np.random.uniform(0.3, 2.5)
    
    # NPK values (based on crop requirements with some variation)
    nitrogen = np.random.uniform(crop_data['npk']['N'][0] * 0.5, crop_data['npk']['N'][1] * 1.3)
    phosphorus = np.random.uniform(crop_data['npk']['P'][0] * 0.5, crop_data['npk']['P'][1] * 1.3)
    potassium = np.random.uniform(crop_data['npk']['K'][0] * 0.5, crop_data['npk']['K'][1] * 1.3)
    
    # Management factors
    irrigation_type = random.choice(IRRIGATION_TYPES)
    seed_variety = random.choice(SEED_VARIETIES)
    fertilizer_amount = np.random.uniform(50, 350)
    pesticide_used = random.choice(['Yes', 'No'])
    
    # Additional weather
    sunshine_hours = np.random.uniform(4, 10)
    wind_speed = np.random.uniform(5, 30)
    pressure = np.random.uniform(99, 103)
    
    # Growing Degree Days
    gdd = max(0, (temperature - 10) * np.random.uniform(90, 150))
    
    # Crop price
    crop_price = np.random.uniform(crop_data['price_range'][0], crop_data['price_range'][1])
    
    # Calculate yield
    yield_value = generate_realistic_yield(
        crop_data, temperature, rainfall, soil_ph, nitrogen, phosphorus,
        potassium, fertilizer_amount, irrigation_type, soil_quality
    )
    
    # Create record
    record = {
        'State': state,
        'District': f'{state}_District_{random.randint(1, 20)}',
        'Year': year,
        'Season': season,
        'Crop': crop,
        'Rainfall_mm': round(rainfall, 2),
        'Temperature_C': round(temperature, 2),
        'Humidity': round(humidity, 2),
        'Sunshine_hours': round(sunshine_hours, 2),
        'GDD': round(gdd, 2),
        'Rainfall_Anomaly': round(np.random.uniform(-0.3, 0.3), 3),
        'Pressure_KPa': round(pressure, 2),
        'Wind_Speed_Kmh': round(wind_speed, 2),
        'Soil_Type': soil_type,
        'Soil_pH': round(soil_ph, 2),
        'Soil_Quality': round(soil_quality, 2),
        'OrganicCarbon': round(organic_carbon, 2),
        'Nitrogen': round(nitrogen, 2),
        'Phosphorus': round(phosphorus, 2),
        'Potassium': round(potassium, 2),
        'Soil_Moisture': round(soil_moisture, 2),
        'Irrigation_Type': irrigation_type,
        'Seed_Variety': seed_variety,
        'Fertilizer_Amount_kg_per_hectare': round(fertilizer_amount, 2),
        'Pesticide_Used': pesticide_used,
        'Crop_Price': round(crop_price, 2),
        'Yield_kg_per_hectare': round(yield_value, 2)
    }
    
    data.append(record)
    
    # Progress indicator
    if (i + 1) % 10000 == 0:
        print(f"   Generated {i + 1:,} records...")

# Create DataFrame
df = pd.DataFrame(data)

print(f"\n✅ Generated {len(df):,} records")

# =============================================================================
# DATASET STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("📊 DATASET STATISTICS")
print("=" * 70)

print(f"\n📋 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print(f"\n🌾 Crops: {df['Crop'].nunique()} unique")
print(df['Crop'].value_counts().head(10))

print(f"\n🗺️ States: {df['State'].nunique()} unique")
print(df['State'].value_counts().head(10))

print(f"\n📅 Years: {df['Year'].min()} - {df['Year'].max()}")
print(df['Year'].value_counts().sort_index())

print(f"\n🌡️ Temperature Range: {df['Temperature_C'].min():.1f}°C - {df['Temperature_C'].max():.1f}°C")
print(f"🌧️ Rainfall Range: {df['Rainfall_mm'].min():.0f}mm - {df['Rainfall_mm'].max():.0f}mm")
print(f"🌾 Yield Range: {df['Yield_kg_per_hectare'].min():.0f} - {df['Yield_kg_per_hectare'].max():.0f} kg/ha")

print("\n📊 Numeric Columns Summary:")
print(df.describe().round(2).to_string())

# =============================================================================
# SAVE DATASET
# =============================================================================

print("\n" + "=" * 70)
print("💾 SAVING DATASET")
print("=" * 70)

# Save to CSV
output_path = 'unified_dataset.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved: {output_path}")
print(f"   Size: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Also save to Dataset folder
dataset_folder_path = '../Dataset/synthetic_crop_data.csv'
df.to_csv(dataset_folder_path, index=False)
print(f"✅ Saved: {dataset_folder_path}")

# Backup original if exists
import shutil
import os
if os.path.exists('unified_dataset_BACKUP.csv'):
    os.remove('unified_dataset_BACKUP.csv')
    print("   Removed old backup")

print("\n" + "=" * 70)
print("✅ SYNTHETIC DATASET GENERATION COMPLETE!")
print("=" * 70)

print(f"""
📊 SUMMARY:
   - Total Records: {len(df):,}
   - Crops: {df['Crop'].nunique()}
   - States: {df['State'].nunique()}
   - Years: {df['Year'].min()} - {df['Year'].max()}
   - All values are complete (no missing data)

🔧 NEXT STEPS:
   Run 'python full_workflow.py' to train the ML model on this dataset!
""")
