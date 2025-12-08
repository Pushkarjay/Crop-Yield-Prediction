"""
Data Generation Module - Creates Synthetic Dataset
Author: Pushkarjay Ajay
"""

import pandas as pd
import numpy as np
import random
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    STATES, CROPS, SOIL_TYPES, IRRIGATION_TYPES, SEED_VARIETIES, 
    SEASONS, TARGET_ROWS, RANDOM_SEED, BASE_DIR
)
from src.utils import print_header, print_section, print_success, print_info


def generate_synthetic_dataset(n_rows=TARGET_ROWS, save_path=None):
    """
    Generate synthetic crop yield dataset with realistic correlations
    
    Args:
        n_rows: Number of records to generate
        save_path: Path to save the CSV file
        
    Returns:
        pandas DataFrame with synthetic data
    """
    print_header("SYNTHETIC DATASET GENERATOR")
    print_info(f"Generating {n_rows:,} synthetic crop records...")
    
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    records = []
    state_names = list(STATES.keys())
    crop_names = list(CROPS.keys())
    
    rows_per_combo = n_rows // (len(state_names) * len(crop_names)) + 1
    
    for state in state_names:
        state_config = STATES[state]
        
        for crop in crop_names:
            crop_config = CROPS[crop]
            
            for _ in range(rows_per_combo):
                if len(records) >= n_rows:
                    break
                    
                record = generate_single_record(state, state_config, crop, crop_config)
                records.append(record)
        
        if len(records) >= n_rows:
            break
    
    # Trim to exact size
    records = records[:n_rows]
    
    df = pd.DataFrame(records)
    
    print_success(f"Generated {len(df):,} records")
    print_info(f"Columns: {len(df.columns)}")
    print_info(f"Crops: {df['Crop'].nunique()}")
    print_info(f"States: {df['State'].nunique()}")
    
    # Save if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print_success(f"Saved to: {save_path}")
    
    return df


def generate_single_record(state, state_config, crop, crop_config):
    """Generate a single synthetic record"""
    
    year = np.random.randint(2015, 2025)
    season = random.choice(crop_config['seasons'])
    
    # Weather features based on state
    temp_min, temp_max = state_config['temp_range']
    rain_min, rain_max = state_config['rainfall']
    ph_min, ph_max = state_config['soil_ph']
    
    temperature = np.random.uniform(temp_min, temp_max)
    rainfall = np.random.uniform(rain_min, rain_max)
    humidity = np.random.uniform(40, 90)
    sunshine = np.random.uniform(4, 10)
    gdd = np.random.uniform(1000, 3000)
    pressure = np.random.uniform(95, 105)
    wind_speed = np.random.uniform(5, 25)
    
    # Soil features
    soil_ph = np.random.uniform(ph_min, ph_max)
    soil_type = random.choice(SOIL_TYPES)
    soil_quality = np.random.uniform(50, 95)
    organic_carbon = np.random.uniform(0.3, 2.5)
    soil_moisture = np.random.uniform(20, 60)
    
    # NPK nutrients
    npk = crop_config['npk']
    nitrogen = np.random.uniform(*npk['N'])
    phosphorus = np.random.uniform(*npk['P'])
    potassium = np.random.uniform(*npk['K'])
    
    # Management
    irrigation_type = random.choice(IRRIGATION_TYPES)
    seed_variety = random.choice(SEED_VARIETIES)
    fertilizer = np.random.uniform(50, 350)
    pesticide = np.random.uniform(0, 20)
    
    # Economic
    crop_price = np.random.uniform(*crop_config['price_range'])
    
    # Calculate yield with realistic correlations
    base_yield = np.random.uniform(*crop_config['base_yield'])
    
    # Temperature effect
    opt_temp_min, opt_temp_max = crop_config['optimal_temp']
    temp_effect = calculate_condition_effect(temperature, opt_temp_min, opt_temp_max)
    
    # Rainfall effect
    opt_rain_min, opt_rain_max = crop_config['optimal_rainfall']
    rain_effect = calculate_condition_effect(rainfall, opt_rain_min, opt_rain_max)
    
    # pH effect
    opt_ph_min, opt_ph_max = crop_config['optimal_ph']
    ph_effect = calculate_condition_effect(soil_ph, opt_ph_min, opt_ph_max)
    
    # Irrigation effect
    irrigation_effect = {'Drip': 1.15, 'Sprinkler': 1.10, 'Canal': 1.05, 'Rainfed': 0.85}[irrigation_type]
    
    # Seed variety effect
    seed_effect = {'Local': 0.85, 'Improved': 1.0, 'Hybrid': 1.20}[seed_variety]
    
    # Fertilizer effect
    fert_effect = 0.9 + (fertilizer / 350) * 0.3
    
    # Calculate final yield
    yield_value = (
        base_yield 
        * temp_effect 
        * rain_effect 
        * ph_effect 
        * irrigation_effect 
        * seed_effect 
        * fert_effect 
        * np.random.uniform(0.85, 1.15)  # Random variation
    )
    
    yield_value = max(100, yield_value)  # Minimum yield
    
    return {
        'Year': year,
        'State': state,
        'Crop': crop,
        'Season': season,
        'Temperature_C': round(temperature, 2),
        'Rainfall_mm': round(rainfall, 2),
        'Humidity': round(humidity, 2),
        'Sunshine_Hours': round(sunshine, 2),
        'GDD': round(gdd, 2),
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
        'Fertilizer_Amount': round(fertilizer, 2),
        'Pesticide_Usage': round(pesticide, 2),
        'Crop_Price': round(crop_price, 2),
        'Yield_kg_per_hectare': round(yield_value, 2)
    }


def calculate_condition_effect(value, opt_min, opt_max):
    """Calculate yield effect based on how close value is to optimal range"""
    if opt_min <= value <= opt_max:
        return 1.0
    elif value < opt_min:
        return max(0.5, 1 - (opt_min - value) / opt_min * 0.5)
    else:
        return max(0.5, 1 - (value - opt_max) / opt_max * 0.5)


if __name__ == "__main__":
    from src.utils import log_output, get_log_filename
    
    log_file = get_log_filename("data_generation")
    
    with log_output(log_file):
        save_path = os.path.join(BASE_DIR, 'unified_dataset.csv')
        df = generate_synthetic_dataset(save_path=save_path)
        
        # Also save to Dataset folder
        dataset_copy = os.path.join(os.path.dirname(BASE_DIR), 'Dataset', 'synthetic_crop_data.csv')
        df.to_csv(dataset_copy, index=False)
        print_success(f"Also saved to: {dataset_copy}")
    
    print(f"\n📄 Log saved to: {log_file}")
