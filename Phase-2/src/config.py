"""
Configuration and Constants for Crop Yield Prediction
Author: Pushkarjay Ajay
"""

import os

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for dir_path in [MODEL_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
TARGET_ROWS = 75000
RANDOM_SEED = 42

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
    'Sunflower': {
        'optimal_temp': (20, 30), 'optimal_rainfall': (400, 800), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (60, 90), 'P': (40, 60), 'K': (30, 50)},
        'base_yield': (1000, 1800), 'seasons': ['Kharif', 'Rabi'],
        'price_range': (4000, 5000)
    },
    'Chickpea': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (400, 700), 'optimal_ph': (6.0, 8.0),
        'npk': {'N': (15, 30), 'P': (40, 70), 'K': (20, 40)},
        'base_yield': (800, 1500), 'seasons': ['Rabi'],
        'price_range': (4500, 6000)
    },
    'Pigeonpea': {
        'optimal_temp': (20, 30), 'optimal_rainfall': (600, 1000), 'optimal_ph': (5.5, 7.0),
        'npk': {'N': (15, 30), 'P': (50, 80), 'K': (20, 40)},
        'base_yield': (700, 1200), 'seasons': ['Kharif'],
        'price_range': (5500, 7000)
    },
    'Lentil': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (300, 500), 'optimal_ph': (6.0, 8.0),
        'npk': {'N': (15, 25), 'P': (40, 60), 'K': (20, 30)},
        'base_yield': (600, 1200), 'seasons': ['Rabi'],
        'price_range': (4000, 5500)
    },
    'Mungbean': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (400, 700), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (15, 25), 'P': (40, 60), 'K': (20, 30)},
        'base_yield': (500, 1000), 'seasons': ['Kharif', 'Summer'],
        'price_range': (6000, 8000)
    },
    'Onion': {
        'optimal_temp': (15, 25), 'optimal_rainfall': (400, 700), 'optimal_ph': (6.0, 7.0),
        'npk': {'N': (80, 120), 'P': (40, 70), 'K': (60, 100)},
        'base_yield': (15000, 25000), 'seasons': ['Rabi', 'Kharif'],
        'price_range': (800, 2500)
    },
    'Potato': {
        'optimal_temp': (15, 22), 'optimal_rainfall': (400, 800), 'optimal_ph': (5.5, 6.5),
        'npk': {'N': (120, 180), 'P': (60, 100), 'K': (100, 150)},
        'base_yield': (20000, 35000), 'seasons': ['Rabi'],
        'price_range': (600, 1500)
    },
    'Tomato': {
        'optimal_temp': (20, 28), 'optimal_rainfall': (500, 800), 'optimal_ph': (6.0, 7.0),
        'npk': {'N': (100, 150), 'P': (50, 80), 'K': (80, 120)},
        'base_yield': (25000, 40000), 'seasons': ['Rabi', 'Kharif'],
        'price_range': (500, 2000)
    },
    'Banana': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (1000, 2000), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (150, 250), 'P': (60, 100), 'K': (200, 300)},
        'base_yield': (30000, 50000), 'seasons': ['Annual'],
        'price_range': (1000, 2500)
    },
    'Mango': {
        'optimal_temp': (24, 35), 'optimal_rainfall': (800, 1500), 'optimal_ph': (5.5, 7.5),
        'npk': {'N': (100, 150), 'P': (50, 80), 'K': (100, 150)},
        'base_yield': (5000, 10000), 'seasons': ['Annual'],
        'price_range': (3000, 8000)
    },
    'Grapes': {
        'optimal_temp': (15, 30), 'optimal_rainfall': (400, 800), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (100, 150), 'P': (60, 100), 'K': (150, 250)},
        'base_yield': (15000, 25000), 'seasons': ['Annual'],
        'price_range': (4000, 10000)
    },
    'Jute': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (1500, 2500), 'optimal_ph': (6.0, 7.5),
        'npk': {'N': (40, 70), 'P': (20, 40), 'K': (40, 60)},
        'base_yield': (2000, 3000), 'seasons': ['Kharif'],
        'price_range': (4000, 5500)
    },
    'Coconut': {
        'optimal_temp': (25, 35), 'optimal_rainfall': (1500, 3000), 'optimal_ph': (5.5, 7.0),
        'npk': {'N': (50, 100), 'P': (30, 50), 'K': (100, 200)},
        'base_yield': (8000, 15000), 'seasons': ['Annual'],
        'price_range': (15000, 25000)
    },
    'Coffee': {
        'optimal_temp': (18, 28), 'optimal_rainfall': (1500, 2500), 'optimal_ph': (5.0, 6.5),
        'npk': {'N': (80, 120), 'P': (40, 70), 'K': (80, 120)},
        'base_yield': (500, 1500), 'seasons': ['Annual'],
        'price_range': (20000, 35000)
    },
}

# Soil types
SOIL_TYPES = ['Alluvial', 'Black', 'Red', 'Laterite', 'Desert', 'Mountain', 'Saline']

# Irrigation types
IRRIGATION_TYPES = ['Drip', 'Sprinkler', 'Canal', 'Rainfed']

# Seed varieties
SEED_VARIETIES = ['Local', 'Improved', 'Hybrid']

# Seasons
SEASONS = ['Kharif', 'Rabi', 'Summer', 'Annual']

# =============================================================================
# ML CONFIGURATION
# =============================================================================
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature columns for ML model
NUMERIC_FEATURES = [
    'Rainfall_mm', 'Temperature_C', 'Humidity', 'Sunshine_Hours', 'GDD',
    'Pressure_KPa', 'Wind_Speed_Kmh', 'Soil_pH', 'Soil_Quality', 'OrganicCarbon',
    'Nitrogen', 'Phosphorus', 'Potassium', 'Soil_Moisture',
    'Fertilizer_Amount', 'Pesticide_Usage', 'Year', 'Crop_Price'
]

CATEGORICAL_FEATURES = ['Crop', 'State', 'Irrigation_Type', 'Seed_Variety']

TARGET_COLUMN = 'Yield_kg_per_hectare'

# Features used in final model (22 features)
MODEL_FEATURES = [
    'Year', 'Rainfall_mm', 'Temperature_C', 'Humidity', 'Sunshine_Hours', 'GDD',
    'Pressure_KPa', 'Wind_Speed_Kmh', 'Soil_pH', 'Soil_Quality', 'OrganicCarbon',
    'Nitrogen', 'Phosphorus', 'Potassium', 'Soil_Moisture', 'Fertilizer_Amount',
    'Pesticide_Usage', 'Crop_Price', 'Crop_Encoded', 'State_Encoded',
    'Irrigation_Type_Encoded', 'Seed_Variety_Encoded'
]
