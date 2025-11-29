# ============================================
# CROP YIELD PREDICTION API
# Flask REST API for Model Predictions
# Enhanced with Crop, Location & Farm Area Support
# ============================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# ============================================
# Load Model and Preprocessing Objects
# ============================================
import sys
# Get the directory where app.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'model')

try:
    model = joblib.load(os.path.join(MODEL_PATH, 'model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
    feature_list = joblib.load(os.path.join(MODEL_PATH, 'feature_list.pkl'))
    imputer = joblib.load(os.path.join(MODEL_PATH, 'imputer.pkl'))
    model_info = joblib.load(os.path.join(MODEL_PATH, 'model_info.pkl'))
    print("✅ Model and preprocessing objects loaded successfully!")
    print(f"   Model: {model_info['model_name']}")
    print(f"   Features: {feature_list}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ============================================
# Constants for Crop & Location Support
# ============================================

# List of supported crops for multi-crop comparison
SUPPORTED_CROPS = [
    'Rice', 'Wheat', 'Maize', 'Bajra', 'Jowar', 
    'Pulses', 'Soybean', 'Groundnut', 'Cotton', 'Sugarcane'
]

# Area conversion factors
AREA_CONVERSIONS = {
    'Hectare': 1.0,
    'Acre': 0.4047,    # 1 acre = 0.4047 hectares
    'Bigha': 0.2529    # 1 bigha ≈ 0.2529 hectares (varies by region)
}

# Check if model supports crop as a feature
# TODO: Set this to True when model is retrained with crop feature
MODEL_SUPPORTS_CROP_FEATURE = False

# ============================================
# API Routes
# ============================================

@app.route('/')
def home():
    """Home route with API information"""
    return jsonify({
        'status': 'success',
        'message': '🌾 Crop Yield Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Predict crop yield',
            '/features': 'GET - Get list of required features',
            '/model-info': 'GET - Get model information',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Get list of required input features"""
    return jsonify({
        'status': 'success',
        'features': feature_list,
        'total_features': len(feature_list),
        'description': {
            'Rainfall_mm': 'Annual rainfall in millimeters (0-2000)',
            'Temperature_C': 'Average temperature in Celsius (10-50)',
            'Humidity': 'Relative humidity percentage (0-100)',
            'Soil_Quality': 'Soil quality index (0-100)',
            'Nitrogen': 'Soil nitrogen content (0-150)',
            'Phosphorus': 'Soil phosphorus content (0-150)',
            'Potassium': 'Soil potassium content (0-250)',
            'Fertilizer_Amount_kg_per_hectare': 'Fertilizer used (0-500)',
            'Sunshine_hours': 'Daily sunshine hours (0-150)',
            'Soil_Humidity': 'Soil moisture content (0-100)',
            'Irrigation_Schedule': 'Irrigation frequency (0-15)',
            'Seed_Variety': 'Seed variety type (0-5)'
        }
    })


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get model information and performance metrics"""
    if model_info:
        return jsonify({
            'status': 'success',
            'model_info': {
                'name': model_info.get('model_name', 'Unknown'),
                'r2_score': round(model_info.get('r2_score', 0), 4),
                'mae': round(model_info.get('mae', 0), 4),
                'rmse': round(model_info.get('rmse', 0), 4),
                'training_date': model_info.get('training_date', 'Unknown'),
                'training_samples': model_info.get('training_samples', 0),
                'features_used': feature_list
            }
        })
    return jsonify({'status': 'error', 'message': 'Model info not available'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict crop yield based on input features
    Enhanced with crop type, location, and farm area support
    
    Request body (JSON):
    {
        "Rainfall_mm": 500,
        "Temperature_C": 25,
        "Humidity": 70,
        "Soil_Quality": 75,
        "Nitrogen": 40,
        "Phosphorus": 50,
        "Potassium": 30,
        "Fertilizer_Amount_kg_per_hectare": 150,
        "Sunshine_hours": 100,
        "Soil_Humidity": 60,
        "Irrigation_Schedule": 5,
        "Seed_Variety": 1,
        "crop_type": "Rice",           // NEW
        "state": "Bihar",               // NEW
        "district": "Patna",            // NEW (optional)
        "season": "Kharif",             // NEW (optional)
        "agro_climatic_zone": "...",    // NEW (optional)
        "farm_area": 2.5,               // NEW (optional)
        "farm_area_unit": "Hectare",    // NEW (optional)
        "latitude": 25.5941,            // NEW (optional)
        "longitude": 85.1376,           // NEW (optional)
        "elevation": 150,               // NEW (optional)
        "sowing_date": "2025-06-15",    // NEW (optional)
        "expected_harvest_date": "...", // NEW (optional)
        "compare_crops": false          // NEW (optional)
    }
    """
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided. Send JSON with feature values.'
            }), 400
        
        # ============================================
        # Extract new crop & location fields
        # ============================================
        crop_type = data.get('crop_type', 'Unknown')
        state = data.get('state', 'Unknown')
        district = data.get('district', '')
        season = data.get('season', '')
        agro_climatic_zone = data.get('agro_climatic_zone', '')
        
        # Farm area fields
        farm_area = data.get('farm_area')
        farm_area_unit = data.get('farm_area_unit', 'Hectare')
        
        # Spatial fields
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        elevation = data.get('elevation')
        
        # Temporal fields
        sowing_date = data.get('sowing_date')
        expected_harvest_date = data.get('expected_harvest_date')
        
        # Multi-crop comparison flag
        compare_crops = data.get('compare_crops', False)
        
        # ============================================
        # Prepare input features for ML model
        # ============================================
        input_features = []
        missing_features = []
        
        for feature in feature_list:
            if feature in data:
                value = data[feature]
                # Handle None/null values
                if value is None:
                    input_features.append(np.nan)
                else:
                    input_features.append(float(value))
            else:
                # Use median imputation for missing features
                missing_features.append(feature)
                input_features.append(np.nan)
        
        # Convert to numpy array and reshape
        X_input = np.array(input_features).reshape(1, -1)
        
        # Impute missing values
        X_imputed = imputer.transform(X_input)
        
        # Scale features
        X_scaled = scaler.transform(X_imputed)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        # ============================================
        # Calculate total production if farm area provided
        # ============================================
        total_production = None
        if farm_area is not None and farm_area > 0:
            try:
                # Convert to hectares
                conversion_factor = AREA_CONVERSIONS.get(farm_area_unit, 1.0)
                effective_area_ha = float(farm_area) * conversion_factor
                
                # Calculate total production
                total_kg = prediction * effective_area_ha
                total_production = {
                    'farm_area_input': farm_area,
                    'farm_area_unit': farm_area_unit,
                    'effective_area_hectares': round(effective_area_ha, 4),
                    'total_kg': round(total_kg, 2),
                    'total_tons': round(total_kg / 1000, 3),
                    'total_quintals': round(total_kg / 100, 2),
                    'bags_50kg': round(total_kg / 50, 1),
                    'bags_100kg': round(total_kg / 100, 1)
                }
            except (ValueError, TypeError):
                total_production = None
        
        # ============================================
        # Calculate growing period if dates provided
        # ============================================
        growing_period = None
        if sowing_date and expected_harvest_date:
            try:
                sow = datetime.strptime(sowing_date, '%Y-%m-%d')
                harvest = datetime.strptime(expected_harvest_date, '%Y-%m-%d')
                growing_period = (harvest - sow).days
            except (ValueError, TypeError):
                growing_period = None
        
        # ============================================
        # Build response
        # ============================================
        response = {
            'status': 'success',
            'prediction': {
                'yield_kg_per_hectare': round(prediction, 2),
                'yield_tons_per_hectare': round(prediction / 1000, 3),
                'yield_quintals_per_hectare': round(prediction / 100, 2),
                'bags_per_hectare_50kg': round(prediction / 50, 1)
            },
            # Crop & Location info (echoed back for display)
            'crop_info': {
                'crop_type': crop_type,
                'state': state,
                'district': district if district else None,
                'season': season if season else None,
                'agro_climatic_zone': agro_climatic_zone if agro_climatic_zone else None
            },
            # Spatial info
            'spatial_info': {
                'latitude': latitude,
                'longitude': longitude,
                'elevation_m': elevation
            } if (latitude or longitude or elevation) else None,
            # Temporal info
            'temporal_info': {
                'sowing_date': sowing_date,
                'expected_harvest_date': expected_harvest_date,
                'growing_period_days': growing_period
            } if (sowing_date or expected_harvest_date) else None,
            # Total production for farm
            'total_production': total_production,
            'input_features': data,
            'features_imputed': missing_features if missing_features else None,
            # Model support info
            'model_supports_crop_feature': MODEL_SUPPORTS_CROP_FEATURE
        }
        
        # ============================================
        # Multi-Crop Comparison (if requested)
        # ============================================
        if compare_crops:
            comparison_results = []
            
            if MODEL_SUPPORTS_CROP_FEATURE:
                # TODO: Implement actual multi-crop prediction when model supports it
                # For each crop, encode and predict
                for crop in SUPPORTED_CROPS:
                    # This is a placeholder - actual implementation needs crop encoding
                    comparison_results.append({
                        'crop': crop,
                        'yield_kg_per_hectare': round(prediction, 2),  # Placeholder
                        'yield_tons_per_hectare': round(prediction / 1000, 3),
                        'note': 'Actual multi-crop prediction requires model retraining'
                    })
            else:
                # Return simulated variations for demo (since model doesn't use crop)
                # In production, retrain model with crop as feature
                base_yield = prediction
                crop_yield_factors = {
                    'Rice': 1.0,
                    'Wheat': 0.95,
                    'Maize': 1.1,
                    'Bajra': 0.75,
                    'Jowar': 0.8,
                    'Pulses': 0.5,
                    'Soybean': 0.6,
                    'Groundnut': 0.55,
                    'Cotton': 0.4,
                    'Sugarcane': 3.5
                }
                
                for crop in SUPPORTED_CROPS:
                    factor = crop_yield_factors.get(crop, 1.0)
                    crop_yield = base_yield * factor
                    comparison_results.append({
                        'crop': crop,
                        'yield_kg_per_hectare': round(crop_yield, 2),
                        'yield_tons_per_hectare': round(crop_yield / 1000, 3),
                        'bags_per_hectare_50kg': round(crop_yield / 50, 1)
                    })
            
            response['multi_crop_comparison'] = {
                'supported': MODEL_SUPPORTS_CROP_FEATURE,
                'note': 'Multi-crop comparison will be more accurate after model retraining with crop as a feature.' if not MODEL_SUPPORTS_CROP_FEATURE else 'Full multi-crop prediction supported.',
                'results': sorted(comparison_results, key=lambda x: x['yield_kg_per_hectare'], reverse=True)
            }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid input value: {str(e)}'
        }), 400
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple samples
    
    Request body (JSON):
    {
        "samples": [
            {"Rainfall_mm": 500, "Temperature_C": 25, ...},
            {"Rainfall_mm": 600, "Temperature_C": 28, ...}
        ]
    }
    """
    try:
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Provide "samples" array in request body'
            }), 400
        
        samples = data['samples']
        predictions = []
        
        for idx, sample in enumerate(samples):
            input_features = []
            for feature in feature_list:
                value = sample.get(feature, np.nan)
                input_features.append(float(value) if value is not None else np.nan)
            
            X_input = np.array(input_features).reshape(1, -1)
            X_imputed = imputer.transform(X_input)
            X_scaled = scaler.transform(X_imputed)
            pred = max(0, model.predict(X_scaled)[0])
            
            predictions.append({
                'sample_index': idx,
                'yield_kg_per_hectare': round(pred, 2)
            })
        
        return jsonify({
            'status': 'success',
            'total_samples': len(predictions),
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction error: {str(e)}'
        }), 500


# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


# ============================================
# Run Application
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🌾 CROP YIELD PREDICTION API")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /           - API info")
    print("  GET  /health     - Health check")
    print("  GET  /features   - Feature list")
    print("  GET  /model-info - Model info")
    print("  POST /predict    - Single prediction")
    print("  POST /predict-batch - Batch prediction")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
