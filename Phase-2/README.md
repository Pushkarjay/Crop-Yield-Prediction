# 🌾 Phase-2: Production ML Pipeline

**Crop Yield Prediction System - End-to-End Implementation**

Author: **Pushkarjay Ajay**

---

## 📋 Overview

Phase-2 implements the complete production-ready machine learning pipeline including:
- Data preprocessing and feature engineering
- Model training and evaluation
- REST API for predictions
- Interactive web dashboard
- Technical documentation

---

## 📁 Directory Structure

```
Phase-2/
├── api/                          # Flask REST API
│   ├── app.py                    # Main API application
│   └── requirements.txt          # Python dependencies
│
├── dashboard/                    # Frontend Web UI
│   ├── index.html               # Main prediction dashboard
│   ├── technical.html           # Technical documentation page
│   ├── report.html              # Quick view / PPT-style report
│   ├── script.js                # Frontend JavaScript
│   └── style.css                # Styles
│
├── model/                        # Trained Model Artifacts
│   ├── model.pkl                # Gradient Boosting model
│   ├── scaler.pkl               # StandardScaler
│   ├── imputer.pkl              # SimpleImputer (median)
│   ├── label_encoders.pkl       # Categorical encoders
│   ├── feature_list.pkl         # Feature names
│   └── model_info.pkl           # Model metadata
│
├── plots/                        # EDA Visualizations
│   ├── 01_yield_distribution.png
│   ├── 02_correlation_matrix.png
│   ├── 03_crop_yield_comparison.png
│   ├── 04_state_yield_comparison.png
│   ├── 05_weather_soil_yield.png
│   └── 06_feature_importance.png
│
├── full_workflow.py              # Complete ML pipeline script
├── unified_dataset.csv           # Merged & preprocessed dataset
├── Phase-2-EndToEnd.ipynb        # Jupyter notebook version
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Start Backend API

```bash
python app.py
```

Output:
```
✅ Model and preprocessing objects loaded successfully!
🌾 CROP YIELD PREDICTION API
Starting server on http://localhost:5000
```

### 3. Start Frontend

```bash
cd dashboard
python -m http.server 8080
```

Open: `http://localhost:8080`

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/features` | GET | Feature list |
| `/model-info` | GET | Model metrics |
| `/predict` | POST | Single prediction |
| `/predict-batch` | POST | Batch predictions |

### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Rainfall_mm": 500,
    "Temperature_C": 25,
    "Humidity": 70,
    "Soil_Quality": 80,
    "Nitrogen": 50,
    "Phosphorus": 30,
    "Potassium": 40,
    "Fertilizer_Amount_kg_per_hectare": 150,
    "Sunshine_hours": 6,
    "Soil_Humidity": 45,
    "Irrigation_Schedule": 5,
    "Seed_Variety": 2,
    "crop_type": "Rice",
    "state": "Bihar",
    "compare_crops": true
  }'
```

---

## 📊 ML Features

### Active ML Features (12)

| Category | Features | Data Quality |
|----------|----------|--------------|
| **Weather** | Rainfall_mm, Temperature_C, Humidity, Sunshine_hours | 🟢 Good (3000+) |
| **Soil** | Soil_Quality, Nitrogen, Phosphorus, Potassium, Soil_Humidity | 🟡 Mixed |
| **Management** | Fertilizer_Amount, Irrigation_Schedule, Seed_Variety | 🟢 Good |

### Data Quality Summary

| Quality | Features | Sample Count |
|---------|----------|--------------|
| 🟢 Good | 8 features | 3,000 - 7,099 |
| 🟡 Partial | 4 features | 800 - 2,299 |
| ⚪ Contextual | crop, state, etc. | Display only |

---

## 🤖 Model Performance

**Best Model: Gradient Boosting Regressor**

| Metric | Value |
|--------|-------|
| R² Score | 0.9750 |
| MAE | 37.72 kg/ha |
| RMSE | 52.20 kg/ha |
| Training Samples | 5,687 |
| Test Samples | 1,422 |

---

## 🌐 Dashboard Features

1. **Prediction Form** - Enter crop parameters
2. **Backend Status** - Shows API availability
3. **Field Status Toggle** - Shows which fields are ML-active
4. **Multi-Crop Comparison** - Compare yields across crops
5. **Total Production Calculator** - Farm area calculations
6. **Technical Docs** - Detailed documentation
7. **Quick View** - PPT-style project report

---

## 📁 Files Description

### Python Scripts

| File | Description |
|------|-------------|
| `full_workflow.py` | Complete 13-step ML pipeline |
| `api/app.py` | Flask REST API server |

### Model Artifacts

| File | Description |
|------|-------------|
| `model.pkl` | Trained Gradient Boosting model |
| `scaler.pkl` | StandardScaler for normalization |
| `imputer.pkl` | SimpleImputer for missing values |
| `feature_list.pkl` | List of 12 feature names |

### Dashboard Pages

| File | Description |
|------|-------------|
| `index.html` | Main prediction interface |
| `technical.html` | Technical documentation |
| `report.html` | PPT-style quick view |

---

## 🔧 Retraining the Model

To retrain with updated data:

```bash
python full_workflow.py
```

This will:
1. Load and merge datasets
2. Preprocess features
3. Train multiple models
4. Save best model to `model/`
5. Generate plots in `plots/`

---

## 📝 Notes

- Backend must be running for predictions to work
- Dashboard shows "Backend Offline" if API is not available
- Model imputes missing values using median strategy
- Contextual fields (crop, state) are for display only

---

## 👨‍💻 Author

**Pushkarjay Ajay**

---

<p align="center">
  🌾 Crop Yield Prediction System - Phase 2
</p>
