# 🌾 Crop Yield Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered agricultural yield forecasting system that predicts crop yields based on environmental conditions, soil properties, and farm management practices.

**Author:** Pushkarjay Ajay

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Live Demo](#live-demo)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting crop yield. It combines data from multiple agricultural datasets, performs intelligent feature engineering, trains ensemble models, and serves predictions via a REST API with an interactive dashboard.

### Key Highlights

- 📊 **75,000 samples** synthetic dataset with 22 crops across 20 Indian states
- 🤖 **91.95% accuracy** (R² Score) with Gradient Boosting
- 🌐 **REST API** for seamless integration
- 📱 **Responsive Dashboard** with real-time predictions
- 📈 **Multi-crop comparison** feature
- 🔢 **Total production calculator** with unit conversions

---

## 📁 Project Structure

```
Crop-Yield-Prediction/
├── Dataset/                      # Raw data files
│   ├── agricultural_yield_test.csv
│   ├── Data.csv
│   └── indiancrop_dataset.csv
│
├── Phase-1/                      # Initial exploration & documentation
│   ├── crop yield prediction.ipynb
│   ├── README.md
│   └── *.pdf, *.pptx            # Project documentation
│
├── Phase-2/                      # Production ML pipeline
│   ├── src/                     # Source code modules
│   │   ├── config.py            # Configuration & constants
│   │   ├── utils.py             # Logging utilities
│   │   ├── data_generation.py   # Synthetic dataset generator
│   │   ├── visualization.py     # Plot generation
│   │   ├── training.py          # ML training pipeline
│   │   └── outlier_analysis.py  # Outlier detection
│   ├── api/                      # Flask REST API
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── dashboard/                # Frontend UI
│   │   ├── index.html           # Main dashboard
│   │   ├── technical.html       # Technical documentation
│   │   ├── report.html          # Quick view presentation
│   │   ├── script.js
│   │   └── style.css
│   ├── model/                    # Trained model artifacts
│   ├── plots/                    # EDA visualizations (8 plots)
│   ├── logs/                     # Terminal output logs
│   ├── run_pipeline.py           # Master pipeline runner
│   ├── unified_dataset.csv       # 75K synthetic dataset
│   └── README.md
│
└── README.md                     # This file
```

---

## ✨ Features

### 🌡️ Prediction Inputs
- **Weather**: Rainfall, Temperature, Humidity, Sunshine hours, GDD, Pressure, Wind Speed
- **Soil**: Quality, pH, Organic Carbon, Soil Moisture, N-P-K nutrients
- **Management**: Fertilizer amount, Irrigation type, Seed variety, Pesticide usage
- **Location**: State, Crop type, Year, Crop price

### 📊 Dashboard Features
- Real-time yield predictions
- Backend status indicator
- Multi-crop yield comparison
- Total production calculator (Hectare/Acre/Bigha)
- Data field status toggle (shows which fields are ML-active)
- Interactive visualizations
- Technical documentation
- PPT-style quick view report

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.11, Flask 3.1, scikit-learn |
| **ML Model** | Gradient Boosting Regressor |
| **Data** | pandas, numpy, joblib |
| **Visualization** | matplotlib, seaborn |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **API** | REST, JSON, CORS-enabled |

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Clone Repository
```bash
git clone https://github.com/Pushkarjay/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction
```

### Setup Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Install Dependencies
```bash
cd Phase-2/api
pip install -r requirements.txt
```

---

## 💻 Usage

### Start Backend API
```bash
cd Phase-2/api
python app.py
```
The API will run at `http://localhost:5000`

### Start Frontend (Development)
```bash
cd Phase-2/dashboard
python -m http.server 8080
```
Open `http://localhost:8080` in your browser

### Make a Prediction (API)
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
    "Seed_Variety": 2
  }'
```

---

## 🌐 Live Demo

### GitHub Pages (Frontend Only)
🔗 **[Live Dashboard](https://pushkarjay.github.io/Crop-Yield-Prediction/Phase-2/dashboard/)**

> ⚠️ **Note:** The live demo frontend is hosted on GitHub Pages. The prediction functionality requires the backend API to be running locally. Follow these steps:

1. Clone the repository
2. Start the backend API (`python app.py`)
3. Use the live dashboard OR open local `index.html`
4. The dashboard will show "✓ Backend Online" when API is ready

### Hosting Frontend on GitHub Pages

To host the frontend yourself:

1. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Source: Deploy from branch
   - Branch: `main` → `/root` or `/docs`
   - Save

2. **Access URL:**
   ```
   https://<username>.github.io/Crop-Yield-Prediction/Phase-2/dashboard/
   ```

3. **Backend Requirement:**
   - The frontend alone cannot make predictions
   - Users must run `python Phase-2/api/app.py` locally
   - Dashboard shows backend status indicator

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check & model status |
| `/features` | GET | List of required features |
| `/model-info` | GET | Model metrics & info |
| `/predict` | POST | Single prediction |
| `/predict-batch` | POST | Batch predictions |

### Response Example
```json
{
  "status": "success",
  "prediction": {
    "yield_kg_per_hectare": 636.21,
    "yield_tons_per_hectare": 0.636,
    "bags_per_hectare_50kg": 12.7
  },
  "crop_info": {
    "crop_type": "Rice",
    "state": "Bihar"
  },
  "total_production": {
    "total_kg": 3181.05,
    "total_tons": 3.181
  }
}
```

---

## 📈 Model Performance

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Gradient Boosting** 🏆 | **0.9750** | **37.72** | **52.20** |
| Random Forest | 0.9680 | 42.15 | 58.90 |
| XGBoost | 0.9620 | 45.80 | 64.20 |
| SVR | 0.8950 | 78.30 | 106.50 |
| Linear Regression | 0.7820 | 112.40 | 153.80 |

### Feature Importance (Top 5)
1. Fertilizer Amount (28.5%)
2. Rainfall (18.2%)
3. Temperature (14.7%)
4. Soil Quality (12.3%)
5. Irrigation Schedule (9.8%)

---

## 📸 Screenshots

### Dashboard
![Dashboard](Phase-2/plots/01_yield_distribution.png)

### Feature Analysis
![Correlation](Phase-2/plots/02_correlation_matrix.png)

### Model Performance
![Feature Importance](Phase-2/plots/06_feature_importance.png)

---

## 🚀 Future Enhancements

- [ ] Add crop type as ML feature (requires model retraining)
- [ ] Integrate weather API for real-time data
- [ ] Add geo-spatial prediction maps
- [ ] Mobile application development
- [ ] Cloud deployment (AWS/GCP)
- [ ] Historical yield tracking
- [ ] Seasonal pattern analysis

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Pushkarjay Ajay**

- GitHub: [@Pushkarjay](https://github.com/Pushkarjay)

---

## 🙏 Acknowledgments

- Agricultural datasets from various open sources
- scikit-learn for ML algorithms
- Flask for API framework
- The open-source community

---

<p align="center">
  Made with ❤️ for Agriculture
</p>
