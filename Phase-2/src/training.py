"""
Model Training Module - ML Pipeline for Crop Yield Prediction
Author: Pushkarjay Ajay
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, PLOTS_DIR, BASE_DIR,
    TEST_SIZE, CV_FOLDS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES
)
from src.utils import (
    print_header, print_section, print_success, print_warning, 
    print_error, print_info
)
from src.visualization import (
    create_feature_importance_plot, create_prediction_analysis_plot
)


class CropYieldModel:
    """
    Machine Learning model for crop yield prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        
    def preprocess_data(self, df, fit=True):
        """
        Preprocess the data for model training/prediction
        
        Args:
            df: pandas DataFrame with raw data
            fit: Whether to fit encoders/scaler (True for training, False for inference)
            
        Returns:
            X: Feature matrix
            y: Target variable (if Yield column exists)
        """
        print_section("Data Preprocessing")
        
        df_processed = df.copy()
        
        # Handle categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df_processed.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df_processed[col] = df_processed[col].apply(
                            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                        )
        
        # Select features
        available_features = [f for f in MODEL_FEATURES if f in df_processed.columns]
        self.feature_names = available_features
        
        X = df_processed[available_features].values
        
        # Scale numeric features
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        print_success(f"Preprocessed {X.shape[0]:,} samples with {X.shape[1]} features")
        
        # Return X and y if target exists
        if 'Yield_kg_per_hectare' in df_processed.columns:
            y = df_processed['Yield_kg_per_hectare'].values
            return X, y
        return X
    
    def train(self, X, y):
        """
        Train the Gradient Boosting model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            dict: Training metrics
        """
        print_section("Model Training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        print_info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Initialize Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        print_info("Training Gradient Boosting Regressor...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train))
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
            }
        }
        
        # Cross-validation
        print_info(f"Running {CV_FOLDS}-fold cross-validation...")
        cv_scores = cross_val_score(
            GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42),
            X_train, y_train, cv=CV_FOLDS, scoring='r2'
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Print results
        print_success("Training Complete!")
        print(f"\n{'='*50}")
        print(f"  TRAINING METRICS")
        print(f"{'='*50}")
        print(f"  R² Score:  {metrics['train']['r2']:.4f}")
        print(f"  MAE:       {metrics['train']['mae']:,.0f} kg/ha")
        print(f"  RMSE:      {metrics['train']['rmse']:,.0f} kg/ha")
        
        print(f"\n{'='*50}")
        print(f"  TEST METRICS")
        print(f"{'='*50}")
        print(f"  R² Score:  {metrics['test']['r2']:.4f}")
        print(f"  MAE:       {metrics['test']['mae']:,.0f} kg/ha")
        print(f"  RMSE:      {metrics['test']['rmse']:,.0f} kg/ha")
        
        print(f"\n{'='*50}")
        print(f"  CROSS-VALIDATION")
        print(f"{'='*50}")
        print(f"  CV R² Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        return metrics, y_test, y_pred_test
    
    def compare_models(self, X, y):
        """
        Compare multiple ML models
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            DataFrame: Model comparison results
        """
        print_section("Model Comparison")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=10, random_state=42),
        }
        
        results = []
        
        for name, model in models.items():
            print_info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results.append({
                'Model': name,
                'R² Score': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('R² Score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"  MODEL COMPARISON RESULTS")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def save_model(self, filepath=None):
        """Save model and preprocessors to disk"""
        if not self.is_trained:
            print_error("Model not trained yet!")
            return
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, 'crop_yield_model.pkl')
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print_success(f"Model saved to: {filepath}")
    
    def load_model(self, filepath=None):
        """Load model and preprocessors from disk"""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, 'crop_yield_model.pkl')
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print_success(f"Model loaded from: {filepath}")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            print_error("Model not trained yet!")
            return None
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        return None


def run_training_pipeline(data_path=None):
    """
    Run the complete training pipeline
    
    Args:
        data_path: Path to the dataset CSV
        
    Returns:
        CropYieldModel: Trained model
    """
    print_header("CROP YIELD MODEL TRAINING PIPELINE")
    
    # Load data
    if data_path is None:
        data_path = os.path.join(BASE_DIR, 'unified_dataset.csv')
    
    print_info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print_success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Initialize model
    model = CropYieldModel()
    
    # Preprocess
    X, y = model.preprocess_data(df, fit=True)
    
    # Compare models
    model.compare_models(X, y)
    
    # Train best model
    metrics, y_test, y_pred = model.train(X, y)
    
    # Create visualizations
    print_section("Creating Visualizations")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance is not None:
        print(f"\n{feature_importance.head(10).to_string(index=False)}")
        create_feature_importance_plot(
            model.model.feature_importances_,
            model.feature_names
        )
    
    # Prediction analysis
    create_prediction_analysis_plot(
        y_test, y_pred,
        'Gradient Boosting',
        metrics['test']
    )
    
    # Save model
    model.save_model()
    
    print_header("TRAINING COMPLETE")
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║           FINAL MODEL PERFORMANCE                    ║
    ╠══════════════════════════════════════════════════════╣
    ║  Model:     Gradient Boosting Regressor              ║
    ║  R² Score:  {metrics['test']['r2']:.4f}                                ║
    ║  MAE:       {metrics['test']['mae']:,.0f} kg/ha                           ║
    ║  RMSE:      {metrics['test']['rmse']:,.0f} kg/ha                          ║
    ║  CV R²:     {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})                    ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    return model


if __name__ == "__main__":
    from src.utils import log_output, get_log_filename
    
    log_file = get_log_filename("model_training")
    
    with log_output(log_file):
        model = run_training_pipeline()
    
    print(f"\n📄 Log saved to: {log_file}")
