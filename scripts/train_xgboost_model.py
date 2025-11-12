#!/usr/bin/env python3
"""
XGBoost Model Training Script

This script trains an XGBoost model for product category prediction
and saves it to models/xgboost_product_recommender.pkl using pickle.

Usage: python scripts/train_xgboost_model.py
"""

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

from product_recommender import ProductRecommender

def main():
    """Train and save XGBoost model"""
    print("XGBOOST MODEL TRAINING")
    print("=" * 30)
    
    # Initialize recommender
    recommender = ProductRecommender(random_state=42)
    
    # Load data
    print("Loading data...")
    df = recommender.load_data()
    if df is None:
        print("Error: Could not load data!")
        return 1
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = recommender.prepare_data(df)
    
    # Train XGBoost model
    print("Training XGBoost model...")
    recommender.train_model(X_train, y_train, use_grid_search=True)
    
    # Evaluate model
    print("Evaluating model...")
    results = recommender.evaluate_model(X_test, y_test)
    
    # Save model using pickle
    print("Saving XGBoost model...")
    success = recommender.save_model()
    
    if success:
        print("SUCCESS: XGBoost model trained and saved!")
        print("Model saved to: models/xgboost_product_recommender.pkl")
        print(f"Final accuracy: {results['accuracy']:.1%}")
    else:
        print("ERROR: Failed to save model!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
