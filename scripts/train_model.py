#!/usr/bin/env python3
"""
MODEL TRAINING SCRIPT

This script trains and saves the XGBoost product recommendation model
for use in production pipelines and deployment.

Usage: python scripts/train_model.py

Author: AI Assistant  
Date: November 2025
"""

import sys
import os
import pandas as pd

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

try:
    from product_recommender import ProductRecommender
    print("ProductRecommender imported successfully!")
except ImportError as e:
    print(f"Error importing ProductRecommender: {e}")
    print("Please ensure the src/models directory is accessible.")
    sys.exit(1)

def train_and_save_model():
    """Train XGBoost model and save for production use"""
    
    print("="*50)
    print("    XGBOOST MODEL TRAINING PIPELINE")
    print("="*50)
    print()
    
    # Initialize recommender
    print("1. Initializing ProductRecommender...")
    recommender = ProductRecommender(random_state=42)
    
    # Load data
    print("2. Loading training data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'merged_customer_data.csv')
    df = recommender.load_data(data_path=data_path)
    
    if df is None:
        print("ERROR: Could not load training data!")
        print("Please ensure merged_customer_data.csv exists in data/processed/")
        return False
    
    # Prepare data
    print("3. Preparing features...")
    X_train, X_test, y_train, y_test = recommender.prepare_data(df)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Train model with grid search optimization
    print("4. Training optimized XGBoost model...")
    print("   This may take 5-10 minutes for best results...")
    recommender.train_model(X_train, y_train, use_grid_search=True)
    
    # Evaluate model
    print("5. Evaluating model performance...")
    results = recommender.evaluate_model(X_test, y_test)
    
    print("\nMODEL PERFORMANCE:")
    print("-" * 25)
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    
    # Save model
    print("6. Saving trained model...")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'product_recommender.pkl')
    
    if recommender.save_model(model_path):
        print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*40)
        print(f"Model saved to: {model_path}")
        print("Model is ready for:")
        print("  - Production deployment")
        print("  - Pipeline integration") 
        print("  - Interactive predictions")
        print("  - Batch processing")
        print()
        print("Usage examples:")
        print("  python scripts/product_predictor.py  # Interactive system")
        print("  python scripts/batch_predict.py     # Batch predictions")
        return True
    else:
        print("ERROR: Failed to save model!")
        return False

def main():
    """Main execution function"""
    try:
        success = train_and_save_model()
        if success:
            print("Training pipeline completed successfully!")
            return 0
        else:
            print("Training pipeline failed!")
            return 1
    except Exception as e:
        print(f"Fatal error in training pipeline: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
