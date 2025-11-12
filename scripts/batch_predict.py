#!/usr/bin/env python3
"""
BATCH PREDICTION SCRIPT

Load saved XGBoost model and make predictions on batch data.
Perfect for production pipelines and automated processing.

Usage: python scripts/batch_predict.py [input_file] [output_file]

Author: AI Assistant
Date: November 2025
"""

import sys
import os
import pandas as pd
import argparse

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

try:
    from product_recommender import ProductRecommender
except ImportError as e:
    print(f"Error importing ProductRecommender: {e}")
    sys.exit(1)

def batch_predict(input_file, output_file=None):
    """
    Make batch predictions using saved model
    
    Parameters:
    - input_file: CSV file with customer data
    - output_file: Output CSV file (optional)
    
    Returns:
    - DataFrame with predictions
    """
    
    print("BATCH PREDICTION PIPELINE")
    print("="*30)
    
    # Load the trained model
    print("1. Loading trained model...")
    recommender = ProductRecommender(random_state=42)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'product_recommender.pkl')
    
    if not recommender.load_model(model_path):
        print("ERROR: Could not load trained model!")
        print("Please run: python scripts/train_model.py")
        return None
    
    # Load input data
    print("2. Loading batch input data...")
    try:
        df = pd.read_csv(input_file)
        print(f"   Loaded {len(df)} customer records")
    except Exception as e:
        print(f"ERROR: Could not load input file: {e}")
        return None
    
    # Make predictions
    print("3. Making predictions...")
    predictions = []
    probabilities_list = []
    
    for idx, row in df.iterrows():
        try:
            # Convert row to customer data format
            customer_data = row.to_dict()
            
            # Make prediction
            prediction, probabilities = recommender.predict_product_category(customer_data)
            
            predictions.append(prediction if prediction else 'Unknown')
            probabilities_list.append(probabilities if probabilities else {})
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} records...")
                
        except Exception as e:
            print(f"   Warning: Failed to predict for record {idx}: {e}")
            predictions.append('Error')
            probabilities_list.append({})
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['predicted_category'] = predictions
    
    # Add probability columns
    if probabilities_list and probabilities_list[0]:
        categories = list(probabilities_list[0].keys())
        for category in categories:
            results_df[f'prob_{category.lower()}'] = [
                probs.get(category, 0.0) for probs in probabilities_list
            ]
    
    # Add confidence score
    results_df['confidence'] = [
        max(probs.values()) if probs else 0.0 
        for probs in probabilities_list
    ]
    
    print("4. Processing results...")
    successful_predictions = sum(1 for p in predictions if p not in ['Unknown', 'Error'])
    print(f"   Successful predictions: {successful_predictions}/{len(df)} ({successful_predictions/len(df)*100:.1f}%)")
    
    # Save results
    if output_file:
        print("5. Saving results...")
        try:
            results_df.to_csv(output_file, index=False)
            print(f"   Results saved to: {output_file}")
        except Exception as e:
            print(f"   Warning: Could not save to {output_file}: {e}")
    
    print("\nBATCH PREDICTION COMPLETED!")
    return results_df

def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(description='Batch prediction using saved XGBoost model')
    parser.add_argument('input_file', nargs='?', 
                       default='../data/processed/merged_customer_data.csv',
                       help='Input CSV file with customer data')
    parser.add_argument('output_file', nargs='?',
                       default='../data/processed/predictions.csv', 
                       help='Output CSV file for predictions')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), args.input_file))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), args.output_file))
    
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return 1
    
    # Run batch prediction
    try:
        results = batch_predict(input_file, output_file)
        if results is not None:
            print("\nSample predictions:")
            print(results[['predicted_category', 'confidence']].head())
            return 0
        else:
            return 1
    except Exception as e:
        print(f"ERROR: Batch prediction failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
