#!/usr/bin/env python3
"""
PRODUCT CATEGORY PREDICTOR - ALL-IN-ONE SCRIPT

Complete product category prediction system with multiple modes:
- Quick Test (30 seconds)
- Interactive Predictions (2-3 minutes)
- Business Demo (built-in samples)
- Full Optimization (5-10 minutes for max accuracy)

Usage: python scripts/product_predictor.py

Author: AI Assistant
Date: November 2025
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_processing'))

try:
    from product_recommender import ProductRecommender
    print("ProductRecommender imported successfully!")
except ImportError as e:
    print(f"Error importing ProductRecommender: {e}")
    print("Please ensure the src/models directory is accessible.")
    sys.exit(1)

class UnifiedProductPredictor:
    """All-in-one product category prediction system"""
    
    def __init__(self):
        self.recommender = None
        self.model_loaded = False
        self.available_platforms = ['Twitter', 'Facebook', 'Instagram', 'LinkedIn', 'TikTok']
        self.available_sentiments = ['Positive', 'Neutral', 'Negative']
        self.available_categories = ['Sports', 'Electronics', 'Books', 'Groceries', 'Clothing']
        self.current_mode = None
        
    def display_welcome(self):
        """Display welcome screen"""
        print("=" * 65)
        print("PRODUCT CATEGORY PREDICTOR - ALL-IN-ONE SYSTEM")
        print("=" * 65)
        print()
        print("CHOOSE YOUR MODE:")
        print("=" * 20)
        print("1. QUICK TEST (30 seconds)")
        print("   -> Verify system works, basic functionality test")
        print()
        print("2. INTERACTIVE PREDICTOR (2-3 minutes)")  
        print("   -> Full interactive system with optimized training")
        print("   -> RECOMMENDED for daily use and demonstrations")
        print()
        print("3. BUSINESS DEMO (2 minutes)")
        print("   -> Pre-built demo with sample customers")
        print("   -> Perfect for presentations and training")
        print()
        print("4. MAXIMUM ACCURACY (5-10 minutes)")
        print("   -> Full hyperparameter optimization")
        print("   -> Best possible accuracy for production use")
        print()
        print("5. EXIT")
        print()
        
    def quick_test_mode(self):
        """30-second system verification"""
        print("QUICK TEST MODE - SYSTEM VERIFICATION")
        print("=" * 45)
        print("Testing all core components in 30 seconds...")
        print()
        
        try:
            # Test 1: Basic functionality
            print("Testing basic functionality...")
            recommender = ProductRecommender(random_state=42)
            print("ProductRecommender initialization: SUCCESS")
            
            # Test 2: Data loading
            print("Testing data loading...")
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'merged_customer_data.csv')
            df = recommender.load_data(data_path=data_path)
            
            if df is not None:
                print(f"Data loading: SUCCESS ({df.shape[0]} rows, {df.shape[1]} columns)")
            else:
                print("Data loading: FAILED")
                return False
            
            # Test 3: Fast model training
            print("Testing fast model training...")
            small_df = df.sample(n=min(100, len(df)), random_state=42)
            X_train, X_test, y_train, y_test = recommender.prepare_data(small_df, test_size=0.3)
            
            # Ultra-fast model
            import xgboost as xgb
            recommender.model = xgb.XGBClassifier(
                n_estimators=10, max_depth=3, learning_rate=0.3,
                random_state=42, n_jobs=-1
            )
            recommender.model.fit(X_train, y_train)
            recommender.model_trained = True
            print("Fast model training: SUCCESS")
            
            # Test 4: Prediction
            print("Testing predictions...")
            test_customer = {
                'social_media_platform': 'Instagram',
                'engagement_score': 85, 'purchase_interest_score': 4.2,
                'review_sentiment': 'Positive', 'purchase_amount': 280,
                'customer_rating': 4.5, 'customer_id': 9999,
                'transaction_id': 9999, 'purchase_date': '2024-01-15'
            }
            
            prediction, probabilities = recommender.predict_product_category(test_customer)
            
            if prediction and probabilities:
                print(f"Prediction test: SUCCESS")
                print(f"   Sample prediction: {prediction} ({max(probabilities.values()):.1%})")
            else:
                print("Prediction test: FAILED")
                return False
            
            print()
            print("ALL TESTS PASSED!")
            print("System is fully operational and ready for use!")
            print()
            return True
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False
    
    def train_optimized_model(self, mode='balanced'):
        """Train model with different optimization levels"""
        print("MODEL TRAINING")
        print("=" * 20)
        
        if mode == 'balanced':
            print("Training optimized model (2-3 minutes)...")
        elif mode == 'maximum':
            print("Training with maximum accuracy optimization (5-10 minutes)...")
            print("Using full hyperparameter optimization...")
        
        # Initialize and load data
        self.recommender = ProductRecommender(random_state=42)
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'merged_customer_data.csv')
        df = self.recommender.load_data(data_path=data_path)
        
        if df is None:
            print("Error: Could not load training data!")
            return False
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.recommender.prepare_data(df)
        
        if mode == 'maximum':
            # Use full hyperparameter optimization
            print("Running comprehensive hyperparameter optimization...")
            self.recommender.train_model(X_train, y_train, use_grid_search=True)
        else:
            # Use optimized balanced parameters
            print("Using optimized balanced parameters...")
            import xgboost as xgb
            self.recommender.model = xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1,
                random_state=42, objective='multi:softprob',
                eval_metric='mlogloss', n_jobs=-1
            )
            self.recommender.model.fit(X_train, y_train)
            self.recommender.model_trained = True
        
        # Evaluate model
        results = self.recommender.evaluate_model(X_test, y_test)
        
        self.model_loaded = True
        print(f"Model trained successfully!")
        print(f"   Accuracy: {results['accuracy']:.1%}")
        print(f"   F1-Score: {results['f1_score']:.3f}")
        print(f"   Performance: {'Excellent' if results['accuracy'] > 0.7 else 'Good' if results['accuracy'] > 0.6 else 'Fair'}")
        print()
        return True
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("INTERACTIVE PREDICTOR MODE")
        print("=" * 32)
        print()
        
        if not self.model_loaded:
            if not self.train_optimized_model('balanced'):
                return
        
        while True:
            print("INTERACTIVE MENU")
            print("=" * 18)
            print("1. Predict for new customer")
            print("2. View sample predictions")
            print("3. Model information")
            print("4. Back to main menu")
            print()
            
            try:
                choice = input("Enter your choice (1-4): ").strip()
                
                if choice == '1':
                    self.get_custom_prediction()
                elif choice == '2':
                    self.show_sample_predictions()
                elif choice == '3':
                    self.show_model_info()
                elif choice == '4':
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            
            except KeyboardInterrupt:
                print("\n\nReturning to main menu...")
                break
    
    def get_custom_prediction(self):
        """Get user input and make prediction"""
        print("\nCUSTOMER DATA INPUT")
        print("=" * 25)
        print("Please enter customer information:")
        print()
        
        customer_data = {}
        
        # Platform
        print("1. Social Media Platform:")
        for i, platform in enumerate(self.available_platforms, 1):
            print(f"   {i}. {platform}")
        
        while True:
            try:
                choice = int(input("   Enter choice (1-5): "))
                if 1 <= choice <= 5:
                    customer_data['social_media_platform'] = self.available_platforms[choice-1]
                    break
                else:
                    print("   Please enter a number between 1 and 5.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Engagement Score
        print(f"\n2. Engagement Score (50-100):")
        while True:
            try:
                score = float(input("   Enter engagement score: "))
                if 50 <= score <= 100:
                    customer_data['engagement_score'] = score
                    break
                else:
                    print("   Please enter a value between 50 and 100.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Purchase Interest
        print(f"\n3. Purchase Interest Score (1.0-5.0):")
        while True:
            try:
                score = float(input("   Enter interest score: "))
                if 1.0 <= score <= 5.0:
                    customer_data['purchase_interest_score'] = score
                    break
                else:
                    print("   Please enter a value between 1.0 and 5.0.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Sentiment
        print("\n4. Review Sentiment:")
        for i, sentiment in enumerate(self.available_sentiments, 1):
            print(f"   {i}. {sentiment}")
        
        while True:
            try:
                choice = int(input("   Enter choice (1-3): "))
                if 1 <= choice <= 3:
                    customer_data['review_sentiment'] = self.available_sentiments[choice-1]
                    break
                else:
                    print("   Please enter a number between 1 and 3.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Purchase Amount
        print(f"\n5. Purchase Amount ($100-$500):")
        while True:
            try:
                amount = float(input("   Enter amount ($): "))
                if 100 <= amount <= 500:
                    customer_data['purchase_amount'] = amount
                    break
                else:
                    print("   Please enter a value between $100 and $500.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Customer Rating
        print(f"\n6. Customer Rating (1.0-5.0):")
        while True:
            try:
                rating = float(input("   Enter rating: "))
                if 1.0 <= rating <= 5.0:
                    customer_data['customer_rating'] = rating
                    break
                else:
                    print("   Please enter a value between 1.0 and 5.0.")
            except ValueError:
                print("   Please enter a valid number.")
        
        # Add automatic fields
        customer_data['customer_id'] = 9999
        customer_data['transaction_id'] = 9999
        customer_data['purchase_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Make prediction
        print("\nProcessing prediction...")
        prediction, probabilities = self.recommender.predict_product_category(customer_data)
        
        if prediction and probabilities:
            self.display_prediction_results(customer_data, prediction, probabilities)
        else:
            print("Could not generate prediction. Please try again.")
    
    def display_prediction_results(self, customer_data, prediction, probabilities):
        """Display formatted prediction results"""
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        # Customer Summary
        print("\nCUSTOMER PROFILE:")
        print("-" * 25)
        print(f"Platform:        {customer_data['social_media_platform']}")
        print(f"Engagement:      {customer_data['engagement_score']}/100")
        print(f"Interest Score:  {customer_data['purchase_interest_score']:.1f}/5.0")
        print(f"Sentiment:       {customer_data['review_sentiment']}")
        print(f"Purchase Amount: ${customer_data['purchase_amount']:.2f}")
        print(f"Customer Rating: {customer_data['customer_rating']:.1f}/5.0")
        
        # Prediction
        max_prob = max(probabilities.values())
        confidence_level = "Very High" if max_prob >= 0.8 else "High" if max_prob >= 0.6 else "Medium" if max_prob >= 0.4 else "Low"
        
        print(f"\nRECOMMENDATION:")
        print("-" * 20)
        print(f"Product Category: {prediction}")
        print(f"Confidence Level: {confidence_level}")
        print(f"Certainty:        {max_prob:.1%}")
        
        # Detailed probabilities with visual bars
        print(f"\nALL CATEGORIES:")
        print("-" * 20)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (category, prob) in enumerate(sorted_probs, 1):
            bar_length = int(prob * 25)
            bar = "█" * bar_length + "░" * (25 - bar_length)
            star = " *" if category == prediction else ""
            print(f"{i}. {category:<12} {bar} {prob:6.1%}{star}")
        
        # Business recommendations
        print(f"\nBUSINESS RECOMMENDATIONS:")
        print("-" * 30)
        self.generate_business_insights(customer_data, prediction, max_prob)
        
        print("\n" + "=" * 60)
    
    def generate_business_insights(self, customer_data, prediction, confidence):
        """Generate actionable business recommendations"""
        insights = []
        
        # Confidence-based
        if confidence >= 0.6:
            insights.append(f"High confidence - Target {prediction} category campaigns")
            insights.append(f"Personalize {prediction} product recommendations")
        else:
            insights.append("Medium confidence - Consider multi-category approach")
        
        # Platform-specific
        platform = customer_data['social_media_platform']
        platform_strategies = {
            'Instagram': 'Use visual content and stories for engagement',
            'TikTok': 'Create short-form video content',
            'Facebook': 'Leverage community groups and detailed targeting',
            'Twitter': 'Use real-time engagement and trending topics',
            'LinkedIn': 'Focus on professional and B2B positioning'
        }
        insights.append(platform_strategies.get(platform, "Optimize platform-specific content"))
        
        # Engagement-based
        engagement = customer_data['engagement_score']
        if engagement >= 80:
            insights.append("High engagement - Consider premium products & influencer partnerships")
        elif engagement <= 60:
            insights.append("Focus on engagement improvement strategies")
        
        # Display insights
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
    
    def business_demo_mode(self):
        """Pre-built business demonstration"""
        print("BUSINESS DEMO MODE")
        print("=" * 22)
        print("Running demo with pre-defined sample customers...")
        print()
        
        if not self.model_loaded:
            if not self.train_optimized_model('balanced'):
                return
        
        demo_customers = [
            {
                'name': 'Tech Enthusiast Sarah',
                'description': 'High engagement Twitter user interested in electronics',
                'social_media_platform': 'Twitter',
                'engagement_score': 92, 'purchase_interest_score': 4.5,
                'review_sentiment': 'Positive', 'purchase_amount': 450,
                'customer_rating': 4.8, 'expected': 'Electronics'
            },
            {
                'name': 'Fitness Coach Mike',
                'description': 'Instagram influencer promoting active lifestyle',
                'social_media_platform': 'Instagram',
                'engagement_score': 88, 'purchase_interest_score': 4.2,
                'review_sentiment': 'Positive', 'purchase_amount': 280,
                'customer_rating': 4.6, 'expected': 'Sports'
            },
            {
                'name': 'Book Lover Emma',
                'description': 'Avid reader sharing reviews on Facebook',
                'social_media_platform': 'Facebook',
                'engagement_score': 72, 'purchase_interest_score': 3.8,
                'review_sentiment': 'Positive', 'purchase_amount': 85,
                'customer_rating': 4.3, 'expected': 'Books'
            },
            {
                'name': 'Budget Shopper Tom',
                'description': 'Price-conscious family shopper',
                'social_media_platform': 'Facebook',
                'engagement_score': 58, 'purchase_interest_score': 3.1,
                'review_sentiment': 'Neutral', 'purchase_amount': 120,
                'customer_rating': 3.7, 'expected': 'Groceries'
            }
        ]
        
        correct_predictions = 0
        
        for i, customer in enumerate(demo_customers, 1):
            print(f"CUSTOMER {i}: {customer['name']}")
            print(f"Profile: {customer['description']}")
            print("-" * 50)
            
            # Prepare data
            customer_data = {
                'social_media_platform': customer['social_media_platform'],
                'engagement_score': customer['engagement_score'],
                'purchase_interest_score': customer['purchase_interest_score'],
                'review_sentiment': customer['review_sentiment'],
                'purchase_amount': customer['purchase_amount'],
                'customer_rating': customer['customer_rating'],
                'customer_id': 9990 + i, 'transaction_id': 9990 + i,
                'purchase_date': '2024-01-15'
            }
            
            # Make prediction
            prediction, probabilities = self.recommender.predict_product_category(customer_data)
            
            if prediction and probabilities:
                is_correct = prediction == customer['expected']
                if is_correct:
                    correct_predictions += 1
                
                print(f"Expected:   {customer['expected']}")
                print(f"Predicted:  {prediction} {'[CORRECT]' if is_correct else '[INCORRECT]'}")
                print(f"Confidence: {max(probabilities.values()):.1%}")
                print(f"Platform:   {customer['social_media_platform']}")
                print()
                
                # Top 3 predictions
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                print("Top 3 Categories:")
                for j, (category, prob) in enumerate(sorted_probs[:3], 1):
                    star = " *" if category == prediction else ""
                    print(f"  {j}. {category:<12} {prob:6.1%}{star}")
            else:
                print("Prediction failed!")
            
            print()
        
        # Demo summary
        print("=" * 50)
        print("DEMO SUMMARY")
        print("=" * 50)
        print(f"Demo Results: {correct_predictions}/{len(demo_customers)} correct ({correct_predictions/len(demo_customers):.0%})")
        print(f"Model Status: {'Excellent' if correct_predictions/len(demo_customers) >= 0.75 else 'Good' if correct_predictions/len(demo_customers) >= 0.5 else 'Needs improvement'}")
        print()
    
    def show_sample_predictions(self):
        """Show quick sample predictions"""
        print("\nSAMPLE PREDICTIONS")
        print("=" * 22)
        
        samples = [
            {
                'name': 'Tech Professional', 'social_media_platform': 'LinkedIn',
                'engagement_score': 88, 'purchase_interest_score': 4.3,
                'review_sentiment': 'Positive', 'purchase_amount': 400,
                'customer_rating': 4.7
            },
            {
                'name': 'Creative Influencer', 'social_media_platform': 'Instagram',
                'engagement_score': 92, 'purchase_interest_score': 4.6,
                'review_sentiment': 'Positive', 'purchase_amount': 250,
                'customer_rating': 4.8
            }
        ]
        
        for i, customer in enumerate(samples, 1):
            customer_data = customer.copy()
            customer_data.update({
                'customer_id': 8000 + i, 'transaction_id': 8000 + i,
                'purchase_date': '2024-01-15'
            })
            
            prediction, probabilities = self.recommender.predict_product_category(customer_data)
            
            print(f"\nSample {i}: {customer['name']}")
            print("-" * 25)
            if prediction:
                print(f"Platform:   {customer['social_media_platform']}")
                print(f"Engagement: {customer['engagement_score']}")
                print(f"Predicted:  {prediction} ({max(probabilities.values()):.1%})")
            else:
                print("Prediction failed")
    
    def show_model_info(self):
        """Display model information"""
        print(f"\nMODEL INFORMATION")
        print("=" * 22)
        print(f"Model Type:       XGBoost Classifier")
        print(f"Categories:       {', '.join(self.available_categories)}")
        print(f"Features:         8 total (5 numerical, 3 categorical)")
        print(f"Training Mode:    {self.current_mode or 'Optimized'}")
        print(f"Status:           {'Ready' if self.model_loaded else 'Not loaded'}")
        print(f"Platforms:        {', '.join(self.available_platforms)}")
    
    def main_menu(self):
        """Main application loop"""
        while True:
            self.display_welcome()
            
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    self.current_mode = 'Quick Test'
                    if self.quick_test_mode():
                        input("\nPress Enter to continue...")
                
                elif choice == '2':
                    self.current_mode = 'Interactive'
                    self.interactive_mode()
                
                elif choice == '3':
                    self.current_mode = 'Business Demo'
                    self.business_demo_mode()
                    input("\nPress Enter to continue...")
                
                elif choice == '4':
                    self.current_mode = 'Maximum Accuracy'
                    if self.train_optimized_model('maximum'):
                        print("Model trained with maximum accuracy!")
                        print("You can now use the interactive mode with best performance.")
                        self.interactive_mode()
                
                elif choice == '5':
                    print("\nThank you for using Product Category Predictor!")
                    print("System successfully demonstrated all capabilities!")
                    break
                
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                    input("Press Enter to continue...")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")

def main():
    """Application entry point"""
    try:
        predictor = UnifiedProductPredictor()
        predictor.main_menu()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
