import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# XGBoost import - REQUIRED (no fallback)
import xgboost as xgb
print("XGBoost imported successfully")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import our data processing functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from merge_data import prepare_features

class ProductRecommender:
    """
    XGBoost-based product category predictor (XGBoost ONLY - no fallbacks)
    
    Predicts which product category (Sports, Electronics, Books, Groceries, Clothing)
    a customer is most likely to purchase based on their:
    - Social media engagement patterns
    - Purchase history and behavior
    - Sentiment analysis results
    
    Uses XGBoost (eXtreme Gradient Boosting) exclusively.
    Model saved as pickle to models/xgboost_product_recommender.pkl
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Product Recommender
        
        Parameters:
        - random_state: Ensures reproducible results across runs
        """
        self.random_state = random_state
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.target_encoder = None
        self.feature_cols = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.model_trained = False
        
    def load_data(self, data_path='../../data/processed/merged_customer_data.csv', merged_df=None):
        """
        Load the merged customer dataset from file or use provided DataFrame
        
        Parameters:
        - data_path: Path to the merged customer data CSV (optional if merged_df provided)
        - merged_df: Pre-loaded merged DataFrame (optional)
        
        Returns:
        - df: Loaded or provided pandas DataFrame
        """
        # Use provided DataFrame if available
        if merged_df is not None:
            df = merged_df.copy()
            print("Using provided merged dataset!")
            print(f"Dataset shape: {df.shape}")
            print(f"Target variable: product_category")
            print(f"Product categories: {sorted(df['product_category'].unique())}")
            print(f"Category distribution:")
            print(df['product_category'].value_counts().sort_index())
            return df
            
        # Otherwise load from file
        try:
            # Convert relative path to absolute path
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
            df = pd.read_csv(abs_path)
            
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {df.shape}")
            print(f"Target variable: product_category")
            print(f"Product categories: {sorted(df['product_category'].unique())}")
            print(f"Category distribution:")
            print(df['product_category'].value_counts().sort_index())
            
            return df
            
        except FileNotFoundError:
            print(f"Error: Could not find dataset at {abs_path}")
            print("Please ensure the merged dataset exists in data/processed/")
            return None
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def prepare_data(self, df, target_column='product_category', test_size=0.2):
        """
        Prepare data for machine learning training
        
        Parameters:
        - df: Input DataFrame with merged customer data
        - target_column: Column to predict (product_category)
        - test_size: Fraction of data for testing (0.2 = 20%)
        
        Returns:
        - X_train, X_test, y_train, y_test: Train/test splits
        """
        print("Preparing features for machine learning...")
        
        # Use the prepare_features function from merge_data.py
        X, y, label_encoders, scaler, target_encoder, feature_cols, numerical_cols, categorical_cols = prepare_features(
            df, target_column
        )
        
        # Store preprocessing objects for later use
        self.label_encoders = label_encoders
        self.scaler = scaler
        self.target_encoder = target_encoder
        self.feature_cols = feature_cols
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Split data into training and testing sets
        # stratify=y ensures balanced representation of each category in both sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Data preparation completed!")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples") 
        print(f"Features: {X_train.shape[1]} ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, use_grid_search=True):
        """
        Train the XGBoost model with hyperparameter optimization
        
        Parameters:
        - X_train: Training features
        - y_train: Training target variable
        - use_grid_search: Whether to perform hyperparameter tuning (recommended)
        """
        # Store training data for overfitting analysis
        self._X_train_stored = X_train
        self._y_train_stored = y_train
        
        print("Training XGBoost model...")
        self._train_xgboost_model(X_train, y_train, use_grid_search)
        
        self.model_trained = True
        print("XGBoost model training completed!")
    
    def _train_xgboost_model(self, X_train, y_train, use_grid_search):
        """Train XGBoost model with hyperparameter optimization"""
        if use_grid_search:
            print("Performing XGBoost hyperparameter optimization...")
            
            # Define hyperparameter grid for XGBoost optimization
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.1, 0.15, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 2]
            }
            
            # Create XGBoost classifier
            xgb_model = xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Perform Grid Search with 3-fold cross-validation
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='f1_weighted', 
                n_jobs=-1, verbose=1
            )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Use the best model found
            self.model = grid_search.best_estimator_
            
            print(f"Best XGBoost hyperparameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
            
        else:
            print("Training XGBoost with baseline configuration...")
            # Baseline XGBoost configuration
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
    

    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance using multiple metrics
        
        Parameters:
        - X_test: Testing features  
        - y_test: Testing target variable
        
        Returns:
        - results: Dictionary with performance metrics
        """
        if not self.model_trained:
            print("Error: Model must be trained before evaluation!")
            return None
            
        print("XGBOOST MODEL EVALUATION")
        print("=" * 40)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate core metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        loss = log_loss(y_test, y_pred_proba)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'log_loss': loss,
            'precision': precision,
            'recall': recall,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Display results with business assessment
        print(f"CORE PERFORMANCE METRICS:")
        print("-" * 30)
        print(f"Overall Performance:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Log Loss:  {loss:.4f}")
        
        # Business assessment
        if accuracy >= 0.8:
            performance_level = "EXCELLENT"
            business_impact = "Ready for production deployment"
        elif accuracy >= 0.6:
            performance_level = "GOOD"
            business_impact = "Suitable with monitoring"
        elif accuracy >= 0.4:
            performance_level = "MODERATE"
            business_impact = "Needs improvement before deployment"
        else:
            performance_level = "POOR"
            business_impact = "Requires significant improvement"
            
        print(f"\nBusiness Assessment:")
        print(f"  Performance Level: {performance_level}")
        print(f"  Business Impact:   {business_impact}")
        
        # Detailed classification report
        print(f"\nDETAILED CLASSIFICATION REPORT:")
        print("-" * 35)
        if self.target_encoder:
            target_names = self.target_encoder.classes_
        else:
            target_names = None
        
        report = classification_report(y_test, y_pred, target_names=target_names)
        print(report)
        
        # Overfitting analysis
        print(f"\nOVERFITTING ANALYSIS:")
        print("-" * 25)
        train_accuracy = self.model.score(self._X_train_stored, self._y_train_stored) if hasattr(self, '_X_train_stored') else None
        if train_accuracy:
            overfitting_gap = train_accuracy - accuracy
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:     {accuracy:.4f}")
            print(f"Overfitting Gap:   {overfitting_gap:.4f}")
            
            if overfitting_gap > 0.3:
                overfitting_status = "POOR - Significant overfitting"
                action = "Increase regularization or collect more data"
            elif overfitting_gap > 0.15:
                overfitting_status = "MODERATE - Some overfitting"
                action = "Consider regularization tuning"
            else:
                overfitting_status = "GOOD - Minimal overfitting"
                action = "Model generalizes well"
                
            print(f"Status: {overfitting_status}")
            print(f"Action: {action}")
        else:
            print("Training accuracy not available for overfitting analysis")
        
        print(f"\nModel evaluation completed!")
        print(f"XGBoost model shows {performance_level.lower()} performance")
        
        return results
    
    def plot_feature_importance(self, top_n=10):
        """
        Plot and display the most important features for prediction
        
        Parameters:
        - top_n: Number of top features to display (default: 10)
        """
        if not self.model_trained:
            print("Error: Model must be trained before plotting feature importance!")
            return
        
        print(f"XGBOOST FEATURE IMPORTANCE ANALYSIS")
        print("=" * 42)
            
        # Get feature importances
        importances = self.model.feature_importances_
        feature_names = self.feature_cols
        
        # Create DataFrame for easier handling
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Add importance percentages
        feature_importance_df['importance_percentage'] = (
            feature_importance_df['importance'] / feature_importance_df['importance'].sum() * 100
        )
        
        # Display top features with business insights
        print(f"TOP {top_n} MOST IMPORTANT FEATURES:")
        print("-" * 35)
        for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f} ({row['importance_percentage']:.1f}%)")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize the plot
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances - XGBoost Product Predictor', 
                 fontsize=14, fontweight='bold')
        
        # Set y-axis labels
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        # Invert y-axis to show highest importance at top
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save plot
        plot_path = '../../reports/xgboost_feature_importance.png'
        abs_plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), plot_path))
        os.makedirs(os.path.dirname(abs_plot_path), exist_ok=True)
        plt.savefig(abs_plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {abs_plot_path}")
        plt.show()
        
        # Business insights
        print(f"\nBUSINESS INSIGHTS FROM FEATURE IMPORTANCE:")
        print("-" * 45)
        
        top_feature = feature_importance_df.iloc[0]
        second_feature = feature_importance_df.iloc[1] if len(feature_importance_df) > 1 else None
        top_5_importance = feature_importance_df.head(5)['importance'].sum()
        
        print("Key Business Drivers:")
        print(f"  Primary driver: {top_feature['feature']} ({top_feature['importance_percentage']:.1f}% of decisions)")
        if second_feature is not None:
            print(f"  Secondary driver: {second_feature['feature']} ({second_feature['importance_percentage']:.1f}% of decisions)")
        print(f"  Top 5 features drive {top_5_importance/importances.sum()*100:.1f}% of all predictions")
        
        print(f"\nFeature importance analysis completed!")
        
        return feature_importance_df
    
    def plot_confusion_matrix(self, results):
        """
        Plot confusion matrix to visualize prediction accuracy by category
        
        Parameters:
        - results: Results dictionary from evaluate_model()
        """
        if results is None:
            print("Error: No evaluation results provided!")
            return
            
        print(f"CONFUSION MATRIX ANALYSIS:")
        print("-" * 30)
            
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Get class names
        if self.target_encoder:
            class_names = self.target_encoder.classes_
        else:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_test)))]
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Number of Predictions'})
        
        plt.title('XGBoost Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Category', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Category', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = '../../reports/xgboost_confusion_matrix.png'
        abs_plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), plot_path))
        plt.savefig(abs_plot_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {abs_plot_path}")
        plt.show()
        
        # Per-class performance analysis
        print(f"\nPer-class Performance:")
        for i, class_name in enumerate(class_names):
            class_predictions = np.sum(cm[i, :])
            if class_predictions > 0:
                class_accuracy = cm[i, i] / class_predictions
                class_precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
                print(f"  {class_name}:")
                print(f"    Accuracy:  {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
                print(f"    Precision: {class_precision:.3f}")
        
        print(f"Confusion matrix analysis completed!")
        
        return cm
    
    def save_model(self, model_path='../../models/xgboost_product_recommender.pkl'):
        """
        Save the trained XGBoost model and all preprocessing components using pickle
        
        Parameters:
        - model_path: Path where to save the model file (default: models/xgboost_product_recommender.pkl)
        
        Returns:
        - bool: True if saved successfully, False otherwise
        """
        if not self.model_trained:
            print("Error: No trained model to save. Train the model first.")
            return False
            
        try:
            # Convert relative path to absolute path
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            # Package all necessary components
            model_package = {
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'target_encoder': self.target_encoder,
                'feature_cols': self.feature_cols,
                'numerical_cols': self.numerical_cols,
                'categorical_cols': self.categorical_cols,
                'model_trained': self.model_trained,
                'random_state': self.random_state,
                'model_type': 'XGBoost'
            }
            
            # Save using pickle.dump() as requested
            with open(abs_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f"XGBoost model saved successfully to: {abs_path}")
            print(f"Model type: XGBoost")
            print(f"Features: {len(self.feature_cols)} total")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path='../../models/xgboost_product_recommender.pkl'):
        """
        Load a previously trained XGBoost model and all preprocessing components using pickle
        
        Parameters:
        - model_path: Path to the saved model file (default: models/xgboost_product_recommender.pkl)
        
        Returns:
        - bool: True if loaded successfully, False otherwise
        """
        try:
            # Convert relative path to absolute path
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
            
            if not os.path.exists(abs_path):
                print(f"Model file not found at: {abs_path}")
                return False
            
            # Load the model package using pickle.load()
            with open(abs_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Restore all components
            self.model = model_package['model']
            self.label_encoders = model_package['label_encoders']
            self.scaler = model_package['scaler']
            self.target_encoder = model_package['target_encoder']
            self.feature_cols = model_package['feature_cols']
            self.numerical_cols = model_package['numerical_cols']
            self.categorical_cols = model_package['categorical_cols']
            self.model_trained = model_package['model_trained']
            self.random_state = model_package.get('random_state', 42)
            
            model_type = model_package.get('model_type', 'XGBoost')
            
            print(f"XGBoost model loaded successfully from: {abs_path}")
            print(f"Model type: {model_type}")
            print(f"Features: {len(self.feature_cols)} total")
            print(f"Categories: {list(self.target_encoder.classes_)}")
            print("XGBoost model ready for predictions!")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def model_exists(self, model_path='../../models/xgboost_product_recommender.pkl'):
        """
        Check if a saved XGBoost model exists at the specified path
        
        Parameters:
        - model_path: Path to check for saved model (default: models/xgboost_product_recommender.pkl)
        
        Returns:
        - bool: True if model file exists, False otherwise
        """
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
        return os.path.exists(abs_path)

    def predict_product_category(self, customer_data):
        """
        Predict product category for new customer data
        
        Parameters:
        - customer_data: Dictionary or DataFrame with customer features
        
        Returns:
        - prediction: Predicted product category
        - probability: Confidence scores for each category
        """
        if not self.model_trained:
            print("Error: Model must be trained before making predictions!")
            return None, None
            
        # Convert to DataFrame if dictionary
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.feature_cols) - set(customer_df.columns)
        if missing_features:
            print(f"Error: Missing required features: {missing_features}")
            return None, None
        
        # Apply same preprocessing as training data
        customer_processed = customer_df[self.feature_cols].copy()
        
        # Encode categorical variables
        for col in self.categorical_cols:
            if col in customer_processed.columns:
                customer_processed[col] = self.label_encoders[col].transform(
                    customer_processed[col].astype(str)
                )
        
        # Scale numerical variables
        if len(self.numerical_cols) > 0:
            customer_processed[self.numerical_cols] = self.scaler.transform(
                customer_processed[self.numerical_cols]
            )
        
        # Make prediction
        prediction_encoded = self.model.predict(customer_processed)[0]
        probabilities = self.model.predict_proba(customer_processed)[0]
        
        # Decode prediction if needed
        if self.target_encoder:
            prediction = self.target_encoder.inverse_transform([prediction_encoded])[0]
            class_names = self.target_encoder.classes_
        else:
            prediction = prediction_encoded
            class_names = [f'Class {i}' for i in range(len(probabilities))]
        
        # Create probability dictionary
        prob_dict = dict(zip(class_names, probabilities))
        
        return prediction, prob_dict


def train_product_recommendation_model():
    """
    Main function to train and evaluate the XGBoost product recommendation model
    
    This function orchestrates the entire machine learning pipeline:
    1. Data loading and preparation
    2. XGBoost model training with hyperparameter optimization
    3. Model evaluation and visualization
    4. Model saving for production use
    """
    print("Starting XGBoost Product Recommendation Model Training")
    print("=" * 60)
    
    # Initialize the recommender
    recommender = ProductRecommender(random_state=42)
    
    # Step 1: Load the merged dataset
    df = recommender.load_data()
    if df is None:
        return None
    
    # Step 2: Prepare data for machine learning
    X_train, X_test, y_train, y_test = recommender.prepare_data(df)
    
    # Step 3: Train the XGBoost model
    recommender.train_model(X_train, y_train, use_grid_search=True)
    
    # Step 4: Evaluate model performance
    results = recommender.evaluate_model(X_test, y_test)
    
    # Step 5: Generate insights and visualizations
    feature_importance = recommender.plot_feature_importance(top_n=10)
    recommender.plot_confusion_matrix(results)
    
    # Step 6: Save the trained model
    model_saved = recommender.save_model()
    
    # Step 7: Demo prediction
    print("\nDemo Prediction:")
    print("=" * 30)
    
    # Example customer data for prediction
    demo_customer = {
        'social_media_platform': 'Instagram',
        'engagement_score': 85,
        'purchase_interest_score': 4.2,
        'review_sentiment': 'Positive',
        'transaction_id': 9999,
        'purchase_amount': 350,
        'purchase_date': '2024-01-15',
        'customer_rating': 4.5
    }
    
    predicted_category, probabilities = recommender.predict_product_category(demo_customer)
    
    if predicted_category:
        print(f"Customer Profile: {demo_customer['social_media_platform']} user")
        print(f"Engagement Score: {demo_customer['engagement_score']}")
        print(f"Average Purchase: ${demo_customer['purchase_amount']}")
        print(f"Sentiment: {demo_customer['review_sentiment']}")
        print()
        print(f"Predicted Category: {predicted_category}")
        print(f"Top Predictions:")
        
        # Sort probabilities and show top 3
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for i, (category, prob) in enumerate(sorted_probs[:3], 1):
            print(f"   {i}. {category}: {prob:.2%}")
    
    print("\nXGBoost model training completed successfully!")
    print("=" * 60)
    
    return recommender, results


if __name__ == "__main__":
    # Train the XGBoost model when script is run directly
    model, evaluation_results = train_product_recommendation_model()
