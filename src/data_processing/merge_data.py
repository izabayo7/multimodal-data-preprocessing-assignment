import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_datasets():
    """
    Load customer social profiles and transactions datasets from CSV files
    
    Purpose:
    - Reads two separate CSV files containing customer data
    - Returns both datasets as pandas DataFrames for further processing
    
    Returns:
    - social_df: DataFrame with customer social media information
    - transactions_df: DataFrame with customer transaction history
    - (None, None): If files cannot be found or loaded
    """
    try:
        # Load customer social profiles dataset
        # Contains: customer_id_new, social_media_platform, engagement_score, 
        #          purchase_interest_score, review_sentiment
        social_df = pd.read_csv('../../data/raw/customer_social_profiles.csv')
        
        # Load customer transactions dataset  
        # Contains: customer_id_legacy, transaction_id, purchase_amount,
        #          purchase_date, product_category, customer_rating
        transactions_df = pd.read_csv('../../data/raw/customer_transactions.csv')
        
        # Display success message and dataset dimensions
        print("Datasets loaded successfully!")
        print(f"Social profiles shape: {social_df.shape}")
        print(f"Transactions shape: {transactions_df.shape}")
        return social_df, transactions_df
        
    except FileNotFoundError as e:
        # Handle case where CSV files don't exist
        print(f"Error loading datasets: {e}")
        print("Please ensure the CSV files are in the data/raw/ directory")
        return None, None

def clean_data(df, dataset_name):
    """
    Clean the dataset by handling missing values, duplicates, and data types
    
    Purpose:
    - Removes duplicate rows to avoid data redundancy
    - Fills missing values using appropriate statistical measures
    - Ensures data quality for machine learning models
    
    Parameters:
    - df: pandas DataFrame to be cleaned
    - dataset_name: string name for logging purposes
    
    Cleaning Strategy:
    - Numerical columns: Fill missing values with median (robust to outliers)
    - Categorical columns: Fill missing values with mode (most frequent value)
    - If no mode exists: Fill with 'Unknown'
    
    Returns:
    - df_clean: Cleaned pandas DataFrame with no missing values or duplicates
    """
    print(f"\nCleaning {dataset_name} dataset...")
    print(f"Original shape: {df.shape}")
    
    # STEP 1: Remove duplicate rows
    # Duplicates can skew analysis and model training
    df_clean = df.drop_duplicates()
    print(f"After removing duplicates: {df_clean.shape}")
    
    # STEP 2: Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # STEP 3: Fill missing values based on data type
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            # For numerical columns: Use median (less sensitive to outliers than mean)
            # Example: engagement_score, purchase_amount, customer_rating
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            # For categorical columns: Use mode (most frequent value)
            # Example: social_media_platform, review_sentiment, product_category
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col].fillna(mode_val[0], inplace=True)
            else:
                # If no mode exists (all values unique), use 'Unknown'
                df_clean[col].fillna('Unknown', inplace=True)
    
    # STEP 4: Verify cleaning was successful
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    
    return df_clean

def standardize_customer_ids(social_df, transactions_df):
    """
    Standardize customer IDs between datasets to enable proper merging
    
    Problem:
    - Social profiles use: 'customer_id_new' with format 'A178', 'A190', etc.
    - Transactions use: 'customer_id_legacy' with format '178', '190', etc.
    - Different formats prevent direct merging
    
    Solution:
    - Extract numeric part from social profiles IDs (A178 -> 178)
    - Rename legacy IDs to standard 'customer_id' column
    - Create common 'customer_id' column in both datasets
    
    Parameters:
    - social_df: DataFrame with customer_id_new column
    - transactions_df: DataFrame with customer_id_legacy column
    
    Returns:
    - social_df: DataFrame with standardized customer_id column
    - transactions_df: DataFrame with standardized customer_id column
    """
    # STEP 1: Extract numeric part from alphanumeric IDs
    # Uses regex to find digits (\d+) and convert to integer
    # Example: 'A178' -> '178' -> 178
    social_df['customer_id'] = social_df['customer_id_new'].str.extract('(\d+)').astype(int)
    
    # STEP 2: Rename legacy ID column to standard name
    # Direct copy since legacy IDs are already numeric
    transactions_df['customer_id'] = transactions_df['customer_id_legacy']
    
    # STEP 3: Remove original ID columns to avoid confusion
    social_df = social_df.drop('customer_id_new', axis=1)
    transactions_df = transactions_df.drop('customer_id_legacy', axis=1)
    
    # STEP 4: Display standardization results
    print(f"Social profiles unique customers: {social_df['customer_id'].nunique()}")
    print(f"Transactions unique customers: {transactions_df['customer_id'].nunique()}")
    
    return social_df, transactions_df

def merge_customer_data():
    """
    Merge customer social profiles and transactions data
    
    Purpose: Combines two datasets on customer_id to create unified customer view
    Returns: merged_df with both social media and transaction information
    """
    # Load both datasets from CSV files
    social_df, transactions_df = load_datasets()
    
    if social_df is None or transactions_df is None:
        return None
    
    # Display initial dataset info for verification
    print("\nInitial dataset overview:")
    print("Social Profiles columns:", list(social_df.columns))
    print("Transactions columns:", list(transactions_df.columns))
    
    # Make customer IDs compatible for merging
    social_df, transactions_df = standardize_customer_ids(social_df, transactions_df)
    
    # Clean both datasets (remove duplicates, handle missing values)
    social_clean = clean_data(social_df, "Social Profiles")
    transactions_clean = clean_data(transactions_df, "Transactions")
    
    # Display merge statistics
    print("\nMerging datasets...")
    print(f"Social profiles: {social_clean.shape}")
    print(f"Transactions: {transactions_clean.shape}")
    
    # Analyze customer overlap between datasets
    social_customers = set(social_clean['customer_id'])
    transaction_customers = set(transactions_clean['customer_id'])
    common_customers = social_customers.intersection(transaction_customers)
    
    print(f"Customers only in social profiles: {len(social_customers - transaction_customers)}")
    print(f"Customers only in transactions: {len(transaction_customers - social_customers)}")
    print(f"Common customers: {len(common_customers)}")
    
    # Perform inner join to keep only customers in both datasets
    merged_df = pd.merge(social_clean, transactions_clean, on='customer_id', how='inner')
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Verify merge results
    print(f"Unique customers in merged dataset: {merged_df['customer_id'].nunique()}")
    print(f"Total records: {len(merged_df)}")
    
    # Save merged dataset to processed directory
    output_dir = '../../data/processed/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'merged_customer_data.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\nMerged dataset saved to: {output_path}")
    
    return merged_df

def prepare_features(merged_df, target_column):
    """
    Prepare features for machine learning models
    
    Purpose: Converts raw data into ML-ready format (all numerical)
    Process: Separates features/target, encodes categories, scales numbers
    Returns: X, y, encoders, scaler, target_encoder, column_lists
    """
    # Separate target variable (what we want to predict)
    y = merged_df[target_column]
    
    # Select feature columns (exclude customer_id and target)
    feature_cols = [col for col in merged_df.columns if col not in ['customer_id', target_column]]
    X = merged_df[feature_cols].copy()
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    # Convert categorical text to numbers (e.g., 'Twitter' -> 4)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Standardize numerical features (mean=0, std=1)
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Encode target variable if categorical (e.g., 'Sports' -> 4)
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
    else:
        y_encoded = y
        target_encoder = None
    
    return X, y_encoded, label_encoders, scaler, target_encoder, feature_cols, list(numerical_cols), list(categorical_cols)

if __name__ == "__main__":
    # Run the merge process
    merged_data = merge_customer_data()
    if merged_data is not None:
        print("\n" + "="*50)
        print("DATA MERGING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final dataset shape: {merged_data.shape}")
        print("\nColumns in merged dataset:")
        for i, col in enumerate(merged_data.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nFirst 5 rows of merged dataset:")
        print(merged_data.head().to_string())
        
        print(f"\nDataset info:")
        print(merged_data.info())
    else:
        print("Data merging failed!")
