# Product Category Prediction Scripts

This directory contains the complete XGBoost-based product prediction system with model persistence.

## Main Scripts

### `product_predictor.py` - Interactive Prediction System
**Complete product category prediction system with multiple modes:**

**Features:**
- Quick Test (30 seconds) - System verification
- Interactive Predictor (2-3 minutes) - Daily use with custom input
- Business Demo (2 minutes) - Pre-built sample demonstrations  
- Maximum Accuracy (5-10 minutes) - Full optimization for production
- Automatic model loading/saving
- Real-time predictions with confidence scores
- Business recommendations and insights

```bash
python scripts/product_predictor.py
```

### `train_model.py` - Model Training Pipeline
**Dedicated script for training and saving production models:**

**Features:**
- Optimized XGBoost training with grid search
- Model persistence for production deployment
- Performance evaluation and reporting
- Automatic model saving to `models/` directory

```bash
python scripts/train_model.py
```

### `batch_predict.py` - Batch Prediction Pipeline
**Production script for automated predictions:**

**Features:**
- Batch processing of CSV files
- Automatic model loading
- Confidence scores and probabilities
- Perfect for automated pipelines

```bash
# Batch predictions with custom files
python scripts/batch_predict.py input.csv output.csv

# Use default files
python scripts/batch_predict.py
```

## Model Persistence

**Model Lifecycle:**
1. Train model with `train_model.py` 
2. Model saved to `models/product_recommender.pkl`
3. All scripts auto-load existing models
4. Production-ready deployment

## Model Performance

**Current Status:**
- Accuracy: 58-67% (depending on optimization level)
- Categories: Sports, Electronics, Books, Groceries, Clothing
- Features: 8 total (social media platform, engagement, sentiment, purchase behavior)
- Dataset: 213 customer records

**Known Issues:**
- Low accuracy due to limited dataset size
- Class imbalance affects some category predictions
- Limited feature set may not capture full customer complexity

**Improvement Recommendations:**
- Collect more diverse customer behavioral data (target 1000+ samples)
- Add temporal/seasonal purchase patterns
- Include product-specific features
- Implement ensemble methods or deep learning approaches

**Features:**
- Tests all core functionality
- Minimal model training
- Validates imports and data loading
- Perfect for troubleshooting

**Usage:**
```bash
# Activate virtual environment  
source venv/bin/activate

# Run quick test
python scripts/test_fast.py
```

### 4. `demo_optimized.py` - 2-Minute Balanced Demo 🚀
Optimized demo with good speed/accuracy balance (2 minutes):

**Features:**
- Reasonable model training (balanced parameters)
- 5 realistic customer examples
- ~63% accuracy performance
- Business-friendly demonstration

**Usage:**
```bash
# Activate virtual environment  
source venv/bin/activate

# Run optimized demo
python scripts/demo_optimized.py
```

### 5. `demo_predict.py` - Original Demo
Original demonstration script (may be slower)

### 6. `run_authentication_system.py` - Authentication Pipeline
Original authentication system script (pre-existing).

## ⏱️ Script Performance & Timing

| Script | Duration | Purpose | Accuracy | Best For |
|--------|----------|---------|----------|----------|
| `test_fast.py` | 30 seconds | System verification | Basic | Troubleshooting, testing |
| `predict_interactive.py` | 2-3 minutes | **RECOMMENDED** Interactive | ~63% | **Daily use, demos** |
| `demo_optimized.py` | 2 minutes | Business demo | ~63% | Presentations, training |
| `predict_product.py` | 5-10 minutes | Production system | ~67% | Maximum accuracy needed |
| `demo_predict.py` | Variable | Legacy demo | ~63% | Compatibility |

**🎯 QUICK START RECOMMENDATION:** 
1. **First time?** Run `test_fast.py` to verify everything works
2. **Want to try it?** Use `predict_interactive.py` - perfect balance of speed and features
3. **Need max accuracy?** Use `predict_product.py` for production deployment

## Customer Data Input Format

When using the interactive predictor, you'll be prompted to enter:

| Field | Range/Options | Description |
|-------|---------------|-------------|
| Social Media Platform | Twitter, Facebook, Instagram, LinkedIn, TikTok | Customer's primary platform |
| Engagement Score | 50-100 | Social media engagement level |
| Purchase Interest Score | 1.0-5.0 | Interest in making purchases |
| Review Sentiment | Positive, Neutral, Negative | Customer sentiment analysis |
| Purchase Amount | $100-$500 | Typical purchase amount |
| Customer Rating | 1.0-5.0 | Average customer satisfaction rating |

## Model Performance

The XGBoost model achieves:
- **Accuracy**: ~63-67% (depending on optimization)
- **Categories**: Sports, Electronics, Books, Groceries, Clothing
- **Features**: 8 total (5 numerical, 3 categorical)
- **Training**: Hyperparameter optimized with cross-validation

## Sample Predictions

### Tech Enthusiast
- Platform: Twitter, Engagement: 92, Amount: $450, Sentiment: Positive
- **Predicted**: Sports (43.7% confidence)

### Fitness Enthusiast  
- Platform: Instagram, Engagement: 85, Amount: $280, Sentiment: Positive
- **Predicted**: Sports (42.7% confidence)

### Budget Shopper
- Platform: Facebook, Engagement: 65, Amount: $120, Sentiment: Neutral  
- **Predicted**: Sports (86.0% confidence)

## Business Recommendations

The predictor provides actionable business insights:

### High Confidence Predictions (>60%)
- Target specific category marketing campaigns
- Personalize product recommendations
- Focus on retention strategies

### Platform-Specific Strategies
- **Instagram**: Visual content and stories
- **TikTok**: Short-form video content
- **Facebook**: Community groups and detailed targeting
- **Twitter**: Real-time engagement and trending hashtags
- **LinkedIn**: Professional and B2B positioning

### Engagement-Based Actions
- **High Engagement (80+)**: Premium products, influencer partnerships
- **Low Engagement (<60)**: Engagement improvement strategies

## Technical Requirements

- Python 3.8+
- Virtual environment with packages from `requirements.txt`
- XGBoost, scikit-learn, pandas, numpy
- Training data: `data/processed/merged_customer_data.csv`

## Setup Instructions

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install xgboost  # if not in requirements.txt
   ```

3. **Run scripts:**
   ```bash
   # Quick demo
   python scripts/demo_predict.py
   
   # Full interactive version
   python scripts/predict_product.py
   ```

## Output Examples

### Prediction Results
```
============================================================
           PREDICTION RESULTS
============================================================

CUSTOMER PROFILE SUMMARY:
-------------------------
Platform:        Instagram
Engagement:      85/100
Interest Score:  4.2/5.0
Sentiment:       Positive
Purchase Amount: $280.00
Customer Rating: 4.5/5.0

PREDICTED PRODUCT CATEGORY:
------------------------------
RECOMMENDATION: Sports
Confidence:     High
Certainty:      42.7%

DETAILED CATEGORY PROBABILITIES:
-----------------------------------
1. Sports        ████████████████████ 42.7% ★
2. Clothing      ██████████████░░░░░░ 32.2%  
3. Books         ███░░░░░░░░░░░░░░░░░ 14.7%  
4. Electronics   ██░░░░░░░░░░░░░░░░░░  6.0%  
5. Groceries     █░░░░░░░░░░░░░░░░░░░  4.4%  

BUSINESS RECOMMENDATIONS:
----------------------------
1. Target Sports category marketing campaigns
2. Personalize product recommendations for Sports
3. High engagement - ideal for premium products
4. Consider influencer partnerships
5. Use visual content and stories for product promotion
```

## Notes

- The full interactive version takes longer due to hyperparameter optimization
- The demo version is faster but uses basic model parameters
- Models are trained fresh each time (consider implementing model persistence for production)
- All predictions include confidence scores and business recommendations
- Scripts include comprehensive error handling and user-friendly interfaces
