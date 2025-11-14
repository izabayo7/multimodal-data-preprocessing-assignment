# Multimodal Authentication & Product Recommendation System

**Formative 2 - Group 6**  
**African Leadership University**

---

## 📋 Quick Navigation

| Component                   | Location                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------ |
| **Data Analysis**           | [`notebooks/01_data_merge_eda.ipynb`](notebooks/01_data_merge_eda.ipynb)             |
| **Merged Dataset**          | [`data/processed/merged_customer_data.csv`](data/processed/merged_customer_data.csv) |
| **Images**                  | [`data/images/`](data/images/) (3 expressions × 4 members)                           |
| **Image Features**          | [`data/features/image_features.csv`](data/features/image_features.csv)               |
| **Audio Recordings**        | [`data/audio/`](data/audio/) (2 phrases × 4 members)                                 |
| **Audio Features**          | [`data/processed/audio_features.csv`](data/processed/audio_features.csv)             |
| **Model Training**          | [`notebooks/`](notebooks/) - face, voice, product models                             |
| **Authentication Pipeline** | [`src/pipeline/authentication_pipeline.py`](src/pipeline/authentication_pipeline.py) |
| **CLI Application**         | [`scripts/run_authentication_system.py`](scripts/run_authentication_system.py)       |
| **Demo Video**              | [Watch on YouTube](https://youtu.be/OCy7BjODDCc)                                     |

---

## 👥 Team Members - Group 6

| Name                  | Role   | Contributions                                                          |
| --------------------- | ------ | ---------------------------------------------------------------------- |
| **Alice Mukarwema**   | Task 1 | Data merge, EDA, XGBoost product recommendation model                  |
| **Yassin Hagenimana** | Task 2 | Image collection, augmentation, VGG16+RF face recognition              |
| **Hirwa Brian**       | Task 3 | Audio collection, feature extraction, Random Forest voice verification |
| **Cedric Izabayo**    | Task 4 | Pipeline integration, CLI app, system testing & demo                   |

---

## 📦 Project Overview

A **multimodal authentication system** that combines:

- 🔐 **Face Recognition** (VGG16 + Random Forest)
- 🎤 **Voice Verification** (Random Forest with MFCCs)
- 🛍️ **Product Recommendation** (XGBoost Classifier)

**Authentication Flow:**

```
Face Image → Face Recognition → Product Generation → Voice Audio → Voice Verification → Display Product
     ↓              ↓                    ↓                  ↓              ↓                   ↓
  Capture      Identify User      Predict Product       Capture      Confirm User        Show Result
```

**Security:** Product recommendations are only revealed after successful biometric verification (face + voice).

---

## 📁 Repository Structure

```
multimodal-data-preprocessing-assignment/
│
├── 📊 data/
│   ├── raw/                              # Original datasets
│   │   ├── customer_social_profiles.csv  # Customer demographic data
│   │   └── customer_transactions.csv     # Purchase history
│   │
│   ├── processed/                        # ✅ DELIVERABLE: Cleaned & merged data
│   │   ├── merged_customer_data.csv      # ← Main merged dataset
│   │   ├── audio_features.csv            # ← Extracted audio features
│   │   └── audio_plots/                  # Waveforms & spectrograms
│   │
│   ├── features/                         # ✅ DELIVERABLE: Feature files
│   │   └── image_features.csv            # ← VGG16 embeddings + metadata
│   │
│   ├── images/                           # ✅ DELIVERABLE: Team member photos
│   │   ├── Alice/       (neutral, smile, surprised)
│   │   ├── Armstrong/   (neutral, smile, surprised)
│   │   ├── cedric/      (neutral, smile, surprised)
│   │   ├── yassin/      (neutral, smile, surprised)
│   │   ├── test/        (Additional test images)
│   │   └── unauthorized/ (Negative examples)
│   │
│   └── audio/                            # ✅ DELIVERABLE: Voice recordings
│       ├── alice/       (2 phrases)
│       ├── Armstrong/   (2 phrases)
│       ├── cedric/      (2 phrases)
│       └── yassin/      (2 phrases)
│
├── 📓 notebooks/                         # ✅ DELIVERABLE: Jupyter notebooks
│   ├── 01_data_merge_eda.ipynb           # ← EDA, cleaning, merge (Alice)
│   ├── face_recognition_model.ipynb      # ← Face model training (Yassin)
│   └── voice_recognition_model.ipynb     # ← Voice model training (Brian)
│
├── 🤖 models/                            # ✅ DELIVERABLE: Trained models
│   ├── xgboost_product_recommender.pkl   # ← Product recommendation (Alice)
│   ├── face_recognition_model.pkl        # ← Face recognition (Yassin)
│   ├── face_recognition_scaler.pkl       # ← Face feature scaler
│   ├── face_recognition_metadata.json    # ← Label encoder mapping
│   ├── voice_model.joblib                # ← Voice classification (Brian)
│   ├── voice_scaler.joblib               # ← Voice feature scaler
│   └── voice_encoder.joblib              # ← Voice label encoder
│
├── 🐍 src/                               # Source code modules
│   ├── pipeline/
│   │   └── authentication_pipeline.py    # ✅ DELIVERABLE: Main integration
│   ├── data_processing/
│   │   ├── merge_data.py                 # Data merging logic
│   │   ├── image_processing.py           # Image augmentation & features
│   │   └── audio_processing.py           # Audio augmentation & features
│   ├── models/
│   │   └── product_recommender.py        # Product model class
│   └── utils/
│
├── 🚀 scripts/                           # ✅ DELIVERABLE: CLI application
│   ├── run_authentication_system.py      # ← Main command-line app
│   ├── predict_face.py                   # Face prediction module
│   ├── batch_predict.py                  # Batch processing
│   ├── batch_predict_face.py             # Batch face recognition
│   └── train_xgboost_model.py            # Product model training
│
├── 📄 requirements.txt                   # Python dependencies
└── 📖 README.md                          # This file
```

---

## 🎯 Project Features

### Data Analysis & Processing

**EDA Notebook:** [`notebooks/01_data_merge_eda.ipynb`](notebooks/01_data_merge_eda.ipynb)

- Summary statistics for customer datasets
- Variable types and data validation
- **Visualizations:**
  - Distribution plots (age, purchase amounts)
  - Correlation heatmaps
  - Outlier detection with box plots
- Comprehensive insights and interpretations

**Data Pipeline:** [`data/processed/merged_customer_data.csv`](data/processed/merged_customer_data.csv)

- Null values handled with appropriate strategies
- Duplicates removed
- Data types corrected (dates, categorical variables)
- Merged on customer_id with validation checks

### Image Data

**Dataset:** [`data/images/`](data/images/)

- **12 images total:** 4 team members × 3 expressions each
  - Expressions: neutral, smile, surprised
  - Format: High-resolution JPEG
  - Consistent naming convention
  - Additional test and unauthorized samples

**Processing:** [`notebooks/face_recognition_model.ipynb`](notebooks/face_recognition_model.ipynb)

**Features:** [`data/features/image_features.csv`](data/features/image_features.csv)

- **Augmentations:**
  - Rotation (±15°)
  - Horizontal flip
  - Brightness adjustment
  - Zoom variations
- **VGG16 embeddings:** 512-dimensional feature vectors
- Features exported with labels in CSV format

### Audio Data

**Dataset:** [`data/audio/`](data/audio/)

- **8 recordings total:** 4 members × 2 phrases each
- Format: Clean WAV files (16kHz sampling rate)
- Phrases for voice verification

**Processing:** [`notebooks/voice_recognition_model.ipynb`](notebooks/voice_recognition_model.ipynb)

**Features:** [`data/processed/audio_features.csv`](data/processed/audio_features.csv)

- **Visualizations:**
  - Waveform plots for temporal analysis
  - Spectrogram plots for frequency analysis
  - Saved in [`data/processed/audio_plots/`](data/processed/audio_plots/)
- **Augmentations:**
  - Time stretching (0.8x, 1.2x speed)
  - Pitch shifting (±2 semitones)
  - Background noise injection
- **Features extracted:**
  - 13 MFCCs (Mel-Frequency Cepstral Coefficients)
  - Spectral roll-off
  - RMS energy

### Machine Learning Models

**Models Directory:** [`models/`](models/)

| Model                      | Algorithm             | Files                             | Performance        |
| -------------------------- | --------------------- | --------------------------------- | ------------------ |
| **Face Recognition**       | VGG16 + Random Forest | `face_recognition_model.pkl`      | 96-100% confidence |
| **Voice Verification**     | Random Forest         | `voice_model.joblib`              | 68-91% confidence  |
| **Product Recommendation** | XGBoost Classifier    | `xgboost_product_recommender.pkl` | 90%+ confidence    |

**Integration:** All models work together in the authentication pipeline

**Multimodal Flow:**

```python
# Flow in authentication_pipeline.py
1. Face Recognition → Identify user (fail → Access Denied)
2. Product Prediction → Generate recommendation (hidden)
3. Voice Verification → Confirm identity (fail → Access Denied)
4. Display Product → Reveal recommendation only if both pass
```

**Security:** Product recommendations hidden until complete biometric verification

### System Implementation

**CLI Application:** [`scripts/run_authentication_system.py`](scripts/run_authentication_system.py)

**Available Modes:**

```bash
# Test all team members
python scripts/run_authentication_system.py --mode test

# Interactive authentication
python scripts/run_authentication_system.py --mode interactive

# Batch processing
python scripts/run_authentication_system.py --mode batch
```

**Features:**

- ✅ Authorized user flow (face + voice → product revealed)
- ❌ Unauthorized face detection (access denied)
- ❌ Unauthorized voice detection (access denied)
- Clean CLI interface with user prompts

---

## 🚀 Quick Start Guide

### Installation

```bash
# Clone repository
git clone https://github.com/izabayo7/multimodal-data-preprocessing-assignment.git
cd multimodal-data-preprocessing-assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Note: Mac users may need libomp for XGBoost
brew install libomp
```

### Running the System

**1. Test with all team members (recommended first run):**

```bash
python scripts/run_authentication_system.py --mode test
```

**2. Interactive authentication:**

```bash
python scripts/run_authentication_system.py --mode interactive
```

Follow prompts to:

- Select user (Alice, Armstrong, cedric, yassin)
- Authenticate with face + voice
- Receive personalized product recommendation

**3. Batch processing:**

```bash
python scripts/run_authentication_system.py --mode batch
```

---

## 📊 Deliverables

| Component          | Location                                  | Description                           |
| ------------------ | ----------------------------------------- | ------------------------------------- |
| **Merged Dataset** | `data/processed/merged_customer_data.csv` | Cleaned & merged customer data        |
| **Image Features** | `data/features/image_features.csv`        | VGG16 embeddings for face recognition |
| **Audio Features** | `data/processed/audio_features.csv`       | MFCCs + spectral features for voice   |
| **Face Model**     | `models/face_recognition_model.pkl`       | VGG16 + Random Forest classifier      |
| **Voice Model**    | `models/voice_model.joblib`               | Random Forest voice classifier        |
| **Product Model**  | `models/xgboost_product_recommender.pkl`  | XGBoost product recommender           |
| **Pipeline**       | `src/pipeline/authentication_pipeline.py` | Integration of all 3 models           |
| **CLI App**        | `scripts/run_authentication_system.py`    | Command-line interface                |
| **EDA Notebook**   | `notebooks/01_data_merge_eda.ipynb`       | Data analysis & visualization         |
| **Face Notebook**  | `notebooks/face_recognition_model.ipynb`  | Face model training                   |
| **Voice Notebook** | `notebooks/voice_recognition_model.ipynb` | Voice model training                  |

---

## 🎥 Demo Video

**Watch the system in action:** [YouTube Demo](https://youtu.be/OCy7BjODDCc)

**Demo showcases:**

1. ❌ Unauthorized face rejection
2. ❌ Unauthorized voice rejection
3. ✅ Complete successful authentication flow
4. 📦 Product recommendation display

---

## 🛠️ Technical Details

### Dependencies

- **Python:** 3.8+
- **Deep Learning:** TensorFlow, Keras (VGG16)
- **ML Libraries:** scikit-learn, XGBoost
- **Audio Processing:** librosa, soundfile
- **Image Processing:** OpenCV, Pillow
- **Data Science:** pandas, numpy, matplotlib, seaborn

### Model Architectures

**1. Face Recognition (Yassin)**

- Base: VGG16 (pre-trained on ImageNet)
- Classifier: Random Forest
- Features: 512-dimensional embeddings
- Augmentations: Rotation, flip, brightness, zoom

**2. Voice Verification (Brian)**

- Algorithm: Random Forest
- Features: 13 MFCCs + spectral rolloff + RMS energy
- Augmentations: Time stretch, pitch shift, noise

**3. Product Recommendation (Alice)**

- Algorithm: XGBoost Classifier
- Input: Customer transaction history + demographics
- Output: Product category prediction
- Classes: Books, Groceries, Sports, etc.

---

## 📝 Team Contributions

### Alice Mukarwema (Task 1)

- Downloaded and merged customer datasets
- Performed EDA with 3+ visualizations
- Cleaned data (nulls, duplicates, types)
- Trained XGBoost product recommendation model
- Saved merged data and model artifacts

### Yassin Hagenimana (Task 2)

- Collected 12 face images (3 per member)
- Implemented image augmentation pipeline
- Extracted VGG16 embeddings
- Trained Random Forest face classifier
- Created face recognition prediction module

### Hirwa Brian (Task 3)

- Recorded 8 voice samples (2 per member)
- Generated waveform and spectrogram plots
- Implemented audio augmentation
- Extracted MFCC features
- Trained Random Forest voice classifier

### Cedric Izabayo (Task 4)

- Integrated all 3 models into unified pipeline
- Built CLI authentication system
- Implemented multimodal decision logic
- Conducted system testing (authorized/unauthorized)
- Recorded demo video and documentation

---

**Submission Date:** November 14, 2025  
**Course:** Data Science - Formative 2  
**Project:** Multimodal Data Preprocessing & Authentication System

---
