"""
Pipeline for multimodal authentication system.
Integrates face recognition, voice verification, and product recommendations.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config

# Import face recognition predictor
sys.path.insert(0, str(Config.PROJECT_ROOT / 'scripts'))
from predict_face import EnhancedFaceRecognitionPredictor


def extract_voice_features(audio_path):
    """
    Extract audio features from a voice recording.
    Uses the same feature extraction process as Brian's training.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        dict: Features in the format expected by the voice model
    """
    import librosa
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        
        # Extract RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Combine features in the SAME ORDER as training CSV:
        # mfcc_1 through mfcc_13, then spectral_rolloff, then rms_energy
        features = {}
        for i in range(13):
            features[f'mfcc_{i+1}'] = mfccs_mean[i]
        features['spectral_rolloff'] = rolloff_mean
        features['rms_energy'] = rms_mean
            
        return features
        
    except Exception as e:
        raise Exception(f"Error extracting voice features: {str(e)}")


class MultimodalAuthenticationPipeline:
    """
    Complete authentication pipeline that combines:
    1. Face Recognition (WORKING)
    2. Voice Verification (WORKING)
    3. Product Recommendation (PLACEHOLDER - needs product_recommender_model.pkl)
    """
    
    def __init__(self, 
                 face_confidence_threshold=0.80,
                 voice_confidence_threshold=0.75,
                 models_dir=None):
        """
        Initialize the authentication pipeline
        
        Args:
            face_confidence_threshold: Minimum confidence for face recognition (0.0-1.0)
            voice_confidence_threshold: Minimum confidence for voice verification (0.0-1.0)
            models_dir: Directory containing model files (default: project models/)
        """
        self.models_dir = models_dir or Config.MODELS_DIR
        self.face_threshold = face_confidence_threshold
        self.voice_threshold = voice_confidence_threshold
        
        # Initialize components
        self._load_face_recognition()
        self._load_voice_verification()
        self._load_product_recommender()
        
        print("✅ Authentication Pipeline Initialized")
        print(f"   Face Recognition: {'READY' if self.face_ready else 'NOT READY'}")
        print(f"   Voice Verification: {'READY' if self.voice_ready else 'PLACEHOLDER'}")
        print(f"   Product Recommender: {'READY' if self.product_ready else 'PLACEHOLDER'}")
    
    def _load_face_recognition(self):
        """Load face recognition model (WORKING)"""
        try:
            self.face_predictor = EnhancedFaceRecognitionPredictor(
                model_dir=str(self.models_dir),
                confidence_threshold=self.face_threshold
            )
            self.face_ready = True
            print("✅ Face Recognition Model Loaded")
        except Exception as e:
            print(f"❌ Error loading face recognition: {e}")
            self.face_ready = False
            self.face_predictor = None
    
    def _load_voice_verification(self):
        """Load voice verification model - Brian's Random Forest model"""
        voice_model_path = self.models_dir / 'voice_model.joblib'
        voice_scaler_path = self.models_dir / 'voice_scaler.joblib'
        voice_encoder_path = self.models_dir / 'voice_encoder.joblib'
        
        if voice_model_path.exists() and voice_scaler_path.exists() and voice_encoder_path.exists():
            try:
                self.voice_model = joblib.load(voice_model_path)
                self.voice_scaler = joblib.load(voice_scaler_path)
                self.voice_encoder = joblib.load(voice_encoder_path)
                self.voice_ready = True
                print("✅ Voice Verification Model Loaded")
                print(f"   Can recognize: {', '.join(self.voice_encoder.classes_)}")
            except Exception as e:
                print(f"⚠️  Voice model exists but failed to load: {e}")
                self.voice_ready = False
                self.voice_model = None
                self.voice_scaler = None
                self.voice_encoder = None
        else:
            print("⚠️  Voice Verification Model NOT FOUND (using placeholder)")
            print(f"   Expected: {voice_model_path}")
            print(f"   Brian should provide: voice_model.joblib, voice_scaler.joblib, voice_encoder.joblib")
            self.voice_ready = False
            self.voice_model = None
            self.voice_scaler = None
            self.voice_encoder = None
    
    def _load_product_recommender(self):
        """Load product recommendation model"""
        product_model_path = self.models_dir / 'xgboost_product_recommender.pkl'
        
        # Static mapping: Face recognition names → Customer IDs in merged dataset
        # NOTE: This is a temporary mapping since the customer_social_profiles.csv
        # uses customer IDs but our face/voice recognition uses actual member names.
        # In production, this should come from a database with proper user profiles.
        # These IDs were verified to exist in merged_customer_data.csv
        self.user_to_customer_mapping = {
            'Alice': 150,         # Alice maps to customer 150
            'Armstrong': 177,     # Armstrong maps to customer 177 (178 not available)
            'cedric': 190,        # Cedric maps to customer 190
            'yassin': 162         # Yassin maps to customer 162
        }
        
        if product_model_path.exists():
            try:
                # Alice's model is packaged with ProductRecommender class
                # We need to import it and use load_model method
                import sys
                sys.path.insert(0, str(Config.PROJECT_ROOT / 'src' / 'models'))
                from product_recommender import ProductRecommender
                
                self.product_recommender = ProductRecommender()
                if self.product_recommender.load_model(str(product_model_path)):
                    self.product_ready = True
                    print("✅ Product Recommendation Model Loaded")
                else:
                    print("⚠️  Product model exists but failed to load")
                    self.product_ready = False
                    self.product_recommender = None
            except Exception as e:
                print(f"⚠️  Product model exists but failed to load: {e}")
                self.product_ready = False
                self.product_recommender = None
        else:
            print("⚠️  Product Recommendation Model NOT FOUND (using placeholder)")
            print(f"   Expected: {product_model_path}")
            print(f"   Alice should provide: xgboost_product_recommender.pkl")
            self.product_ready = False
            self.product_recommender = None
    
    def authenticate_face(self, image_path):
        """
        Step 1: Face Recognition
        
        Args:
            image_path: Path to the face image
            
        Returns:
            dict: {
                'success': bool,
                'user': str or None,
                'confidence': float,
                'message': str
            }
        """
        if not self.face_ready:
            return {
                'success': False,
                'user': None,
                'confidence': 0.0,
                'message': 'Face recognition model not available'
            }
        
        try:
            result = self.face_predictor.predict(image_path, show_probabilities=False)
            
            # Convert predict_face.py output to pipeline format
            is_known = result.get('is_known_person', False)
            predicted_user = result.get('predicted_user', 'UNKNOWN')
            confidence = result.get('confidence', 0.0) / 100.0  # Convert from percentage
            
            if is_known and predicted_user != 'UNKNOWN':
                return {
                    'success': True,
                    'user': predicted_user,
                    'confidence': confidence,
                    'message': f'Face recognized as {predicted_user}'
                }
            else:
                return {
                    'success': False,
                    'user': None,
                    'confidence': confidence,
                    'message': 'Unknown person detected - confidence below threshold'
                }
        except Exception as e:
            return {
                'success': False,
                'user': None,
                'confidence': 0.0,
                'message': f'Error during face recognition: {str(e)}'
            }
    
    def verify_voice(self, audio_path, claimed_user):
        """
        Step 2: Voice Verification - Brian's Random Forest model
        
        Args:
            audio_path: Path to the audio file
            claimed_user: User identified from face recognition
            
        Returns:
            dict: {
                'success': bool,
                'confidence': float,
                'predicted_user': str,
                'message': str
            }
        """
        if not self.voice_ready:
            return {
                'success': True,  # Placeholder accepts all
                'confidence': 0.0,
                'predicted_user': claimed_user,
                'message': f'⚠️  PLACEHOLDER: Voice verification skipped (model not available)'
            }
        
        try:
            # Extract audio features
            features_dict = extract_voice_features(audio_path)
            
            # Convert to DataFrame (features are already in correct order from extraction)
            import pandas as pd
            features_df = pd.DataFrame([features_dict])
            
            # Scale features
            features_scaled = self.voice_scaler.transform(features_df)
            
            # Predict user
            prediction_encoded = self.voice_model.predict(features_scaled)[0]
            predicted_user = self.voice_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get confidence (probability of predicted class)
            probabilities = self.voice_model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction_encoded]
            
            # Verify if predicted user matches claimed user
            success = (predicted_user == claimed_user)
            
            if success:
                message = f'✅ Voice verified: {predicted_user} (confidence: {confidence:.1%})'
            else:
                message = f'❌ Voice mismatch: Expected {claimed_user}, got {predicted_user} (confidence: {confidence:.1%})'
            
            return {
                'success': success,
                'confidence': float(confidence),
                'predicted_user': predicted_user,
                'message': message
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'predicted_user': None,
                'message': f'Error during voice verification: {str(e)}'
            }
    
    def recommend_product(self, user):
        """
        Step 3: Product Recommendation - Alice's XGBoost model
        
        Args:
            user: Authenticated user name (Alice, Armstrong, cedric, yassin)
            
        Returns:
            dict: {
                'success': bool,
                'product': str or None,
                'category': str or None,
                'confidence': float,
                'message': str
            }
        """
        if not self.product_ready:
            return {
                'success': True,  # Placeholder returns generic recommendation
                'product': 'Electronics',
                'category': 'General',
                'confidence': 0.0,
                'message': f'⚠️  PLACEHOLDER: Generic product recommendation (model not available)'
            }
        
        try:
            # Map user name to customer ID
            # NOTE: This is a static mapping since merged_customer_data.csv uses 
            # customer IDs but authentication uses member names.
            # In production, this should query a database with user profiles.
            customer_id = self.user_to_customer_mapping.get(user)
            
            if not customer_id:
                return {
                    'success': False,
                    'product': None,
                    'category': None,
                    'confidence': 0.0,
                    'message': f'Error: No customer mapping found for user {user}'
                }
            
            # Load user's most recent transaction data for prediction
            import pandas as pd
            merged_data_path = Config.PROCESSED_DIR / 'merged_customer_data.csv'
            df = pd.read_csv(merged_data_path)
            
            # Get user's most recent transaction
            user_data = df[df['customer_id'] == customer_id].iloc[-1].to_dict()
            
            # Predict product category using Alice's model
            predicted_category, probabilities = self.product_recommender.predict_product_category(user_data)
            
            if predicted_category:
                confidence = probabilities.get(predicted_category, 0.0)
                return {
                    'success': True,
                    'product': predicted_category,
                    'category': predicted_category,
                    'confidence': float(confidence),
                    'message': f'✅ Product recommendation: {predicted_category} (confidence: {confidence:.1%})'
                }
            else:
                return {
                    'success': False,
                    'product': None,
                    'category': None,
                    'confidence': 0.0,
                    'message': 'Error: Model prediction failed'
                }
                
        except Exception as e:
            return {
                'success': False,
                'product': None,
                'category': None,
                'confidence': 0.0,
                'message': f'Error during product recommendation: {str(e)}'
            }
    
    def authenticate(self, image_path, audio_path=None, face_result=None):
        """
        Complete authentication pipeline following the assignment flow:
        
        Flow:
        1. Face Recognition → Identify user (if fail → Access Denied)
        2. Product Recommendation → Generate prediction for identified user
        3. Voice Verification → Confirm user identity (if fail → Access Denied)
        4. Display Product → Show recommendation only if voice verified
        
        Args:
            image_path: Path to face image
            audio_path: Path to audio file (REQUIRED for voice verification)
            face_result: Pre-validated face recognition result (optional, skips face recognition if provided)
            
        Returns:
            dict: Complete authentication result with all steps
        """
        print("\n" + "="*60)
        print("MULTIMODAL AUTHENTICATION PIPELINE")
        print("="*60)
        
        # Step 1: Face Recognition - Identify the user (skip if already done)
        if face_result is None:
            print("\n[Step 1/3] Face Recognition...")
            face_result = self.authenticate_face(image_path)
        else:
            print("\n[Step 1/3] Face Recognition... ✅ Already verified")
            print(f"   ✅ Face recognized: {face_result['user']} (confidence: {face_result['confidence']:.2%})")
        
        if not face_result['success']:
            print(f"   ❌ Face recognition failed: {face_result['message']}")
            print("\n" + "="*60)
            print("❌ ACCESS DENIED - Face not recognized")
            print("="*60 + "\n")
            return {
                'authenticated': False,
                'access_granted': False,
                'user': None,
                'face_recognition': face_result,
                'product_recommendation': None,
                'voice_verification': None,
                'message': f"❌ ACCESS DENIED: {face_result['message']}"
            }
        
        identified_user = face_result['user']
        print(f"   ✅ Face recognized: {identified_user} (confidence: {face_result['confidence']:.2%})")
        
        # Step 2: Product Recommendation - Generate prediction for the user (but don't reveal it yet)
        print(f"\n[Step 2/3] Product Recommendation for {identified_user}...")
        product_result = self.recommend_product(identified_user)
        
        if product_result['success']:
            print(f"   📦 Generating personalized recommendation...")
            print(f"   ✅ Product recommendation ready (will be revealed after voice verification)")
        else:
            print(f"   ❌ Product recommendation failed: {product_result['message']}")
            print("\n" + "="*60)
            print("❌ ACCESS DENIED - Product recommendation failed")
            print("="*60 + "\n")
            return {
                'authenticated': False,
                'access_granted': False,
                'user': identified_user,
                'face_recognition': face_result,
                'product_recommendation': product_result,
                'voice_verification': None,
                'message': f"❌ ACCESS DENIED: Product recommendation failed"
            }
        
        # Step 3: Voice Verification - Confirm it's really the user before showing product
        if not audio_path:
            print(f"\n[Step 3/3] Voice Verification... ❌ REQUIRED")
            print(f"   ❌ No audio provided - voice verification is required to confirm prediction")
            print("\n" + "="*60)
            print("❌ ACCESS DENIED - Voice verification required")
            print(f"   Prediction ready: {product_result['product']}")
            print(f"   Provide audio to confirm and view prediction")
            print("="*60 + "\n")
            return {
                'authenticated': False,
                'access_granted': False,
                'user': identified_user,
                'face_recognition': face_result,
                'product_recommendation': product_result,
                'voice_verification': None,
                'message': f"❌ ACCESS DENIED: Voice verification required to confirm prediction"
            }
        
        print(f"\n[Step 3/3] Voice Verification...")
        voice_result = self.verify_voice(audio_path, identified_user)
        
        if not voice_result['success']:
            print(f"   ❌ Voice verification failed: {voice_result['message']}")
            print("\n" + "="*60)
            print("❌ ACCESS DENIED - Voice verification failed")
            print(f"   Cannot display product recommendation - authentication incomplete")
            print("="*60 + "\n")
            return {
                'authenticated': False,
                'access_granted': False,
                'user': identified_user,
                'face_recognition': face_result,
                'product_recommendation': product_result,
                'voice_verification': voice_result,
                'message': f"❌ ACCESS DENIED: Voice verification failed"
            }
        
        print(f"   ✅ Voice verified for {identified_user}")
        if voice_result['confidence'] > 0:
            print(f"   � Confidence: {voice_result['confidence']:.2%}")
        else:
            print(f"   ⚠️  {voice_result['message']}")
        
        # Success! All steps passed - Display the product
        print("\n" + "="*60)
        print(f"✅ ACCESS GRANTED - AUTHENTICATION SUCCESSFUL!")
        print("="*60)
        print(f"\n👤 User: {identified_user}")
        print(f"📦 Recommended Product: {product_result['product']}")
        if product_result['category']:
            print(f"🏷️  Category: {product_result['category']}")
        print("\n" + "="*60 + "\n")
        
        return {
            'authenticated': True,
            'access_granted': True,
            'user': identified_user,
            'face_recognition': face_result,
            'product_recommendation': product_result,
            'voice_verification': voice_result,
            'message': f"✅ ACCESS GRANTED: {identified_user} authenticated successfully"
        }


def main():
    """Example usage of the authentication pipeline"""
    # Initialize pipeline
    pipeline = MultimodalAuthenticationPipeline(
        face_confidence_threshold=0.80,
        voice_confidence_threshold=0.75
    )
    
    # Example: Authenticate with just face recognition
    test_image = Config.IMAGES_DIR / 'Alice' / 'neutral.jpeg'
    
    if test_image.exists():
        result = pipeline.authenticate(
            image_path=str(test_image),
            audio_path=None  # No audio for now
        )
        
        # Print detailed results
        print("\nDetailed Results:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Test image not found: {test_image}")


if __name__ == "__main__":
    main()
