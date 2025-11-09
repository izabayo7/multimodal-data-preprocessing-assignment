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


class MultimodalAuthenticationPipeline:
    """
    Complete authentication pipeline that combines:
    1. Face Recognition (WORKING)
    2. Voice Verification (PLACEHOLDER - needs voice_verification_model.pkl)
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
        """Load voice verification model (PLACEHOLDER)"""
        voice_model_path = self.models_dir / 'voice_verification_model.pkl'
        voice_scaler_path = self.models_dir / 'voice_verification_scaler.pkl'
        
        if voice_model_path.exists() and voice_scaler_path.exists():
            try:
                self.voice_model = joblib.load(voice_model_path)
                self.voice_scaler = joblib.load(voice_scaler_path)
                self.voice_ready = True
                print("✅ Voice Verification Model Loaded")
            except Exception as e:
                print(f"⚠️  Voice model exists but failed to load: {e}")
                self.voice_ready = False
                self.voice_model = None
                self.voice_scaler = None
        else:
            print("⚠️  Voice Verification Model NOT FOUND (using placeholder)")
            print(f"   Expected: {voice_model_path}")
            print(f"   Brian should provide: voice_verification_model.pkl & voice_verification_scaler.pkl")
            self.voice_ready = False
            self.voice_model = None
            self.voice_scaler = None
    
    def _load_product_recommender(self):
        """Load product recommendation model (PLACEHOLDER)"""
        product_model_path = self.models_dir / 'product_recommender_model.pkl'
        
        if product_model_path.exists():
            try:
                self.product_model = joblib.load(product_model_path)
                self.product_ready = True
                print("✅ Product Recommendation Model Loaded")
            except Exception as e:
                print(f"⚠️  Product model exists but failed to load: {e}")
                self.product_ready = False
                self.product_model = None
        else:
            print("⚠️  Product Recommendation Model NOT FOUND (using placeholder)")
            print(f"   Expected: {product_model_path}")
            print(f"   Alice should provide: product_recommender_model.pkl")
            self.product_ready = False
            self.product_model = None
    
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
        Step 2: Voice Verification (PLACEHOLDER)
        
        Args:
            audio_path: Path to the audio file
            claimed_user: User identified from face recognition
            
        Returns:
            dict: {
                'success': bool,
                'confidence': float,
                'message': str
            }
        """
        if not self.voice_ready:
            return {
                'success': True,  # Placeholder accepts all
                'confidence': 0.0,
                'message': f'⚠️  PLACEHOLDER: Voice verification skipped (model not available)'
            }
        
        # TODO: When Brian provides voice_verification_model.pkl, implement actual verification:
        # 1. Extract audio features using librosa (MFCCs, spectral features)
        # 2. Scale features using voice_scaler
        # 3. Predict using voice_model
        # 4. Check if predicted user matches claimed_user
        # 5. Return confidence score
        
        try:
            # Actual implementation when model is available
            # features = extract_voice_features(audio_path)
            # features_scaled = self.voice_scaler.transform(features)
            # prediction = self.voice_model.predict(features_scaled)
            # confidence = self.voice_model.predict_proba(features_scaled).max()
            
            # For now, return placeholder
            return {
                'success': True,
                'confidence': 0.0,
                'message': f'⚠️  PLACEHOLDER: Voice verification for {claimed_user} skipped'
            }
        except Exception as e:
            return {
                'success': False,
                'confidence': 0.0,
                'message': f'Error during voice verification: {str(e)}'
            }
    
    def recommend_product(self, user):
        """
        Step 3: Product Recommendation (PLACEHOLDER)
        
        Args:
            user: Authenticated user name
            
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
        
        # TODO: When Alice provides product_recommender_model.pkl, implement actual recommendation:
        # 1. Load user's transaction history from merged_customer_data.csv
        # 2. Extract features (engagement_score, purchase_interest, sentiment, etc.)
        # 3. Predict product category using product_model
        # 4. Return top recommended product with confidence
        
        try:
            # Actual implementation when model is available
            # user_data = load_user_data(user)
            # features = extract_features(user_data)
            # prediction = self.product_model.predict(features)
            # confidence = self.product_model.predict_proba(features).max()
            
            # For now, return placeholder
            return {
                'success': True,
                'product': 'Electronics',
                'category': 'General',
                'confidence': 0.0,
                'message': f'⚠️  PLACEHOLDER: Product recommendation for {user} (model pending)'
            }
        except Exception as e:
            return {
                'success': False,
                'product': None,
                'category': None,
                'confidence': 0.0,
                'message': f'Error during product recommendation: {str(e)}'
            }
    
    def authenticate(self, image_path, audio_path=None):
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
            
        Returns:
            dict: Complete authentication result with all steps
        """
        print("\n" + "="*60)
        print("MULTIMODAL AUTHENTICATION PIPELINE")
        print("="*60)
        
        # Step 1: Face Recognition - Identify the user
        print("\n[Step 1/3] Face Recognition...")
        face_result = self.authenticate_face(image_path)
        
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
        
        # Step 2: Product Recommendation - Generate prediction for the user
        print(f"\n[Step 2/3] Product Recommendation for {identified_user}...")
        product_result = self.recommend_product(identified_user)
        
        if product_result['success']:
            print(f"   📦 Product predicted: {product_result['product']}")
            if product_result['confidence'] > 0:
                print(f"   📊 Confidence: {product_result['confidence']:.2%}")
            else:
                print(f"   ⚠️  {product_result['message']}")
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
            print(f"   Prediction was ready: {product_result['product']}")
            print(f"   But voice did not match {identified_user}")
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
