#!/usr/bin/env python3
"""
Multimodal Authentication System - Interactive CLI

Command-line interface for the complete authentication pipeline.
Supports face recognition, voice verification, and product recommendations.

Usage:
    python run_authentication_system.py
    python run_authentication_system.py --image path/to/image.jpg
    python run_authentication_system.py --image path/to/image.jpg --audio path/to/audio.wav
"""

import sys
import os
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline.authentication_pipeline import MultimodalAuthenticationPipeline


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print(" " * 15 + "MULTIMODAL AUTHENTICATION SYSTEM")
    print(" " * 10 + "Face Recognition + Voice Verification + Product AI")
    print("="*70 + "\n")


def print_result_summary(result):
    """Print formatted authentication result"""
    print("\n" + "="*70)
    print(" " * 25 + "AUTHENTICATION RESULT")
    print("="*70)
    
    if result['access_granted']:
        print(f"\n✅ Status: ACCESS GRANTED")
        print(f"👤 User: {result['user']}")
        
        # Face recognition details
        face = result['face_recognition']
        print(f"\n📸 Face Recognition:")
        print(f"   Confidence: {face['confidence']:.2%}")
        print(f"   Status: ✅ {face['message']}")
        
        # Product recommendation details
        product = result['product_recommendation']
        print(f"\n📦 Product Recommendation:")
        print(f"   Product: {product['product']}")
        if product['category']:
            print(f"   Category: {product['category']}")
        if product['confidence'] > 0:
            print(f"   Confidence: {product['confidence']:.2%}")
        
        # Voice verification details
        voice = result['voice_verification']
        print(f"\n🎤 Voice Verification:")
        if voice['confidence'] > 0:
            print(f"   Confidence: {voice['confidence']:.2%}")
        print(f"   Status: ✅ {voice['message']}")
        
    else:
        print(f"\n❌ Status: ACCESS DENIED")
        print(f"❌ Reason: {result['message']}")
        
        # Show what passed/failed
        if result['face_recognition']:
            face = result['face_recognition']
            if face['success']:
                print(f"\n� Face Recognition: ✅ PASSED ({face['user']})")
            else:
                print(f"\n📸 Face Recognition: ❌ FAILED")
        
        if result['product_recommendation']:
            product = result['product_recommendation']
            if product['success']:
                print(f"📦 Product Recommendation: ✅ READY ({product['product']})")
            else:
                print(f"📦 Product Recommendation: ❌ FAILED")
        
        if result['voice_verification']:
            voice = result['voice_verification']
            print(f"🎤 Voice Verification: ❌ FAILED")
        else:
            print(f"🎤 Voice Verification: ⚠️  NOT PROVIDED (required)")
    
    print("\n" + "="*70 + "\n")


def interactive_mode(pipeline):
    """Run in interactive mode with prompts"""
    print_banner()
    print("Running in INTERACTIVE mode\n")
    
    # Step 1: Get image path
    print("Step 1: Face Recognition")
    print(f"Available users: {', '.join(Config.DEFAULT_USERS)}")
    
    while True:
        image_path = input("\nEnter path to face image (or 'q' to quit): ").strip()
        
        if image_path.lower() == 'q':
            print("Exiting...")
            return
        
        image_path = Path(image_path).expanduser()
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            print("Try again or enter 'q' to quit.")
            continue
        
        break
    
    # First, verify the face
    print("\n🔄 Checking face recognition...")
    face_result = pipeline.authenticate_face(str(image_path))
    
    if not face_result['success']:
        print(f"\n❌ Face recognition failed: {face_result['message']}")
        print("❌ ACCESS DENIED\n")
        
        # Ask to try again
        retry = input("Try another image? (y/n): ").strip().lower()
        if retry == 'y':
            interactive_mode(pipeline)
        return
    
    identified_user = face_result['user']
    print(f"✅ Face recognized: {identified_user} (confidence: {face_result['confidence']:.2%})")
    
    # Step 2: Now that face is recognized, ask for voice verification
    print(f"\nStep 2: Voice Verification for {identified_user}")
    print("Voice verification is required to confirm prediction and grant access")
    
    while True:
        audio_path = input(f"Enter path to {identified_user}'s audio file (or 'skip' to see access denied): ").strip()
        
        if audio_path.lower() == 'skip':
            # Run without audio to show the denial
            print("\n🔄 Processing without voice verification...")
            result = pipeline.authenticate(
                image_path=str(image_path),
                audio_path=None
            )
            print_result_summary(result)
            break
        
        audio_path = Path(audio_path).expanduser()
        if not audio_path.exists():
            print(f"❌ Audio file not found: {audio_path}")
            print("Try again or type 'skip' to proceed without audio.")
            continue
        
        # Run complete authentication
        print("\n🔄 Running complete authentication pipeline...")
        result = pipeline.authenticate(
            image_path=str(image_path),
            audio_path=str(audio_path)
        )
        print_result_summary(result)
        break
    
    # Ask to continue
    continue_choice = input("Authenticate another user? (y/n): ").strip().lower()
    if continue_choice == 'y':
        interactive_mode(pipeline)


def batch_mode(pipeline, image_path, audio_path=None):
    """Run in batch mode with provided paths"""
    print_banner()
    print("Running in BATCH mode\n")
    
    # Validate paths
    image_path = Path(image_path).expanduser()
    if not image_path.exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    if audio_path:
        audio_path = Path(audio_path).expanduser()
        if not audio_path.exists():
            print(f"⚠️  Warning: Audio not found: {audio_path}")
            print("Continuing without voice verification...")
            audio_path = None
        else:
            audio_path = str(audio_path)
    
    # Run authentication
    result = pipeline.authenticate(
        image_path=str(image_path),
        audio_path=audio_path
    )
    
    # Print result
    print_result_summary(result)
    
    return result


def test_mode(pipeline):
    """Run test mode with sample data"""
    print_banner()
    print("Running in TEST mode with sample data\n")
    print("⚠️  Note: Tests will include audio files for complete flow\n")
    
    test_cases = []
    
    # Find test images and corresponding audio
    for user in Config.DEFAULT_USERS:
        user_img_dir = Config.IMAGES_DIR / user
        user_audio_dir = Config.AUDIO_DIR / user
        
        if user_img_dir.exists():
            for img in user_img_dir.glob('*.jp*g'):
                # Try to find corresponding audio
                audio_file = None
                if user_audio_dir.exists():
                    # Look for any audio file for this user
                    audio_files = list(user_audio_dir.glob('*.wav'))
                    if audio_files:
                        audio_file = str(audio_files[0])
                
                test_cases.append({
                    'name': f"{user} - {img.stem}",
                    'image': str(img),
                    'audio': audio_file
                })
                break  # Just one image per user
    
    if not test_cases:
        print("❌ No test images found in data/images/")
        sys.exit(1)
    
    print(f"Found {len(test_cases)} test case(s)\n")
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_cases)}: {test['name']}")
        if test['audio']:
            print(f"  With audio: {Path(test['audio']).name}")
        else:
            print(f"  ⚠️  No audio available - will fail voice verification")
        print(f"{'='*70}")
        
        result = pipeline.authenticate(
            image_path=test['image'],
            audio_path=test['audio']
        )
        
        # Brief summary
        if result['access_granted']:
            status = "✅ PASS"
            user = result['user']
            product = result['product_recommendation']['product']
            print(f"\n{status} - {test['name']}: {user} → {product}")
            passed += 1
        else:
            status = "❌ FAIL"
            print(f"\n{status} - {test['name']}: {result['message']}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(test_cases)} test(s)")
    print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multimodal Authentication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python run_authentication_system.py
  
  Batch mode with face only:
    python run_authentication_system.py --image data/images/Alice/neutral.jpeg
  
  Batch mode with face and voice:
    python run_authentication_system.py --image data/images/Alice/neutral.jpeg --audio data/audio/Alice/confirm.wav
  
  Test mode:
    python run_authentication_system.py --test
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to face image for authentication'
    )
    
    parser.add_argument(
        '--audio', '-a',
        type=str,
        help='Path to audio file for voice verification (optional)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run in test mode with sample data'
    )
    
    parser.add_argument(
        '--face-threshold',
        type=float,
        default=0.80,
        help='Face recognition confidence threshold (default: 0.80)'
    )
    
    parser.add_argument(
        '--voice-threshold',
        type=float,
        default=0.75,
        help='Voice verification confidence threshold (default: 0.75)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = MultimodalAuthenticationPipeline(
            face_confidence_threshold=args.face_threshold,
            voice_confidence_threshold=args.voice_threshold
        )
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Determine mode
    if args.test:
        # Test mode
        test_mode(pipeline)
    elif args.image:
        # Batch mode
        result = batch_mode(pipeline, args.image, args.audio)
        if args.json:
            print("\nJSON Output:")
            print(json.dumps(result, indent=2, default=str))
    else:
        # Interactive mode
        try:
            interactive_mode(pipeline)
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()
