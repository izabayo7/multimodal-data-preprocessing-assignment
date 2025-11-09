import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# --- Configuration ---
AUDIO_DIR = '../data/audio' 
OUTPUT_CSV = '../data/processed/audio_features.csv'
PLOT_DIR = '../data/processed/audio_plots/'

# Create the plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

# Helper function to extract features
def extract_features(y, sr):
    """Extracts MFCCs, Spectral Rolloff, and RMS Energy from an audio array"""
    try:
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1) 
        
        # 2. Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        
        # 3. Energy (RMS)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        
        # Combining all features into a dictionary
        features = {
            'spectral_rolloff': rolloff_mean,
            'rms_energy': rms_mean,
        }
        for i in range(13):
            features[f'mfcc_{i+1}'] = mfccs_mean[i]
            
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Augmentation options
def get_augmentations(y, sr):
    """Yields original, pitch-shifted, and noise-added audio arrays"""
    # Original
    yield "original", y
    
    # Aug 1: Pitch Shift (4 semitones up)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    yield "pitch_shift", y_pitch
    
    # Aug 2: Add Background Noise
    noise = 0.005 * np.random.randn(len(y))
    y_noise = y + noise
    yield "noise", y_noise

# Helper function to parse phrase from filename
def get_phrase_from_filename(filename):
    """Parses 'confirm' or 'approve' from filename, handling different formats"""
    filename_lower = filename.lower()
    if "confirm" in filename_lower:
        return "confirm"
    if "approve" in filename_lower:
        return "approve"
    return "unknown"

# Main script to build the CSV
all_features = []

print(f"Starting feature extraction from: {AUDIO_DIR}")
print(f"Saving plots to: {PLOT_DIR}")

for root, dirs, files in os.walk(AUDIO_DIR):
    for filename in files:
        # Process only .wav files
        if filename.endswith('.wav'):
            
            member_id = os.path.basename(root)
            
            phrase = get_phrase_from_filename(filename)
            
            if phrase == "unknown":
                print(f"Skipping file with unknown phrase: {os.path.join(root, filename)}")
                continue
                
            file_path = os.path.join(root, filename)
            
            try:
                # Load the original file
                y_orig, sr_orig = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
            # Create a clean base filename for the plots
            plot_basename = f"{member_id}_{phrase}_{filename.replace('.wav', '')}"

            # Plot the Waveform
            try:
                plt.figure(figsize=(12, 4))
                librosa.display.waveshow(y_orig, sr=sr_orig)
                plt.title(f'Waveform: {member_id} - {phrase}')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, f"{plot_basename}_waveform.png"))
                plt.close()
            
                # Plot the Spectrogram
                S = librosa.feature.melspectrogram(y=y_orig, sr=sr_orig, n_mels=128)
                S_db = librosa.power_to_db(S, ref=np.max)
                
                plt.figure(figsize=(12, 4))
                librosa.display.specshow(S_db, sr=sr_orig, x_axis='time', y_axis='mel')
                plt.title(f'Mel Spectrogram: {member_id} - {phrase}')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, f"{plot_basename}_spectrogram.png"))
                plt.close()
            
            except Exception as e:
                print(f"Error generating plot for {file_path}: {e}")


            # Process original + augmentations for the CSV
            for aug_name, y_aug in get_augmentations(y_orig, sr_orig):
                
                # Extract features from the (potentially augmented) audio
                features = extract_features(y_aug, sr_orig)
                
                if features:
                    # Create a row of data
                    row = {
                        'filename': filename,
                        'member_id': member_id, # This is your LABEL
                        'phrase': phrase,
                        'augmentation': aug_name,
                    }
                    
                    # Add all the extracted features (MFCCs, rolloff, rms)
                    row.update(features)
                    
                    all_features.append(row)

# Create and save the DataFrame
df = pd.DataFrame(all_features)

if not df.empty:
    # Reorder columns to be nice
    feature_cols = [f'mfcc_{i+1}' for i in range(13)] + ['spectral_rolloff', 'rms_energy']
    info_cols = ['filename', 'member_id', 'phrase', 'augmentation']
    
    # Ensure all columns exist before trying to order them
    final_cols = info_cols + [col for col in feature_cols if col in df.columns]
    
    df = df[final_cols]

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccessfully processed {len(df)} audio samples.")
    print(f"Features saved to {OUTPUT_CSV}")
    print("\nDataFrame Head:")
    print(df.head())
else:
    print("\nNo features were extracted. Please check your audio directory and file formats.")

print(f"\nAll plots saved to {PLOT_DIR}")