"""
AeroGuard TinyML Dataset Creator
Creates a small subset for rapid prototyping and testing
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm
import random

print("=" * 70)
print("🎯 AeroGuard TinyML Dataset Creator")
print("=" * 70)

# Configuration
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 1000
WINDOW_OVERLAP_MS = 500
N_MFCC = 13

# Subset sizes (keep it small for TinyML testing)
COUGHVID_SAMPLES = 60  # Will give ~40% of dataset
ESC50_HUMAN_SAMPLES = 30  # Will give ~30% of dataset
ESC50_BACKGROUND_SAMPLES = 30  # Will give ~30% of dataset

OUTPUT_DIR = Path(r"c:\HS\TML1\TinyML_Dataset")
COUGHVID_DIR = Path(r"c:\HS\TML1\public_dataset")
ESC50_AUDIO_DIR = Path(r"c:\HS\TML1\ESC-50-master\audio")
ESC50_META = Path(r"c:\HS\TML1\ESC-50-master\meta\esc50.csv")

# ESC-50 category mapping
ESC50_HUMAN_NOISE = [34, 35, 36, 37, 38]  # Laughing, Crying, Sneezing, Breathing, Coughing
ESC50_BACKGROUND = [0, 1, 2, 11, 12, 13, 14, 40, 41, 42, 43, 44, 45]  # Various ambient sounds

def normalize_audio(audio_path, sr=16000):
    """Load and normalize audio to 16kHz, 16-bit, mono"""
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if len(y) == 0:
            return None
        # Normalize amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        # Convert to 16-bit PCM
        y = np.int16(y * 32767)
        return y
    except Exception as e:
        print(f"  ⚠️ Error loading {audio_path.name}: {e}")
        return None

def create_windows(audio, sr=16000, window_size_ms=1000, overlap_ms=500):
    """Create overlapping windows"""
    window_samples = int(sr * window_size_ms / 1000)
    hop_samples = int(sr * (window_size_ms - overlap_ms) / 1000)
    
    windows = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        window = audio[start:start + window_samples]
        if len(window) == window_samples:
            windows.append(window)
    
    return windows

def extract_mfcc(window, sr=16000, n_mfcc=13):
    """Extract MFCC features"""
    window_float = window.astype(float) / 32768
    mfcc = librosa.feature.mfcc(
        y=window_float,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=160,
        n_mels=128
    )
    return mfcc

def process_files(file_list, class_name, max_files):
    """Process a list of files and create windows"""
    all_windows = []
    metadata = []
    
    # Sample random files
    sampled_files = random.sample(file_list, min(max_files, len(file_list)))
    
    print(f"\n📊 Processing {class_name} ({len(sampled_files)} files)...")
    
    for file_path in tqdm(sampled_files, desc=f"  {class_name}"):
        audio = normalize_audio(file_path, sr=SAMPLE_RATE)
        if audio is None:
            continue
        
        windows = create_windows(audio, sr=SAMPLE_RATE, 
                                window_size_ms=WINDOW_SIZE_MS,
                                overlap_ms=WINDOW_OVERLAP_MS)
        
        for i, window in enumerate(windows):
            all_windows.append(window)
            metadata.append({
                'source_file': file_path.name,
                'class': class_name,
                'window_id': i,
                'duration_ms': WINDOW_SIZE_MS,
                'sample_rate': SAMPLE_RATE
            })
    
    print(f"  ✓ Created {len(all_windows)} windows from {len(sampled_files)} files")
    return all_windows, metadata

# Step 1: Load ESC-50 metadata
print("\n📂 Step 1: Loading ESC-50 metadata...")
esc50_df = pd.read_csv(ESC50_META)
print(f"  ✓ Loaded {len(esc50_df)} entries")

# Step 2: Collect file lists
print("\n📂 Step 2: Collecting files...")

# COUGHVID files (for Cough class)
coughvid_files = list(COUGHVID_DIR.glob("*.wav"))
print(f"  ✓ Found {len(coughvid_files)} COUGHVID files")

# ESC-50 Human Noise files
human_noise_entries = esc50_df[esc50_df['target'].isin(ESC50_HUMAN_NOISE)]
human_noise_files = [ESC50_AUDIO_DIR / row['filename'] for _, row in human_noise_entries.iterrows()]
human_noise_files = [f for f in human_noise_files if f.exists()]
print(f"  ✓ Found {len(human_noise_files)} ESC-50 Human Noise files")

# ESC-50 Background files
background_entries = esc50_df[esc50_df['target'].isin(ESC50_BACKGROUND)]
background_files = [ESC50_AUDIO_DIR / row['filename'] for _, row in background_entries.iterrows()]
background_files = [f for f in background_files if f.exists()]
print(f"  ✓ Found {len(background_files)} ESC-50 Background files")

# Step 3: Process files
print("\n📊 Step 3: Processing audio files...")
random.seed(42)  # For reproducibility

cough_windows, cough_metadata = process_files(coughvid_files, "Cough", COUGHVID_SAMPLES)
human_windows, human_metadata = process_files(human_noise_files, "Human_Noise", ESC50_HUMAN_SAMPLES)
background_windows, background_metadata = process_files(background_files, "Background", ESC50_BACKGROUND_SAMPLES)

# Combine all data
all_windows = cough_windows + human_windows + background_windows
all_metadata = cough_metadata + human_metadata + background_metadata

print(f"\n✓ Total windows created: {len(all_windows)}")
print(f"  • Cough: {len(cough_windows)} ({len(cough_windows)/len(all_windows)*100:.1f}%)")
print(f"  • Human_Noise: {len(human_windows)} ({len(human_windows)/len(all_windows)*100:.1f}%)")
print(f"  • Background: {len(background_windows)} ({len(background_windows)/len(all_windows)*100:.1f}%)")

# Step 4: Create train/test split
print("\n📂 Step 4: Creating train/test split (80/20 stratified)...")
metadata_df = pd.DataFrame(all_metadata)

train_indices, test_indices = train_test_split(
    range(len(all_windows)),
    test_size=0.2,
    stratify=metadata_df['class'],
    random_state=42
)

print(f"  ✓ Train: {len(train_indices)} samples")
print(f"  ✓ Test: {len(test_indices)} samples")

# Step 5: Save files
print("\n📂 Step 5: Saving processed dataset...")

# Create directory structure
for class_name in ["Cough", "Human_Noise", "Background"]:
    (OUTPUT_DIR / class_name / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / class_name / "test").mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "features").mkdir(parents=True, exist_ok=True)

# Save audio files and extract features
sample_counter = {'Cough': 0, 'Human_Noise': 0, 'Background': 0}
saved_metadata = []

print("\n  Saving train set...")
for idx in tqdm(train_indices, desc="  Train"):
    window = all_windows[idx]
    meta = all_metadata[idx]
    class_name = meta['class']
    
    # Generate filename
    sample_id = sample_counter[class_name]
    filename = f"{class_name[0]}{sample_id:04d}.wav"
    sample_counter[class_name] += 1
    
    # Save audio
    output_path = OUTPUT_DIR / class_name / "train" / filename
    sf.write(output_path, window.astype(float) / 32768, SAMPLE_RATE)
    
    # Extract and save MFCC
    mfcc = extract_mfcc(window, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_path = OUTPUT_DIR / "features" / f"{filename.replace('.wav', '_mfcc.npy')}"
    np.save(mfcc_path, mfcc)
    
    # Save metadata
    saved_metadata.append({
        'filename': filename,
        'class': class_name,
        'split': 'train',
        'source_file': meta['source_file'],
        'window_id': meta['window_id'],
        'duration_ms': meta['duration_ms'],
        'sample_rate': meta['sample_rate'],
        'mfcc_file': mfcc_path.name
    })

print("\n  Saving test set...")
sample_counter = {'Cough': 0, 'Human_Noise': 0, 'Background': 0}
for idx in tqdm(test_indices, desc="  Test"):
    window = all_windows[idx]
    meta = all_metadata[idx]
    class_name = meta['class']
    
    # Generate filename
    sample_id = sample_counter[class_name]
    filename = f"{class_name[0]}_test_{sample_id:04d}.wav"
    sample_counter[class_name] += 1
    
    # Save audio
    output_path = OUTPUT_DIR / class_name / "test" / filename
    sf.write(output_path, window.astype(float) / 32768, SAMPLE_RATE)
    
    # Extract and save MFCC
    mfcc = extract_mfcc(window, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mfcc_path = OUTPUT_DIR / "features" / f"{filename.replace('.wav', '_mfcc.npy')}"
    np.save(mfcc_path, mfcc)
    
    # Save metadata
    saved_metadata.append({
        'filename': filename,
        'class': class_name,
        'split': 'test',
        'source_file': meta['source_file'],
        'window_id': meta['window_id'],
        'duration_ms': meta['duration_ms'],
        'sample_rate': meta['sample_rate'],
        'mfcc_file': mfcc_path.name
    })

# Save metadata CSV
metadata_df = pd.DataFrame(saved_metadata)
metadata_csv_path = OUTPUT_DIR / "metadata" / "dataset_metadata.csv"
metadata_df.to_csv(metadata_csv_path, index=False)
print(f"\n  ✓ Metadata saved: {metadata_csv_path}")

# Step 6: Generate summary statistics
print("\n" + "=" * 70)
print("📊 DATASET SUMMARY")
print("=" * 70)

print("\n📁 Directory Structure:")
print(f"  {OUTPUT_DIR}/")
for class_name in ["Cough", "Human_Noise", "Background"]:
    train_count = len(list((OUTPUT_DIR / class_name / "train").glob("*.wav")))
    test_count = len(list((OUTPUT_DIR / class_name / "test").glob("*.wav")))
    print(f"  ├── {class_name}/")
    print(f"  │   ├── train/ ({train_count} files)")
    print(f"  │   └── test/ ({test_count} files)")

feature_count = len(list((OUTPUT_DIR / "features").glob("*.npy")))
print(f"  ├── features/ ({feature_count} MFCC files)")
print(f"  └── metadata/ (1 CSV file)")

print("\n📊 Class Distribution:")
for split in ['train', 'test']:
    split_data = metadata_df[metadata_df['split'] == split]
    print(f"\n  {split.upper()} set ({len(split_data)} samples):")
    for class_name in ["Cough", "Human_Noise", "Background"]:
        count = len(split_data[split_data['class'] == class_name])
        pct = (count / len(split_data)) * 100
        print(f"    • {class_name}: {count} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("✅ TinyML Dataset Creation Complete!")
print("=" * 70)

print(f"\n📂 Dataset location: {OUTPUT_DIR}")
print(f"📊 Total samples: {len(metadata_df)}")
print(f"📊 Total size: ~{len(metadata_df) * 32000 / (1024*1024):.1f} MB (audio)")
print(f"\n🚀 Next steps:")
print(f"  1. Use this dataset to train a lightweight model")
print(f"  2. Model training should take 5-10 minutes")
print(f"  3. Perfect for testing TinyML deployment")
print(f"\n💡 To load the dataset:")
print(f"  metadata = pd.read_csv('{metadata_csv_path}')")
print(f"  # Use 'filename' and 'class' columns for training")
