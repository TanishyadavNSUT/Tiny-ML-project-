"""
AeroGuard Data Pipeline Demonstration
Quick demo showing the complete audio processing pipeline
"""

import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("=" * 70)
print("🎯 AeroGuard Data Pipeline Demonstration")
print("=" * 70)

# Step 1: Load and normalize audio
print("\n📊 Step 1: Loading sample audio file...")
sample_file = Path(r"c:\HS\TML1\ESC-50-master\audio\1-100032-A-0.wav")
audio, sr = librosa.load(sample_file, sr=16000, mono=True)
print(f"✓ Loaded: {sample_file.name}")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration: {len(audio) / sr:.2f} seconds")
print(f"  Original shape: {audio.shape}")

# Normalize
if np.max(np.abs(audio)) > 0:
    audio = audio / np.max(np.abs(audio))
audio_16bit = np.int16(audio * 32767)
print(f"✓ Normalized to 16-bit PCM")
print(f"  Value range: [{np.min(audio_16bit)}, {np.max(audio_16bit)}]")

# Step 2: Create windows
print("\n📊 Step 2: Creating 1-second windows with 500ms overlap...")
window_size_ms = 1000
overlap_ms = 500
window_samples = int(sr * window_size_ms / 1000)
hop_samples = int(sr * (window_size_ms - overlap_ms) / 1000)

windows = []
for start in range(0, len(audio_16bit) - window_samples + 1, hop_samples):
    window = audio_16bit[start:start + window_samples]
    if len(window) == window_samples:
        windows.append(window)

print(f"✓ Created {len(windows)} windows")
print(f"  Window size: {window_samples} samples ({window_size_ms}ms)")
print(f"  Hop size: {hop_samples} samples ({overlap_ms}ms overlap)")

# Step 3: Extract MFCC from first window
print("\n📊 Step 3: Extracting MFCC features from first window...")
n_mfcc = 13
window_float = windows[0].astype(float) / 32768
mfcc = librosa.feature.mfcc(
    y=window_float,
    sr=sr,
    n_mfcc=n_mfcc,
    n_fft=512,
    hop_length=160,
    n_mels=128
)
print(f"✓ MFCC extracted")
print(f"  Shape: {mfcc.shape} (13 coefficients × {mfcc.shape[1]} time frames)")
print(f"  Value range: [{np.min(mfcc):.2f}, {np.max(mfcc):.2f}]")

# Step 4: Dataset statistics
print("\n📊 Step 4: Dataset inventory...")
coughvid_dir = Path(r"c:\HS\TML1\public_dataset")
esc50_dir = Path(r"c:\HS\TML1\ESC-50-master\audio")

coughvid_files = list(coughvid_dir.glob("*.wav"))
esc50_files = list(esc50_dir.glob("*.wav"))

print(f"✓ COUGHVID: {len(coughvid_files)} audio files")
print(f"✓ ESC-50: {len(esc50_files)} audio files")
print(f"✓ Total: {len(coughvid_files) + len(esc50_files)} audio files available")

# Step 5: Expected output
print("\n📊 Step 5: Expected processing results...")
print(f"✓ Target class distribution:")
print(f"  • Cough: 40%")
print(f"  • Human_Noise: 30%")
print(f"  • Background: 30%")
print(f"✓ Train/test split: 80/20 (stratified)")
print(f"✓ Expected windows per file: ~{len(windows)} (for 5-second audio)")
print(f"✓ Expected total windows: ~2,000-3,000 (after processing)")

print("\n" + "=" * 70)
print("✅ Pipeline demonstration complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Run the full AeroGuard_DataProcessor.py to process all files")
print("2. This will create the Project_AeroGuard_Data/ directory structure")
print("3. Then train the model using the processed data")
print("\nNote: Full processing will take 4-8 hours depending on your hardware.")
