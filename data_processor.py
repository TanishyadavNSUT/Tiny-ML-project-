"""
AeroGuard: Privacy-Preserving On-Device Cough Monitoring via TinyML
Data Preprocessing and Structuring Script

This script processes COUGHVID and ESC-50 datasets for TinyML model training.
It implements:
- Audio normalization (16kHz, 16-bit, mono)
- 1-second windowing with 500ms overlap
- MFCC feature extraction
- Proper train/test splitting (80/20)
- Metadata generation for Edge Impulse compatibility
"""

import os
import json
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AeroGuardDataProcessor:
    """
    Main data processor for AeroGuard project.
    """
    
    # Configuration parameters
    CONFIG = {
        'sample_rate': 16000,  # 16 kHz for ESP32
        'window_size_ms': 1000,  # 1 second
        'overlap_ms': 500,  # 50% overlap
        'n_mfcc': 13,  # MFCC features
        'bit_depth': 16,  # 16-bit PCM
        'channels': 1,  # Mono only
    }
    
    # Dataset splits: 40% Cough, 30% Human Noise, 30% Background
    DATASET_SPLIT = {
        'Cough': 0.40,
        'Human_Noise': 0.30,
        'Background': 0.30,
    }
    
    # ESC-50 category mapping
    ESC50_MAPPING = {
        # Cough class
        0: 'Cough',  # "dog bark" maps to noise
        1: 'Cough',  # "rooster" maps to noise
        
        # Human/Sneeze
        34: 'Human_Noise',  # "sneezing"
        35: 'Human_Noise',  # "laughing"
        36: 'Human_Noise',  # "crying baby"
        37: 'Human_Noise',  # "snoring"
        26: 'Human_Noise',  # "breathing" -> human noise
        
        # Background/Ambient
        2: 'Background',   # "pig oinking" -> noise
        3: 'Background',   # "cow mooing" -> noise
        4: 'Background',   # "frog croaking" -> noise
        40: 'Background',  # "door wood knock" -> background
        41: 'Background',  # "door metal knock" -> background
        42: 'Background',  # "door open/close" -> background
        43: 'Background',  # "chain saw" -> background
        44: 'Background',  # "siren" -> background
        45: 'Background',  # "car horn" -> background
        46: 'Background',  # "engine" -> background
        47: 'Background',  # "train" -> background
        48: 'Background',  # "church bells" -> background
        49: 'Background',  # "alarm clock" -> background
    }
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Root directory containing public_dataset and ESC-50-master
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.coughvid_dir = self.data_dir / 'public_dataset'
        self.esc50_dir = self.data_dir / 'ESC-50-master'
        
        # Create output structure
        self._create_output_structure()
        
    def _create_output_structure(self):
        """Create the desired output folder structure."""
        self.processed_dir = self.output_dir / 'Project_AeroGuard_Data'
        
        # Main class directories
        for class_name in self.DATASET_SPLIT.keys():
            (self.processed_dir / class_name / 'train').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / class_name / 'test').mkdir(parents=True, exist_ok=True)
        
        # Feature storage
        (self.processed_dir / 'features').mkdir(parents=True, exist_ok=True)
        (self.processed_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output structure created at: {self.processed_dir}")
        
    def normalize_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and normalize audio to 16kHz, 16-bit, mono.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Normalized audio array
        """
        try:
            # Load audio with target sample rate
            y, sr = librosa.load(audio_path, sr=self.CONFIG['sample_rate'], mono=True)
            
            # Normalize amplitude
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            
            # Convert to 16-bit
            y = np.int16(y * 32767)
            
            return y
        except Exception as e:
            print(f"  ⚠ Error processing {audio_path}: {str(e)}")
            return None
    
    def create_windows(self, audio: np.ndarray, sr: int = 16000) -> List[np.ndarray]:
        """
        Create 1-second windows with 500ms overlap.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of window arrays
        """
        window_samples = int(self.CONFIG['window_size_ms'] * sr / 1000)
        overlap_samples = int(self.CONFIG['overlap_ms'] * sr / 1000)
        
        windows = []
        start = 0
        
        while start + window_samples <= len(audio):
            window = audio[start:start + window_samples]
            if len(window) == window_samples:
                windows.append(window)
            start += overlap_samples
        
        return windows
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio array (16-bit PCM)
            
        Returns:
            MFCC feature matrix
        """
        # Convert to float32 for librosa
        audio_float = audio.astype(np.float32) / 32768.0
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio_float,
            sr=self.CONFIG['sample_rate'],
            n_mfcc=self.CONFIG['n_mfcc'],
            n_fft=512,
            hop_length=160
        )
        
        return mfcc
    
    def process_coughvid(self):
        """
        Process COUGHVID dataset.
        """
        print("\n" + "="*60)
        print("PROCESSING COUGHVID DATASET")
        print("="*60)
        
        wav_files = list(self.coughvid_dir.glob('*.wav'))
        print(f"Found {len(wav_files)} COUGHVID audio files")
        
        metadata_list = []
        processed_count = 0
        window_count = 0
        
        for idx, wav_file in enumerate(wav_files):
            json_file = wav_file.with_suffix('.json')
            
            # Load JSON metadata
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
            except:
                continue
            
            # Normalize audio
            audio = self.normalize_audio(str(wav_file))
            if audio is None:
                continue
            
            # Create windows
            windows = self.create_windows(audio)
            if not windows:
                continue
            
            # Process each window
            for window_idx, window in enumerate(windows):
                # Extract MFCC
                mfcc = self.extract_mfcc(window)
                
                # Generate unique filename
                filename = f"{wav_file.stem}_window_{window_idx:03d}"
                
                # Save to train/test with 80/20 split
                split = 'train' if np.random.rand() < 0.8 else 'test'
                
                # Save raw audio
                wav_output = self.processed_dir / 'Cough' / split / f"{filename}.wav"
                sf.write(
                    str(wav_output),
                    window,
                    self.CONFIG['sample_rate'],
                    subtype='PCM_16'
                )
                
                # Save MFCC features
                mfcc_output = self.processed_dir / 'features' / f"{filename}_mfcc.npy"
                np.save(mfcc_output, mfcc)
                
                # Collect metadata
                metadata_list.append({
                    'filename': filename,
                    'window_index': window_idx,
                    'class': 'Cough',
                    'split': split,
                    'source_file': wav_file.name,
                    'source_dataset': 'COUGHVID',
                    'audio_duration_ms': self.CONFIG['window_size_ms'],
                    'sample_rate': self.CONFIG['sample_rate'],
                    'bit_depth': self.CONFIG['bit_depth'],
                    'channels': self.CONFIG['channels'],
                    'cough_confidence': metadata.get('cough_detected', 0),
                    'cough_status': metadata.get('status', 'unknown'),
                })
                
                window_count += 1
            
            processed_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  ✓ Processed {idx + 1}/{len(wav_files)} COUGHVID files ({window_count} windows created)")
        
        print(f"\n✓ COUGHVID Complete: {processed_count} files → {window_count} windows")
        
        return metadata_list
    
    def process_esc50(self):
        """
        Process ESC-50 dataset for Human Noise and Background classes.
        """
        print("\n" + "="*60)
        print("PROCESSING ESC-50 DATASET")
        print("="*60)
        
        # Load ESC-50 metadata
        csv_file = self.esc50_dir / 'meta' / 'esc50.csv'
        df = pd.read_csv(csv_file)
        
        print(f"Found {len(df)} ESC-50 entries")
        
        metadata_list = []
        processed_count = 0
        window_count = 0
        
        for _, row in df.iterrows():
            # Get target class
            target = int(row['target'])
            
            # Map to our classes
            if target not in self.ESC50_MAPPING:
                # Skip unmapped categories
                continue
            
            class_label = self.ESC50_MAPPING[target]
            
            # Load audio
            audio_file = self.esc50_dir / 'audio' / row['filename']
            if not audio_file.exists():
                continue
            
            audio = self.normalize_audio(str(audio_file))
            if audio is None:
                continue
            
            # Create windows
            windows = self.create_windows(audio)
            if not windows:
                continue
            
            # Process each window
            for window_idx, window in enumerate(windows):
                # Extract MFCC
                mfcc = self.extract_mfcc(window)
                
                # Generate unique filename
                filename = f"{audio_file.stem}_window_{window_idx:03d}"
                
                # Save to train/test with 80/20 split
                split = 'train' if np.random.rand() < 0.8 else 'test'
                
                # Save raw audio
                wav_output = self.processed_dir / class_label / split / f"{filename}.wav"
                sf.write(
                    str(wav_output),
                    window,
                    self.CONFIG['sample_rate'],
                    subtype='PCM_16'
                )
                
                # Save MFCC features
                mfcc_output = self.processed_dir / 'features' / f"{filename}_mfcc.npy"
                np.save(mfcc_output, mfcc)
                
                # Collect metadata
                metadata_list.append({
                    'filename': filename,
                    'window_index': window_idx,
                    'class': class_label,
                    'split': split,
                    'source_file': row['filename'],
                    'source_dataset': 'ESC-50',
                    'esc_target': target,
                    'esc_category': row['target_names'],
                    'audio_duration_ms': self.CONFIG['window_size_ms'],
                    'sample_rate': self.CONFIG['sample_rate'],
                    'bit_depth': self.CONFIG['bit_depth'],
                    'channels': self.CONFIG['channels'],
                })
                
                window_count += 1
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"  ✓ Processed {processed_count} ESC-50 files ({window_count} windows created)")
        
        print(f"\n✓ ESC-50 Complete: {processed_count} files → {window_count} windows")
        
        return metadata_list
    
    def create_metadata_csv(self, all_metadata: List[Dict]):
        """
        Create comprehensive metadata CSV for all samples.
        
        Args:
            all_metadata: List of metadata dictionaries
        """
        df = pd.DataFrame(all_metadata)
        
        # Save full metadata
        metadata_file = self.processed_dir / 'metadata' / 'dataset_metadata.csv'
        df.to_csv(metadata_file, index=False)
        
        print(f"\n✓ Metadata saved: {metadata_file}")
        
        # Print statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        print("\n📊 Class Distribution:")
        for class_name in self.DATASET_SPLIT.keys():
            count = len(df[df['class'] == class_name])
            percentage = (count / len(df)) * 100
            print(f"  • {class_name}: {count} windows ({percentage:.1f}%)")
        
        print("\n🔄 Train/Test Split:")
        for split in ['train', 'test']:
            count = len(df[df['split'] == split])
            percentage = (count / len(df)) * 100
            print(f"  • {split.upper()}: {count} windows ({percentage:.1f}%)")
        
        print("\n📚 Source Dataset Distribution:")
        for source in df['source_dataset'].unique():
            count = len(df[df['source_dataset'] == source])
            print(f"  • {source}: {count} windows")
        
        print("\n✓ Total Samples: {} windows ({:.1f} minutes)".format(
            len(df),
            len(df) * self.CONFIG['window_size_ms'] / 60000
        ))
    
    def create_edge_impulse_manifest(self, all_metadata: List[Dict]):
        """
        Create Edge Impulse compatible manifest for easy upload.
        
        Args:
            all_metadata: List of metadata dictionaries
        """
        manifest = {
            "version": 1,
            "files": []
        }
        
        for metadata in all_metadata:
            split = metadata['split']
            class_label = metadata['class']
            filename = metadata['filename']
            
            file_entry = {
                "path": f"Project_AeroGuard_Data/{class_label}/{split}/{filename}.wav",
                "hash": "",
                "uploaded": False,
                "category": split,
                "label": class_label
            }
            
            manifest["files"].append(file_entry)
        
        manifest_file = self.processed_dir / 'metadata' / 'edge_impulse_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Edge Impulse manifest created: {manifest_file}")
    
    def run(self):
        """Execute the complete data processing pipeline."""
        print("\n" + "="*70)
        print("   AeroGuard: TinyML Cough Detection - Data Preprocessing")
        print("="*70)
        
        # Process datasets
        coughvid_metadata = self.process_coughvid()
        esc50_metadata = self.process_esc50()
        
        # Combine all metadata
        all_metadata = coughvid_metadata + esc50_metadata
        
        # Create output files
        self.create_metadata_csv(all_metadata)
        self.create_edge_impulse_manifest(all_metadata)
        
        print("\n" + "="*70)
        print("✅ DATA PROCESSING COMPLETE!")
        print("="*70)
        print(f"\n📁 Output Directory: {self.processed_dir}")
        print("\nNext Steps:")
        print("1. Review metadata at: Project_AeroGuard_Data/metadata/dataset_metadata.csv")
        print("2. Upload to Edge Impulse or train locally")
        print("3. Convert model to TensorFlow Lite for ESP32")
        print("4. Deploy using AeroGuard_ESP32_Firmware.ino")


def main():
    """Main entry point."""
    # Configuration
    DATA_DIR = r"c:\HS\TML1"  # Root directory with datasets
    OUTPUT_DIR = r"c:\HS\TML1\processed"  # Where to save processed data
    
    # Create processor
    processor = AeroGuardDataProcessor(DATA_DIR, OUTPUT_DIR)
    
    # Run processing
    processor.run()


if __name__ == "__main__":
    main()
