"""
AeroGuard Configuration File
Centralized settings for the entire project
"""

# ============================================================================
# AUDIO PROCESSING PARAMETERS
# ============================================================================

SAMPLE_RATE = 16000  # Hz - standard for ESP32 and speech recognition
BIT_DEPTH = 16  # PCM 16-bit signed integer
CHANNELS = 1  # Mono only (reduces processing)

# Windowing Parameters
WINDOW_SIZE_MS = 1000  # 1 second (captures full cough duration)
WINDOW_OVERLAP_MS = 500  # 50% overlap (ensures no edge-case misses)

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================

# MFCC (Mel-Frequency Cepstral Coefficient) settings
N_MFCC = 13  # Number of coefficients (12-13 is standard)
N_FFT = 512  # FFT window size
HOP_LENGTH = 160  # Samples between successive frames
N_MELS = 128  # Number of mel bands

# ============================================================================
# DATASET PARAMETERS
# ============================================================================

# Dataset composition ratios
DATASET_SPLIT = {
    'Cough': 0.40,       # 40% - COUGHVID dataset
    'Human_Noise': 0.30, # 30% - ESC-50 (sneezes, laughs, breathing)
    'Background': 0.30,  # 30% - ESC-50 (ambient, mechanical)
}

# Train/Test split
TRAIN_TEST_SPLIT = 0.80  # 80% training, 20% validation

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Model architecture
MODEL_TYPE = 'CNN1D'  # 1D Convolutional Neural Network

CONV_LAYERS = [
    {'filters': 32, 'kernel_size': 3},
    {'filters': 64, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
]

DENSE_LAYERS = [
    {'units': 128, 'activation': 'relu', 'dropout': 0.3},
]

NUM_CLASSES = 3  # [Cough, Human_Noise, Background]

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'

# Early stopping
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs

# ============================================================================
# ESP32 DEPLOYMENT PARAMETERS
# ============================================================================

ESP32_FLASH_SIZE = '4MB'  # ESP32 flash memory
ESP32_RAM = 520  # KB available for model + buffers
MAX_MODEL_SIZE = 1200  # KB - must fit in SPRAM

# I2S Configuration
I2S_SAMPLE_RATE = 16000  # Hz
I2S_BUFFER_SIZE = 4096  # Samples per buffer

# Microphone Pinout (INMP441 → ESP32)
I2S_SCK_PIN = 14   # Serial Clock
I2S_SD_PIN = 32    # Serial Data
I2S_WS_PIN = 15    # Word Select (LRCLK)

# Display Settings (optional)
OLED_SDA = 21
OLED_SCL = 22
OLED_WIDTH = 128
OLED_HEIGHT = 64

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

# Confidence threshold
DETECTION_THRESHOLD = 0.70  # Only count if model is >70% confident

# Cough counting
COUGH_SMOOTHING = 3  # Require 3 consecutive detections to count as 1 cough
DEBOUNCE_MS = 500  # Minimum time between counted coughs

# ============================================================================
# DATA PATHS
# ============================================================================

# Input datasets
COUGHVID_PATH = 'public_dataset'
ESC50_PATH = 'ESC-50-master'

# Output paths
PROCESSED_DATA_PATH = 'Project_AeroGuard_Data'
MODELS_PATH = 'models'
LOGS_PATH = 'logs'

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'aeroguard.log'

# Save training history plots
SAVE_PLOTS = True
PLOT_FORMAT = 'png'  # jpg, png, svg

# Save model checkpoints
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQ = 5  # Save every N epochs

# ============================================================================
# EDGE IMPULSE SETTINGS (optional)
# ============================================================================

# If using Edge Impulse for cloud training/deployment
EDGE_IMPULSE_API_KEY = ''  # Get from Edge Impulse dashboard
EDGE_IMPULSE_PROJECT_ID = ''  # Your project ID

# ============================================================================
# HARDWARE BENCHMARKS (Expected values)
# ============================================================================

EXPECTED_ACCURACY = 0.90  # >90% on test set
EXPECTED_INFERENCE_TIME_MS = 120  # ~120ms per window on ESP32
EXPECTED_MODEL_SIZE_KB = 800  # ~800 KB quantized

# ============================================================================
# AUGMENTATION SETTINGS (Optional for better generalization)
# ============================================================================

AUGMENTATION_ENABLED = False  # Enable data augmentation

AUGMENTATION_PARAMS = {
    'time_stretch_rate': (0.85, 1.15),  # Speed up/slow down
    'pitch_shift_semitones': (-2, 2),    # Shift pitch
    'gaussian_noise_std': 0.001,         # Add white noise
}

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

# Cross-validation (if using)
K_FOLDS = 5  # 5-fold cross-validation

# Metric thresholds for "passing" validation
MIN_ACCURACY = 0.88
MIN_PRECISION = 0.87
MIN_RECALL = 0.87
MIN_F1_SCORE = 0.87

print("""
✓ AeroGuard Configuration loaded
  • Sample Rate: 16 kHz
  • Window: 1000ms @ 500ms overlap
  • Features: 13 MFCC
  • Dataset: 40/30/30 (Cough/Human/Background)
  • Model: 1D CNN (32→64→128 filters)
  • Target: >90% accuracy on ESP32
""")
