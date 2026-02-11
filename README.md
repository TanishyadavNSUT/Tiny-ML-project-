# AeroGuard

Privacy-preserving on-device cough monitoring using TinyML.

## Overview

AeroGuard detects cough events directly on an ESP32 microcontroller using a lightweight 1D CNN. Audio is processed and classified on-device -- raw audio never leaves the hardware, ensuring complete privacy.

**Key results:**

| Metric | Value |
|--------|-------|
| Test accuracy | 96.68% |
| Model size (INT8) | 28 KB |
| Parameters | 13,219 |
| Inference latency | <150 ms |
| Hardware cost | ~500 INR |

**Classes:** Cough, Human Noise, Background

## Hardware

| Component | Part | Interface |
|-----------|------|-----------|
| Microcontroller | ESP32-WROOM-32 | -- |
| Microphone | INMP441 | I2S |
| Display (optional) | SSD1306 OLED | I2C |

### Wiring (INMP441 to ESP32)

| INMP441 | ESP32 | Function |
|---------|-------|----------|
| VDD | 3.3V | Power |
| GND | GND | Ground |
| SCK | GPIO 14 | I2S Clock |
| SD | GPIO 32 | I2S Data |
| WS | GPIO 15 | I2S Word Select |
| L/R | GND | Left channel |

## Pipeline

1. **Normalise** -- Resample to 16 kHz mono, 16-bit PCM
2. **Window** -- 1-second sliding windows, 500 ms overlap
3. **MFCC** -- 13 coefficients, FFT size 512, hop 160 -> shape (101, 13)
4. **Split** -- 80/20 stratified train/test (1,082 / 271 samples)
5. **Train** -- 1D CNN, Adam optimiser, 34 epochs with early stopping
6. **Quantise** -- TFLite INT8 for ESP32 deployment

## Model Architecture

```
Input (101, 13)
  -> Conv1D(32) + BatchNorm + MaxPool(4)
  -> Conv1D(64) + BatchNorm + MaxPool(4)
  -> Conv1D(128) + GlobalAvgPool
  -> Dense(128) + Dropout(0.3)
  -> Dense(3, softmax)
```

## Project Structure

```
.
├── config.py               # Project configuration
├── create_tiny_dataset.py   # Dataset creation from raw audio
├── data_processor.py        # Audio preprocessing and MFCC extraction
├── data_analysis.ipynb      # Exploratory data analysis notebook
├── train_model.py           # Model training and export
├── demo_pipeline.py         # End-to-end demo
├── report.pdf               # Project report
├── models/
│   ├── best_model.keras
│   ├── aeroguard_model.tflite
│   ├── aeroguard_model_quantized.tflite
│   ├── label_mapping.json
│   ├── training_report.json
│   ├── training_history.png
│   └── confusion_matrix.png
└── TinyML_Dataset/
    ├── Cough/
    ├── Human_Noise/
    ├── Background/
    ├── features/
    └── metadata/
```

## Setup

```bash
pip install librosa soundfile numpy pandas scikit-learn tensorflow tqdm
```

## Usage

```bash
# Create dataset from raw audio
python create_tiny_dataset.py

# Process audio and extract features
python data_processor.py

# Train model
python train_model.py

# Run demo
python demo_pipeline.py
```

## Model Outputs

| File | Size | Purpose |
|------|------|---------|
| best_model.keras | 51.6 KB | Full-precision model |
| aeroguard_model.tflite | 59.7 KB | TFLite float32 |
| aeroguard_model_quantized.tflite | 28.0 KB | TFLite INT8 (deployment) |

## References

- Orlandini et al., "COUGHVID: A Large-Scale Machine Learning Study on Cough Detection and Diagnosis," 2021
- Piczak et al., "ESC: Dataset for Environmental Sound Classification," 2015
- Warden & Situnayake, "TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers," 2019

## License

CC BY 4.0 -- Free for academic and non-commercial use.
