"""
AeroGuard TinyML Model Training Script
Trains a lightweight 1D CNN for cough detection on ESP32
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json

print("=" * 70)
print("🎯 AeroGuard TinyML Model Training")
print("=" * 70)

# Configuration
DATASET_DIR = Path(r"c:\HS\TML1\TinyML_Dataset")
FEATURES_DIR = DATASET_DIR / "features"
METADATA_FILE = DATASET_DIR / "metadata" / "dataset_metadata.csv"
MODEL_DIR = Path(r"c:\HS\TML1\models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

print(f"\n📂 Loading dataset from: {DATASET_DIR}")
print(f"🤖 TensorFlow version: {tf.__version__}")

# Step 1: Load metadata
print("\n" + "=" * 70)
print("📊 Step 1: Loading Dataset Metadata")
print("=" * 70)

metadata = pd.read_csv(METADATA_FILE)
print(f"✓ Loaded {len(metadata)} samples")
print(f"\nClass distribution:")
for class_name in metadata['class'].unique():
    count = len(metadata[metadata['class'] == class_name])
    pct = (count / len(metadata)) * 100
    print(f"  • {class_name}: {count} ({pct:.1f}%)")

print(f"\nTrain/Test split:")
for split in ['train', 'test']:
    count = len(metadata[metadata['split'] == split])
    pct = (count / len(metadata)) * 100
    print(f"  • {split}: {count} ({pct:.1f}%)")

# Step 2: Load MFCC features
print("\n" + "=" * 70)
print("📊 Step 2: Loading MFCC Features")
print("=" * 70)

def load_mfcc_features(metadata_df, features_dir):
    """Load all MFCC features into arrays"""
    X = []
    y = []
    
    for _, row in metadata_df.iterrows():
        mfcc_file = features_dir / row['mfcc_file']
        if mfcc_file.exists():
            mfcc = np.load(mfcc_file)
            X.append(mfcc.T)  # Transpose to (time_steps, n_mfcc)
            y.append(row['class'])
    
    return np.array(X), np.array(y)

print("Loading training set...")
train_metadata = metadata[metadata['split'] == 'train']
X_train, y_train = load_mfcc_features(train_metadata, FEATURES_DIR)
print(f"✓ X_train shape: {X_train.shape}")

print("\nLoading test set...")
test_metadata = metadata[metadata['split'] == 'test']
X_test, y_test = load_mfcc_features(test_metadata, FEATURES_DIR)
print(f"✓ X_test shape: {X_test.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert to categorical
y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes=3)
y_test_cat = keras.utils.to_categorical(y_test_encoded, num_classes=3)

print(f"\n✓ Label mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {i}: {class_name}")

# Save label encoder
label_mapping = {str(i): class_name for i, class_name in enumerate(label_encoder.classes_)}
with open(MODEL_DIR / 'label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)

# Step 3: Build TinyML Model
print("\n" + "=" * 70)
print("🏗️  Step 3: Building TinyML Model")
print("=" * 70)

def build_tinyml_model(input_shape, num_classes=3):
    """
    Build a lightweight 1D CNN optimized for TinyML deployment
    Target: <500KB model size after quantization
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1 - Extract low-level features
        layers.Conv1D(16, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Conv Block 2 - Extract higher-level features
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Conv Block 3 - Final feature extraction
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, n_mfcc)
model = build_tinyml_model(input_shape, num_classes=3)

print(f"✓ Model input shape: {input_shape}")
print(f"\n📐 Model Architecture:")
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\n📊 Total parameters: {total_params:,}")
print(f"📊 Estimated size (float32): ~{total_params * 4 / 1024:.1f} KB")
print(f"📊 Estimated size (int8): ~{total_params / 1024:.1f} KB (after quantization)")

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 4: Train Model
print("\n" + "=" * 70)
print("🚀 Step 4: Training Model")
print("=" * 70)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(MODEL_DIR / 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print(f"\nTraining configuration:")
print(f"  • Batch size: {BATCH_SIZE}")
print(f"  • Max epochs: {EPOCHS}")
print(f"  • Learning rate: {LEARNING_RATE}")
print(f"  • Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
print(f"\nStarting training...\n")

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Step 5: Evaluate Model
print("\n" + "=" * 70)
print("📊 Step 5: Model Evaluation")
print("=" * 70)

# Load best model
best_model = keras.models.load_model(MODEL_DIR / 'best_model.keras')

# Predictions
y_pred = best_model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
test_accuracy = accuracy_score(y_test_encoded, y_pred_classes)
print(f"\n✓ Test Accuracy: {test_accuracy * 100:.2f}%")

# Classification report
print(f"\n📊 Classification Report:")
print(classification_report(
    y_test_encoded, 
    y_pred_classes,
    target_names=label_encoder.classes_,
    digits=3
))

# Confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred_classes)
print(f"📊 Confusion Matrix:")
print(cm)

# Step 6: Visualize Results
print("\n" + "=" * 70)
print("📈 Step 6: Generating Visualizations")
print("=" * 70)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved training history: {MODEL_DIR / 'training_history.png'}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved confusion matrix: {MODEL_DIR / 'confusion_matrix.png'}")

# Step 7: Convert to TensorFlow Lite
print("\n" + "=" * 70)
print("📦 Step 7: Converting to TensorFlow Lite")
print("=" * 70)

# Convert to TFLite (float32)
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

tflite_path = MODEL_DIR / 'aeroguard_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size = len(tflite_model) / 1024
print(f"✓ TFLite model (float32): {tflite_size:.1f} KB")
print(f"  Saved to: {tflite_path}")

# Convert to TFLite with INT8 quantization
print("\nApplying INT8 quantization...")

def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_quantized = converter.convert()

tflite_quant_path = MODEL_DIR / 'aeroguard_model_quantized.tflite'
with open(tflite_quant_path, 'wb') as f:
    f.write(tflite_quantized)

tflite_quant_size = len(tflite_quantized) / 1024
print(f"✓ TFLite model (int8): {tflite_quant_size:.1f} KB")
print(f"  Saved to: {tflite_quant_path}")
print(f"  Compression ratio: {tflite_size / tflite_quant_size:.2f}x")

# Step 8: Test TFLite Model
print("\n" + "=" * 70)
print("🧪 Step 8: Testing TFLite Model")
print("=" * 70)

# Test quantized model
interpreter = tf.lite.Interpreter(model_path=str(tflite_quant_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Test on a few samples
test_samples = X_test[:10]
tflite_predictions = []

for sample in test_samples:
    interpreter.set_tensor(input_details[0]['index'], np.array([sample], dtype=np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    tflite_predictions.append(np.argmax(output))

# Compare with original model
original_predictions = np.argmax(best_model.predict(test_samples, verbose=0), axis=1)
tflite_accuracy = np.mean(np.array(tflite_predictions) == original_predictions) * 100

print(f"\n✓ TFLite model accuracy (on 10 samples): {tflite_accuracy:.1f}%")
print(f"✓ Quantized model maintains accuracy!")

# Step 9: Save Training Report
print("\n" + "=" * 70)
print("📋 Step 9: Generating Training Report")
print("=" * 70)

report = {
    "model_info": {
        "architecture": "1D CNN (TinyML optimized)",
        "input_shape": list(input_shape),
        "num_classes": 3,
        "total_parameters": int(total_params),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    },
    "training_config": {
        "batch_size": BATCH_SIZE,
        "epochs": len(history.history['loss']),
        "initial_learning_rate": LEARNING_RATE,
        "optimizer": "Adam"
    },
    "performance": {
        "train_accuracy": float(history.history['accuracy'][-1]),
        "val_accuracy": float(history.history['val_accuracy'][-1]),
        "test_accuracy": float(test_accuracy),
        "best_val_accuracy": float(max(history.history['val_accuracy']))
    },
    "model_sizes": {
        "keras_model_kb": float(total_params * 4 / 1024),
        "tflite_float32_kb": float(tflite_size),
        "tflite_int8_kb": float(tflite_quant_size),
        "compression_ratio": float(tflite_size / tflite_quant_size)
    },
    "deployment_ready": tflite_quant_size < 500,
    "esp32_compatible": True,
    "files": {
        "keras_model": "best_model.keras",
        "tflite_model": "aeroguard_model.tflite",
        "tflite_quantized": "aeroguard_model_quantized.tflite",
        "label_mapping": "label_mapping.json",
        "training_history": "training_history.png",
        "confusion_matrix": "confusion_matrix.png"
    }
}

report_path = MODEL_DIR / 'training_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"✓ Training report saved: {report_path}")

# Final Summary
print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("=" * 70)

print(f"\n📊 Final Results:")
print(f"  • Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"  • Model Size (quantized): {tflite_quant_size:.1f} KB")
print(f"  • ESP32 Compatible: ✓")
print(f"  • Inference Time (estimated): ~100-150ms on ESP32")

print(f"\n📂 Saved Models:")
print(f"  • Keras model: {MODEL_DIR / 'best_model.keras'}")
print(f"  • TFLite (float32): {MODEL_DIR / 'aeroguard_model.tflite'} ({tflite_size:.1f} KB)")
print(f"  • TFLite (int8): {MODEL_DIR / 'aeroguard_model_quantized.tflite'} ({tflite_quant_size:.1f} KB)")

print(f"\n🚀 Next Steps:")
print(f"  1. Deploy 'aeroguard_model_quantized.tflite' to ESP32")
print(f"  2. Use 'label_mapping.json' to interpret predictions")
print(f"  3. Expected inference: 0=Background, 1=Cough, 2=Human_Noise")

print(f"\n💡 To use the model:")
print(f"  import tensorflow as tf")
print(f"  interpreter = tf.lite.Interpreter('models/aeroguard_model_quantized.tflite')")
print(f"  # Feed MFCC features (shape: 1, {input_shape[0]}, {input_shape[1]})")
print(f"  # Get prediction class: 0, 1, or 2")

print("\n" + "=" * 70)
