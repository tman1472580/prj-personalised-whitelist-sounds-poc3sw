"""
Yamnet-256 Training Script for FSD50K Dataset
Replicates STM32 approach: backbone reduction + pruning + int8 quantization
"""

import os
import json
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow_model_optimization.quantization.keras import quantize_model
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset_root': '/Users/tanzimmohammad/Documents/GitHub/FSD50K Dataset/root',
    'target_classes': [
        "Accelerating_and_revving_and_vroom",
        "Aircraft",
        "Fire",
        "Crying_and_sobbing",
        "Dog",
        "Rain",
        "Chicken_and_rooster",
        "Waves_and_surf",
        "Sneeze",
        "Clock",
        "Child_speech_and_kid_speaking"
    ],
    'sr': 16000,  # Sample rate
    'fft_size': 400,  # 25ms at 16kHz
    'hop_length': 160,  # 10ms at 16kHz
    'n_mels': 64,
    'n_frames': 96,
    'mel_min_hz': 125,
    'mel_max_hz': 7500,
    'batch_size': 32,
    'epochs': 1,
    'learning_rate': 0.001,
    'output_dir': './yamnet256_model',
}

# ============================================================================
# STEP 1: DATA LOADING & PREPROCESSING
# ============================================================================

class FSD50KDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset_root = Path(config['dataset_root'])
        self.class_map = self._create_class_mapping()
        
    def _create_class_mapping(self):
        """Map target classes to indices"""
        return {cls: idx for idx, cls in enumerate(self.config['target_classes'])}
    
    def _read_ground_truth(self, csv_path):
        """Read FSD50K ground truth CSV"""
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def _normalize_class_name(self, name):
        """Normalize class names for comparison"""
        return name.strip().lower()
    
    def load_audio_file(self, file_path):
        """Load and resample audio file using soundfile"""
        try:
            # Load with soundfile (works with Python 3.13)
            audio, sr = sf.read(str(file_path))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.config['sr']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sr'])
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def create_mel_spectrogram(self, audio):
        """Create mel-spectrogram matching original Yamnet preprocessing"""
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config['sr'],
            n_fft=self.config['fft_size'],
            hop_length=self.config['hop_length'],
            n_mels=self.config['n_mels'],
            fmin=self.config['mel_min_hz'],
            fmax=self.config['mel_max_hz'],
            power=2.0
        )
        # Convert to dB scale
        S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
        return S_db
    
    def extract_patches(self, mel_spec, overlap=0.5):
        """
        Extract (n_mels, n_frames, 1) patches from mel-spectrogram with overlap.
        Handles audio of ANY length - very short audio is padded automatically.
        
        Args:
            mel_spec: Mel-spectrogram (n_mels, n_frames)
            overlap: Overlap ratio (0.5 = 50% overlap for smoother predictions)
        
        Returns:
            List of patches with shape (n_mels, n_frames, 1)
        """
        patches = []
        n_frames = mel_spec.shape[1]
        
        # Calculate stride based on overlap
        stride = int(self.config['n_frames'] * (1 - overlap))
        stride = max(1, stride)  # Ensure at least 1 frame stride
        
        # CRITICAL: Pad to at least one patch if needed (handles very short audio)
        if n_frames < self.config['n_frames']:
            pad_amount = self.config['n_frames'] - n_frames
            mel_spec = np.pad(
                mel_spec,
                ((0, 0), (0, pad_amount)),
                mode='constant',
                constant_values=np.min(mel_spec)  # Pad with minimum value instead of zero
            )
            n_frames = mel_spec.shape[1]
            duration_s = (n_frames - pad_amount) * self.config['hop_length'] / self.config['sr']
            print(f"[INFO] Padded short audio: {n_frames - pad_amount} -> {n_frames} frames ({duration_s:.3f}s)")
        
        # Extract overlapping patches
        for start_frame in range(0, n_frames - self.config['n_frames'] + 1, stride):
            patch = mel_spec[:, start_frame:start_frame + self.config['n_frames']]
            if patch.shape == (self.config['n_mels'], self.config['n_frames']):
                patches.append(patch[..., np.newaxis])
        
        # This should never happen now, but safety check
        if not patches:
            print("[WARNING] No patches extracted - creating zero patch")
            patches = [np.zeros((self.config['n_mels'], self.config['n_frames'], 1), dtype=np.float32)]
        
        return patches

    def prepare_audio(self, file_path, overlap=0.5):
        """
        Prepare full audio for inference: load -> mel-spec -> patches.
        
        Args:
            file_path: Path to audio file
            overlap: Patch overlap ratio (default 0.5 for smoother predictions)
        """
        audio = self.load_audio_file(file_path)
        if audio is None:
            return None
        
        mel_spec = self.create_mel_spectrogram(audio)
        patches = self.extract_patches(mel_spec, overlap=overlap)
        
        # Keep scaling consistent with training
        patches = np.array(patches, dtype=np.float32) / 255.0
        return patches

    def prepare_audio_segment(self, file_path, start_s: float, end_s: float, overlap=0.5):
        """
        Prepare ONLY the requested time segment for inference.
        
        Args:
            file_path: Path to audio file
            start_s: Start time in seconds
            end_s: End time in seconds
            overlap: Patch overlap ratio (default 0.5 for smoother predictions)
        """
        audio = self.load_audio_segment(file_path, start_s, end_s)
        if audio is None:
            return None
        mel_spec = self.create_mel_spectrogram(audio)
        patches = self.extract_patches(mel_spec, overlap=overlap)
        patches = np.array(patches, dtype=np.float32) / 255.0
        return patches
    
    def load_audio_segment(self, file_path, start_s, end_s):
        """Load specific segment of audio file"""
        try:
            audio, sr = sf.read(str(file_path))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sr != self.config['sr']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sr'])
            
            # Extract segment
            start_sample = int(start_s * self.config['sr'])
            end_sample = int(end_s * self.config['sr'])
            audio = audio[start_sample:end_sample]
            
            return audio
        except Exception as e:
            print(f"Error loading segment from {file_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load entire dataset with target classes - NO SKIPPING SHORT AUDIO"""
        X, y = [], []
        class_samples = {cls: 0 for cls in self.config['target_classes']}
        
        # Load dev set
        dev_csv = self.dataset_root / 'FSD50K.ground_truth' / 'dev.csv'
        dev_audio_dir = self.dataset_root / 'FSD50K.dev_audio'
        
        print(f"[DEBUG] Looking in: {dev_audio_dir}")
        print(f"[DEBUG] CSV exists: {dev_csv.exists()}")
        print(f"[DEBUG] Audio dir exists: {dev_audio_dir.exists()}")
        
        dev_data = self._read_ground_truth(dev_csv)
        print(f"[DEBUG] Total rows in CSV: {len(dev_data)}")
        
        skipped_no_labels = 0
        skipped_no_target = 0
        skipped_no_audio_file = 0
        skipped_audio_load_error = 0
        processed_short_audio = 0
        
        for idx, row in enumerate(dev_data):
            if idx % 5000 == 0:
                print(f"[DEBUG] Processing row {idx}/{len(dev_data)}...")
            
            # FSD50K CSV format: fname, labels, mids, split
            fname = row['fname'].strip()
            labels_str = row['labels'].strip()
            
            if not labels_str:
                skipped_no_labels += 1
                continue
            
            # Labels are comma-separated class names (not IDs!)
            label_names = [x.strip() for x in labels_str.split(',') if x.strip()]
            
            # Audio file path
            audio_file = dev_audio_dir / f"{fname}.wav"
            
            # Check if any label matches target classes
            target_label = None
            for label_name in label_names:
                if label_name in self.class_map:
                    target_label = label_name
                    break
            
            if target_label is None:
                skipped_no_target += 1
                continue
            
            if not audio_file.exists():
                skipped_no_audio_file += 1
                continue
            
            # Load audio
            audio = self.load_audio_file(str(audio_file))
            if audio is None:
                skipped_audio_load_error += 1
                continue
            
            # REMOVED: Short audio check - now handled by padding in extract_patches
            if len(audio) < self.config['sr']:
                processed_short_audio += 1
            
            # Extract patches and create labels
            mel_spec = self.create_mel_spectrogram(audio)
            patches = self.extract_patches(mel_spec)
            
            for patch in patches:
                X.append(patch)
                y.append(self.class_map[target_label])
                class_samples[target_label] += 1
        
        print(f"\n[DEBUG STATS]")
        print(f"  Skipped (no labels): {skipped_no_labels}")
        print(f"  Skipped (no target class): {skipped_no_target}")
        print(f"  Skipped (no audio file): {skipped_no_audio_file}")
        print(f"  Skipped (audio load error): {skipped_audio_load_error}")
        print(f"  Processed short audio (<1s): {processed_short_audio}")
        
        print(f"\n[+] Loaded {len(X)} patches from {len(set(y))} unique classes")
        print("\nSamples per class:")
        for cls, count in class_samples.items():
            if count > 0:
                print(f"  {cls}: {count}")
        
        return np.array(X), np.array(y)


# ============================================================================
# STEP 2: BUILD REDUCED YAMNET-256 ARCHITECTURE
# ============================================================================

def build_yamnet256(num_classes):
    """
    Build Yamnet-256 with reduced architecture:
    - Reduced embedding dimension (256 vs 1024)
    - Pruned layers
    - Bottleneck architecture
    """
    
    inputs = layers.Input(shape=(64, 96, 1), name='mel_spectrogram')
    
    # Expand dims if needed
    x = inputs
    
    # Block 1: Depthwise separable conv (reduced filters)
    x = layers.SeparableConv2D(
        32, (3, 3), padding='same', activation='relu', name='sep_conv_1'
    )(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_1')(x)
    
    # Block 2: Depthwise separable conv
    x = layers.SeparableConv2D(
        64, (3, 3), padding='same', activation='relu', name='sep_conv_2'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_2')(x)
    
    # Block 3: Depthwise separable conv
    x = layers.SeparableConv2D(
        128, (3, 3), padding='same', activation='relu', name='sep_conv_3'
    )(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_3')(x)
    
    # Block 4: Depthwise separable conv (reduced)
    x = layers.SeparableConv2D(
        256, (3, 3), padding='same', activation='relu', name='sep_conv_4'
    )(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.MaxPooling2D((2, 2), name='pool_4')(x)
    
    # Global pooling for embedding
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Embedding layer (256-d, vs original 1024-d)
    embeddings = layers.Dense(256, activation='relu', name='embedding')(x)
    
    # Classification head (transfer learning friendly)
    x = layers.Dropout(0.3, name='dropout')(embeddings)
    predictions = layers.Dense(
        num_classes, activation='softmax', name='predictions'
    )(x)
    
    model = Model(inputs=inputs, outputs=predictions, name='yamnet256')
    return model, embeddings


# ============================================================================
# STEP 3: TRANSFER LEARNING & TRAINING
# ============================================================================

def train_yamnet256(X_train, y_train, X_val, y_val, num_classes, config):
    """Train Yamnet-256 with transfer learning approach"""
    
    print("[*] Building Yamnet-256 architecture...")
    model, embedding_layer = build_yamnet256(num_classes)
    model.summary()
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(config['output_dir'], 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train
    print("[*] Training Yamnet-256...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history


# ============================================================================
# STEP 4: PRUNING FOR SIZE REDUCTION
# ============================================================================

def prune_model(model, pruning_rate=0.3):
    """Apply weight pruning to reduce model size"""
    from tensorflow_model_optimization.sparsity import keras as sparsity
    
    print(f"[*] Applying {pruning_rate*100}% pruning...")
    
    try:
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=pruning_rate,
                begin_step=0,
                end_step=1000
            )
        }
        
        # Clone model to avoid issues
        pruned_model = keras.models.clone_model(model)
        pruned_model = sparsity.prune_low_magnitude(pruned_model, **pruning_params)
        return pruned_model
    except Exception as e:
        print(f"[!] Pruning failed: {e}")
        print("[*] Skipping pruning, proceeding directly to quantization...")
        return model


# ============================================================================
# STEP 5: INT8 QUANTIZATION (TFLite)
# ============================================================================

def quantize_to_tflite(model, X_train):
    """Quantize model to int8 and convert to TFLite"""
    
    print("[*] Converting to TFLite with int8 quantization...")
    
    # Create representative dataset for quantization
    def representative_dataset_gen():
        for i in range(min(500, len(X_train))):
            # Ensure float32
            sample = X_train[i:i+1].astype(np.float32)
            yield [sample]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # IMPORTANT: Set the representative data generator
    converter.representative_data = representative_dataset_gen
    
    # For full integer quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_model = converter.convert()
        print("[+] Quantization successful!")
        return tflite_model
    except Exception as e:
        print(f"[!] Full int8 quantization failed: {e}")
        print("[*] Falling back to dynamic range quantization...")
        
        # Fallback to dynamic range quantization (less aggressive but more compatible)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("[+] Dynamic quantization successful!")
        return tflite_model


# ============================================================================
# STEP 6: MAIN EXECUTION
# ============================================================================

def main():
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    num_classes = len(CONFIG['target_classes'])
    float32_model_path = os.path.join(CONFIG['output_dir'], 'yamnet256_float32.h5')
    
    # Check if model already exists
    if os.path.exists(float32_model_path):
        print(f"[*] Loading pre-trained model from {float32_model_path}")
        model = keras.models.load_model(float32_model_path)
        print("[+] Model loaded successfully!")
        
        # Load minimal data just for quantization
        print("[*] Loading FSD50K dataset for quantization...")
        loader = FSD50KDataLoader(CONFIG)
        X, y = loader.load_dataset()
        
        if len(X) == 0:
            print("[!] No data found. Check dataset path and class names.")
            return
        
        X = X.astype(np.float32) / 255.0
        
        # Evaluate on a subset
        X_subset = X[:min(1000, len(X))]
        y_subset = y[:min(1000, len(y))]
        
        eval_loss, eval_acc = model.evaluate(X_subset, y_subset, verbose=0)
        print(f"[+] Evaluation Accuracy: {eval_acc*100:.2f}%")
        
    else:
        print("[*] No pre-trained model found. Training from scratch...")
        
        # Load data
        print("[*] Loading FSD50K dataset...")
        loader = FSD50KDataLoader(CONFIG)
        X, y = loader.load_dataset()
        
        if len(X) == 0:
            print("[!] No data found. Check dataset path and class names.")
            return
        
        print(f"[+] Loaded {len(X)} samples from {len(np.unique(y))} classes")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Normalize
        X_train = X_train.astype(np.float32) / 255.0
        X_val = X_val.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # Train model
        model, history = train_yamnet256(
            X_train, y_train, X_val, y_val, num_classes, CONFIG
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n[+] Test Accuracy: {test_acc*100:.2f}%")
        
        # Save float32 model
        model.save(float32_model_path)
    
    # Apply pruning (or skip if it fails)
    pruned_model = prune_model(model)
    
    # For quantization, we need some training data
    # If we just loaded the model, load a small subset
    if os.path.exists(float32_model_path) and 'X_train' not in locals():
        print("[*] Using loaded data for quantization...")
        X_train = X[:min(5000, len(X))].astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32) / 255.0 if 'X_train' in locals() else X[:min(5000, len(X))].astype(np.float32) / 255.0
    
    # Quantize to int8 TFLite
    tflite_model = quantize_to_tflite(pruned_model, X_train)
    tflite_path = os.path.join(CONFIG['output_dir'], 'yamnet256_int8.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Model size comparison
    float32_size = os.path.getsize(
        os.path.join(CONFIG['output_dir'], 'yamnet256_float32.h5')
    ) / (1024 * 1024)
    int8_size = os.path.getsize(tflite_path) / (1024 * 1024)
    
    print(f"\n[+] Model Sizes:")
    print(f"   Float32: {float32_size:.2f} MB")
    print(f"   Int8 TFLite: {int8_size:.2f} MB")
    print(f"   Compression: {(1 - int8_size/float32_size)*100:.1f}%")
    
    print(f"\n[+] Models saved to {CONFIG['output_dir']}")
    print(f"   - yamnet256_float32.h5")
    print(f"   - yamnet256_int8.tflite")


if __name__ == '__main__':
    main()