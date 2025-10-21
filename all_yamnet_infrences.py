"""
Yamnet-256 Inference & Evaluation Script
- Evaluate models on test data (eval split)
- Run full-file or time-segment inference (HH:MM:SS(.ms))
- Compare with original YAMNet (explicit class mapping)
- Show test sample distribution by class
- Print & save classification reports for Float32 and INT8 (and Original YAMNet)
"""

import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
from pathlib import Path
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_dir': './yamnet256_model',
    'dataset_root': '/Users/tanzimmohammad/Documents/GitHub/FSD50K Dataset/root',  # <-- change me
    'yamnet_hub_url': 'https://tfhub.dev/google/yamnet/1',
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
    # Audio / feature params
    'sr': 16000,
    'fft_size': 400,
    'hop_length': 160,
    'n_mels': 64,
    'n_frames': 96,
    'mel_min_hz': 125,
    'mel_max_hz': 7500,
}

# ============================================================================
# HELPERS
# ============================================================================

def _parse_hhmmss_to_seconds(s: str) -> float:
    """
    Accepts "HH:MM:SS", "HH:MM:SS.mmm", "MM:SS(.mmm)", or "SS(.mmm)"
    Returns seconds as float.
    """
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = "0", parts[0], parts[1]
    else:
        h, m, sec = "0", "0", parts[0]
    return int(h) * 3600 + int(m) * 60 + float(sec)

# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.class_map = {cls: idx for idx, cls in enumerate(config['target_classes'])}
        self.reverse_class_map = {idx: cls for cls, idx in self.class_map.items()}
    
    def load_audio(self, file_path):
        """Load full audio and resample to target sr (robust: sf.read -> librosa fallback)."""
        try:
            audio, sr = sf.read(str(file_path))
            # Convert to mono
            if hasattr(audio, "shape") and len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        except Exception as e:
            print(f"[soundfile] {e} -> falling back to librosa.load")
            audio, sr = librosa.load(file_path, sr=None, mono=True, dtype=np.float32)

        if sr != self.config['sr']:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sr'])

        return audio.astype(np.float32, copy=False)

    def load_audio_segment(self, file_path, start_s: float, end_s: float):
        """Load only a segment [start_s, end_s) directly at target sr."""
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s")
        duration = end_s - start_s
        try:
            audio, _ = librosa.load(
                file_path,
                sr=self.config['sr'],
                mono=True,
                offset=start_s,
                duration=duration,
                dtype=np.float32,
            )
            if audio is None or audio.size == 0:
                print(f"[!] Empty segment {start_s:.3f}s–{end_s:.3f}s for {file_path}")
                return None
            return audio.astype(np.float32, copy=False)
        except Exception as e:
            print(f"Error loading segment from {file_path}: {e}")
            return None
    
    def create_mel_spectrogram(self, audio):
        """Create mel-spectrogram (in dB)."""
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
        S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
        return S_db
    
    def extract_patches(self, mel_spec):
        """Extract non-overlapping (n_mels, n_frames, 1) patches from mel-spectrogram."""
        patches = []
        n_frames = mel_spec.shape[1]
        
        # Pad to at least one patch
        if n_frames < self.config['n_frames']:
            mel_spec = np.pad(
                mel_spec,
                ((0, 0), (0, self.config['n_frames'] - n_frames)),
                mode='constant'
            )
            n_frames = mel_spec.shape[1]
        
        step = self.config['n_frames']
        for start_frame in range(0, n_frames - self.config['n_frames'] + 1, step):
            patch = mel_spec[:, start_frame:start_frame + self.config['n_frames']]
            if patch.shape == (self.config['n_mels'], self.config['n_frames']):
                patches.append(patch[..., np.newaxis])
        
        if not patches:
            patches = [np.zeros((self.config['n_mels'], self.config['n_frames'], 1), dtype=np.float32)]
        
        return patches
    
    def prepare_audio(self, file_path):
        """Prepare full audio for inference: load -> mel-spec -> patches."""
        audio = self.load_audio(file_path)
        if audio is None:
            return None
        
        mel_spec = self.create_mel_spectrogram(audio)
        patches = self.extract_patches(mel_spec)
        
        # Keep scaling consistent with training
        patches = np.array(patches, dtype=np.float32) / 255.0
        return patches

    def prepare_audio_segment(self, file_path, start_s: float, end_s: float):
        """Prepare ONLY the requested time segment for inference."""
        audio = self.load_audio_segment(file_path, start_s, end_s)
        if audio is None:
            return None
        mel_spec = self.create_mel_spectrogram(audio)
        patches = self.extract_patches(mel_spec)
        patches = np.array(patches, dtype=np.float32) / 255.0
        return patches


# ============================================================================
# ORIGINAL YAMNET INFERENCE (EXPLICIT CLASS MAPPING)
# ============================================================================

class OriginalYAMNetInference:
    def __init__(self, config):
        self.config = config
        self.processor = AudioProcessor(config)
        self.yamnet_model = None
        self.yamnet_class_names = []
        self.class_mapping = {}  # target_name -> [yamnet_indices]
        
        print("[*] Loading original YAMNet from TensorFlow Hub...")
        try:
            self.yamnet_model = hub.load(config['yamnet_hub_url'])
            class_map_path = self.yamnet_model.class_map_path().numpy()
            self.yamnet_class_names = self._load_yamnet_classes(class_map_path)
            print(f"[+] Original YAMNet loaded with {len(self.yamnet_class_names)} classes")

            self.class_mapping = self._create_explicit_mapping()
            self._validate_and_log_mapping()
        except Exception as e:
            print(f"[!] Failed to load original YAMNet: {e}")
            import traceback; traceback.print_exc()
    
    def _load_yamnet_classes(self, class_map_path):
        """Load YAMNet class names from CSV."""
        class_names = []
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    def _create_explicit_mapping(self):
        """
        Explicit mapping from your 11 target classes to YAMNet (AudioSet) display names.
        These names must exactly match the YAMNet class map (case-insensitive).
        """
        mapping_names = {
            'Accelerating_and_revving_and_vroom': ['Accelerating, revving, vroom'],
            'Aircraft': ['Aircraft', 'Fixed-wing aircraft, airplane', 'Helicopter'],
            'Fire': ['Fire'],
            'Crying_and_sobbing': ['Crying, sobbing', 'Whimper (dog)', 'Baby cry, infant cry'],
            'Dog': ['Dog', 'Bark', 'Howl', 'Bow-wow', 'Growling'],
            'Rain': ['Rain', 'Raindrop', 'Rain on surface'],
            'Chicken_and_rooster': ['Chicken, rooster', 'Cluck', 'Crowing, cock-a-doodle-doo'],
            'Waves_and_surf': ['Waves, surf', 'Ocean', 'Splash, splatter'],
            'Sneeze': ['Sneeze'],
            'Clock': ['Clock', 'Tick-tock', 'Alarm clock'],
            'Child_speech_and_kid_speaking': ['Child speech, kid speaking', 'Speech', 'Babbling'],
        }
        # Resolve names -> indices
        lower_to_idx = {n.lower(): i for i, n in enumerate(self.yamnet_class_names)}
        class_to_indices = {}
        for target, yamnet_names in mapping_names.items():
            idxs = []
            for yn in yamnet_names:
                idx = lower_to_idx.get(yn.lower())
                if idx is not None:
                    idxs.append(idx)
            class_to_indices[target] = sorted(set(idxs))
        return class_to_indices

    def _validate_and_log_mapping(self):
        print("\n[*] YAMNet class mapping (resolved indices):")
        for target in self.config['target_classes']:
            idxs = self.class_mapping.get(target, [])
            if not idxs:
                print(f"  {target}: NO MATCH FOUND")
            else:
                printable = [f"{self.yamnet_class_names[i]} (idx:{i})" for i in idxs]
                print(f"  {target}: {printable}")

    def predict(self, file_path):
        """Predict with original YAMNet and map to target classes via explicit pooled indices."""
        if self.yamnet_model is None:
            return None
        
        try:
            audio = self.processor.load_audio(file_path)
            if audio is None:
                return None
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            scores, embeddings, spectrogram = self.yamnet_model(audio)
            if scores is None or scores.shape[0] == 0:
                print(f"[!] YAMNet returned empty scores for {file_path}")
                return None
            
            clip_scores = np.mean(scores.numpy(), axis=0)

            # Pool (max) over mapped indices per target class
            target_scores = np.zeros(len(self.config['target_classes']), dtype=np.float32)
            for t_idx, target in enumerate(self.config['target_classes']):
                idxs = self.class_mapping.get(target, [])
                if idxs:
                    target_scores[t_idx] = float(np.max(clip_scores[idxs]))
                else:
                    target_scores[t_idx] = 0.0

            if np.all(target_scores == 0):
                top_y = int(np.argmax(clip_scores))
                print(f"[!] All pooled target scores are zero. Top YAMNet: "
                      f"{self.yamnet_class_names[top_y]} ({clip_scores[top_y]:.4f})")
            
            class_idx = int(np.argmax(target_scores))
            confidence = float(target_scores[class_idx])
            class_name = self.processor.reverse_class_map[class_idx]
            
            return {
                'class': class_name,
                'class_idx': class_idx,
                'confidence': confidence,
                'all_scores': {self.processor.reverse_class_map[i]: float(target_scores[i])
                               for i in range(len(target_scores))}
            }
        except Exception as e:
            print(f"[!] Error in original YAMNet prediction for {file_path}: {e}")
            import traceback; traceback.print_exc()
            return None


# ============================================================================
# MODEL INFERENCE (YAMNet-256)
# ============================================================================

class YamnetInference:
    def __init__(self, config):
        self.config = config
        self.processor = AudioProcessor(config)
        
        # Load models
        float32_path = os.path.join(config['model_dir'], 'yamnet256_float32.h5')
        tflite_path = os.path.join(config['model_dir'], 'yamnet256_int8.tflite')
        
        print("[*] Loading YAMNet-256 models...")
        self.float32_model = tf.keras.models.load_model(float32_path)
        print("[+] Float32 model loaded")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[+] TFLite model loaded")
    
    def infer_float32(self, patches):
        """Inference with float32 Keras model."""
        predictions = self.float32_model.predict(patches, verbose=0)
        return predictions
    
    def infer_tflite(self, patches):
        """Inference with TFLite model (handles float or int input/output)."""
        predictions = []

        in_info = self.input_details[0]
        out_info = self.output_details[0]
        in_dtype = in_info['dtype']
        out_dtype = out_info['dtype']
        in_scale, in_zp = in_info.get('quantization', (0.0, 0))
        out_scale, out_zp = out_info.get('quantization', (0.0, 0))

        for patch in patches:
            # Prepare input according to dtype
            if in_dtype in (np.int8, np.uint8):
                if in_scale == 0:
                    raise ValueError("TFLite input has integer dtype but zero quantization scale.")
                patch_q = np.round(patch / in_scale + in_zp).astype(in_dtype)
                input_value = patch_q[np.newaxis, ...]
            elif in_dtype == np.float32:
                input_value = patch[np.newaxis, ...].astype(np.float32)
            else:
                raise ValueError(f"Unhandled TFLite input dtype: {in_dtype}")

            # Set input & run
            self.interpreter.set_tensor(in_info['index'], input_value)
            self.interpreter.invoke()

            # Get output and dequantize if needed
            output_data = self.interpreter.get_tensor(out_info['index'])
            if out_dtype in (np.int8, np.uint8):
                if out_scale == 0:
                    raise ValueError("TFLite output has integer dtype but zero quantization scale.")
                output_data = (output_data.astype(np.float32) - out_zp) * out_scale

            predictions.append(output_data[0])

        return np.array(predictions)
    
    def aggregate_predictions(self, predictions):
        """Aggregate patch predictions to clip-level prediction."""
        return np.mean(predictions, axis=0)
    
    def predict(self, file_path, use_tflite=False):
        """Predict class for an entire audio file."""
        patches = self.processor.prepare_audio(file_path)
        if patches is None:
            return None
        
        predictions = self.infer_tflite(patches) if use_tflite else self.infer_float32(patches)
        clip_pred = self.aggregate_predictions(predictions)
        class_idx = int(np.argmax(clip_pred))
        confidence = float(clip_pred[class_idx])
        class_name = self.processor.reverse_class_map[class_idx]
        
        return {
            'class': class_name,
            'class_idx': class_idx,
            'confidence': confidence,
            'all_scores': {self.processor.reverse_class_map[i]: float(clip_pred[i]) 
                           for i in range(len(clip_pred))}
        }

    def predict_segment(self, file_path, start_s: float, end_s: float, use_tflite=False):
        """Predict class for a time segment [start_s, end_s)."""
        patches = self.processor.prepare_audio_segment(file_path, start_s, end_s)
        if patches is None:
            return None
        predictions = self.infer_tflite(patches) if use_tflite else self.infer_float32(patches)
        clip_pred = self.aggregate_predictions(predictions)
        class_idx = int(np.argmax(clip_pred))
        confidence = float(clip_pred[class_idx])
        class_name = self.processor.reverse_class_map[class_idx]
        return {
            'class': class_name,
            'class_idx': class_idx,
            'confidence': confidence,
            'all_scores': {self.processor.reverse_class_map[i]: float(clip_pred[i])
                           for i in range(len(clip_pred))}
        }


# ============================================================================
# EVALUATION
# ============================================================================

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.dataset_root = Path(config['dataset_root'])
        self.inference = YamnetInference(config)
        self.original_yamnet = OriginalYAMNetInference(config)
    
    def load_test_samples(self, num_samples=100, balanced=True):
        """Load test samples from FSD50K (EVAL split using eval.csv with label NAMES)."""
        csv_path = self.dataset_root / 'FSD50K.ground_truth' / 'eval.csv'
        audio_dir = self.dataset_root / 'FSD50K.eval_audio'
        
        class_map = self.inference.processor.class_map
        
        if balanced:
            samples_per_class = max(1, num_samples // len(self.config['target_classes']))
            class_samples = {cls: [] for cls in self.config['target_classes']}
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row['fname'].strip()
                    labels_str = row['labels'].strip()
                    if not labels_str:
                        continue
                    
                    label_names = [x.strip() for x in labels_str.split(',') if x.strip()]
                    
                    target_label = None
                    for label_name in label_names:
                        if label_name in class_map:
                            target_label = label_name
                            break
                    if target_label is None:
                        continue
                    if len(class_samples[target_label]) >= samples_per_class:
                        continue
                    
                    audio_file = audio_dir / f"{fname}.wav"
                    if not audio_file.exists():
                        continue
                    
                    class_samples[target_label].append({
                        'file': str(audio_file),
                        'true_class': target_label,
                        'true_idx': class_map[target_label]
                    })
                    if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                        break
            
            samples = []
            for cls_list in class_samples.values():
                samples.extend(cls_list)
            for cls, cls_list in class_samples.items():
                if len(cls_list) < samples_per_class:
                    print(f"[!] Warning: Only found {len(cls_list)}/{samples_per_class} samples for class '{cls}'")
        else:
            samples = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if len(samples) >= num_samples:
                        break
                    fname = row['fname'].strip()
                    labels_str = row['labels'].strip()
                    if not labels_str:
                        continue
                    label_names = [x.strip() for x in labels_str.split(',') if x.strip()]
                    target_label = None
                    for label_name in label_names:
                        if label_name in class_map:
                            target_label = label_name
                            break
                    if target_label is None:
                        continue
                    audio_file = audio_dir / f"{fname}.wav"
                    if not audio_file.exists():
                        continue
                    samples.append({
                        'file': str(audio_file),
                        'true_class': target_label,
                        'true_idx': class_map[target_label]
                    })
        return samples
    
    def print_class_distribution(self, samples):
        """Print distribution of test samples by class."""
        class_counts = Counter([s['true_class'] for s in samples])
        
        print("\n" + "=" * 60)
        print("TEST SAMPLE DISTRIBUTION BY CLASS")
        print("=" * 60)
        print(f"{'Class':<40} {'Count':>8}")
        print("-" * 60)
        
        for class_name in self.config['target_classes']:
            count = class_counts.get(class_name, 0)
            print(f"{class_name:<40} {count:>8}")
        
        print("-" * 60)
        print(f"{'TOTAL':<40} {len(samples):>8}")
        print("=" * 60)

    def _report_and_confusion(self, y_true_, y_pred_, tag: str):
        """Print & save classification report and confusion matrix for a model tag."""
        valid = y_pred_ != -1
        if valid.sum() == 0:
            print(f"\n[*] {tag}: No valid predictions to evaluate.")
            return None

        labels_all = list(range(len(self.config['target_classes'])))
        target_names = self.config['target_classes']

        acc = accuracy_score(y_true_[valid], y_pred_[valid])
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {tag}")
        print("=" * 60)
        print(f"{tag} Accuracy: {acc*100:.2f}%")

        print(f"\n[*] Detailed Classification Report ({tag}):")
        report_str = classification_report(
            y_true_[valid], y_pred_[valid],
            labels=labels_all,
            target_names=target_names,
            zero_division=0,
            digits=2
        )
        print(report_str)

        # Save report
        out_dir = self.config.get('model_dir', '.')
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, f"classification_report_{tag.replace(' ', '_').lower()}.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"[+] Saved report to {report_path}")

        # Confusion matrix
        cm = confusion_matrix(y_true_[valid], y_pred_[valid], labels=labels_all)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {tag}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = os.path.join(out_dir, f"confusion_matrix_{tag.replace(' ', '_').lower()}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[+] Saved confusion matrix to {cm_path}")

        return acc
    
    def evaluate(self, num_samples=100, compare_original=False, balanced=True):
        """Evaluate model on test set (eval split)."""
        print(f"\n[*] Loading {num_samples} test samples (balanced={balanced})...")
        samples = self.load_test_samples(num_samples, balanced=balanced)
        if not samples:
            print("[!] No test samples found")
            return
        
        print(f"[+] Loaded {len(samples)} samples")
        self.print_class_distribution(samples)
        
        y_true = []
        y_pred_f32 = []
        y_pred_tflite = []
        y_pred_original = []
        
        print(f"\n[*] Running inference...")
        failed_original = 0
        for i, sample in enumerate(samples):
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(samples)}")
            
            # Float32
            result_f32 = self.inference.predict(sample['file'], use_tflite=False)
            y_pred_f32.append(result_f32['class_idx'] if result_f32 else -1)
            
            # INT8 (TFLite)
            result_tflite = self.inference.predict(sample['file'], use_tflite=True)
            y_pred_tflite.append(result_tflite['class_idx'] if result_tflite else -1)
            
            # Original YAMNet
            if compare_original and self.original_yamnet.yamnet_model is not None:
                result_original = self.original_yamnet.predict(sample['file'])
                if result_original:
                    y_pred_original.append(result_original['class_idx'])
                else:
                    y_pred_original.append(-1)
                    failed_original += 1
            else:
                y_pred_original.append(-1)
            
            y_true.append(sample['true_idx'])
        
        if compare_original and failed_original > 0:
            print(f"\n[!] Original YAMNet failed on {failed_original}/{len(samples)} samples")
        
        # Convert to arrays
        y_true = np.array(y_true)
        y_pred_f32 = np.array(y_pred_f32)
        y_pred_tflite = np.array(y_pred_tflite)

        # Reports for both models (and original)
        acc_f32 = self._report_and_confusion(y_true, y_pred_f32, "YAMNet-256 Float32")
        acc_tflite = self._report_and_confusion(y_true, y_pred_tflite, "YAMNet-256 INT8 (TFLite)")

        if acc_f32 is not None and acc_tflite is not None:
            print("\n" + "=" * 60)
            print("QUANTIZATION SUMMARY")
            print("=" * 60)
            print(f"YAMNet-256 Float32 Accuracy:  {acc_f32*100:.2f}%")
            print(f"YAMNet-256 INT8 (TFLite) Acc: {acc_tflite*100:.2f}%")
            print(f"Quantization Loss:            {(acc_f32 - acc_tflite)*100:.2f}%")

        if compare_original and len(y_pred_original) > 0:
            y_pred_original = np.array(y_pred_original)
            self._report_and_confusion(y_true, y_pred_original, "Original YAMNet")


# ============================================================================
# DEMO INFERENCE (FULL FILE & SEGMENT)
# ============================================================================

def demo_inference_segment(audio_file, start_s, end_s, compare_original=False):
    """Run inference on a time segment [start_s, end_s)."""
    print(f"\n[*] Running inference on segment {start_s:.3f}s–{end_s:.3f}s of: {audio_file}")
    inference = YamnetInference(CONFIG)

    print("\n[*] Float32 Model (segment):")
    r1 = inference.predict_segment(audio_file, start_s, end_s, use_tflite=False)
    if r1:
        print(f"  Class: {r1['class']}")
        print(f"  Confidence: {r1['confidence']:.4f}")

    print("\n[*] INT8 (TFLite) Model (segment):")
    r2 = inference.predict_segment(audio_file, start_s, end_s, use_tflite=True)
    if r2:
        print(f"  Class: {r2['class']}")
        print(f"  Confidence: {r2['confidence']:.4f}")
    
    # ADD THIS BLOCK:
    if compare_original:
        original_yamnet = OriginalYAMNetInference(CONFIG)
        print("\n[*] Original YAMNet Model (segment):")
        processor = AudioProcessor(CONFIG)
        audio_segment = processor.load_audio_segment(audio_file, start_s, end_s)
        if audio_segment is not None:
            scores, _, _ = original_yamnet.yamnet_model(audio_segment)
            if scores is not None and scores.shape[0] > 0:
                clip_scores = np.mean(scores.numpy(), axis=0)
                target_scores = np.zeros(len(CONFIG['target_classes']), dtype=np.float32)
                for t_idx, target in enumerate(CONFIG['target_classes']):
                    idxs = original_yamnet.class_mapping.get(target, [])
                    if idxs:
                        target_scores[t_idx] = float(np.max(clip_scores[idxs]))
                class_idx = int(np.argmax(target_scores))
                confidence = float(target_scores[class_idx])
                class_name = processor.reverse_class_map[class_idx]
                print(f"  Class: {class_name}")
                print(f"  Confidence: {confidence:.4f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="YAMNet-256 inference/evaluation")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file for demo inference")
    parser.add_argument("--start", type=str, help="Segment start time (HH:MM:SS(.ms))")
    parser.add_argument("--end", type=str, help="Segment end time (HH:MM:SS(.ms))")
    parser.add_argument("--num-samples", type=int, default=200, help="Eval samples if no audio_file")
    parser.add_argument("--compare-original", action="store_true", 
                       help="Compare with original YAMNet model")
    parser.add_argument("--unbalanced", action="store_true",
                       help="Use unbalanced sampling (default is balanced)")
    args = parser.parse_args()

    if args.audio_file:
        if args.start and args.end:
            start_s = _parse_hhmmss_to_seconds(args.start)
            end_s = _parse_hhmmss_to_seconds(args.end)
            demo_inference_segment(args.audio_file, start_s, end_s, compare_original=args.compare_original)  # Pass the flag!
        else:
            demo_inference(args.audio_file, compare_original=args.compare_original)
    else:
        evaluator = Evaluator(CONFIG)
        evaluator.evaluate(
            num_samples=args.num_samples, 
            compare_original=args.compare_original,
            balanced=not args.unbalanced
        )
