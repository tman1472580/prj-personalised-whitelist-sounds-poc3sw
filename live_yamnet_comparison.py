#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Live microphone comparison: Original YAMNet vs YAMNet-256 (TFLite int8 + Keras float32)

Shows top 5 predictions from each model in real-time.

Usage:
  python live_yamnet_comparison.py \
    --tflite-model /Users/tanzimmohammad/Documents/GitHub/Caudeverison_yamnet_training/yamnet256_model/yamnet256_int8.tflite \
    --keras-model /Users/tanzimmohammad/Documents/GitHub/Caudeverison_yamnet_training/yamnet256_model/yamnet256_float32.h5 \
    --labels /path/to/yamnet256_labels.txt
"""

import os
import sys
import csv
import queue
import argparse
import datetime
import signal
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
import librosa

import tensorflow as tf
import tensorflow_hub as hub

# ---------------------------
# Audio / windowing params
# ---------------------------
TARGET_SR = 16000
WIN_SEC = 0.96
WIN_SAMPLES = int(TARGET_SR * WIN_SEC)
UI_UPDATE_MS = 300  # ms

# Log-mel params for YAMNet-256 (64x96 patches) - MUST match training config
N_MELS = 64
N_FFT = 512
HOP_SAMPLES = 160        # 10 ms at 16kHz (hop_length in training)
WIN_SAMPLES_STFT = 400   # 25 ms at 16kHz (fft_size in training)
PATCH_FRAMES = 96        # 0.96 s / 10 ms (n_frames in training)
FMIN = 125.0             # mel_min_hz in training
FMAX = 7500.0            # mel_max_hz in training

# YAMNet-256 classes (11 classes)
YAMNET256_CLASSES = [
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
]

# ---------------------------
# Label loaders
# ---------------------------
def load_label_names(path: Optional[str]) -> List[str]:
    """Load labels from text file (one per line)."""
    if not path:
        return YAMNET256_CLASSES
    p = Path(path)
    if not p.exists():
        print(f"WARNING: label file not found: {path}, using defaults")
        return YAMNET256_CLASSES
    try:
        with p.open("r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return labels if labels else YAMNET256_CLASSES
    except Exception as e:
        print(f"WARNING: could not parse labels from {path}: {e}")
        return YAMNET256_CLASSES

# ---------------------------
# TFHub YAMNet
# ---------------------------
def load_yamnet_and_names():
    print("🔹 Loading original YAMNet from TFHub…")
    ym = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = ym.class_map_path().numpy().decode("utf-8")
    names = []
    with tf.io.gfile.GFile(class_map_path, "r") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            names.append(row[2])  # display_name
    return ym, names

def yamnet_scores(yamnet_module, wave_16k: np.ndarray):
    wf = tf.convert_to_tensor(wave_16k, dtype=tf.float32)
    scores, _, _ = yamnet_module(wf)  # (F,521)
    return tf.reduce_mean(scores, axis=0).numpy()

# ---------------------------
# Log-mel computation
# ---------------------------
def compute_logmel64(audio_16k: np.ndarray) -> np.ndarray:
    """Compute 64x96 log-mel spectrogram matching training preprocessing."""
    # Use librosa.feature.melspectrogram to match training exactly
    S = librosa.feature.melspectrogram(
        y=audio_16k,
        sr=TARGET_SR,
        n_fft=WIN_SAMPLES_STFT,  # 400 (25ms at 16kHz)
        hop_length=HOP_SAMPLES,   # 160 (10ms)
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    
    # Convert to dB scale (matches training)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)  # [64, T]
    
    # Normalize by dividing by 255.0 (matches training preprocessing)
    S_db = S_db.astype(np.float32) / 255.0
    
    T = S_db.shape[1]
    if T < PATCH_FRAMES:
        # Pad if too short
        pad = np.tile(S_db[:, -1:], (1, PATCH_FRAMES - T)) if T > 0 else np.zeros((N_MELS, PATCH_FRAMES), dtype=np.float32)
        mel = np.concatenate([S_db, pad], axis=1)
    else:
        # Take last 96 frames
        mel = S_db[:, -PATCH_FRAMES:]
    
    return mel  # [64,96]

# ---------------------------
# TFLite utilities
# ---------------------------
Interpreter = None
try:
    from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter
    Interpreter = TFInterpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter as RTInterpreter
        Interpreter = RTInterpreter
    except Exception:
        pass

def _safe_qparams(tensor_det):
    """Return (scale, zero_point) with safe defaults."""
    qp = tensor_det.get("quantization_parameters", {}) or {}
    scales = qp.get("scales", None)
    zps = qp.get("zero_points", None)

    def first_or_default(arr, default):
        try:
            if arr is None:
                return default
            if hasattr(arr, "size"):
                return float(arr[0]) if arr.size else default
            return float(arr[0]) if len(arr) else default
        except Exception:
            return default

    scale = first_or_default(scales, 1.0)
    zp = int(first_or_default(zps, 0))
    return scale, zp

class TFLiteModel:
    def __init__(self, model_path: str):
        if Interpreter is None:
            raise RuntimeError("No TFLite interpreter available.")
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = _safe_qparams(self.in_det)
        self.out_scale, self.out_zp = _safe_qparams(self.out_det)

    def predict(self, mel64x96: np.ndarray) -> np.ndarray:
        x = mel64x96[None, :, :, None].astype(np.float32)  # [1,64,96,1]
        
        # Quantize input if needed
        if self.in_det['dtype'] == np.int8:
            x_q = np.round(x / self.in_scale + self.in_zp).astype(np.int32)
            x_q = np.clip(x_q, -128, 127).astype(np.int8)
            self.interp.set_tensor(self.in_det['index'], x_q)
        else:
            self.interp.set_tensor(self.in_det['index'], x)
        
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det['index'])
        
        # Dequantize output if needed
        if self.out_det['dtype'] == np.int8:
            y = (y.astype(np.float32) - self.out_zp) * self.out_scale
        
        # Handle multi-dimensional outputs
        y = np.array(y).reshape(-1)
        return y

# ---------------------------
# Audio streaming
# ---------------------------
q = queue.Queue()
buf = np.zeros(WIN_SAMPLES, dtype=np.float32)
actual_sr = [TARGET_SR]

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    mono = indata[:, 0].astype(np.float32, copy=True)
    q.put(mono)

def to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == TARGET_SR:
        return x
    return librosa.resample(x, orig_sr=sr_in, target_sr=TARGET_SR)

# ---------------------------
# Tk UI
# ---------------------------
try:
    import tkinter as tk
    from tkinter import ttk, font as tkfont
except Exception:
    print("tkinter is required. Install with: conda install -c conda-forge tk")
    sys.exit(1)

def format_top5(scores: np.ndarray, labels: List[str], top_k: int = 5) -> List[tuple]:
    """Return top-k (label, score) pairs."""
    if len(scores) == 0:
        return []
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return [(labels[i] if i < len(labels) else f"Class_{i}", scores[i]) for i in top_idx]

def make_model_column(parent, title: str, row: int, col: int):
    """Create a column showing top 5 predictions."""
    frame = ttk.LabelFrame(parent, text=title, padding=10)
    frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
    
    # Use monospace font for alignment
    labels = []
    for i in range(5):
        lbl = ttk.Label(frame, text=f"{i+1}. ...", width=45, font=("Courier", 11))
        lbl.pack(anchor="w", pady=2)
        labels.append(lbl)
    
    return frame, labels

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Live YAMNet vs YAMNet-256 comparison")
    parser.add_argument("--tflite-model", type=str, 
                       default="/Users/tanzimmohammad/Documents/GitHub/Caudeverison_yamnet_training/yamnet256_model/yamnet256_int8.tflite",
                       help="Path to YAMNet-256 TFLite int8 model")
    parser.add_argument("--keras-model", type=str,
                       default="/Users/tanzimmohammad/Documents/GitHub/Caudeverison_yamnet_training/yamnet256_model/yamnet256_float32.h5",
                       help="Path to YAMNet-256 Keras float32 model")
    parser.add_argument("--labels", type=str, default=None,
                       help="Path to YAMNet-256 labels file (optional)")
    parser.add_argument("--update-ms", type=int, default=UI_UPDATE_MS,
                       help="UI update interval (ms)")
    args = parser.parse_args()

    # Load labels
    y256_labels = load_label_names(args.labels)
    print(f"📋 Loaded {len(y256_labels)} YAMNet-256 labels")

    # Load models
    print("🔹 Loading original YAMNet from TFHub…")
    yamnet_module, yamnet_labels = load_yamnet_and_names()
    print(f"✅ YAMNet loaded ({len(yamnet_labels)} classes)")

    print(f"🔹 Loading YAMNet-256 TFLite model: {args.tflite_model}")
    tflite_model = TFLiteModel(args.tflite_model)
    print("✅ TFLite model loaded")

    print(f"🔹 Loading YAMNet-256 Keras model: {args.keras_model}")
    keras_model = tf.keras.models.load_model(args.keras_model, compile=False)
    print("✅ Keras model loaded")

    # UI setup
    root = tk.Tk()
    root.title("Live Audio Classification: YAMNet vs YAMNet-256")
    root.geometry("1400x400")

    container = ttk.Frame(root, padding=10)
    container.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)
    container.columnconfigure(1, weight=1)
    container.columnconfigure(2, weight=1)

    # Create three columns
    yamnet_frame, yamnet_lbls = make_model_column(container, "Original YAMNet (521 classes)", 0, 0)
    tflite_frame, tflite_lbls = make_model_column(container, "YAMNet-256 TFLite int8 (11 classes)", 0, 1)
    keras_frame, keras_lbls = make_model_column(container, "YAMNet-256 Keras float32 (11 classes)", 0, 2)

    # Status bar
    status_frame = ttk.Frame(root, padding=(10, 5))
    status_frame.grid(row=1, column=0, sticky="ew")
    status_var = tk.StringVar(value="Initializing…")
    ttk.Label(status_frame, textvariable=status_var).pack(side="left")

    # Control
    stream = None
    app_running = [True]

    def cleanup(*_args):
        app_running[0] = False
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", cleanup)
    try:
        signal.signal(signal.SIGINT, lambda s, f: cleanup())
        signal.signal(signal.SIGTERM, lambda s, f: cleanup())
    except Exception:
        pass
    root.bind("<Escape>", lambda e: cleanup())
    root.bind("<q>", lambda e: cleanup())
    root.bind("<Q>", lambda e: cleanup())

    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / np.sum(ex)

    def update_loop():
        if not app_running[0]:
            return

        # Drain audio queue
        global buf
        while not q.empty():
            chunk = q.get()
            c16 = to_16k(chunk, actual_sr[0])
            if len(c16) >= WIN_SAMPLES:
                buf[:] = c16[-WIN_SAMPLES:]
            elif len(c16) > 0:
                shift = len(c16)
                buf[:-shift] = buf[shift:]
                buf[-shift:] = c16

        # Compute mel spectrogram for YAMNet-256 models
        mel = compute_logmel64(buf)

        # Original YAMNet prediction
        try:
            scores_yamnet = yamnet_scores(yamnet_module, buf)
            top5_yamnet = format_top5(scores_yamnet, yamnet_labels)
            for i, (label, score) in enumerate(top5_yamnet):
                yamnet_lbls[i].config(text=f"{i+1}. {label[:35]:35s} {score*100:5.1f}%")
        except Exception as e:
            print(f"YAMNet error: {e}")
            for i in range(5):
                yamnet_lbls[i].config(text=f"{i+1}. Error")

        # TFLite YAMNet-256 prediction
        try:
            scores_tflite = tflite_model.predict(mel)
            # Models were trained with softmax activation, so outputs are already probabilities
            # Just normalize to ensure they sum to 1
            scores_tflite = scores_tflite / (np.sum(scores_tflite) + 1e-10)
            
            top5_tflite = format_top5(scores_tflite, y256_labels)
            for i, (label, score) in enumerate(top5_tflite):
                tflite_lbls[i].config(text=f"{i+1}. {label[:35]:35s} {score*100:5.1f}%")
        except Exception as e:
            print(f"TFLite error: {e}")
            import traceback
            traceback.print_exc()
            for i in range(5):
                tflite_lbls[i].config(text=f"{i+1}. Error")

        # Keras YAMNet-256 prediction
        try:
            input_keras = mel[None, :, :, None]  # [1,64,96,1]
            scores_keras = keras_model.predict(input_keras, verbose=0)[0]
            # Models were trained with softmax activation, so outputs are already probabilities
            # Just normalize to ensure they sum to 1
            scores_keras = scores_keras / (np.sum(scores_keras) + 1e-10)
            
            top5_keras = format_top5(scores_keras, y256_labels)
            for i, (label, score) in enumerate(top5_keras):
                keras_lbls[i].config(text=f"{i+1}. {label[:35]:35s} {score*100:5.1f}%")
        except Exception as e:
            print(f"Keras error: {e}")
            import traceback
            traceback.print_exc()
            for i in range(5):
                keras_lbls[i].config(text=f"{i+1}. Error")

        root.after(args.update_ms, update_loop)

    # Start audio
    try:
        try:
            with sd.InputStream(samplerate=TARGET_SR, channels=1, callback=audio_callback) as probe:
                actual_sr[0] = int(probe.samplerate)
        except Exception:
            with sd.InputStream(channels=1, callback=audio_callback) as probe:
                actual_sr[0] = int(probe.samplerate)

        status_var.set(f"🎙️ Listening… Input SR: {actual_sr[0]} Hz | Press Esc/Q or close window to exit")
        stream = sd.InputStream(samplerate=actual_sr[0], channels=1, callback=audio_callback)
        stream.start()

        print("🎙️ Listening to microphone… (Esc/Q/Close to exit)")
        update_loop()
        root.mainloop()
    finally:
        cleanup()

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    main()