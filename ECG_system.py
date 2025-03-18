import sys
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Bidirectional, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.signal
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Global Parameters ---
NUM_CLASSES = 6           # 6 conditions: Normal, STEMI, Sustained VT, Unstable AFib, Bradycardia, Torsades
SAMPLE_LENGTH = 360       # 1-second ECG at 360 Hz

# --- Synthetic Data Generation Functions for Training ---
def generate_synthetic_ecg(label: int) -> np.ndarray:
    """
    Generate a synthetic ECG signal for a given label.
    Labels:
      0: Normal         - Regular sine wave with noise.
      1: STEMI          - Sine wave with an added ST-elevation segment.
      2: Sustained VT   - Faster heart rate simulated by higher frequency sine.
      3: Unstable AFib  - Chaotic, noisy waveform.
      4: Bradycardia    - Slower heart rate (lower frequency sine).
      5: Torsades       - Irregular waveform with phase shifts.
    """
    t = np.linspace(0, 1, SAMPLE_LENGTH)
    base = np.sin(2 * np.pi * 5 * t)  # Base frequency ~5 Hz for a normal beat
    noise = 0.05 * np.random.randn(SAMPLE_LENGTH)
    signal = base + noise

    if label == 1:  # STEMI: add ST-elevation in mid-window
        start = int(0.45 * SAMPLE_LENGTH)
        end = int(0.55 * SAMPLE_LENGTH)
        signal[start:end] += 0.5
    elif label == 2:  # Sustained VT: simulate faster heart rate (higher frequency)
        signal = np.sin(2 * np.pi * 8 * t) + noise
    elif label == 3:  # Unstable AFib: add more randomness
        signal += 0.2 * np.random.randn(SAMPLE_LENGTH)
    elif label == 4:  # Bradycardia: simulate slower heart rate (lower frequency)
        signal = np.sin(2 * np.pi * 2 * t) + noise
    elif label == 5:  # Torsades: simulate irregular phase shifts
        phase_shift = np.random.randn() * 0.5
        signal = np.sin(2 * np.pi * 5 * t + phase_shift) + noise
    return signal

def generate_dataset(num_samples: int = 2000) -> (np.ndarray, np.ndarray):
    """
    Generate a dataset of synthetic ECG signals with corresponding labels.
    """
    X, y = [], []
    for _ in range(num_samples):
        label = np.random.randint(0, NUM_CLASSES)
        ecg_signal = generate_synthetic_ecg(label)
        X.append(ecg_signal)
        y.append(label)
    X = np.array(X).reshape(-1, SAMPLE_LENGTH, 1)  # shape: (samples, 360, 1)
    y = np.array(y)
    return X, y

# --- Model Architecture ---
def build_emergency_model(input_shape: tuple = (SAMPLE_LENGTH, 1)) -> tf.keras.Model:
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# --- Emergency ECG Detector for Real-Time Processing ---
class EmergencyECGDetector:
    def _init_(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.buffer = deque(maxlen=360 * 5)  # 5-second buffer (5*360 samples)
        self.emergency_log = []
        
    def _preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Apply basic filtering to the raw ECG signal."""
        baseline = scipy.signal.medfilt(signal, kernel_size=71)
        filtered = signal - baseline
        b, a = scipy.signal.butter(3, [0.5, 40], btype='bandpass', fs=360)
        filtered = scipy.signal.filtfilt(b, a, filtered)
        sos = scipy.signal.iirnotch(50, 30, 360)
        filtered = scipy.signal.sosfiltfilt(sos, filtered)
        return filtered

    def _pan_tompkins_qrs(self, signal: np.ndarray) -> np.ndarray:
        """A simple placeholder QRS detection."""
        threshold = np.mean(signal) + np.std(signal)
        peaks = np.where(signal > threshold)[0]
        if peaks.size == 0:
            return np.array([])
        qrs = [peaks[0]]
        for idx in peaks[1:]:
            if idx - qrs[-1] > 20:
                qrs.append(idx)
        return np.array(qrs)

    def _measure_st_elevation(self, signal: np.ndarray, qrs_peaks: np.ndarray) -> float:
        """Placeholder for ST-segment measurement."""
        st_values = []
        for peak in qrs_peaks:
            start = min(peak + 10, len(signal)-20)
            end = start + 10
            st_values.append(np.mean(signal[start:end]))
        return np.mean(st_values) if st_values else 0.0

    def _calculate_sqi(self, signal: np.ndarray) -> float:
        """Compute a simple Signal Quality Index (SQI)."""
        noise = np.std(signal[:50])
        amplitude = np.max(signal) - np.min(signal)
        return np.clip(amplitude / (noise + 1e-7) / 10, 0, 1)
    
    def _extract_wave_features(self, signal: np.ndarray) -> dict:
        """Extract features for clinical validation."""
        qrs_peaks = self._pan_tompkins_qrs(signal)
        hr = len(qrs_peaks) * 6  # Assume 10-second window -> BPM = beats * 6
        st_elev = self._measure_st_elevation(signal, qrs_peaks)
        return {"heart_rate": hr, "st_elevation": st_elev}
    
    def analyze_ecg(self, signal: np.ndarray) -> dict:
        """Full analysis: preprocess, model prediction, and feature extraction."""
        processed = self._preprocess(signal)
        proc_reshaped = processed[np.newaxis, :, np.newaxis].astype(np.float32)
        probas = self.model.predict(proc_reshaped)
        pred_class = np.argmax(probas)
        confidence = probas[0][pred_class]
        features = self._extract_wave_features(processed)
        sqi = self._calculate_sqi(processed)
        
        # For deployment, clinical thresholds would be applied here.
        icu_required = (pred_class in [1, 2, 3, 5]) and (confidence > 0.85)
        
        return {
            "condition": ["Normal", "STEMI", "Sustained VT", "Unstable AFib", "Bradycardia", "Torsades"][pred_class],
            "confidence": float(confidence),
            "sqi": float(sqi),
            "clinical_features": features,
            "icu_required": icu_required
        }
    
    def real_time_processing(self, new_sample: float):
        """Simulate real-time processing by accumulating samples."""
        self.buffer.append(new_sample)
        if len(self.buffer) >= 360:
            window = np.array(list(self.buffer)[-360:])
            result = self.analyze_ecg(window)
            if result["icu_required"]:
                self._trigger_emergency_alert(result)
            self._log_result(result)
    
    def _trigger_emergency_alert(self, result: dict):
        alert_msg = (
            f"🚨 CRITICAL ALERT: {result['condition']} detected!\n"
            f"HR: {result['clinical_features']['heart_rate']} BPM | "
            f"ST Elev: {result['clinical_features']['st_elevation']:.2f}\n"
            "Take immediate action!"
        )
        print(alert_msg)
    
    def _log_result(self, result: dict):
        log_entry = (
            f"Condition: {result['condition']} | "
            f"Confidence: {result['confidence']:.1%} | "
            f"SQI: {result['sqi']:.2f} | "
            f"ICU Required: {'Yes' if result['icu_required'] else 'No'}"
        )
        logging.info(log_entry)
        self.emergency_log.append(log_entry)

def simulate_ecg_sample() -> float:
    """
    Simulate a single ECG sample from a 1-second synthetic ECG signal.
    Returns one random sample from the generated signal.
    """
    t = np.linspace(0, 1, SAMPLE_LENGTH)
    ecg = 0.5 * np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(SAMPLE_LENGTH)
    if random.random() < 0.1:
        start = int(0.45 * SAMPLE_LENGTH)
        end = int(0.55 * SAMPLE_LENGTH)
        ecg[start:end] += 0.5
    return float(random.choice(ecg))

# --- Main Execution ---
def train_model():
    # Generate synthetic dataset
    X, y = generate_dataset(num_samples=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_emergency_model()
    model.summary()
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the model
    loss, acc, auc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    # Save the model
    model.save("emergency_ecg_model.h5")
    print("Model saved as emergency_ecg_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def run_detection():
    # Load the trained model and initialize the detector
    detector = EmergencyECGDetector("emergency_ecg_model.h5")
    print("Running real-time emergency ECG detection (simulated)...")
    # Simulate real-time streaming for 10 seconds
    for _ in range(360 * 10):
        sample = simulate_ecg_sample()
        detector.real_time_processing(sample)
        time.sleep(1 / 360)

if _name_ == "_main_":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "train":
        train_model()
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "detect":
        run_detection()
    else:
        print("Usage: python emergency_ecg_system.py [train|detect]"