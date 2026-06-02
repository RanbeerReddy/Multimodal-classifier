from scipy.signal import butter, filtfilt
import numpy as np






def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """
    Butterworth bandpass filter applied on all 12 leads.
    signal: (1000, 12)
    Returns: (1000, 12) filtered
    """
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)  # axis=0 = along time


def normalize_per_sample(signal):
    """
    Z-score normalization per sample per lead.
    signal: (1000, 12)
    """
    mean = signal.mean(axis=0, keepdims=True)  # (1, 12)
    std  = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


def preprocess_ecg_batch(X_batch, fs=100, apply_filter=True):
    """
    Apply the full preprocessing pipeline on a batch of ECG signals.
    X_batch: (N, 1000, 12)
    Returns: (N, 1000, 12) preprocessed
    """
    X_out = np.empty_like(X_batch, dtype=np.float32)
    for i in range(len(X_batch)):
        sig = X_batch[i].copy().astype(np.float64)
        if apply_filter:
            sig = bandpass_filter(sig, fs=fs)
        sig = normalize_per_sample(sig)
        X_out[i] = sig.astype(np.float32)
    return X_out
