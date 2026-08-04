"""
Ground truth I/O — load and save WAV files and JSON ground truth data.
"""

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional


def load_wav(filepath: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load a WAV file, optionally resampling."""
    import soundfile as sf
    signal, sr = sf.read(filepath)
    if sr != target_sr:
        import librosa
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return signal, sr


def save_wav(filepath: str, signal: np.ndarray, sr: int = 16000) -> None:
    """Save signal as WAV file."""
    import soundfile as sf
    sf.write(filepath, signal, sr)


def save_ground_truth(filepath: str,
                      echo_path: np.ndarray,
                      params: dict) -> None:
    """Save echo path and parameters as JSON."""
    data = {
        'echo_path': echo_path.tolist(),
        'params': params,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_ground_truth(filepath: str) -> Tuple[np.ndarray, dict]:
    """Load echo path and parameters from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return np.array(data['echo_path']), data.get('params', {})
