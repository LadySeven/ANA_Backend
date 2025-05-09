import librosa
import numpy as np

def augment_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr = 22050)

    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}")

    if not isinstance(y, np.ndarray) or len(y) < 2048 or np.allclose(y, 0):
        raise ValueError("Audio data is invalid or too short.")

    augmented = [(y, sr)]  # Always include original

    # Add noise
    y_noise = y + np.random.randn(len(y)) * 0.005
    augmented.append((y_noise, sr))

    # Pitch Shift
    try:
        y_shift = librosa.effects.pitch_shift(y = y, sr = sr, n_steps = 2)
        augmented.append((y_shift, sr))
    except Exception as e:
        print(f"Skipping pitch shift: {e}")

    # Try time stretch
    try:
        y_stretch = librosa.effects.time_stretch(y, rate=0.8)
        augmented.append((y_stretch, sr))
    except Exception as e:
        print(f"Skipping time stretch: {e}")

    return augmented
