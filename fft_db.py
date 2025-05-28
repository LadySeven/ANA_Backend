import librosa
import numpy as np

def compute_fft_db(file_path, n_fft = 2048, hop_length = 512, sr = 22050):
    y, sr = librosa.load(file_path, sr = sr)

    # Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
    magnitude = np.abs(stft)

    # Calculate Real-World dB SPL (Sound Pressure Level)
    reference_pressure = 20e-6  # 20 µPa (Standard for dB SPL)
    max_mag = np.max(magnitude)

    if max_mag <= 0:
        real_world_db = -np.inf

    else:
        real_world_db = 20 * np.log10(np.max(magnitude) / reference_pressure)

    # Peak dB Calculation (Real-World)
    peak_db = real_world_db
    peak_time_index = np.argmax(np.max(magnitude, axis = 0))
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr = sr, hop_length=hop_length)
    peak_time = times[peak_time_index]

    return magnitude, peak_db, peak_time
