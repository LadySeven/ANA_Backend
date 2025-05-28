import librosa
from librosa import feature
import numpy as np
import noisereduce as nr


def spectral_subtraction(y, sr, noise_frames = 10):
    stft = librosa.stft(y)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_est = np.mean(magnitude[:, :noise_frames], axis = 1, keepdims = True)
    clean_magnitude = np.maximum(magnitude - noise_est, 0)
    clean_stft = clean_magnitude * np.exp(1j * phase)
    y_denoised = librosa.istft(clean_stft)
    return y_denoised


def is_stationary(y, sr, threshold = 0.05, duration = 1.0):
    num_samples = int(sr * duration)
    y_segment = y[:num_samples]
    frame_size = 1024
    hop = 512
    energies = [
        np.sum(y_segment[i: i + frame_size] ** 2)
        for i in range(0, len(y_segment) - frame_size, hop)
    ]

    energy_var = np.var(energies)
    return energy_var < threshold           # Kapag low variance ng energy = stationary


def smart_denoise(y, sr):
    if is_stationary(y, sr):
        print("Stationary noise detected - applying spectral subtraction")
        return spectral_subtraction(y, sr)
    else:
        print("Non-stationary noise detected - applying spectral gating")
        return nr.reduce_noise(y = y, sr = sr)


def extract_mfcc (y, sr = 22050, n_mfcc = 13, max_pad_len = 174, include_delta = True):
    y = smart_denoise(y, sr)
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = n_mfcc)

    if include_delta:
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order = 2)
        mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width = ((0, 0), (0, pad_width)), mode = 'constant')

    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc[..., np.newaxis]


def extract_spectrogram(y, sr = 22050, n_fft = 2048, hop_length = 512, max_pad_len = 174):
    y = smart_denoise(y, sr)
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = n_fft, hop_length = hop_length)
    spectrogram_db = librosa.power_to_db(spectrogram, ref = np.max)

    if spectrogram_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - spectrogram_db.shape[1]
        spectrogram_db = np.pad(spectrogram_db, pad_width = ((0, 0), (0, pad_width)), mode = 'constant')

    else:
        spectrogram_db = spectrogram_db[:, :max_pad_len]

    return spectrogram_db[..., np.newaxis]



