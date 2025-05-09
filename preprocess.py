import librosa
from librosa import feature
import numpy as np


def extract_mfcc (y, sr = 22050, n_mfcc = 13, max_pad_len = 174):
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = n_mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width = ((0, 0), (0, pad_width)), mode = 'constant')

    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = mfcc[..., np.newaxis]
    return mfcc


def extract_spectrogram(y, sr = 22050, n_fft = 2048, hop_length = 512, max_pad_len = 174):
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = n_fft, hop_length = hop_length)
    spectrogram_db = librosa.power_to_db(spectrogram, ref = np.max)

    if spectrogram_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - spectrogram_db.shape[1]
        spectrogram_db = np.pad(spectrogram_db, pad_width = ((0, 0), (0, pad_width)), mode = 'constant')

    else:
        spectrogram_db = spectrogram_db[:, :max_pad_len]

    spectrogram_db = spectrogram_db[..., np.newaxis]
    return spectrogram_db


