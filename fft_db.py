import librosa
import numpy as np

def compute_fft_db (file_path):
    y, sr = librosa.load(file_path, sr = 22050)

    stft = librosa.stft(y)
    magnitude = np.abs(stft)

    db = librosa.amplitude_to_db(magnitude, ref = 1.0)

    time_index = np.argmax(np.max(db, axis = 0)) # max db per time frame
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr = sr)
    peak_time = times[time_index]
    peak_db = np.max(db)



    return db, peak_db, peak_time

