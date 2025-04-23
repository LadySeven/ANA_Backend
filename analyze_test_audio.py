import librosa
import numpy as np
import os
import soundfile as sf
from preprocess import extract_mfcc
from fft_db import compute_fft_db

def analyze_test_audio (file_path, model, label_encoder, chunk_duration = 5):
    y, sr = librosa.load(file_path, sr = 22050)
    chunk_samples = chunk_duration * sr
    num_chunks = len(y)

    peak_db = -np.inf
    peak_time = 0
    peak_noise = None
    predictions = []

    for i in range (num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = y [start : end]

        if len(chunk) < chunk_samples:
            break

        temp_chunk_path = "temp_chunk.wav"
        sf.write(temp_chunk_path, chunk, sr)

        mfcc = extract_mfcc(temp_chunk_path)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        pred = model.predict (mfcc)
        pred_class = np.argmax (pred)
        label = label_encoder.inverse_transform([pred_class])[0]
        predictions.append ((i * chunk_duration, label))

        db, max_db, peak_time_chunk = compute_fft_db (temp_chunk_path)

        if max_db > peak_db:
            peak_db = max_db
            peak_time = peak_time_chunk
            peak_noise = label

        os.remove (temp_chunk_path)

    print(f"\n📈 Peak dB reached: {peak_db:.2f} dB at {peak_time}s")
    print(f"🔊 Detected noise at peak: {peak_noise}")
    print("\n🧠 All detections:")

    for time,label in predictions:
        print (f" - At {time:>4}s: {label}")