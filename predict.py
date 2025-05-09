import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import config

from preprocess import extract_mfcc
from fft_db import compute_fft_db
from snr_lsd_eval import computer_snr, computer_lsd

def load_trained_model(model_path = config.MODEL_PATH):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded form {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_audio(file_path):

    try:
        y, sr = librosa.load(file_path, sr = config.SAMPLE_RATE)

        mel_spec = librosa.feature.melspectogram(y = y, sr = sr, n_mels = config.N_MELS, hop_length = config.HOP_LENGTH)

        mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max)
        mel_spec_db = (mel_spec_db -  np.mean(mel_spec_db)) / np.std(mel_spec_db)

        return mel_spec_db

    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def predict_audio_class(file_path, model = None, model_path = config.MODEL_PATH):

    if model is None:
        model = load_trained_model(model_path)
        if model is None:
            return {"error": "Failed to load model"}

    mel_spec = preprocess_audio(file_path)
    if mel_spec is None:
        return {"error": f"Failed to process audio file: {file_path}"}

    mel_spec = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)

    predictions = model.predict(mel_spec)
    predicted_class = np.argmax(predictions)

    db, peak_db, peak_time = compute_fft_db(file_path)
    y, sr = librosa.load(file_path, sr = config.SAMPLE_RATE)
    snr = computer_snr(y, db)
    lsd = computer_lsd(y, db)

    result = {
        "class": config.CATEGORIES[predicted_class],
        "confidence": float(predictions[0][predicted_class]),
        "all_probabilities": {cat: float(prob) for cat, prob in zip(config.CATEGORIES, predictions[0])},
        "peak_db": peak_db,
        "peak_time": peak_time,
        "snr": snr,
        "lsd": lsd
    }

    return result

def predict_directory(directory_path, model = None, model_path = config.MODEL_PATH):
    if model is None:
        model = load_trained_model(model_path)
        if model is None:
            return {"error": "Failed to load model"}

    results = {}

    print(f"Processing files in {directory_path}...")

    if not os.path.exists(directory_path):
        return {"error": f"Directory not found: {directory_path}"}

    files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.wav', '.mp3', '.ogg'))]

    if not files:
        return {"error": f"No audio files found in {directory_path}"}

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        print(f"Processing {file_name}...")

        try:
            result = predict_audio_class(file_path, model = model, model_path = model_path)
            results[file_name] = result
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            results[file_name] = {"error": str(e)}

    summarize_predictions(results)

    return results


def summarize_predictions(results):
    class_counts = {}

    for file_name, result in results.items():
        if "error" not in result:
            class_name = result["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nPrediction Summary:")
    print("=" * 50)

    total_files = len(results)
    errors = sum(1 for result in results.values() if "error" in result)
    successful = total_files - errors

    print(f"Total files processed: {total_files}")
    print(f"Successful predictions: {successful}")
    print(f"Errors: {errors}")

    if successful > 0:
        print("\nClass Distribution:")
        for class_name, count in sorted(class_counts.items(), key = lambda x: x[1], reverse = True):
            percentage = (count / successful) * 100
            print(f" {class_name}: {count} files ({percentage:.1f}%)")


def real_time_prediction(model = None, model_path = config.MODEL_PATH, duration = 5, sr = config.SAMPLE_RATE):
    try:
        import pyaudio
        import wave
        import tempfile

    except ImportError:
        print("Error: pyaudio package is required for real-time prediction.")
        print("Install it using 'pip install pyaudio'.")
        return

    if model is None:
        model = load_trained_model(model_path)
        if model is None:
            print(f"Error: Failed to load model from {model_path}.")
            return

    chunk = 1024
    format = pyaudio.paInt16
    channels = 1

    print(f"Recording {duration} seconds of audio...")

    p = pyaudio.PyAudio()

    stream = p.open(
        format = format,
        channels = channels,
        rate = sr,
        input = True,
        frames_per_buffer = chunk
    )

    frames = []

    for i in range(0, int(sr / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording finished.")

    with tempfile.NamedTemporaryFile(suffix = '.wav', delete = False) as temp_file:
        temp_filename = temp_file.name

        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sr)
        wf.writeframes(b''.join(frames))
        wf.close()

        result = predict_audio_class(temp_filename, model = model, model_path = model_path)
        os.unlink(temp_filename)

        return result

if __name__ == "__main__":
    # Example usage for predicting an audio file
    file_path = 'path_to_your_audio_file.wav'  # Example audio file path

    # Predict a single file
    result = predict_audio_class(file_path)

    if "error" not in result:
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print("All Probabilities:")

        for category, prob in result["all_probabilities"].items():
            print(f"  {category}: {prob * 100:.2f}%")

        print(f"Peak dB: {result['peak_db']} at {result['peak_time']}s")
        print(f"SNR: {result['snr']}")
        print(f"LSD: {result['lsd']}")

    else:
        print(result["error"])

    # Example usage for predicting multiple files in a directory
    directory_path = 'path_to_audio_directory'  # Folder containing multiple audio files
    results = predict_directory(directory_path)

    # Show the results
    for file_name, result in results.items():
        print(f"File: {file_name}, Predicted Class: {result['class']}, Confidence: {result['confidence'] * 100:.2f}%")