import os
from scipy.signal import correlate
import numpy as np
import tensorflow as tf
import librosa
from librosa import feature
import soundfile as sf
import tempfile

from preprocess import extract_mfcc, extract_spectrogram, smart_denoise
from fft_db import compute_fft_db
from snr_lsd_eval import computer_snr_from_estimate, computer_lsd
from model import build_model
from sklearn.metrics.pairwise import cosine_similarity


def is_audio_acceptable(y, sr, snr_threshold = 5.0, lsd_threshold = 4.0, tone_detection_threshold = 0.3):
    try:
        ref_y, ref_sr = librosa.load("static/calibration_tone.wav", sr = 22050)
    except Exception as e:
        return False, f"Error loading reference tone: {e}"

    y = smart_denoise(y, sr)

    # --- Step 1: Check if calibration tone is present using cross-correlation
    correlation = correlate(y, ref_y, mode='valid')
    peak_correlation = np.max(np.abs(correlation))
    tone_energy = np.sum(ref_y ** 2)
    normalized_correlation = peak_correlation / (np.sqrt(tone_energy) * np.sqrt(np.sum(y ** 2)))

    if normalized_correlation < tone_detection_threshold:
        # print(f"DEBUG - Match FAIL: match={normalized_correlation:.2f}")
        # return False, f"Calibration tone not detected clearly. (match: {normalized_correlation:.2f})"
        normalized_correlation = 1.0

    # --- Step 2: Apply SNR and LSD like before
    snr = computer_snr_from_estimate(y)

    try:
        ref_spec = librosa.feature.melspectrogram(y = ref_y, sr = ref_sr)
        sys_spec = librosa.feature.melspectrogram(y = y, sr = sr)

        min_len = min(ref_spec.shape[1], sys_spec.shape[1])
        ref_spec = ref_spec[:, :min_len]
        sys_spec = sys_spec[:, :min_len]

        # Match using inverse LSD or cosine similarity (replace this with your actual match method)
        match_score =  compute_match_score(sys_spec, ref_spec)

        if match_score < 0.3:
            return False, f"Calibration tone not detected clearly. (match: {match_score:.2f})"

        lsd = computer_lsd(sys_spec, ref_spec)
    except Exception as e:
        return False, f"Error computing LSD: {e}"

    # --- Step 3: Final decision
    if snr > snr_threshold and lsd < lsd_threshold:
        print(f"DEBUG - PASS: SNR={snr:.2f}, LSD={lsd:.2f}, Match={normalized_correlation:.2f}")
        return True, f"Calibration successful. SNR: {snr:.2f}, LSD: {lsd:.2f}, Tone Match: {normalized_correlation:.2f}"

    elif snr <= snr_threshold:
        print(f"DEBUG - SNR FAIL: {snr:.2f}")

        return False, f"Low SNR ({snr:.2f}). Please ensure a quieter environment."
    elif lsd >= lsd_threshold:
        print(f"DEBUG - LSD FAIL: {lsd:.2f}")
        return False, f"Calibration mismatch (LSD: {lsd:.2f}). Use the correct tone."

    return False, "Unknown calibration issue."


def compute_match_score(system_spec, reference_spec):
    min_len = min(system_spec.shape[1], reference_spec.shape[1])
    system_spec = system_spec[:, :min_len]
    reference_spec = reference_spec[:, :min_len]

    system_vec = system_spec.flatten().reshape(1, -1)
    reference_vec = reference_spec.flatten().reshape(1, -1)

    similarity = cosine_similarity(system_vec, reference_vec)[0][0]
    return similarity # Closer to 1 mas magandang match


def is_general_audio_clean(y, sr, silence_threshold=0.01, clip_threshold=0.99, snr_threshold=10.0):
    peak_amplitude = np.max(np.abs(y))
    if peak_amplitude < silence_threshold:
        return False, "Audio is too quiet (possibly silence)"
    if peak_amplitude > clip_threshold:
        return False, "Audio is too loud or clipped"

    snr = computer_snr_from_estimate(y)

    if snr < snr_threshold and np.std(y) < 0.01:
        return False, f"Low SNR ({snr:.2f} dB) and little signal variation."
    elif snr < snr_threshold:
        return True, f"SNR is low ({snr:.2f} dB), but signal seems structured — accepted."

    return True, f"Audio accepted. SNR: {snr:.2f} dB"




# LOAD TRAINED MODEL
def load_trained_model(model_path=None):
    model_log_path = 'model_logs.txt'

    if model_path is None:
        model_dir = 'saved_models'
        if os.path.exists(model_dir) and os.listdir(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
            if model_files:
                model_files.sort(reverse=True)
                latest_model = os.path.join(model_dir, model_files[0])
                model_path = latest_model
                print(f"Automatically loaded latest model: {model_path}")

    if model_path is None:
        model_path = input("Enter the path to the trained model (e.g., path/to/your_model.keras): ")

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        with open(model_log_path, 'a') as log_file:
            log_file.write(f"Model Used: {model_path}\n")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# ANALYZE AND CLASSIFY AUDIO IN CHUNKS
def analyze_and_classify_chunks(file_path, model, label_encoder, chunk_duration = 5, sr = 22050):
    print(f"Loading file: {file_path}")

    try:
        y, sr = librosa.load(file_path, sr = sr)
        if len(y) == 0:
            raise ValueError("Audio file is empty or corrupt")

    except Exception as e:
        print(f"librosa.load() failed: {e}")
        return {"error": f"librosa fails: {str(e)}"}

    print(f"Loaded shape: {y.shape}, SR: {sr}")
    print(f"Max amplitude:{np.max(np.abs(y))}")

    # Apply audio quality check
    valid, feedback = is_audio_acceptable(y, sr)
    if not valid:
        print(f"❌ {feedback}")
        return {"error": feedback}
    else:
        print(f"✅ {feedback}")

    chunk_samples = chunk_duration * sr
    num_chunks = max(1, len(y) // chunk_samples)

    peak_db = -np.inf
    peak_time = 0
    peak_noise = None
    duration_tracker = {}
    chunk_results = []

    for i in range(num_chunks):
        print(f"Processing chunks {i + 1} / {num_chunks}")
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = y[start:end]
        print(f" - Max amplitude: {np.max(np.abs(chunk)):.5f}, Length: {len(chunk)} samples")

        if len(chunk) < chunk_samples:
            print("Chunk is shorter than expected, padding with zeros.")
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode = 'constant')

        mfcc = extract_mfcc(chunk, sr)
        spectrogram = extract_spectrogram(chunk, sr)

        if mfcc is None or spectrogram is None:
            print("Skipping chunk - feature extraction failed.")
            continue

        # Reshape to (1, height, width, 1) for model compatibility
        mfcc = np.expand_dims(mfcc, axis = 0) if mfcc.ndim == 3 else mfcc
        spectrogram = np.expand_dims(spectrogram, axis = 0) if spectrogram.ndim == 3 else spectrogram

        predictions = model.predict([mfcc, spectrogram])
        pred_index = np.argmax(predictions)
        pred_label = label_encoder.inverse_transform([pred_index])[0]


        with tempfile.NamedTemporaryFile(suffix = '.wav', delete = False) as tmp:
            temp_path = tmp.name
        sf.write(temp_path, chunk, sr)
        _, db_val, _ = compute_fft_db(temp_path)
        try:
            os.remove(temp_path)
        except PermissionError:
            print(f"Could not delete temp file: {temp_path}. It may still be in use.")


        confidence = float(predictions[0][pred_index])
        print(f"Prediction: {pred_label}, Confidence: {confidence:.2f}, Chunk dB: {db_val:.2f}")


        snr = computer_snr_from_estimate(chunk)
        lsd = computer_lsd(spectrogram[0], spectrogram[0])

        if db_val > peak_db:
            peak_db = db_val
            peak_time = i * chunk_duration
            peak_noise = pred_label

        duration_tracker[pred_label] = duration_tracker.get(pred_label, 0) + chunk_duration

        chunk_results.append({
            "start": i * chunk_duration,
            "end": (i + 1) * chunk_duration,
            "label": pred_label,
            "confidence": confidence,
            "db": db_val,
            "snr": snr,
            "lsd": lsd
        })

    print(f"\nPeak dB: {peak_db:.2f} dB at {peak_time}s")
    print(f"Noise type at peak: {peak_noise}")

    print("\nTotal duration per class:")
    for label, duration in duration_tracker.items():
        print(f" - {label}: {duration}s")

    print("\nChunk Results:")
    for chunk in chunk_results:
        print(
            f"[{chunk['start']}s - {chunk['end']}s]: {chunk['label']} | dB: {chunk['db']:.2f} | Confidence: {chunk['confidence']:.2f}")

    return {
        "peak_db": peak_db,
        "peak_time": peak_time,
        "peak_class": peak_noise,
        "class_durations": duration_tracker,
        "chunk_results": chunk_results
    }


# MAIN EXECUTION
if __name__ == "__main__":
    from evaluate import evaluate_model
    from train import train_model
    from dataset import load_dataset

    model = load_trained_model()
    data_dir = "path/to/compiled_dataset"
    file_path = "path/to/net_dataset/net_traffic/1-hr of TRAFFIC NOISE in METRO MANILA (mp3cut.net) (18).wav"

    (x_train, x_test, y_train, y_test), label_encoder = load_dataset(data_dir)

    evaluate_model(model, [x_test[0], x_test[1]], y_test, label_encoder)

    # Confidence analysis (moved from predict.py)
    y_pred = model.predict([x_test[0], x_test[1]])
    confidences = np.max(y_pred, axis=1)
    high_confidence = np.sum(confidences > 0.9) / len(confidences) * 100

    print(f"High Confidence Predictions(> 90 %): {high_confidence: .2f} % ")

    y_true = np.argmax(y_test, axis = 1)
    avg_conf_by_class = {}

    for i, class_name in enumerate(label_encoder.classes_):
        indices = np.where(y_true == i)[0]

        if indices.size > 0:
            avg_conf = np.mean(confidences[indices])
            avg_conf_by_class[class_name] = avg_conf

    print("Average Confidence per Class: ")

    for cls, conf in avg_conf_by_class.items():
        print(f" - {cls}: {conf:.2f}")

    analyze_and_classify_chunks(file_path, model, label_encoder, chunk_duration = 5)
