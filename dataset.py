import os
from preprocess import extract_mfcc, extract_spectrogram
from data_augmentation import augment_audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_dataset(data_dir):
    x_mfcc, x_spec, y = [], [], []

    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)

        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            try:
                augmented_audio = augment_audio(file_path)

                for y_audio, sr in augmented_audio:
                    if not isinstance(y_audio, np.ndarray):
                        print(f"Skipping non-array audio in {file_path}")
                        continue

                    mfcc = extract_mfcc(y_audio, sr) if y_audio is not None else None
                    spectrogram = extract_spectrogram(y_audio, sr) if y_audio is not None else None


                    if mfcc is not None and spectrogram is not None:
                        x_mfcc.append(mfcc)
                        x_spec.append(spectrogram)
                    elif mfcc is not None:
                        x_mfcc.append(mfcc)
                        x_spec.append(spectrogram)
                    elif spectrogram is not None:
                        x_spec.append(spectrogram)
                        x_mfcc.append(np.zeros_like(spectrogram))
                    else:
                        print(f"Skipping {file_path} - No valid features extracted")

                    y.append(label)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if len(x_mfcc) == 0 or len(y) == 0:
        raise ValueError("Dataset is empty after processing. Check data quality or augment_audio logic.")

    x_mfcc = np.array(x_mfcc)
    x_spec = np.array(x_spec)

    print(f"\n✅ Total Samples: {len(x_mfcc)}")
    print(f"✅ Samples with MFCC only: {np.sum(np.all(x_spec == 0, axis=(1, 2)))}")
    print(f"✅ Samples with Spectrogram only: {np.sum(np.all(x_mfcc == 0, axis=(1, 2)))}")
    print(f"✅ Samples with Both Features: {np.sum((np.any(x_mfcc != 0, axis=(1, 2)) & np.any(x_spec != 0, axis=(1, 2))))}\n")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    x_mfcc_train, x_mfcc_test, x_spec_train, x_spec_test, y_train, y_test = train_test_split(x_mfcc, x_spec, y_categorical, test_size = 0.2, random_state = 42)

    return ([x_mfcc_train, x_spec_train], [x_mfcc_test, x_spec_test], y_train, y_test), label_encoder