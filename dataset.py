import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from preprocess import extract_mfcc


def load_dataset (data_dir):
    x, y = [], []
    labels = os.listdir(data_dir)

    for label in labels:
        folder = os.path.join(data_dir, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            mfcc = extract_mfcc(path)
            x.append (mfcc)
            y.append (label)

    x = np.array(x)[..., np.newaxis]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y = to_categorical(y_encoded)

    return train_test_split(x, y, test_size = 0.2, random_state = 42), le


