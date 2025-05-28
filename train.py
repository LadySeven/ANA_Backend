import tensorflow as tf

from model import build_model
from dataset import load_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

import matplotlib.pyplot as plt
import os


def create_callbacks(model_name = "audio_cnn"):
    os.makedirs("saved_models", exist_ok = True)
    os.makedirs("logs", exist_ok = True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path_keras = f"saved_models/{model_name}_{timestamp}.keras"
    model_path_tflite = f"saved_models/{model_name}_{timestamp}.tflite"
    model_path_tflite_opt = f"saved_models/{model_name}_{timestamp}_optimized.tflite"

    callbacks = [
        ModelCheckpoint(
            model_path_keras,
            monitor = 'val_accuracy',
            save_best_only = True,
            verbose = 1,
            mode = 'max'
        ),

        EarlyStopping(
            monitor = 'val_loss',
            patience = 5,
            restore_best_weights = True,
            verbose = 1
        ),

        ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.5,
            patience = 3,
            verbose = 1,
            min_lr = 1e-6
        )
    ]

    return callbacks, model_path_keras, model_path_tflite, model_path_tflite_opt


def train_model(data_dir, num_blocks = 6):
    (x_train, x_test, y_train, y_test), label_encoder = load_dataset(data_dir)

    model = build_model(x_train[0].shape[1:], x_train[1].shape[1:], y_train.shape[1], num_blocks = num_blocks)

    callbacks, model_path_keras, model_path_tflite, model_path_tflite_opt = create_callbacks()

    history = model.fit(
        [x_train[0], x_train[1]], y_train,
        epochs = 30,
        batch_size = 32,
        validation_data = ([x_test[0], x_test[1]], y_test),
        callbacks = callbacks,
        verbose = 1
    )

    model.save(model_path_keras)

    # CONVERT HERE TO TF-LITE

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(model_path_tflite, "wb") as f:
        f.write(tflite_model)

    # Convert to TF-Lite (Optimized - Quantized)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_optimized_model = converter.convert()

    with open(model_path_tflite_opt, "wb") as f:
        f.write(tflite_optimized_model)

    print(f"Model saved as: {model_path_keras}, {model_path_tflite} and {model_path_tflite_opt}")

    plot_metrics(history)

    return model, x_test, y_test, label_encoder

def plot_metrics(history):
    plt.figure(figsize = (12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()




