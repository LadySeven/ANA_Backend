from model import build_model
from dataset import load_dataset
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def train_model(data_dir):
    (x_train, x_test, y_train, y_test), label_encoder = load_dataset(data_dir)
    model =build_model(x_train.shape[1:], y_train.shape[1])

    early_stopping = EarlyStopping(monitor = 'val_loss', patience =  5, restore_best_weights = True)

    history = model.fit(x_train,
                        y_train,
                        epochs = 30,
                        batch_size = 32,
                        validation_data = (x_test, y_test)
                        )

    plot_metrics(history)

    return model, x_test, y_test, label_encoder

def plot_metrics(history):
    plt.figure (figsize = (12, 6))

    plt.subplot(1, 2, 1)
    plt.plot (history.history['accuracy'], label = 'Training Accuracy')
    plt.plot (history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.title ('Training and Validation Accuracy')
    plt.xlabel ('Epochs')
    plt.ylabel ('Accuracy')
    plt.legend ()

    plt.subplot(1, 2, 2)
    plt.plot (history.history['loss'], label = 'Training Loss')
    plt.plot (history.history['val_loss'], label = 'Validation Loss')
    plt.title ('Loss')
    plt.xlabel ('Epochs')
    plt.ylabel ('Loss')
    plt.legend ()

    plt.tight_layout ()
    plt.show ()
