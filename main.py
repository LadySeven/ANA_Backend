from analyze_test_audio import analyze_test_audio
from train import train_model
from evaluate import evaluate_model

if __name__ == '__main__':
    DATA_DIR = 'path/to/audio_dataset'
    TEST_DIR = 'path/to/test_dataset/traffic sound.wav'

    model, x_test, y_test, label_encoder = train_model(DATA_DIR)
    evaluate_model(model, [x_test[0], x_test[1]], y_test, label_encoder)

    # analyze_test_audio (TEST_DIR, model, label_encoder)