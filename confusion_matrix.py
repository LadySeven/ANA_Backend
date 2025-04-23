import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def show_confusion_matrix (model, x_test, y_test, label_encoder):
    y_pred = model.predict (x_test)
    y_pred_classes = np.argmax (y_pred, axis = 1)
    y_true = np.argmax (y_test, axis = 1)

    cm = confusion_matrix (y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay (confusion_matrix = cm, display_labels = label_encoder.classes_)
    disp.plot (cmap = plt.cm.Blues)
    plt.title ("Confusion Matrix")
    plt.show()