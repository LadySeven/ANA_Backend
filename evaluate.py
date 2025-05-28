from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, x_test, y_test, label_encoder, save_dir = "evaluation_logs"):
    os.makedirs(save_dir, exist_ok = True)

    if isinstance(x_test, list) and len(x_test) == 2:
        y_pred = model.predict([x_test[0], x_test[1]])

    else:
        y_pred = model.predict(x_test)

    y_pred_classes = np.argmax(y_pred, axis = 1)
    y_true = np.argmax(y_test, axis = 1)

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names = label_encoder.classes_)
    print(report)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, label_encoder, save_dir)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average = 'weighted')
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_roc_curve(y_test, y_pred, save_dir)

    # Confidence Analysis
    confidences = np.max(y_pred, axis = 1)
    high_confidence = np.sum(confidences > 0.9) / len(confidences) * 100
    print(f"\n High Confidence Predictions (>90%): {high_confidence:.2f}%")

    avg_conf_by_class = {}

    for i, class_name in enumerate(label_encoder.classes_):
        indices = np.where(y_true == i)[0]
        if indices.size > 0:
            avg_conf = np.mean(confidences[indices])
            avg_conf_by_class[class_name] = avg_conf

    print("\n Average Confidence per Class:")

    for cls, conf in avg_conf_by_class.items():
        print(f" - {cls}: {conf:.2f}")

    plot_roc_curve(y_test, y_pred, save_dir)


def plot_confusion_matrix(cm, label_encoder, save_dir):
    plt.figure(figsize = (10, 8))
    sns.heatmap(cm,
                annot = True,
                fmt = 'd',
                cmap = 'Blues',
                cbar = False,
                xticklabels = label_encoder.classes_,
                yticklabels = label_encoder.classes_)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, save_dir):
    plt.figure(figsize = (10, 8))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range (len(y_test[0])):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label = f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')

    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.show()
