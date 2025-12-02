# src/evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from data_prep import get_data_generators, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

ARTIFACTS_DIR = "../src/artifacts"
MODEL_PATH = "../src/artifacts/best_model.h5"



def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Normalize by row (true labels) for easier reading (optional)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm_norm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j, i, str(value),
                horizontalalignment="center",
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_model():
    # Load the same generators – we only need the validation one here
    train_gen, val_gen = get_data_generators(data_dir="../data")

    # Load best model
    model = load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # True labels (integer indices)
    y_true = val_gen.classes

    # Predictions
    y_prob = model.predict(val_gen)
    y_pred = np.argmax(y_prob, axis=1)

    # Map indices back to class names
    class_indices = train_gen.class_indices  # dict: class_name -> index
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("Class indices:", class_indices)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_out_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    plot_confusion_matrix(cm, class_names, cm_out_path)
    print(f"Confusion matrix saved to {cm_out_path}")


if __name__ == "__main__":
    evaluate_model()
