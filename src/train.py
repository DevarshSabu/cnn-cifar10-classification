# src/train.py

import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_prep import get_data_generators, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
from model import build_cnn_model

ARTIFACTS_DIR = "artifacts"

def plot_history(history, out_dir=ARTIFACTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()


def train_model(epochs=20):
    # Load data
    train_gen, val_gen = get_data_generators(data_dir="../data")

    num_classes = train_gen.num_classes
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {input_shape}")

    # Build model
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_model_path = os.path.join(ARTIFACTS_DIR, "best_model.h5")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=best_model_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_model_path = os.path.join(ARTIFACTS_DIR, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Best model saved to {best_model_path}")

    # Save accuracy/loss plots
    plot_history(history, ARTIFACTS_DIR)

    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}, Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    train_model(epochs=20)
