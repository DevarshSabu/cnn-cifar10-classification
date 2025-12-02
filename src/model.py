# src/model.py

from tensorflow.keras import layers, models, optimizers

def build_cnn_model(input_shape=(128, 128, 3), num_classes=7):
    """
    Builds a Convolutional Neural Network for sports image classification.
    """

    model = models.Sequential()

    # ---- Convolutional Block 1 ----
    model.add(layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same",
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Convolutional Block 2 ----
    model.add(layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Convolutional Block 3 ----
    model.add(layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Fully Connected Part ----
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # ---- Compile ----
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Quick test
    m = build_cnn_model()
    m.summary()
