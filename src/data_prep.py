# src/data_prep.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions and batch size
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def get_data_generators(data_dir="data"):
    """
    Loads training and validation images from directory structure:
    
    data/
        train/
            cricket/
            football/
            basketball/
            tennis/
        val/
            cricket/
            football/
            basketball/
            tennis/
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Data augmentation for training images
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    # Only normalize validation images
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Load training dataset
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Load validation dataset
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen

if __name__ == "__main__":
    train_gen, val_gen = get_data_generators()
    print("Train samples:", train_gen.samples)
    print("Validation samples:", val_gen.samples)
    print("Classes:", train_gen.class_indices)
