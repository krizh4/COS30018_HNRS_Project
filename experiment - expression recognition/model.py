"""16-class CNN: digits 0-9 plus +, -, x, /, (, ) - see symbol_dataset.py
for the label mapping."""

from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 16


def build_symbol_cnn(input_shape=(28, 28, 1), num_classes=NUM_CLASSES) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
