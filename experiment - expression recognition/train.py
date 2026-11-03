"""
Trains the 16-class symbol model (digits + operators + parentheses).

Run:
    python train.py
    python train.py --quick     (tiny subset, fast smoke-test)
"""

import os
import sys
import numpy as np
from tensorflow import keras

from model import build_symbol_cnn
from symbol_dataset import build_combined_dataset, LABEL_NAMES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def train(epochs: int = 8, quick: bool = False):
    (mx_train, my_train), (mx_test, my_test) = load_mnist()

    if quick:
        mx_train, my_train = mx_train[:3000], my_train[:3000]
        mx_test, my_test = mx_test[:800], my_test[:800]
        epochs = 2

    x_train, y_train, x_test, y_test = build_combined_dataset(mx_train, my_train, mx_test, my_test)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"\nTraining set: {x_train.shape[0]} samples | Test set: {x_test.shape[0]} samples")

    model = build_symbol_cnn()
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nOverall test accuracy (digits + operators + parens): {acc:.4f}")

    preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nPer-class accuracy:")
    for label in sorted(LABEL_NAMES):
        mask = y_test == label
        if mask.sum() == 0:
            continue
        class_acc = (preds[mask] == label).mean()
        print(f"  '{LABEL_NAMES[label]}': {class_acc:.2%}  (n={mask.sum()})")

    model.save(os.path.join(MODELS_DIR, "symbol_cnn.keras"))
    print(f"\nSaved model to {os.path.join(MODELS_DIR, 'symbol_cnn.keras')}")
    return model


if __name__ == "__main__":
    train(quick="--quick" in sys.argv)
