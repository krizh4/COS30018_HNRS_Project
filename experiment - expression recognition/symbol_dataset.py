"""
This module provides functions to load and prepare a dataset of handwritten mathematical symbols
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image

import my_preprocessing as pre

LABEL_NAMES = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "+", 11: "-", 12: "x", 13: "/", 14: "(", 15: ")",
}
OPERATOR_FOLDERS = {
    10: "plus", 11: "minus", 12: "multiply", 13: "divide",
    14: "open_paren", 15: "close_paren",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "operators")


def _load_images_from_folder(folder: str):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    images = []
    for p in paths:
        img = Image.open(p)
        if img is not None:
            images.append(img)
    return images


def _prepare_symbol_image(pil_img, method="otsu") -> np.ndarray:
    """Runs a raw photo of one symbol through the SAME pipeline used at
    inference time (symbol_segmentation.py): grayscale -> binarize ->
    invert -> resize_and_pad. Returns a 28x28 float32 array in [0,1]."""
    gray = pre.to_grayscale(pil_img)
    binary = pre.BINARIZATION_METHODS[method](gray)
    inverted = 255 - binary
    padded = pre.resize_and_pad(inverted, size=28)
    return np.array(padded).astype("float32") / 255.0


def _augment(img_28x28: np.ndarray, n_variants: int = 40, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    h, w = img_28x28.shape
    out = [img_28x28]
    img_uint8 = (img_28x28 * 255).astype(np.uint8)
    for _ in range(n_variants - 1):
        angle = rng.uniform(-15, 15)
        tx, ty = rng.uniform(-2, 2), rng.uniform(-2, 2)
        scale = rng.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        warped = cv2.warpAffine(img_uint8, M, (w, h), borderValue=0)
        out.append(warped.astype("float32") / 255.0)
    return out


def _synthetic_symbol(name: str, size: int = 28) -> np.ndarray:
    """Fallback only - a printed-style placeholder, NOT a substitute for
    real handwriting. White ink on black background to match the real
    pipeline's output convention."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    text = {"plus": "+", "minus": "-", "multiply": "x",
            "divide": "/", "open_paren": "(", "close_paren": ")"}[name]
    cv2.putText(canvas, text, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    return canvas.astype("float32") / 255.0


def load_symbol_class(label: int, variants_per_real_sample: int = 40):
    folder_name = OPERATOR_FOLDERS[label]
    folder = os.path.join(DATA_DIR, folder_name)
    raw_images = _load_images_from_folder(folder)

    X = []
    if raw_images:
        print(f"  '{LABEL_NAMES[label]}': found {len(raw_images)} real sample(s) in {folder}")
        for img in raw_images:
            prepped = _prepare_symbol_image(img)
            X.extend(_augment(prepped, n_variants=variants_per_real_sample))
    else:
        print(f"  '{LABEL_NAMES[label]}': NO real samples found in {folder} -> using synthetic placeholder")
        synth = _synthetic_symbol(folder_name)
        X.extend(_augment(synth, n_variants=variants_per_real_sample))

    X = np.array(X, dtype="float32")
    y = np.full(len(X), label, dtype="int64")
    return X, y


def build_combined_dataset(mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test):
    """mnist_x_* are expected already normalized to [0,1], shape (N,28,28)."""
    op_X, op_y = [], []
    print("Building symbol dataset (operators + parentheses):")
    for label in OPERATOR_FOLDERS:
        X, y = load_symbol_class(label)
        op_X.append(X)
        op_y.append(y)
    op_X = np.concatenate(op_X, axis=0)
    op_y = np.concatenate(op_y, axis=0)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(op_X))
    op_X, op_y = op_X[idx], op_y[idx]
    split = int(0.8 * len(op_X))
    op_x_train, op_y_train = op_X[:split], op_y[:split]
    op_x_test, op_y_test = op_X[split:], op_y[split:]

    x_train = np.concatenate([mnist_x_train, op_x_train], axis=0)
    y_train = np.concatenate([mnist_y_train, op_y_train], axis=0)
    x_test = np.concatenate([mnist_x_test, op_x_test], axis=0)
    y_test = np.concatenate([mnist_y_test, op_y_test], axis=0)

    tr_idx = rng.permutation(len(x_train))
    te_idx = rng.permutation(len(x_test))
    return x_train[tr_idx], y_train[tr_idx], x_test[te_idx], y_test[te_idx]
