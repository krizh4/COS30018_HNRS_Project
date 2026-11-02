"""
Classic (non-neural) machine learning models for MNIST, to compare with the neural networks

    python train_classic.py            trains all three
    python train_classic.py svm        trains one

Models:
    knn   k-nearest neighbours: a new digit gets the label most common among the 3
          training digits that look most like it (no real "training", just remembering).
          PCA first shrinks 784 pixels to 50 features, so it is faster and much smaller
    rf    random forest: 100 decision trees that each vote; good with raw pixel values
    svm   support vector machine (RBF kernel) after PCA shrinks 784 pixels to 50 features,
          trained on 20,000 digits because an SVM gets very slow with more

Output:
    checkpoints/<name>.joblib            the trained model (the app and predictor.py load it)
    results/classic_results.csv          accuracy and timing of every model
"""
import csv
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from models.cnn import ROOT

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_FILE = Path("results") / "classic_results.csv"
SEED = 42

CLASSIC_MODELS = {
    "knn": (lambda: make_pipeline(PCA(n_components=50, random_state=SEED),
                                  KNeighborsClassifier(n_neighbors=3, n_jobs=-1)), 60000),
    "rf": (lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                          random_state=SEED), 60000),
    # CalibratedClassifierCV gives the SVM probabilities (the app shows a confidence per digit)
    "svm": (lambda: make_pipeline(PCA(n_components=50, random_state=SEED),
                                  CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"),
                                                         ensemble=False, cv=3)), 20000),
}


def load_mnist(train):
    """MNIST as flat rows of 784 numbers between 0 and 1 (the same scale the pipeline gives)"""
    prefix = "train" if train else "t10k"
    raw = ROOT / "MNIST" / "raw"
    images = np.fromfile(raw / f"{prefix}-images-idx3-ubyte", np.uint8, offset=16)
    labels = np.fromfile(raw / f"{prefix}-labels-idx1-ubyte", np.uint8, offset=8)
    return images.reshape(-1, 784).astype(np.float32) / 255.0, labels


def train_classic(name):
    make_model, train_size = CLASSIC_MODELS[name]
    x_train, y_train = load_mnist(train=True)
    x_test, y_test = load_mnist(train=False)
    pick = np.random.default_rng(SEED).permutation(len(x_train))[:train_size]

    print(f"Training {name} on {train_size} digits ...")
    model = make_model()
    start = time.time()
    model.fit(x_train[pick], y_train[pick])
    train_seconds = time.time() - start

    start = time.time()
    accuracy = 100.0 * (model.predict(x_test) == y_test).mean()
    test_seconds = time.time() - start
    print(f"{name}: test accuracy {accuracy:.2f}%  (train {train_seconds:.0f}s, test {test_seconds:.0f}s)")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, CHECKPOINT_DIR / f"{name}.joblib", compress=3)   # compressed: smaller file

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    new_file = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["model", "train_digits", "test_acc", "train_seconds", "test_seconds"])
        writer.writerow([name, train_size, round(accuracy, 2), round(train_seconds, 1), round(test_seconds, 1)])
    return accuracy


if __name__ == "__main__":
    for model_name in sys.argv[1:] or CLASSIC_MODELS:
        train_classic(model_name)