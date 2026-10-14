"""
Classical machine learning baselines on MNIST (Week 6 lecture), using scikit-learn.

python baselines.py

Each picture is flattened into 784 numbers between 0 and 1.
Output:
    results/baselines.csv           accuracy, F1 and timing for every model
    checkpoints/<name>.joblib       each trained model except k-NN (so the GUI can use it too)
"""
import csv
import time
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from torchvision import datasets

DATA_ROOT = Path("data")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

def load_mnist_as_numpy():
    """Return X_train, y_train, X_test, y_test as NumPy arrays (X values 0..1)."""
    train = datasets.MNIST(DATA_ROOT, train=True, download=False)
    test = datasets.MNIST(DATA_ROOT, train=False, download=False)

    X_train = train.data.numpy().reshape(-1, 784) / 255.0
    X_test = test.data.numpy().reshape(-1, 784) / 255.0

    return X_train, train.targets.numpy(), X_test, test.targets.numpy()

# name -> the untrained model. Change settings here to experiment (e.g. n_neighbors).
BASELINES = {
    "logistic_regression": LogisticRegression(max_iter=200),
    "knn_1": KNeighborsClassifier(n_neighbors=1),
    "knn_3": KNeighborsClassifier(n_neighbors=3),
    "knn_5": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42),
}

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_as_numpy()
    print("train", X_train.shape, "test", X_test.shape)

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    rows = []
    for name, model in BASELINES.items():
        start = time.time()
        model.fit(X_train, y_train)  # learn from the training pictures
        train_seconds = time.time() - start

        start = time.time()
        predictions = model.predict(X_test)  # guess the test pictures
        predict_seconds = time.time() - start

        acc = 100 * accuracy_score(y_test, predictions)
        f1 = 100 * f1_score(y_test, predictions, average="macro")

        print(f"{name:20s} accuracy {acc:.2f}% F1 {f1:.2f} train {train_seconds:.0f}s predict {predict_seconds:.0f}s")
        rows.append([name, round(acc, 2), round(f1, 2), round(train_seconds, 1), round(predict_seconds, 1)])

        # k-NN "training" just stores all 60,000 pictures (~380 MB file), so we don't save it
        if not name.startswith("knn"):
            joblib.dump(model, CHECKPOINT_DIR / f"{name}.joblib")

    with open(RESULTS_DIR / "baselines.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy_%", "macro_f1_%", "train_seconds", "predict_seconds"])
        writer.writerows(rows)
    print("Saved results/baselines.csv")