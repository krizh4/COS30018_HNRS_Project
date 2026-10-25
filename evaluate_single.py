"""
Task 4, part 1 - evaluate every trained model on the 10,000 MNIST test digits.

python evaluate_single.py

Output:
    results/single_digit.csv         one row per model
    results/confusion_<name>.png     which digits get mixed up
    results/report_<name>.txt        precision / recall / F1 for each digit
"""
import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report,
                             precision_recall_fscore_support)
from torchvision import datasets

from predictor import available_models, load_model, predict_digits

RESULTS_DIR = Path("results")

if __name__ == "__main__":
    test = datasets.MNIST("data", train=False, download=False)
    X = test.data.numpy().astype(np.float32) / 255.0  # (10000, 28, 28), values 0..1
    y = test.targets.numpy()
    
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    
    for name in available_models():
        model = load_model(name)
        start = time.time()
        predictions = []
        
        # 1000 at a time to save memory
        for i in range(0, len(X), 1000):
            labels, _ = predict_digits(model, X[i:i + 1000])
            predictions += labels
            
        ms_per_image = 1000 * (time.time() - start) / len(X)
        acc = 100 * accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average="macro")
        
        print(f"{name:20s} accuracy {acc:.2f}% F1 {100 * f1:.2f} {ms_per_image:.3f} ms/image")
        
        rows.append([name, round(acc, 2), round(100 * precision, 2), 
                     round(100 * recall, 2), round(100 * f1, 2), round(ms_per_image, 3)])
        
        with open(RESULTS_DIR / f"report_{name}.txt", "w") as f:
            f.write(classification_report(y, predictions, digits=4))
            
        ConfusionMatrixDisplay.from_predictions(y, predictions, cmap="Blues", colorbar=False)
        plt.title(f"{name} confusion matrix (rows = true digit)")
        plt.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=120, bbox_inches="tight")
        plt.close()
        
    with open(RESULTS_DIR / "single_digit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy_%", "precision_%", "recall_%", "f1_%", "ms_per_image"])
        writer.writerows(rows)
        
    print("Saved results/single_digit.csv and one confusion matrix per model")