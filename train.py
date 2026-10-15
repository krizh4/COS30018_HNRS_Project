"""
Train a neural network on MNIST, test it after every epoch, and save it.

    python train.py cnn

Output:
    checkpoints/<name>.pt          the trained weights (the GUI loads these later)
    results/<name>_log.csv         loss and accuracy after every epoch (for the report)
"""
import csv
import sys
import time
from pathlib import Path

import torch
from torch import nn

from models.cnn import BetterCNN, SimpleCNN, get_loaders
from models.mlp import MLP
from models.vit import TinyViT

# ---- Settings (hyper-parameters) ----
EPOCHS = 5                 # how many full passes over the 60,000 training pictures
BATCH_SIZE = 128           # pictures per weight update
LEARNING_RATE = 0.001      # how big each weight update is
SEED = 42                  # same seed = same result every run, so experiments are fair

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

MODELS = {
    "mlp": MLP,
    "cnn": SimpleCNN,
    "cnn_bn": BetterCNN,
    "vit": TinyViT,
}


def get_device():
    """Use the NVIDIA GPU if PyTorch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, device):
    """Return (average loss, accuracy %) on data the model does NOT learn from."""
    model.eval()                                   # testing mode
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():                          # no learning here, so skip gradients (faster)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            total_loss += loss_fn(scores, labels).item() * len(labels)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, 100.0 * correct / count


def train(name, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Train model `name`, save it, and return its final test accuracy."""
    torch.manual_seed(SEED)
    device = get_device()
    model = MODELS[name]().to(device)              # build the model and move it to the GPU
    train_loader, test_loader = get_loaders(batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()                # measures how wrong the 10 scores are
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # the rule for updating weights

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / f"{name}_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "seconds"])

    print(f"Training {name} on {device} for {epochs} epochs")
    test_acc = 0.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()                              # learning mode
        total_loss, correct, count = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            scores = model(images)                 # 1. guess
            loss = loss_fn(scores, labels)         # 2. how wrong?
            optimizer.zero_grad()                  #    (clear last batch's gradients)
            loss.backward()                        # 3. which way to nudge each weight (backpropagation)
            optimizer.step()                       # 4. nudge

            total_loss += loss.item() * len(labels)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            count += len(labels)

        train_loss, train_acc = total_loss / count, 100.0 * correct / count
        test_loss, test_acc = evaluate(model, test_loader, device)
        seconds = time.time() - start
        print(f"epoch {epoch}/{epochs}  train acc {train_acc:.2f}%  test acc {test_acc:.2f}%  ({seconds:.0f}s)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 4), round(train_acc, 2),
                                    round(test_loss, 4), round(test_acc, 2), round(seconds, 1)])

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name}.pt")
    print(f"Saved checkpoints/{name}.pt  (final test accuracy {test_acc:.2f}%)")
    return test_acc


def load_trained_model(name):
    """Rebuild model `name` and load its saved weights, ready to make predictions."""
    model = MODELS[name]()
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    train(model_name)