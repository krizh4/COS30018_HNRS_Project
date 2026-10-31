"""
MNIST data loading + the CNN models.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# the data/ folder at the top of the repo. MNIST is saved in data/MNIST/raw
ROOT = Path(__file__).parent.parent / "data"

def get_loaders(batch_size=128, augment=False, root=ROOT):
    """
    Download MNIST (first time only) and return (train_loader, test_loader)

    augment = True randomly rotates, shifts and resizes the TRAINING pictures a little, so the model gets used to messy real handwriting. Test pictures are never changed
    """
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ToTensor(),
        ])
    else:
        train_tf = transforms.ToTensor()
    
    test_tf = transforms.ToTensor()          # test pictures stay exactly as they are

    train_set = datasets.MNIST(root, train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
    return train_loader, test_loader

class SimpleCNN(nn.Module):
    """LeNet-style CNN: two convolution blocks, then two fully connected layers."""

    def __init__(self):
        super().__init__()
        # Part 1: find patterns in the picture
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1x28x28  -> 32x28x28 (32 pattern maps)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x28x28 -> 32x14x14 (half size)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x14x14 -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 64x14x14 -> 64x7x7
        )
        # Part 2: turn the patterns into 10 scores
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 64x7x7 -> one list of 3136 numbers
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                           # one score per digit 0-9
        )

    def forward(self, x):
        """What happens when a batch of pictures goes through the model."""
        return self.classifier(self.features(x))

class BetterCNN(nn.Module):
    """Deeper CNN with Batch Normalisation (steadier training) and Dropout (less overfitting)"""

    def __init__(self, dropout=0.3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                         # -> 32x14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                         # -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),                                     # switch off 30% of neurons while training
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),                             # 10 digits (16 later for the extension)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    print("training pictures:", len(train_loader.dataset))
    print("test pictures:", len(test_loader.dataset))

    # take ONE batch and look at it

    images, labels = next(iter(train_loader))
    print("batch of images:", images.shape)    # torch.Size([128, 1, 28, 28])
    print("batch of labels:", labels.shape)    # torch.Size([128])
    print("pixel range:", images.min().item(), "to", images.max().item())
    print("first 10 labels:", labels[:10].tolist())

    # Push the batch through an UNTRAINED model
    model = SimpleCNN()
    scores = model(images)
    print("model output:", scores.shape)               # torch.Size([128, 10])
    print("guesses (untrained):", scores.argmax(dim=1)[:10].tolist())
    print("parameters to learn:", sum(p.numel() for p in model.parameters()))

    # show 16 of them with their labels

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")   # squeeze: (1,28,28) -> (28,28)
        ax.set_title(labels[i].item())
        ax.axis("off")
    plt.tight_layout()
    plt.show()
