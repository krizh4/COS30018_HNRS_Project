"""MNIST data loading + the CNN models.Run this file directly to check the data loads: python models/cnn.py"""

from pathlib import Pathimport matplotlib.pyplot as plt
from torch import nnfrom torch.utils.data import DataLoaderfrom torchvision import datasets, transforms
# data/ folder at the top of the repo (MNIST raw files live in data/MNIST/raw)
ROOT = Path(__file__).parent.parent / "data"
def get_loaders(batch_size=128, augment=False, root=ROOT): 
    """Return (train_loader, test_loader) for MNIST. augment=True randomly rotates, shifts and scales the TRAINING pictures a little, so the model gets used to messy real handwriting. Test pictures are never changed. """ 
    if augment: train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), 
        scale=(0.85, 1.15)), 
        transforms.ToTensor(), 
        ]) 
    else: 
        train_tf = transforms.ToTensor() 
        train_set = datasets.MNIST(root, train=True, download=False, transform=train_tf) 
        test_set = datasets.MNIST(root, train=False, download=False, transform=transforms.ToTensor())

    # num_workers=2 loads batches in the background; set it to 0 if you get errors on Windows 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2) 
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, test_loader

class SimpleCNN(nn.Module): 
    """LeNet-style CNN: two convolution blocks, then two fully connected layers."""
    def __init__(self): super().__init__()
    self.features = nn.Sequential( nn.Conv2d(1, 32, kernel_size=3, padding=1), # 1x28x28 -> 32x28x28 
    nn.ReLU(), 
    nn.MaxPool2d(2), # -> 32x14x14 
    nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 64x14x14 
    nn.ReLU(), 
    nn.MaxPool2d(2), # -> 64x7x7 ) 
    self.classifier = nn.Sequential(
        nn.Flatten(), # -> 3136 numbers 
        nn.Linear(64 * 7 * 7, 128), 
        nn.ReLU(), 
        nn.Linear(128, 10), # one score per digit 0-9 ) 
        
    def forward(self, x): 
        return self.classifier(self.features(x))

class BetterCNN(nn.Module): 
    """Deeper CNN with Batch Normalisation (steadier training) and Dropout (less overfitting).""" 
    def __init__(self, dropout=0.3, num_classes=10): 
        super().__init__() 
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.MaxPool2d(2), # -> 32x14x14 
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), 
            nn.MaxPool2d(2), # -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(dropout), 
            nn.Linear(64 * 7 * 7, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, num_classes), # 10 digits (16 for the extension) 
        ) 
            
    def forward(self, x): 
        return self.classifier(self.features(x))
        
if __name__ == "__main__": 
    train_loader, test_loader = get_loaders() 
    print(len(train_loader.dataset), len(test_loader.dataset))
    images, labels = next(iter(train_loader)) 
    print(images.shape, labels.shape) # torch.Size([128, 1, 28, 28]) 
    torch.Size([128]) 
    print(images.min().item(), images.max().item()) 
    fig, axes = plt.subplots(4, 4, figsize=(6, 6)) 
    for i, ax in enumerate(axes.flat): 
        ax.imshow(images[i].squeeze(), cmap="gray") 
        ax.set_title(labels[i].item()) 
        ax.axis("off") 
    plt.tight_layout() 
    plt.show()

