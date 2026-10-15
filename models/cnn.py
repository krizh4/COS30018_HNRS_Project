"""
MNIST data loading + the CNN models.

Run this file directly to check the data loads:  python models/cnn.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# the data/ folder at the top of the repo. MNIST is saved in data/MNIST/raw
ROOT = Path(__file__).parent.parent / "data"

def get_loaders(batch_size=128, root=ROOT):
    """Download MNIST (first time only) and return (train_loader, test_loader)."""

    # ToTensor turns each 28x28 picture (pixel values 0-255)
    # into a tensor of shape (1, 28, 28) with values 0.0-1.0
    to_tensor = transforms.ToTensor()

    # train=True  -> the 60,000 training pictures
    # train=False -> the 10,000 test pictures the model never learns from
    # download=True fetches the files if they are not already in root/MNIST/raw
    train_set = datasets.MNIST(root, train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST(root, train=False, download=True, transform=to_tensor)

    # A dataLoader hands out the pictures in batches.
    # shuffle = True mixes the training order every epoch so the model can't
    # learn anything from the order. test data never needs shuffling.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)
    return train_loader, test_loader

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

    # show 16 of them with their labels
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")   # squeeze: (1,28,28) -> (28,28)
        ax.set_title(labels[i].item())
        ax.axis("off")
    plt.tight_layout()
    plt.show()
