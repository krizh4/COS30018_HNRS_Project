import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).parent.parent / "data"

def get_loaders(batch_size=128, root=ROOT):
    # build train_set and test_set with datasets.MNIST
    #    train=True | train=False
    #    download=False (have already downloaded)
    #    transform=transforms.ToTensor()
    train_set = datasets.MNIST(root, train=True, download=False, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root, train=False, download=False, transform=transforms.ToTensor())

    # wrap each in DataLoader
    #    - train: shuffle=True, num_workers=4
    #    - test:  shuffle=False, batch_size=512, num_workers=2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # returning both
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_loaders()
    print(len(train_loader.dataset), len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)
    print(images.min().item(), images.max().item())
    print(labels[:8])

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(labels[i].item())
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("mnist_samples.png", dpi=150)
    plt.show()