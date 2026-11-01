"""
"Not a digit" pictures for the 11th class (label 10)

The digit models only know 0-9, so they must pick a digit even for a scribble.
Here we draw random junk (scribbles, dashes, blobs, crossed lines, coils),
run it through the SAME make_mnist_like() the pipeline uses, and label it 10.

    python -m models.junk       # show a few examples
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.preprocessing import make_mnist_like

NOT_A_DIGIT = 10
SIZE = 200          # draw on a 200x200 black canvas, then shrink to 28x28


def scribble(rng, img, pen):
    """A random tangle: many points joined by a thick line"""
    points = rng.integers(20, SIZE - 20, size=(rng.integers(6, 16), 2))
    cv2.polylines(img, [points.astype(np.int32)], False, 255, pen, cv2.LINE_AA)


def dash(rng, img, pen):
    """A flat line (straight or wavy) - lying down, so it never looks like a 1"""
    angle = np.radians(rng.uniform(-30, 30))
    length = rng.uniform(80, 180)
    xs = np.linspace(-length / 2, length / 2, 40)
    ys = rng.uniform(0, 12) * np.sin(xs / rng.uniform(10, 30))       # 0 = straight
    x = SIZE / 2 + xs * np.cos(angle) - ys * np.sin(angle)
    y = SIZE / 2 + xs * np.sin(angle) + ys * np.cos(angle)
    cv2.polylines(img, [np.stack([x, y], 1).astype(np.int32)], False, 255, pen, cv2.LINE_AA)


def blob(rng, img, pen):
    """A filled-in shape (someone colouring in)"""
    points = rng.integers(40, SIZE - 40, size=(rng.integers(4, 8), 2))
    cv2.fillPoly(img, [cv2.convexHull(points.astype(np.int32))], 255)


def crossed_lines(rng, img, pen):
    """3 to 6 lines crossing each other, like # or a crossed-out mistake"""
    for _ in range(rng.integers(3, 7)):
        p1, p2 = rng.integers(20, SIZE - 20, size=(2, 2))
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 255, pen, cv2.LINE_AA)


def coil(rng, img, pen):
    """Several overlapping loops"""
    for _ in range(rng.integers(3, 6)):
        centre = tuple(int(v) for v in rng.integers(60, SIZE - 60, size=2))
        axes = tuple(int(v) for v in rng.integers(20, 60, size=2))
        cv2.ellipse(img, centre, axes, rng.uniform(0, 180), 0, 360, 255, pen, cv2.LINE_AA)


def smooth(points, rounds=3):
    """Chaikin's corner cutting: turns a zig-zag into a smooth curve"""
    for _ in range(rounds):
        a, b = points[:-1], points[1:]
        points = np.concatenate([points[:1], np.stack([0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b], 1).reshape(-1, 2), points[-1:]])
    return points


def smooth_scribble(rng, img, pen):
    """A smooth, loopy pen path - the tricky kind that looks almost like a 3, 8 or 9"""
    points = rng.uniform(20, SIZE - 20, size=(rng.integers(7, 13), 2))
    cv2.polylines(img, [smooth(points).astype(np.int32)], False, 255, pen, cv2.LINE_AA)


def curls(rng, img, pen):
    """Joined-up loops like cursive 'llll' or 'eeee', going in a random direction"""
    loops = rng.uniform(2, 5)
    t = np.linspace(0, 2 * np.pi * loops, 200)
    radius = rng.uniform(20, 40)
    x = (t / t[-1] - 0.5) * rng.uniform(100, 170) + radius * np.cos(t)
    y = radius * rng.uniform(0.6, 1.4) * np.sin(t)
    angle = rng.uniform(0, np.pi)
    xr = SIZE / 2 + x * np.cos(angle) - y * np.sin(angle)
    yr = SIZE / 2 + x * np.sin(angle) + y * np.cos(angle)
    cv2.polylines(img, [np.stack([xr, yr], 1).astype(np.int32)], False, 255, pen, cv2.LINE_AA)


KINDS = [scribble, smooth_scribble, curls, dash, blob, crossed_lines, coil]
WEIGHTS = [0.2, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1]


def make_junk(rng):
    img = np.zeros((SIZE, SIZE), np.uint8)
    kind = KINDS[rng.choice(len(KINDS), p=WEIGHTS)]
    kind(rng, img, int(rng.integers(6, 16)))
    img = (img > 127).astype(np.uint8) * 255          # pure black/white, like preprocess() gives
    return make_mnist_like(img)


class JunkDigits(Dataset):
    """`count` junk pictures, all labelled 10. Same seed = same pictures (for a fair test set)"""

    def __init__(self, count, seed=0):
        rng = np.random.default_rng(seed)
        pictures = np.stack([make_junk(rng) for _ in range(count)])
        self.images = torch.from_numpy(pictures).unsqueeze(1)     # (count, 1, 28, 28) like MNIST tensors

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], NOT_A_DIGIT


class PipelineStyleMNIST(Dataset):
    """MNIST digits sent through make_mnist_like(), so they look like the digits the pipeline cuts out.

    Without this the model learns a shortcut: "processed by make_mnist_like = junk",
    and then calls every real photo digit junk.
    """

    def __init__(self, mnist):
        pictures = []
        for image in mnist.data.numpy():                                  # uint8 28x28, white on black
            big = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            pictures.append(make_mnist_like(((big > 127) * 255).astype(np.uint8)))
        self.images = torch.from_numpy(np.stack(pictures)).unsqueeze(1)
        self.labels = mnist.targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], int(self.labels[i])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = JunkDigits(32)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for ax, (image, _) in zip(axes.flat, data):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()