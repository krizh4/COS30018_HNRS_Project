"""
Handwriting-style pictures of the maths signs  +  -  x  ÷  (  )  for the extension

MNIST only has digits, so we draw the signs ourselves with random wobble, size,
slant and pen thickness, then run them through the SAME make_mnist_like() the
pipeline uses (so they look like the marks the pipeline cuts out)

Labels follow on from cnn_junk: 0-9 digits, 10 = not a digit, 11.. = signs

    python -m models.symbols        shows examples of every sign
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.preprocessing import make_mnist_like

SIGNS = ["+", "-", "×", "÷", "(", ")"]
FIRST_SIGN_LABEL = 11                      # "+" is 11, "-" is 12, ... ")" is 16
SIZE = 200                                 # draw on a 200x200 black canvas


def wobbly_line(rng, img, p1, p2, pen):
    """A hand-drawn straight line: slightly bent, never perfectly straight"""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    middle = (p1 + p2) / 2
    normal = np.array([p1[1] - p2[1], p2[0] - p1[0]])          # at right angles to the line
    bend = middle + normal * rng.uniform(-0.08, 0.08)
    t = np.linspace(0, 1, 30)[:, None]
    curve = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * bend + t ** 2 * p2    # quadratic Bezier
    cv2.polylines(img, [curve.astype(np.int32)], False, 255, pen, cv2.LINE_AA)


def at_angle(centre, length, degrees):
    """The two ends of a line of `length` through `centre`"""
    a = np.radians(degrees)
    d = np.array([np.cos(a), np.sin(a)]) * length / 2
    return centre - d, centre + d


def draw_plus(rng, img, pen):
    c = np.array([SIZE / 2, SIZE / 2]) + rng.uniform(-10, 10, 2)
    tilt = rng.uniform(-12, 12)
    across, down = rng.uniform(60, 160), rng.uniform(60, 160)
    # the crossing point is not always in the exact middle of each stroke
    wobbly_line(rng, img, *at_angle(c + rng.uniform(-15, 15, 2) * [1, 0], across, tilt), pen)
    wobbly_line(rng, img, *at_angle(c + rng.uniform(-15, 15, 2) * [0, 1], down, 90 + tilt), pen)


def draw_minus(rng, img, pen):
    c = np.array([SIZE / 2, SIZE / 2])
    wobbly_line(rng, img, *at_angle(c, rng.uniform(40, 170), rng.uniform(-10, 10)), pen)   # short and thick to long and thin


def draw_times(rng, img, pen):
    c = np.array([SIZE / 2, SIZE / 2])
    tilt = rng.uniform(-12, 12)
    wobbly_line(rng, img, *at_angle(c, rng.uniform(60, 160), 45 + tilt + rng.uniform(-8, 8)), pen)
    wobbly_line(rng, img, *at_angle(c, rng.uniform(60, 160), 135 + tilt + rng.uniform(-8, 8)), pen)


def draw_divide(rng, img, pen):
    c = np.array([SIZE / 2, SIZE / 2])
    length = rng.uniform(60, 160)
    wobbly_line(rng, img, *at_angle(c, length, rng.uniform(-8, 8)), pen)
    radius = int(max(3, pen * rng.uniform(0.6, 1.1)))
    for side in (-1, 1):                                       # one dot above, one below
        gap = length * rng.uniform(0.25, 0.45)
        dot = c + [rng.uniform(-0.1, 0.1) * length, side * gap]
        cv2.circle(img, tuple(int(v) for v in dot), radius, 255, -1, cv2.LINE_AA)


def draw_bracket(rng, img, pen, opening):
    """( is the left part of a tall ellipse, ) is the right part"""
    centre = (int(SIZE / 2 + (40 if opening else -40) * rng.uniform(0.6, 1.2)), SIZE // 2)
    axes = (int(rng.uniform(15, 70)), int(rng.uniform(70, 95)))   # from gentle to very curved brackets
    sweep = rng.uniform(50, 75)                                # how far the arc goes up and down
    start, end = (180 - sweep, 180 + sweep) if opening else (-sweep, sweep)
    cv2.ellipse(img, centre, axes, rng.uniform(-10, 10), start, end, 255, pen, cv2.LINE_AA)


DRAW = [draw_plus, draw_minus, draw_times, draw_divide,
        lambda rng, img, pen: draw_bracket(rng, img, pen, True),
        lambda rng, img, pen: draw_bracket(rng, img, pen, False)]


def make_sign(rng, index):
    img = np.zeros((SIZE, SIZE), np.uint8)
    DRAW[index](rng, img, int(rng.integers(6, 16)))
    img = (img > 127).astype(np.uint8) * 255          # pure black/white, like preprocess() gives
    return make_mnist_like(img)


class MathSymbols(Dataset):
    """`per_sign` pictures of each sign (twice that for brackets). Same seed = same pictures"""

    def __init__(self, per_sign, seed=0):
        rng = np.random.default_rng(seed)
        pictures, labels = [], []
        for index in range(len(SIGNS)):
            count = per_sign * (2 if SIGNS[index] in "()" else 1)    # brackets look like a 1, so show twice as many
            for _ in range(count):
                pictures.append(make_sign(rng, index))
                labels.append(FIRST_SIGN_LABEL + index)
        self.images = torch.from_numpy(np.stack(pictures)).unsqueeze(1)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = MathSymbols(6)
    fig, axes = plt.subplots(len(SIGNS), 6, figsize=(7, 7))
    for ax, (image, label) in zip(axes.flat, data):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(SIGNS[label - FIRST_SIGN_LABEL])
        ax.axis("off")
    plt.tight_layout()
    plt.show()